import json
from typing import List, Optional, Dict
from dataclasses import dataclass, field

@dataclass
class Headline:
    session: int
    bar_ix: int
    text: str
    company: Optional[str] = None
    granular_sector: Optional[str] = None
    sector: Optional[str] = None
    sentiment: Optional[str] = None  # "buy" or "sell"
    sentiment_score: Optional[float] = None # -1.0 to 1.0
    confidence: Optional[float] = None # 0.0 to 1.0
    reasoning: Optional[str] = None
    history_context: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"Headline(session={self.session}, bar_ix={self.bar_ix}, company={self.company}, sector={self.sector}, score={self.sentiment_score})"

    @staticmethod
    def predict_batch(predictor, headlines: List['Headline'], history_map: Dict[str, List['Headline']] = None):
        """
        Predicts sentiment for a batch of headlines in one call.
        """
        if not headlines:
            return

        # Simple company and granular sector extraction
        for h in headlines:
            words = h.text.split()
            if h.company is None and len(words) > 0:
                h.company = words[0]
            if len(words) > 1:
                # If 3rd word exists, is capitalized, and is NOT a common action/title, it's part of the sector
                excluded = {"Chief", "CEO", "CFO", "CTO", "COO", "names", "reports", "secures", "sees", "faces", "opens", "Chairman", "Founder", "President", "Officer"}
                if len(words) > 2 and words[2][0].isupper() and words[2] not in excluded:
                    h.granular_sector = " ".join(words[1:3])
                else:
                    h.granular_sector = words[1]

        # Check if predictor is FinBERT for optimized local batching
        from src.predictor.predictor import FinBertPredictor
        if isinstance(predictor, FinBertPredictor):
            texts = [h.text for h in headlines]
            results = predictor.predict_batch_json(texts)
            for h, res in zip(headlines, results):
                h.sentiment_score = res.get("sentiment_score")
                h.sentiment = res.get("sentiment")
                h.confidence = res.get("confidence")
                h.reasoning = res.get("reasoning")
            return

        # Fallback for LLMs (JSON Batching)
        headlines_data = []
        for i, h in enumerate(headlines):
            headlines_data.append({
                "id": i,
                "text": h.text,
                "session": h.session,
                "bar_ix": h.bar_ix
            })

        prompt = f"""
        Analyze the following financial headlines for a synthetic stock market challenge.
        For each headline:
        1. Identify the primary company mentioned. If none, use "None".
        2. Determine sentiment: "buy" or "sell".
        3. Provide confidence (0.0 to 1.0).
        4. Provide brief reasoning.

        Headlines:
        {json.dumps(headlines_data, indent=2)}

        Respond with a JSON object containing a list called "results" with the same number of items as the input.
        Each item should have: "id", "company", "sentiment", "confidence", "reasoning".
        """

        try:
            response_data = predictor.predict_json(prompt)
            results = response_data.get("results", [])
            
            for res in results:
                idx = res.get("id")
                if idx is not None and 0 <= idx < len(headlines):
                    h = headlines[idx]
                    h.company = res.get("company")
                    if h.company == "None":
                        h.company = None
                    h.sentiment = res.get("sentiment")
                    h.confidence = res.get("confidence")
                    h.reasoning = res.get("reasoning")
        except Exception as e:
            print(f"Error in batch prediction: {e}")
            # Fallback to individual if batch fails? Or just mark as failed.
            for h in headlines:
                if h.sentiment is None:
                    h.reasoning = f"Batch prediction failed: {str(e)}"

    def predict_sentiment(self, predictor, history: List['Headline'] = None):
        """
        Predicts the sentiment (buy/sell) and identifies the company for this headline.
        """
        context_str = ""
        if history:
            context_str = "Historical headlines for this company:\n"
            for h in history:
                context_str += f"- Session {h.session}, Bar {h.bar_ix}: {h.text} (Sentiment: {h.sentiment}, Confidence: {h.confidence})\n"
        
        prompt = f"""
        Analyze the following financial headline for a synthetic stock market challenge.
        
        Headline: "{self.text}"
        Session: {self.session}
        Bar Index: {self.bar_ix}
        
        {context_str}
        
        Task:
        1. Identify the primary company mentioned in the headline. If no company is mentioned, state "None".
        2. Determine the sentiment of the headline: "buy" or "sell". 
           Even if the news seems minor, you MUST choose the most likely direction.
           "buy" means the stock price is more likely to go up.
           "sell" means the stock price is more likely to go down.
           DO NOT use "neutral".
        3. Provide a confidence score between 0.0 and 1.0 for your prediction.
        4. Provide a brief reasoning.
        
        Respond ONLY in the following JSON format:
        {{
            "company": "Company Name",
            "sentiment": "buy/sell",
            "confidence": 0.85,
            "reasoning": "Brief explanation"
        }}
        """
        
        try:
            raw_response = predictor.predict(prompt)
            start = raw_response.find('{')
            end = raw_response.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = raw_response[start:end]
                data = json.loads(json_str)
                self.company = data.get("company")
                if self.company == "None":
                    self.company = None
                self.sentiment = data.get("sentiment")
                self.confidence = data.get("confidence")
                self.reasoning = data.get("reasoning")
            else:
                self.reasoning = f"Failed to parse JSON from response: {raw_response}"
        except Exception as e:
            self.reasoning = f"Error during prediction: {str(e)}"

class HeadlineCollection:
    def __init__(self):
        self.headlines: List[Headline] = []
        self.sessions: Dict[int, List[Headline]] = {}
        self.companies: Dict[str, List[Headline]] = {}

    def add_headline(self, h: Headline):
        self.headlines.append(h)
        if h.session not in self.sessions:
            self.sessions[h.session] = []
        self.sessions[h.session].append(h)
        
        if h.company:
            self._index_by_company(h)

    def _index_by_company(self, h: Headline):
        if h.company not in self.companies:
            self.companies[h.company] = []
        self.companies[h.company].append(h)

    def get_session_headlines(self, session: int) -> List[Headline]:
        return self.sessions.get(session, [])

    def get_company_history(self, company: str, session: int, current_bar: int, global_history: bool = True) -> List[Headline]:
        """
        Returns headlines for a company that occurred before the current point.
        If global_history is True, it looks across all sessions. 
        If False, it only looks within the current session.
        """
        if company not in self.companies:
            return []
        
        if global_history:
            # All headlines from earlier sessions OR same session but earlier bar
            history = [
                h for h in self.companies[company] 
                if h.session < session or (h.session == session and h.bar_ix < current_bar)
            ]
        else:
            # Only same session, earlier bar
            history = [
                h for h in self.companies[company] 
                if h.session == session and h.bar_ix < current_bar
            ]
            
        return sorted(history, key=lambda x: (x.session, x.bar_ix))
