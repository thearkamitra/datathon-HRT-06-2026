import json
from typing import List, Optional, Dict
from dataclasses import dataclass, field

@dataclass
class Headline:
    session: int
    bar_ix: int
    text: str
    company: Optional[str] = None
    sentiment: Optional[str] = None  # "buy" or "sell"
    confidence: Optional[float] = None # 0.0 to 1.0
    reasoning: Optional[str] = None
    history_context: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"Headline(session={self.session}, bar_ix={self.bar_ix}, company={self.company}, sentiment={self.sentiment}, conf={self.confidence})"

    def predict_sentiment(self, predictor, history: List['Headline'] = None):
        """
        Predicts the sentiment (buy/sell) and identifies the company for this headline.
        """
        context_str = ""
        if history:
            context_str = "Previous headlines for this company:\n"
            for h in history:
                context_str += f"- Bar {h.bar_ix}: {h.text} (Sentiment: {h.sentiment}, Confidence: {h.confidence})\n"
        
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

    def get_company_history(self, company: str, session: int, current_bar: int) -> List[Headline]:
        """Returns headlines for a company in a session that occurred before current_bar."""
        if company not in self.companies:
            return []
        
        history = [
            h for h in self.companies[company] 
            if h.session == session and h.bar_ix < current_bar
        ]
        return sorted(history, key=lambda x: x.bar_ix)
