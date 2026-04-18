import pandas as pd
from .models import Headline, HeadlineCollection
from src.predictor.predictor import get_predictor
from typing import List, Optional

class Processor:
    def __init__(self, predictor_provider: str = "gemini", **predictor_kwargs):
        self.collection = HeadlineCollection()
        self.predictor = get_predictor(predictor_provider, **predictor_kwargs)

    def load_data(self, file_path: str):
        df = pd.read_parquet(file_path)
        for _, row in df.iterrows():
            h = Headline(
                session=row['session'],
                bar_ix=row['bar_ix'],
                text=row['headline']
            )
            self.collection.add_headline(h)

    def load_from_df(self, df: pd.DataFrame):
        """
        Loads already processed results from a DataFrame.
        """
        for _, row in df.iterrows():
            h = Headline(
                session=row['session'],
                bar_ix=row['bar_ix'],
                text=row['headline'],
                company=row.get('company'),
                sentiment=row.get('sentiment'),
                confidence=row.get('confidence'),
                reasoning=row.get('reasoning')
            )
            self.collection.add_headline(h)

    def process_headlines(self, session_limit: Optional[int] = None, headline_limit_per_session: Optional[int] = None):
        """
        Processes headlines to identify companies and sentiment.
        """
        sessions = sorted(list(self.collection.sessions.keys()))
        if session_limit:
            sessions = sessions[:session_limit]

        for session_id in sessions:
            headlines = sorted(self.collection.get_session_headlines(session_id), key=lambda x: x.bar_ix)
            if headline_limit_per_session:
                headlines = headlines[:headline_limit_per_session]
            
            print(f"Processing session {session_id} ({len(headlines)} headlines)...")
            
            for h in headlines:
                # 1. Initial prediction to get company
                h.predict_sentiment(self.predictor)
                
                # If company was identified, we might want to re-run with history
                if h.company:
                    # Re-index to ensure history lookup works
                    self.collection._index_by_company(h)
                    
                    history = self.collection.get_company_history(h.company, h.session, h.bar_ix)
                    if history:
                        # Re-predict with history context
                        h.predict_sentiment(self.predictor, history=history)

    def get_results_df(self):
        data = []
        for h in self.collection.headlines:
            if h.sentiment: # Only return processed ones
                data.append({
                    "session": h.session,
                    "bar_ix": h.bar_ix,
                    "headline": h.text,
                    "company": h.company,
                    "sentiment": h.sentiment,
                    "confidence": h.confidence,
                    "reasoning": h.reasoning
                })
        return pd.DataFrame(data)

    def get_company_report(self, company_name: Optional[str] = None):
        """
        Returns a detailed report grouped by company, showing the timeline of 
        headlines and predicted movements across all sessions and bars.
        """
        data = []
        # If no specific company, get all identified companies
        target_companies = [company_name] if company_name else sorted(list(self.collection.companies.keys()))
        
        for co in target_companies:
            headlines = self.collection.companies.get(co, [])
            # Sort by session and then bar_ix for a chronological timeline per company
            sorted_h = sorted(headlines, key=lambda x: (x.session, x.bar_ix))
            
            for h in sorted_h:
                data.append({
                    "company": co,
                    "session": h.session,
                    "bar_ix": h.bar_ix,
                    "sentiment": h.sentiment,
                    "confidence": h.confidence,
                    "headline": h.text,
                    "reasoning": h.reasoning
                })
        
        return pd.DataFrame(data)
