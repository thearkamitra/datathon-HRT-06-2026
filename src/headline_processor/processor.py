import pandas as pd
from .models import Headline, HeadlineCollection
from src.predictor.predictor import get_predictor
from typing import List, Optional, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    def load_processed_data(self, df: pd.DataFrame):
        """
        Updates the current collection with already processed results from a DataFrame.
        """
        # Create a lookup map for faster updates
        lookup: Dict[str, Headline] = {f"{h.session}-{h.bar_ix}-{h.text}": h for h in self.collection.headlines}
        
        for _, row in df.iterrows():
            key = f"{row['session']}-{row['bar_ix']}-{row['headline']}"
            if key in lookup:
                h = lookup[key]
                h.company = row.get('company')
                h.sentiment = row.get('sentiment')
                h.confidence = row.get('confidence')
                h.reasoning = row.get('reasoning')
                
                # Also ensure it's indexed in the companies dict
                if h.company:
                    self.collection._index_by_company(h)

    def _process_single_session(self, session_id: int, headline_limit: Optional[int] = None):
        headlines = sorted(self.collection.get_session_headlines(session_id), key=lambda x: x.bar_ix)
        if headline_limit:
            headlines = headlines[:headline_limit]
        
        for h in headlines:
            # Skip if already processed
            if h.sentiment is not None:
                continue
                
            # 1. Initial prediction to get company
            h.predict_sentiment(self.predictor)
            
            # If company was identified, we might want to re-run with history
            if h.company:
                # Re-index to ensure history lookup works
                self.collection._index_by_company(h)

                history = self.collection.get_company_history(h.company, h.session, h.bar_ix, global_history=True)
                if history:
                    # Re-predict with history context
                    h.predict_sentiment(self.predictor, history=history)

        return session_id

    def process_headlines(self, session_limit: Optional[int] = None, headline_limit_per_session: Optional[int] = None, max_workers: int = 5):
        """
        Processes headlines to identify companies and sentiment.
        Uses ThreadPoolExecutor for concurrent session processing.
        """
        sessions = sorted(list(self.collection.sessions.keys()))
        if session_limit:
            sessions = sessions[:session_limit]

        # Filter sessions that actually need processing
        sessions_to_process = []
        for s in sessions:
            headlines = self.collection.get_session_headlines(s)
            if any(h.sentiment is None for h in headlines):
                sessions_to_process.append(s)

        if not sessions_to_process:
            print("All requested sessions are already processed.")
            return

        print(f"Processing {len(sessions_to_process)} sessions with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_single_session, s, headline_limit_per_session) for s in sessions_to_process]
            for _ in tqdm(as_completed(futures), total=len(sessions_to_process), desc="Processing Sessions"):
                pass

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
            associated_sessions = sorted(list(set(h.session for h in headlines)))
            
            # Sort by session and then bar_ix for a chronological timeline per company
            sorted_h = sorted(headlines, key=lambda x: (x.session, x.bar_ix))
            
            for h in sorted_h:
                data.append({
                    "company": co,
                    "associated_sessions": associated_sessions,
                    "session": h.session,
                    "bar_ix": h.bar_ix,
                    "sentiment": h.sentiment,
                    "confidence": h.confidence,
                    "headline": h.text,
                    "reasoning": h.reasoning
                })
        
        return pd.DataFrame(data)
