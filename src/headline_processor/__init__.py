import os
import pandas as pd
from .processor import Processor
from typing import Optional
from dotenv import load_dotenv

def get_or_process_file(
    file_path: str, 
    provider: str = "gemini", 
    session_limit: Optional[int] = None, 
    headline_limit: Optional[int] = None,
    **predictor_kwargs
) -> Processor:
    """
    Checks if processed data for the given model exists. 
    If yes, loads it. If no, processes it and saves it.
    """
    load_dotenv()
    # Initialize processor to get model name for the cache file
    processor = Processor(predictor_provider=provider, **predictor_kwargs)
    model_name = processor.predictor.get_model_name().replace("/", "_")
    
    # Create a unique cache name based on input file and model
    input_basename = os.path.basename(file_path).replace(".parquet", "")
    cache_path = f"data/{input_basename}_processed_{model_name}.parquet"
    
    if os.path.exists(cache_path):
        print(f"Loading existing processed data from {cache_path}...")
        df = pd.read_parquet(cache_path)
        processor.load_from_df(df)
    else:
        print(f"No existing data found. Processing {file_path}...")
        processor.load_data(file_path)
        # Process everything unless limits are specified
        processor.process_headlines(session_limit=session_limit, headline_limit_per_session=headline_limit)
        
        # Save the results
        results_df = processor.get_results_df()
        if not results_df.empty:
            print(f"Saving processed data to {cache_path}...")
            results_df.to_parquet(cache_path)
    
    return processor

def get_company_analysis(processor: Processor, company_name: Optional[str] = None) -> pd.DataFrame:
    """
    Returns the company-centric report from a processed instance.
    """
    return processor.get_company_report(company_name)
