import pandas as pd
import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Repo root: .../datathon-HRT-06-2026
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from src.predictor.predictor import get_predictor

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Update existing sentiment CSVs with sector mappings.")
    parser.add_argument("--files", nargs="+", required=True, help="CSV files to update (e.g. data/sentiments_*.csv)")
    parser.add_argument("--provider", type=str, default="gemini", help="LLM provider for sector mapping")
    args = parser.parse_args()

    predictor = get_predictor(args.provider)
    
    # 1. Collect all dataframes and unique granular sectors
    dfs = {}
    all_unique_granular = set()

    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        print(f"Reading {file_path}...")
        df = pd.read_csv(file_path)
        
        # Extract granular sector if not present
        if 'granular_sector' not in df.columns:
            def extract_granular(x):
                if not isinstance(x, str): return "None"
                words = x.split()
                if len(words) < 2: return "None"
                
                # Exclude common non-sector words that might be capitalized
                excluded = {"Chief", "CEO", "CFO", "CTO", "names", "reports", "secures", "sees", "faces", "opens"}
                
                # If 3rd word exists, is capitalized, and is NOT in the excluded list, 
                # it is part of the sector (e.g. "Retail Group")
                if len(words) > 2 and words[2][0].isupper() and words[2] not in excluded:
                    return " ".join(words[1:3])
                return words[1]
            
            df['granular_sector'] = df['headline'].apply(extract_granular)
        
        # Ensure sector column exists and is string type
        if 'sector' not in df.columns:
            df['sector'] = None
        df['sector'] = df['sector'].astype(object)
            
        dfs[file_path] = df
        
        # Only map those that don't have a broad sector yet
        to_map = df[df['sector'].isna()]['granular_sector'].unique()
        all_unique_granular.update([s for s in to_map if s != "None"])

    if not all_unique_granular:
        print("No new granular sectors to map.")
        return

    # 2. Map unique granular sectors to broad sectors in ONE call
    print(f"Mapping {len(all_unique_granular)} unique granular sectors using {args.provider}...")
    
    prompt = f"""
    Map the following granular industrial sectors to broad categories 
    (e.g., Technology, Healthcare, Energy, Finance, Consumer Goods, Industrials, etc.).
    
    Sectors: {", ".join(sorted(list(all_unique_granular)))}
    
    Respond with a JSON object where keys are granular sectors and values are broad categories.
    Example: {{"Biosciences": "Healthcare", "Renewables": "Energy"}}
    """
    
    try:
        mapping = predictor.predict_json(prompt)
        print("Mapping received:")
        for k, v in mapping.items():
            print(f"  {k} -> {v}")
            
        # 3. Update dataframes and save
        for file_path, df in dfs.items():
            # Apply mapping to rows where sector is missing
            mask = df['sector'].isna()
            df.loc[mask, 'sector'] = df.loc[mask, 'granular_sector'].map(mapping)
            
            # Reorder columns to match the new standard
            cols = ['session', 'bar_ix', 'headline', 'company', 'granular_sector', 'sector', 'sentiment', 'sentiment_score', 'confidence']
            # Only keep columns that exist (in case reasoning was missing)
            existing_cols = [c for c in cols if c in df.columns]
            df = df[existing_cols]
            
            print(f"Saving updated {file_path}...")
            df.to_csv(file_path, index=False)
            
    except Exception as e:
        print(f"Error mapping sectors: {e}")

if __name__ == "__main__":
    main()
