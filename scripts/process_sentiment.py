import pandas as pd
import argparse
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Repo root: .../datathon-HRT-06-2026
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from src.predictor.predictor import FinBertPredictor

def main():
    parser = argparse.ArgumentParser(description="Batch process sentiments for a parquet file.")
    parser.add_argument("--input", type=str, required=True, help="Input parquet file (e.g. data/headlines_seen_train.parquet)")
    parser.add_argument("--output", type=str, required=True, help="Output csv file (e.g. data/sentiments_train.csv)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for FinBERT (default: 64)")
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    total_rows = len(df)
    print(f"Total headlines to process: {total_rows}")

    print("Initializing FinBERT with MPS/Metal acceleration...")
    predictor = FinBertPredictor()
    
    headlines = df['headline'].tolist()
    results = []

    # Process in batches with progress bar
    print(f"Processing in batches of {args.batch_size}...")
    for i in tqdm(range(0, total_rows, args.batch_size), desc="Sentiment Analysis"):
        batch_texts = headlines[i:i + args.batch_size]
        batch_results = predictor.predict_batch_json(batch_texts)
        results.extend(batch_results)

    # Combine back into a dataframe
    results_df = pd.DataFrame(results)
    
    # Add identifiers and original headlines
    final_df = pd.concat([
        df[['session', 'bar_ix', 'headline']].reset_index(drop=True),
        results_df
    ], axis=1)

    # Use first word for company name, second word (or second and third) for granular sector
    final_df['company'] = final_df['headline'].apply(lambda x: x.split()[0] if x else "None")
    
    def extract_granular(x):
        if not x: return "None"
        words = x.split()
        if len(words) < 2: return "None"
        excluded = {"Chief", "CEO", "CFO", "CTO", "names", "reports", "secures", "sees", "faces", "opens"}
        if len(words) > 2 and words[2][0].isupper() and words[2] not in excluded:
            return " ".join(words[1:3])
        return words[1]
    
    final_df['granular_sector'] = final_df['headline'].apply(extract_granular)
    final_df['sector'] = None # Will be mapped later

    # Reorder columns
    cols = ['session', 'bar_ix', 'headline', 'company', 'granular_sector', 'sector', 'sentiment', 'sentiment_score', 'confidence']
    final_df = final_df[cols]

    print(f"Saving {len(final_df)} results to {args.output}...")
    final_df.to_csv(args.output, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
