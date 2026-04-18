import os
import argparse
from dotenv import load_dotenv
from src.headline_processor import get_or_process_file, get_company_analysis

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Financial Headline Sentiment Analysis")
    parser.add_argument("--provider", type=str, default="gemini", choices=["gemini", "ollama"], help="LLM provider")
    parser.add_argument("--model", type=str, help="Model name (optional)")
    parser.add_argument("--sessions", type=int, help="Number of sessions to process (default: all sessions in file)")
    parser.add_argument("--limit", type=int, help="Headlines per session limit (default: all headlines in session)")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of headlines to batch in one LLM call (default: 5)")
    parser.add_argument("--input", type=str, default="data/headlines_seen_train.parquet", help="Input parquet file")
    parser.add_argument("--company", type=str, help="Specific company to report on (default: all identified companies)")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent workers (default: 5)")
    
    args = parser.parse_args()
    
    predictor_kwargs = {}
    if args.model:
        predictor_kwargs["model_name"] = args.model
        
    try:
        # Returns the processor instance with all data (loads from cache if available)
        processor = get_or_process_file(
            file_path=args.input,
            provider=args.provider,
            session_limit=args.sessions,
            headline_limit=args.limit,
            max_workers=args.workers,
            batch_size=args.batch_size,
            **predictor_kwargs
        )
        
        print("\n--- Company Analysis Report ---")
        report = get_company_analysis(processor, args.company)
        
        if not report.empty:
            # Displaying grouped results
            for company, group in report.groupby("company"):
                sessions = group.iloc[0]["associated_sessions"]
                print(f"\nCompany: {company}")
                print(f"Associated Sessions: {sessions}")
                print("-" * (9 + len(company)))
                for _, row in group.iterrows():
                    move = "▲ BUY" if row['sentiment'] == 'buy' else "▼ SELL"
                    conf = f"({row['confidence']*100:3.0f}%)" if row['confidence'] is not None else ""
                    print(f"S{row['session']} | Bar {row['bar_ix']:3} | {move:7} {conf:6} | {row['headline']}")
        else:
            print("No company data found or processed.")
        
        # Save full results
        full_results = processor.get_results_df()
        full_results.to_csv("sentiment_results.csv", index=False)
        report.to_csv("company_report.csv", index=False)
        print("\nResults saved to sentiment_results.csv and company_report.csv")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
