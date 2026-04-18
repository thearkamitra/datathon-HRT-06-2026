#!/usr/bin/env python3
"""Export data/*.parquet to CSV and a compact text summary for offline reading."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Folder containing parquet files (default: repo data/).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder for CSV + summary (default: <data-dir>/readable_export).",
    )
    args = parser.parse_args()
    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir or (data_dir / "readable_export")
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No .parquet files under {data_dir}")

    summary_lines: list[str] = []
    for pq in parquet_files:
        df = pd.read_parquet(pq)
        stem = pq.stem
        csv_path = out_dir / f"{stem}.csv"
        df.to_csv(csv_path, index=False)

        summary_lines.append(f"=== {stem} ===")
        summary_lines.append(f"rows: {len(df):,}  columns: {len(df.columns)}")
        summary_lines.append("dtypes:")
        for col, dtype in df.dtypes.items():
            summary_lines.append(f"  {col}: {dtype}")
        summary_lines.append("sample (up to 3 rows):")
        sample = df.head(3).to_string(index=False)
        summary_lines.append(sample)
        summary_lines.append("")

    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Wrote {len(parquet_files)} CSV files and {summary_path.name} under {out_dir}")


if __name__ == "__main__":
    main()
