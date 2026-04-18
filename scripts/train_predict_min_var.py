#!/usr/bin/env python3
"""Train min-variance linear model (datathon_minvariance) and write submission CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from datathon_baseline.paths import data_dir as default_data_dir
from datathon_minvariance.predict import fit_and_predict


def main() -> None:
    p = argparse.ArgumentParser(
        description="Min-variance linear train → predict → submission CSV (datathon_minvariance)"
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Parquet folder (default: data/ under repo root).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (Ridge init for SLSQP).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_REPO_ROOT / "Submissions" / "submission_min_var.csv",
        help="Output submission path.",
    )
    args = p.parse_args()

    dd = args.data_dir or default_data_dir()
    dd = dd.resolve()
    if not dd.is_dir():
        raise SystemExit(f"Data directory not found: {dd}")

    sub, res = fit_and_predict(dd, random_state=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.output, index=False)

    print(f"Train Sharpe: {res.train_sharpe:.6f}")
    print(f"Train PnL variance: {res.train_pnl_variance:.8f}")
    print(f"Optimizer: {res.optimizer_message}")
    print(f"Wrote {len(sub)} rows to {args.output.resolve()}")


if __name__ == "__main__":
    main()
