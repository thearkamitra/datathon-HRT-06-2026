#!/usr/bin/env python3
"""Train baseline (ridge / momentum / constant) and write submission CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root: .../datathon-HRT-06-2026
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from datathon_baseline.paths import data_dir as default_data_dir
from datathon_baseline.predict import Method, fit_and_predict


def main() -> None:
    p = argparse.ArgumentParser(description="Baseline train → predict → submission CSV")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Parquet folder (default: data/ under repo root).",
    )
    p.add_argument(
        "--method",
        type=str,
        choices=[m.value for m in Method],
        default=Method.ridge.value,
    )
    p.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge alpha (sklearn). Ignored unless --method ridge.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed for Ridge.")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_REPO_ROOT / "Submissions" / "submission.csv",
        help="Output submission path.",
    )
    args = p.parse_args()

    dd = args.data_dir or default_data_dir()
    dd = dd.resolve()
    if not dd.is_dir():
        raise SystemExit(f"Data directory not found: {dd}")

    method = Method(args.method)
    sub, res = fit_and_predict(dd, method, ridge_reg=args.ridge_alpha, random_state=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.output, index=False)

    print(f"Method: {res.method.value}")
    print(f"Train Sharpe (aligned sign): {res.train_sharpe:.6f}")
    if res.ridge_alpha is not None:
        print(f"Ridge alpha: {res.ridge_alpha}")
    print(f"Wrote {len(sub)} rows to {args.output.resolve()}")


if __name__ == "__main__":
    main()
