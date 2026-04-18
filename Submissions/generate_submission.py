#!/usr/bin/env python3
"""
Build submission.csv in this folder: columns session,target_position.

Session ids are taken from public + private test bar parquet files so they always
match the competition data layout.

Examples:
  cd Submissions && python generate_submission.py
  python generate_submission.py --seed 42
  python generate_submission.py --zeros
  python generate_submission.py --data-dir ../../hrt-eth-zurich-datathon-2026/data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _default_data_dir() -> Path:
    # This file lives in .../datathon-HRT-06-2026/Submissions/
    repo = Path(__file__).resolve().parent.parent
    return repo.parent / "hrt-eth-zurich-datathon-2026" / "data"


def load_test_sessions(data_dir: Path) -> list[int]:
    pub = pd.read_parquet(data_dir / "bars_seen_public_test.parquet", columns=["session"])
    priv = pd.read_parquet(data_dir / "bars_seen_private_test.parquet", columns=["session"])
    ids = sorted(set(pub["session"].unique()) | set(priv["session"].unique()))
    return [int(x) for x in ids]


def main() -> None:
    here = Path(__file__).resolve().parent
    default_out = here / "submission.csv"

    p = argparse.ArgumentParser(description="Generate submission CSV for Zurich Datathon 2026.")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=_default_data_dir(),
        help="Folder with bars_seen_*_test.parquet (default: sibling hrt-eth-zurich-datathon-2026/data).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_out,
        help=f"Output CSV path (default: {default_out}).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible target_position values.",
    )
    p.add_argument(
        "--zeros",
        action="store_true",
        help="Set every target_position to 0 instead of random.",
    )
    p.add_argument(
        "--std",
        type=float,
        default=1.0,
        help="Std dev for Gaussian target_position when random (default: 1). Ignored with --zeros.",
    )
    args = p.parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    sessions = load_test_sessions(data_dir)
    n = len(sessions)
    if args.zeros:
        targets = np.zeros(n, dtype=np.float64)
    else:
        rng = np.random.default_rng(args.seed)
        targets = rng.normal(loc=0.0, scale=args.std, size=n)

    out = pd.DataFrame({"session": sessions, "target_position": targets})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output.resolve()}")


if __name__ == "__main__":
    main()
