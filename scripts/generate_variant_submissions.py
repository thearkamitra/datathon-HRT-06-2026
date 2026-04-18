#!/usr/bin/env python3
"""Build alternative submissions: cluster+Sharpe-linear, HGBR, MLP, linear+HGBR ensemble."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from datathon_baseline.paths import data_dir as default_data_dir
from datathon_sharpe.variant_submissions import (
    submission_cluster_ohe_sharpe_linear,
    submission_ensemble_linear_hgbr,
    submission_hgbr_R,
    submission_mlp_R,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate variant Sharpe submission CSVs")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO / "Submissions",
        help="Directory for CSV files (default: Submissions/)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--l1-ratio", type=float, default=0.0)
    p.add_argument("--clusters", type=int, default=8, help="KMeans clusters for cluster_ohe variant")
    p.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable augment-test-proxy (same as CLI --no-augment-test-proxy).",
    )
    args = p.parse_args()

    dd = args.data_dir or default_data_dir()
    dd = dd.resolve()
    if not dd.is_dir():
        raise SystemExit(f"Data directory not found: {dd}")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    aug = not args.no_augment

    print("Data:", dd)
    print("Augment test proxy:", aug)
    print()

    # 1) KMeans one-hot + Sharpe-linear
    sub, sh = submission_cluster_ohe_sharpe_linear(
        dd,
        n_clusters=args.clusters,
        ridge_reg=args.ridge_alpha,
        l1_ratio=args.l1_ratio,
        random_state=args.seed,
        augment_test_with_proxy=aug,
    )
    path = out_dir / "submission_sharpe_cluster_ohe.csv"
    sub.to_csv(path, index=False)
    print(f"1) {path.name}  (train Sharpe sessions, competition R): {sh:.6f}")

    # 2) HGBR
    sub, sh = submission_hgbr_R(dd, random_state=args.seed, augment_test_with_proxy=aug)
    path = out_dir / "submission_sharpe_hgbr.csv"
    sub.to_csv(path, index=False)
    print(f"2) {path.name}  train Sharpe (pred R vs R): {sh:.6f}")

    # 3) MLP
    sub, sh = submission_mlp_R(dd, random_state=args.seed, augment_test_with_proxy=aug)
    path = out_dir / "submission_sharpe_mlp.csv"
    sub.to_csv(path, index=False)
    print(f"3) {path.name}  train Sharpe (pred R vs R): {sh:.6f}")

    # 4) Ensemble
    sub, sh_lin, sh_ens = submission_ensemble_linear_hgbr(
        dd,
        ridge_reg=args.ridge_alpha,
        l1_ratio=args.l1_ratio,
        random_state=args.seed,
        augment_test_with_proxy=aug,
        linear_weight=0.5,
    )
    path = out_dir / "submission_sharpe_ensemble_lin_hgbr.csv"
    sub.to_csv(path, index=False)
    print(f"4) {path.name}  train Sharpe linear: {sh_lin:.6f}  ensemble: {sh_ens:.6f}")

    print()
    print(f"Wrote 4 files under {out_dir}")


if __name__ == "__main__":
    main()
