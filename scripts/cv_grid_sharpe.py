#!/usr/bin/env python3
"""Session-level K-fold CV grid for Sharpe-linear (ridge, l1, optional optimizer label)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import pandas as pd

from datathon_baseline.paths import data_dir as default_data_dir
from datathon_sharpe.cv_grid import cv_grid_results_to_dataframe, run_sharpe_linear_cv_grid


def _parse_float_list(s: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def main() -> None:
    p = argparse.ArgumentParser(
        description="K-fold CV on train sessions; each fit includes full augment rows when enabled."
    )
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument(
        "--ridge-alphas",
        type=str,
        default="0.5,1.0,2.0,5.0",
        help="Comma-separated ridge warm-start alphas.",
    )
    p.add_argument(
        "--l1-ratios",
        type=str,
        default="0.0",
        help="Comma-separated ElasticNet l1_ratio values (0=Ridge-only).",
    )
    p.add_argument(
        "--labels",
        type=str,
        default="identity",
        help="Comma-separated: identity and/or r2_sign_100",
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable augment-test-proxy in the CV loader (default: augment on).",
    )
    p.add_argument(
        "--mse-anchor-lambdas",
        type=str,
        default="0.0",
        help="Comma-separated MSE-anchor λ values for Sharpe-linear (default: 0.0 only).",
    )
    p.add_argument(
        "-o",
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save results table.",
    )
    args = p.parse_args()

    dd = args.data_dir or default_data_dir()
    dd = dd.resolve()
    if not dd.is_dir():
        raise SystemExit(f"Data directory not found: {dd}")

    mse_anchor_lambdas = _parse_float_list(args.mse_anchor_lambdas)
    if any(x < 0.0 for x in mse_anchor_lambdas):
        raise SystemExit("--mse-anchor-lambdas must be non-negative.")

    ridge_alphas = _parse_float_list(args.ridge_alphas)
    l1_ratios = _parse_float_list(args.l1_ratios)
    labels_raw = [x.strip() for x in args.labels.split(",") if x.strip()]
    allowed = {"identity", "r2_sign_100"}
    for lb in labels_raw:
        if lb not in allowed:
            raise SystemExit(f"Unknown label {lb!r}; use {allowed}")
    labels = tuple(labels_raw)  # type: ignore[assignment]

    print("Data:", dd)
    print("Augment test proxy:", not args.no_augment)
    print("K:", args.n_splits)
    print("Ridge alphas:", ridge_alphas)
    print("L1 ratios:", l1_ratios)
    print("Optimizer labels:", labels)
    print("MSE anchor lambdas:", mse_anchor_lambdas)
    print()

    rows = run_sharpe_linear_cv_grid(
        dd,
        ridge_alphas=ridge_alphas,
        l1_ratios=l1_ratios,
        sharpe_optimizer_labels=labels,
        mse_anchor_lambdas=mse_anchor_lambdas,
        n_splits=args.n_splits,
        random_state=args.seed,
        augment_test_with_proxy=not args.no_augment,
    )

    df = cv_grid_results_to_dataframe(rows)
    show = df.drop(columns=["fold_val_sharpes"])
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(show.to_string(index=False))
    print()
    best = rows[0]
    print(
        "Best (mean val Sharpe): "
        f"ridge={best.ridge_alpha}, l1={best.l1_ratio}, label={best.sharpe_optimizer_label}, "
        f"mse_λ={best.mse_anchor_lambda} "
        f"-> {best.mean_val_sharpe:.6f} (+/- {best.std_val_sharpe:.6f})"
    )
    print("Fold scores:", [round(x, 4) for x in best.fold_val_sharpes])

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        out = show.copy()
        for j in range(args.n_splits):
            out[f"fold_{j}"] = [r.fold_val_sharpes[j] for r in rows]
        out.to_csv(args.output_csv, index=False)
        print(f"Wrote {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
