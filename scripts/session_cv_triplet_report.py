#!/usr/bin/env python3
"""
Session-level K-fold CV for robustness: sharpe_linear vs prob_sign vs prob_sign_sharpe.

Val Sharpe uses only held-out **train** sessions (competition R). Augment test rows (20k)
are in fold training fits only — same as production when augment is on.

Writes Submissions/sessionCV_triplet_REPORT.txt and sessionCV_triplet_results.csv by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import pandas as pd

from datathon_baseline.paths import data_dir as default_data_dir
from datathon_sharpe.session_cv_compare import run_session_cv_triplet


def _parse_int_list(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def main() -> None:
    p = argparse.ArgumentParser(description="Session CV triplet (Sharpe / prob_sign / prob_sign_sharpe)")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--l1-ratio", type=float, default=0.0)
    p.add_argument("--mse-anchor-lambda", type=float, default=0.0)
    p.add_argument(
        "--sharpe-optimizer-label",
        type=str,
        default="identity",
        choices=["identity", "r2_sign_100"],
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument(
        "--cv-seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated KFold shuffle seeds.",
    )
    p.add_argument("--no-augment", action="store_true")
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Default: repo Submissions/",
    )
    args = p.parse_args()

    dd = args.data_dir or default_data_dir()
    dd = dd.resolve()
    if not dd.is_dir():
        raise SystemExit(f"Data directory not found: {dd}")

    cv_seeds = _parse_int_list(args.cv_seeds)
    augment = not args.no_augment
    out_dir = (args.output_dir or (_REPO / "Submissions")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    lines: list[str] = [
        "Session CV triplet — val Sharpe on held-out TRAIN sessions only (competition R).",
        f"data_dir={dd}",
        f"ridge_alpha={args.ridge_alpha}  l1_ratio={args.l1_ratio}  mse_lambda={args.mse_anchor_lambda}",
        f"sharpe_optimizer_label={args.sharpe_optimizer_label}",
        f"n_splits={args.n_splits}  augment_test_with_proxy={augment}",
        "",
        "prob_sign_sharpe: per fold, logistic + optimize α for tanh(α·logit(p)) on that fold's fit rows.",
        "",
    ]

    for cv_seed in cv_seeds:
        sh, ps, pss = run_session_cv_triplet(
            dd,
            ridge_reg=args.ridge_alpha,
            l1_ratio=args.l1_ratio,
            mse_anchor_lambda=args.mse_anchor_lambda,
            sharpe_optimizer_label=args.sharpe_optimizer_label,  # type: ignore[arg-type]
            n_splits=args.n_splits,
            cv_random_state=cv_seed,
            augment_test_with_proxy=augment,
        )
        lines.append(
            f"cv_seed={cv_seed}  sharpe_linear: {sh.mean_val_sharpe:.6f} +/- {sh.std_val_sharpe:.6f}  "
            f"folds={[round(x, 4) for x in sh.fold_val_sharpes]}"
        )
        lines.append(
            f"cv_seed={cv_seed}  prob_sign:          {ps.mean_val_sharpe:.6f} +/- {ps.std_val_sharpe:.6f}  "
            f"folds={[round(x, 4) for x in ps.fold_val_sharpes]}"
        )
        lines.append(
            f"cv_seed={cv_seed}  prob_sign_sharpe:   {pss.mean_val_sharpe:.6f} +/- {pss.std_val_sharpe:.6f}  "
            f"folds={[round(x, 4) for x in pss.fold_val_sharpes]}"
        )
        lines.append("")
        for r in (sh, ps, pss):
            rows.append(
                {
                    "cv_seed": cv_seed,
                    "method": r.method_label,
                    "mean_val_sharpe": r.mean_val_sharpe,
                    "std_val_sharpe": r.std_val_sharpe,
                    "fold_val_sharpes": r.fold_val_sharpes,
                }
            )

    df = pd.DataFrame(rows)
    for label in (
        "sharpe_linear",
        "distributional_mono_prob_sign",
        "distributional_mono_prob_sign_sharpe",
    ):
        sub = df[df["method"] == label]["mean_val_sharpe"].to_numpy()
        if sub.size:
            lines.append(
                f"--- Across cv_seed: {label} ---  "
                f"mean_of_means={float(np.mean(sub)):.6f}  "
                f"std_across_seeds={float(np.std(sub)):.6f}  "
                f"min={float(np.min(sub)):.6f}  max={float(np.max(sub)):.6f}"
            )
    lines.append("")
    lines.append(
        "Lower std_across_seeds = less sensitivity to KFold shuffle. "
        "prob_sign_sharpe fits α inside each fold — can track train Sharpe more closely; "
        "compare val spread vs prob_sign to judge overfitting to fold noise."
    )

    report_path = out_dir / "sessionCV_triplet_REPORT.txt"
    csv_path = out_dir / "sessionCV_triplet_results.csv"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    df.to_csv(csv_path, index=False)

    print("\n".join(lines))
    print(f"\nWrote {report_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
