#!/usr/bin/env python3
"""
(1) Session-level K-fold CV: Sharpe-linear vs prob_sign on held-out train sessions (competition R).

(3) Stability: repeat CV with multiple KFold shuffle seeds; write submissions with multiple fit seeds.

Outputs:
  - Submissions/sessionCV_REPORT.txt          — human-readable CV + stability summary
  - Submissions/sessionCV_results.csv         — one row per (method, cv_seed)
  - Submissions/sessionCV_*_fitseed*.csv      — submission CSVs (clear filenames)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import pandas as pd

from datathon_baseline.paths import data_dir as default_data_dir
from datathon_baseline.predict import Method
from datathon_sharpe.session_cv_compare import run_session_cv_pair
from datathon_sharpe.train_model import fit_full_train_and_submission


def _parse_int_list(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def main() -> None:
    p = argparse.ArgumentParser(
        description="Session CV (Sharpe vs prob_sign) + multi-seed submissions"
    )
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
        help="Comma-separated KFold random_state values (stability across splits).",
    )
    p.add_argument(
        "--fit-seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated random_state values for full-train submission fits.",
    )
    p.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable augment-test-proxy (match --no-augment-test-proxy in CLI).",
    )
    p.add_argument(
        "--submissions-dir",
        type=Path,
        default=None,
        help="Default: <repo>/Submissions",
    )
    args = p.parse_args()

    dd = args.data_dir or default_data_dir()
    dd = dd.resolve()
    if not dd.is_dir():
        raise SystemExit(f"Data directory not found: {dd}")

    cv_seeds = _parse_int_list(args.cv_seeds)
    fit_seeds = _parse_int_list(args.fit_seeds)
    augment = not args.no_augment
    sub_dir = (args.submissions_dir or (_REPO / "Submissions")).resolve()
    sub_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    report_lines: list[str] = [
        "Session-level CV (K-fold over train session ids; val Sharpe = held-out train, competition R).",
        f"data_dir={dd}",
        f"ridge_alpha={args.ridge_alpha}  l1_ratio={args.l1_ratio}  mse_anchor_lambda={args.mse_anchor_lambda}",
        f"sharpe_optimizer_label={args.sharpe_optimizer_label}",
        f"n_splits={args.n_splits}  augment_test_with_proxy={augment}",
        "",
    ]

    for cv_seed in cv_seeds:
        sh, ps = run_session_cv_pair(
            dd,
            ridge_reg=args.ridge_alpha,
            l1_ratio=args.l1_ratio,
            mse_anchor_lambda=args.mse_anchor_lambda,
            sharpe_optimizer_label=args.sharpe_optimizer_label,
            n_splits=args.n_splits,
            cv_random_state=cv_seed,
            augment_test_with_proxy=augment,
        )
        report_lines.append(
            f"cv_seed={cv_seed}  |  sharpe_linear: {sh.mean_val_sharpe:.6f} +/- {sh.std_val_sharpe:.6f}  "
            f"folds={['%.4f' % x for x in sh.fold_val_sharpes]}"
        )
        report_lines.append(
            f"cv_seed={cv_seed}  |  prob_sign:     {ps.mean_val_sharpe:.6f} +/- {ps.std_val_sharpe:.6f}  "
            f"folds={['%.4f' % x for x in ps.fold_val_sharpes]}"
        )
        report_lines.append("")
        rows.append(
            {
                "cv_seed": cv_seed,
                "method": "sharpe_linear",
                "mean_val_sharpe": sh.mean_val_sharpe,
                "std_val_sharpe": sh.std_val_sharpe,
                "fold_val_sharpes": sh.fold_val_sharpes,
            }
        )
        rows.append(
            {
                "cv_seed": cv_seed,
                "method": "prob_sign",
                "mean_val_sharpe": ps.mean_val_sharpe,
                "std_val_sharpe": ps.std_val_sharpe,
                "fold_val_sharpes": ps.fold_val_sharpes,
            }
        )

    # Stability summary across cv_seeds
    df = pd.DataFrame(rows)
    sharpe_means = df[df["method"] == "sharpe_linear"]["mean_val_sharpe"].to_numpy()
    ps_means = df[df["method"] == "prob_sign"]["mean_val_sharpe"].to_numpy()
    report_lines.append("--- Stability across cv_seed (distribution of mean val Sharpe per fold-run) ---")
    report_lines.append(
        f"sharpe_linear: mean_of_means={sharpe_means.mean():.6f}  std_across_cv_seeds={sharpe_means.std():.6f}  "
        f"min={sharpe_means.min():.6f}  max={sharpe_means.max():.6f}"
    )
    report_lines.append(
        f"prob_sign:     mean_of_means={ps_means.mean():.6f}  std_across_cv_seeds={ps_means.std():.6f}  "
        f"min={ps_means.min():.6f}  max={ps_means.max():.6f}"
    )
    report_lines.append("")
    report_lines.append(
        "Interpretation: lower std_across_cv_seeds suggests more stable ranking across random KFold shuffles."
    )

    report_path = sub_dir / "sessionCV_REPORT.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))

    csv_path = sub_dir / "sessionCV_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {report_path}")
    print(f"Wrote {csv_path}")

    # Full-train submissions (multiple fit seeds)
    def _num_tag(x: float) -> str:
        return f"{x:g}".replace(".", "p")

    aug_tag = "aug" if augment else "noaug"
    mse_tag = f"mse{_num_tag(args.mse_anchor_lambda)}"
    ra_tag = _num_tag(args.ridge_alpha)
    for fit_seed in fit_seeds:
        _, sub_sh, _ = fit_full_train_and_submission(
            dd,
            Method.sharpe_linear,
            ridge_reg=args.ridge_alpha,
            l1_ratio=args.l1_ratio,
            random_state=fit_seed,
            augment_test_with_proxy=augment,
            sharpe_optimizer_label=args.sharpe_optimizer_label,
            mse_anchor_lambda=args.mse_anchor_lambda,
        )
        name_sh = (
            f"sessionCV_{aug_tag}_sharpe_ridge{ra_tag}_{mse_tag}_"
            f"opt{args.sharpe_optimizer_label}_fitseed{fit_seed}.csv"
        )
        path_sh = sub_dir / name_sh
        sub_sh.to_csv(path_sh, index=False)
        print(f"Wrote {path_sh} ({len(sub_sh)} rows)")

        _, sub_ps, _ = fit_full_train_and_submission(
            dd,
            Method.distributional_mono,
            ridge_reg=args.ridge_alpha,
            random_state=fit_seed,
            augment_test_with_proxy=augment,
            distributional_policy="prob_sign",
        )
        name_ps = f"sessionCV_{aug_tag}_prob_sign_ridge{ra_tag}_fitseed{fit_seed}.csv"
        path_ps = sub_dir / name_ps
        sub_ps.to_csv(path_ps, index=False)
        print(f"Wrote {path_ps} ({len(sub_ps)} rows)")


if __name__ == "__main__":
    main()
