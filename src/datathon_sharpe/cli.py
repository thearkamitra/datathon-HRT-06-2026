"""CLI: full-train submission + 25/25 train-session Sharpe monitoring + optional test CSV split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from datathon_baseline.paths import data_dir as default_data_dir
from datathon_baseline.predict import Method
from datathon_sharpe.cv import run_cv_report, split_test_csv_sessions_25_25


def main() -> None:
    p = argparse.ArgumentParser(
        description="Baseline submission + 25/25 session split (train Sharpe CV + optional test CSV split)"
    )
    p.add_argument("--data-dir", type=Path, default=None, help="Parquet folder (default: data/).")
    p.add_argument(
        "--method",
        type=str,
        choices=[m.value for m in Method],
        default=Method.sharpe_linear.value,
    )
    p.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge / ElasticNet alpha for --method ridge; Sharpe-linear warm-start (higher = stronger penalty).",
    )
    p.add_argument(
        "--l1-ratio",
        type=float,
        default=0.0,
        help="Sharpe-linear warm-start: L1 share in ElasticNet (0=Ridge-only, 1=Lasso). Ignored for other methods.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed (model + default split seed).")
    p.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed for 25/25 session draw (default: same as --seed).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_REPO_ROOT / "Submissions" / "submission_sharpe.csv",
    )
    p.add_argument(
        "--dump-test-split",
        type=Path,
        default=None,
        help="If set, write JSON with 25+25 session ids from merged public/private test CSVs.",
    )
    p.add_argument(
        "--within-session-split",
        action="store_true",
        help="Features from seen bars 0–24 only; train label R = close_49/close_24-1 (proxy, not competition R).",
    )
    p.add_argument(
        "--augment-test-proxy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Augment fit: train 50-bar + competition R; add test sessions with 25-bar + proxy R (default: on). Submission still 50-bar test. Use --no-augment-test-proxy to disable.",
    )
    p.add_argument(
        "--sharpe-optimizer-label",
        type=str,
        choices=["identity", "r2_sign_100"],
        default="identity",
        help="Sharpe-linear only: label inside optimizer (Ridge+SLSQP). "
        "'identity' = R; 'r2_sign_100' = R**2*100*sign(R). Reported train Sharpes still use raw R.",
    )
    p.add_argument(
        "--use-cnn",
        action="store_true",
        help="Train a 1D CNN on OHLC sequences (MSE on R), add cnn_r_pred to features; Sharpe-linear stacks on all columns.",
    )
    p.add_argument(
        "--cnn-epochs",
        type=int,
        default=40,
        help="Training epochs for --use-cnn (default: 40).",
    )
    p.add_argument(
        "--mse-anchor-lambda",
        type=float,
        default=0.0,
        help="Sharpe-linear only: λ for -Sharpe + λ·MSE(w, w_ridge); 0 = original unit-sphere objective.",
    )
    p.add_argument(
        "--distributional-policy",
        type=str,
        choices=["prob_sign", "quantile_median", "rank_score"],
        default="prob_sign",
        help="distributional_mono only: prob_sign (default), quantile_median, or rank_score.",
    )
    args = p.parse_args()

    dd = args.data_dir or default_data_dir()
    dd = dd.resolve()
    if not dd.is_dir():
        raise SystemExit(f"Data directory not found: {dd}")

    if args.within_session_split and args.augment_test_proxy:
        raise SystemExit(
            "Cannot combine --within-session-split with augment (on by default). "
            "Add --no-augment-test-proxy when using --within-session-split."
        )
    if not 0.0 <= args.l1_ratio <= 1.0:
        raise SystemExit("--l1-ratio must be between 0 and 1.")
    if args.mse_anchor_lambda < 0.0:
        raise SystemExit("--mse-anchor-lambda must be non-negative.")

    method = Method(args.method)
    split_seed = args.split_seed if args.split_seed is not None else args.seed

    sub, report = run_cv_report(
        dd,
        method,
        ridge_reg=args.ridge_alpha,
        l1_ratio=args.l1_ratio,
        random_state=args.seed,
        split_seed=split_seed,
        within_session_split=args.within_session_split,
        augment_test_with_proxy=args.augment_test_proxy,
        sharpe_optimizer_label=args.sharpe_optimizer_label,
        use_cnn=args.use_cnn,
        cnn_epochs=args.cnn_epochs,
        mse_anchor_lambda=args.mse_anchor_lambda,
        distributional_policy=args.distributional_policy,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.output, index=False)

    print(f"Method: {method.value}")
    if method == Method.sharpe_linear:
        print(f"Warm-start: alpha={args.ridge_alpha}, l1_ratio={args.l1_ratio} (ElasticNet if l1_ratio>0)")
        print(f"Sharpe optimizer label: {args.sharpe_optimizer_label}")
        if args.mse_anchor_lambda > 0.0:
            print(f"MSE anchor λ (to Ridge w): {args.mse_anchor_lambda}")
        if args.use_cnn:
            print(f"CNN session head: on (epochs={args.cnn_epochs})")
    if method == Method.distributional_mono:
        print(
            f"Distributional policy: {args.distributional_policy} "
            f"(ridge_reg={args.ridge_alpha} as C⁻¹ / quantile α / Ridge α)"
        )
    if args.within_session_split:
        print(
            "Mode: within-session — features bars 0–24, train R = close_49/close_24-1 (proxy; not competition R)."
        )
    elif args.augment_test_proxy:
        print(
            "Mode: augment — train 50-bar + competition R; extra rows = test 25-bar + proxy R; submit with 50-bar test features."
        )
    ts_label = "full train"
    if args.within_session_split:
        ts_label = "full train (proxy R)"
    elif args.augment_test_proxy:
        ts_label = "full fit (train+test aug, mixed R types)"
    else:
        ts_label = "full train, same as datathon_baseline"
    print(f"Train Sharpe ({ts_label}): {report.train_result.train_sharpe:.6f}")

    if args.within_session_split:
        r_note = "proxy R (seen 2nd half), all 1000 sessions"
    else:
        r_note = "competition R, all 1000 train sessions"
    print(f"Train Sharpe ({r_note}): {report.sharpe_train_all_sessions:.6f}")

    print(f"Train Sharpe (25-session 'train' block):      {report.sharpe_block_train:.6f}")
    print(f"Train Sharpe (25-session 'label' block):      {report.sharpe_block_label:.6f}")
    print(f"Sessions — train block: {report.sessions_train_block.tolist()}")
    print(f"Sessions — label block: {report.sessions_label_block.tolist()}")
    print(f"Wrote {len(sub)} rows to {args.output.resolve()}")

    if args.dump_test_split is not None:
        a, b = split_test_csv_sessions_25_25(random_state=split_seed)
        payload = {
            "seed": split_seed,
            "sessions_train_block": a.tolist(),
            "sessions_label_block": b.tolist(),
            "note": "No official R for test sessions; use for subset submission / debugging.",
        }
        args.dump_test_split.parent.mkdir(parents=True, exist_ok=True)
        args.dump_test_split.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote test CSV 25/25 split to {args.dump_test_split.resolve()}")


if __name__ == "__main__":
    main()
