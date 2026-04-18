#!/usr/bin/env python3
"""
Linear Shapley-style feature attributions for ``sharpe_linear`` and ``ridge``.

Uses the **independent linear SHAP** formula on the **scaled** feature block
(same space as ``_fit_linear_sharpe`` / sklearn Pipeline):

  phi_j(x) = w_j * (x_scaled_j - E[x_scaled_j])

where ``w`` is the linear weight on scaled inputs (no intercept in phi; the
intercept is absorbed in the baseline expectation of the model output).

No ``shap`` package required. For ``momentum`` / ``constant``, use
``--method`` and note only one feature carries signal (or none).

Examples::

  PYTHONPATH=src python Tools/shapley_analysis.py --data-dir data --method sharpe_linear
  PYTHONPATH=src python Tools/shapley_analysis.py -o Tools/out/shap_summary.csv
  PYTHONPATH=src python Tools/shapley_analysis.py --no-augment-test-proxy  # train rows only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from datathon_baseline.predict import Method, _fit_linear_sharpe
from datathon_sharpe.path_features import FEATURE_COLUMNS_SHARPE
from datathon_sharpe.training_table import load_training_feature_matrices


def _shap_linear_scaled(X_scaled: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Shape (n_samples, n_features). phi_ij = coef_j * (x_ij - mean_j)."""
    mu = X_scaled.mean(axis=0)
    return (X_scaled - mu) * coef


def analyze(
    data_dir: Path,
    method: Method,
    *,
    ridge_alpha: float,
    l1_ratio: float,
    random_state: int,
    within_session_split: bool,
    augment_test_with_proxy: bool,
) -> tuple[pd.DataFrame, dict]:
    _, feat_fit = load_training_feature_matrices(
        data_dir,
        within_session_split=within_session_split,
        augment_test_with_proxy=augment_test_with_proxy,
    )
    X_raw = feat_fit[FEATURE_COLUMNS_SHARPE].to_numpy(dtype=np.float64)
    R = feat_fit["R"].to_numpy(dtype=np.float64)
    n = len(feat_fit)

    meta: dict = {"n_rows": n, "method": method.value}

    if method == Method.sharpe_linear:
        scaler, beta, msg = _fit_linear_sharpe(
            X_raw,
            R,
            random_state=random_state,
            ridge_alpha=ridge_alpha,
            l1_ratio=l1_ratio,
        )
        meta["sharpe_opt_message"] = msg
        meta["ridge_alpha_warmstart"] = ridge_alpha
        meta["l1_ratio_warmstart"] = l1_ratio
        Xs = scaler.transform(X_raw)
        coef = beta[1:].astype(np.float64)
        phi = _shap_linear_scaled(Xs, coef)
    elif method == Method.ridge:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=ridge_alpha, random_state=random_state)),
            ]
        )
        pipe.fit(X_raw, R)
        Xs = pipe.named_steps["scaler"].transform(X_raw)
        coef = pipe.named_steps["ridge"].coef_.astype(np.float64)
        phi = _shap_linear_scaled(Xs, coef)
        meta["ridge_alpha"] = ridge_alpha
    elif method == Method.momentum:
        # Single effective feature: cum_ret as signal; approximate as 1D linear in cum_ret only
        cr = feat_fit["cum_ret"].to_numpy(dtype=np.float64)
        mu = cr.mean()
        phi_1d = (cr - mu) * 1.0
        idx = FEATURE_COLUMNS_SHARPE.index("cum_ret")
        phi_full = np.zeros((n, len(FEATURE_COLUMNS_SHARPE)), dtype=np.float64)
        phi_full[:, idx] = phi_1d
        phi = phi_full
        meta["note"] = "Only cum_ret column has non-zero linear SHAP (identity signal)."
    elif method == Method.constant:
        phi = np.zeros((n, len(FEATURE_COLUMNS_SHARPE)), dtype=np.float64)
        meta["note"] = "Constant signal: zero SHAP by construction."
    else:
        raise ValueError(method)

    mean_abs = np.abs(phi).mean(axis=0)
    mean_val = phi.mean(axis=0)
    summary = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS_SHARPE,
            "mean_abs_shap": mean_abs,
            "mean_shap": mean_val,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    return summary, meta


def main() -> None:
    p = argparse.ArgumentParser(description="Linear SHAP-style feature importance (scaled space)")
    p.add_argument("--data-dir", type=Path, default=None, help="Parquet folder (default: ../data)")
    p.add_argument(
        "--method",
        type=str,
        choices=[m.value for m in Method],
        default=Method.sharpe_linear.value,
    )
    p.add_argument(
        "--ridge-alpha",
        type=float,
        default=5.0,
        help="Ridge / ElasticNet alpha; Sharpe-linear warm-start (default 5).",
    )
    p.add_argument(
        "--l1-ratio",
        type=float,
        default=0.15,
        help="Sharpe-linear warm-start ElasticNet l1_ratio (0=Ridge-only).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--within-session-split", action="store_true")
    p.add_argument(
        "--augment-test-proxy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include test sessions with proxy R in fit matrix (default: on).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write summary CSV (default: Tools/shapley_summary_<method>.csv)",
    )
    args = p.parse_args()

    dd = args.data_dir or (_REPO_ROOT / "data")
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

    method = Method(args.method)
    out = args.output or (
        _REPO_ROOT / "Tools" / f"shapley_summary_{method.value}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    summary, meta = analyze(
        dd,
        method,
        ridge_alpha=args.ridge_alpha,
        l1_ratio=args.l1_ratio,
        random_state=args.seed,
        within_session_split=args.within_session_split,
        augment_test_with_proxy=args.augment_test_proxy,
    )

    summary.to_csv(out, index=False)
    print(f"Wrote {len(summary)} features to {out.resolve()}")
    print(f"Rows in fit matrix: {meta['n_rows']}")
    for k, v in meta.items():
        if k not in ("n_rows",):
            print(f"  {k}: {v}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
