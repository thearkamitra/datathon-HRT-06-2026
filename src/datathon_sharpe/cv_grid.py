"""Session-level K-fold CV for Sharpe-linear hyperparameters (competition R on held-out train sessions)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from datathon_baseline.metrics import sharpe
from datathon_baseline.predict import _fit_linear_sharpe
from datathon_sharpe.sentiment_features import FEATURE_COLUMNS_SHARPE
from datathon_sharpe.sharpe_label_transforms import SharpeOptimizerLabel, transform_r_for_optimizer
from datathon_sharpe.training_table import load_training_feature_matrices


@dataclass
class CVGridResultRow:
    ridge_alpha: float
    l1_ratio: float
    sharpe_optimizer_label: str
    mse_anchor_lambda: float
    mean_val_sharpe: float
    std_val_sharpe: float
    fold_val_sharpes: list[float]


def _fit_val_sharpe(
    feat_fit: pd.DataFrame,
    feat_main_val: pd.DataFrame,
    *,
    ridge_reg: float,
    l1_ratio: float,
    sharpe_optimizer_label: SharpeOptimizerLabel,
    mse_anchor_lambda: float,
    random_state: int,
) -> float:
    """Fit on ``feat_fit`` (train + optional augment rows), Sharpe on held-out **train** sessions only."""
    X_fit = feat_fit[FEATURE_COLUMNS_SHARPE].to_numpy(dtype=np.float64)
    R_fit = feat_fit["R"].to_numpy(dtype=np.float64)
    R_opt = transform_r_for_optimizer(R_fit, sharpe_optimizer_label)
    scaler, beta, _msg = _fit_linear_sharpe(
        X_fit,
        R_opt,
        random_state=random_state,
        ridge_alpha=ridge_reg,
        l1_ratio=l1_ratio,
        mse_anchor_lambda=mse_anchor_lambda,
    )
    Xv = feat_main_val[FEATURE_COLUMNS_SHARPE].to_numpy(dtype=np.float64)
    Rv = feat_main_val["R"].to_numpy(dtype=np.float64)
    Xd = np.column_stack(
        [np.ones(len(feat_main_val), dtype=np.float64), scaler.transform(Xv)]
    )
    f = Xd @ beta
    return float(sharpe(f * Rv))


def run_sharpe_linear_cv_grid(
    data_dir: Path,
    *,
    ridge_alphas: Sequence[float] = (0.5, 1.0, 2.0, 5.0),
    l1_ratios: Sequence[float] = (0.0,),
    sharpe_optimizer_labels: Sequence[SharpeOptimizerLabel] = ("identity",),
    mse_anchor_lambdas: Sequence[float] = (0.0,),
    n_splits: int = 5,
    random_state: int = 0,
    augment_test_with_proxy: bool = True,
) -> list[CVGridResultRow]:
    """
    K-fold CV over **train session ids** only. Each fit uses:

    - rows from ``feat_main`` whose session is in the fold's train split, **plus**
    - all augment test rows (if ``augment_test_with_proxy``), same as production.

    Validation Sharpe uses **competition R** on held-out train sessions (``f * R``).

    ``mse_anchor_lambdas``: grid over Sharpe-linear MSE-anchor weights (same meaning as
    ``train_model.fit_full_train_and_submission(..., mse_anchor_lambda=...)``).
    """
    for lam in mse_anchor_lambdas:
        if float(lam) < 0.0:
            raise ValueError(f"mse_anchor_lambdas must be non-negative, got {lam!r}")
    feat_main, feat_fit = load_training_feature_matrices(
        data_dir,
        within_session_split=False,
        augment_test_with_proxy=augment_test_with_proxy,
    )
    train_sess = np.sort(feat_main["session"].unique())
    train_set = set(int(s) for s in train_sess)
    feat_aug = feat_fit[~feat_fit["session"].isin(train_sess)].copy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results: list[CVGridResultRow] = []

    for ridge, l1, label, mse_lam in product(
        ridge_alphas, l1_ratios, sharpe_optimizer_labels, mse_anchor_lambdas
    ):
        fold_scores: list[float] = []
        for fold_idx, (tr_i, va_i) in enumerate(kf.split(np.arange(len(train_sess)))):
            S_train = train_sess[tr_i]
            S_val = train_sess[va_i]
            fm_tr = feat_main[feat_main["session"].isin(S_train)]
            fm_va = feat_main[feat_main["session"].isin(S_val)]
            if feat_aug.empty:
                fold_fit = fm_tr
            else:
                fold_fit = pd.concat([fm_tr, feat_aug], ignore_index=True)
            rs = random_state + 1000 * fold_idx
            sc = _fit_val_sharpe(
                fold_fit,
                fm_va,
                ridge_reg=float(ridge),
                l1_ratio=float(l1),
                sharpe_optimizer_label=label,
                mse_anchor_lambda=float(mse_lam),
                random_state=rs,
            )
            fold_scores.append(sc)
        results.append(
            CVGridResultRow(
                ridge_alpha=float(ridge),
                l1_ratio=float(l1),
                sharpe_optimizer_label=str(label),
                mse_anchor_lambda=float(mse_lam),
                mean_val_sharpe=float(np.mean(fold_scores)),
                std_val_sharpe=float(np.std(fold_scores)),
                fold_val_sharpes=fold_scores,
            )
        )

    results.sort(key=lambda r: r.mean_val_sharpe, reverse=True)
    return results


def cv_grid_results_to_dataframe(rows: list[CVGridResultRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ridge_alpha": r.ridge_alpha,
                "l1_ratio": r.l1_ratio,
                "sharpe_optimizer_label": r.sharpe_optimizer_label,
                "mse_anchor_lambda": r.mse_anchor_lambda,
                "mean_val_sharpe": r.mean_val_sharpe,
                "std_val_sharpe": r.std_val_sharpe,
                "fold_val_sharpes": r.fold_val_sharpes,
            }
            for r in rows
        ]
    )


__all__ = [
    "CVGridResultRow",
    "cv_grid_results_to_dataframe",
    "run_sharpe_linear_cv_grid",
]
