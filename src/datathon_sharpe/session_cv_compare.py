"""Session-level K-fold CV: Sharpe-linear vs prob_sign vs prob_sign_sharpe on held-out train sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from datathon_baseline.metrics import sharpe
from datathon_baseline.predict import _fit_linear_sharpe
from datathon_sharpe.distributional_mono import fit_distributional_mono
from datathon_sharpe.sentiment_features import FEATURE_COLUMNS_SHARPE
from datathon_sharpe.sharpe_label_transforms import SharpeOptimizerLabel, transform_r_for_optimizer
from datathon_sharpe.training_table import load_training_feature_matrices


@dataclass
class SessionCVResult:
    """Val Sharpe on held-out **train** sessions (competition R); folds are session-disjoint."""

    method_label: str
    ridge_alpha: float
    l1_ratio: float
    mse_anchor_lambda: float
    sharpe_optimizer_label: str
    n_splits: int
    cv_random_state: int
    augment_test_with_proxy: bool
    fold_val_sharpes: list[float]
    mean_val_sharpe: float
    std_val_sharpe: float


def _fit_val_sharpe_fold(
    feat_fit: pd.DataFrame,
    feat_main_val: pd.DataFrame,
    *,
    ridge_reg: float,
    l1_ratio: float,
    sharpe_optimizer_label: SharpeOptimizerLabel,
    mse_anchor_lambda: float,
    random_state: int,
) -> float:
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


def _fit_val_prob_sign_fold(
    feat_fit: pd.DataFrame,
    feat_main_val: pd.DataFrame,
    *,
    ridge_reg: float,
    random_state: int,
) -> float:
    X_fit = feat_fit[FEATURE_COLUMNS_SHARPE].to_numpy(dtype=np.float64)
    R_fit = feat_fit["R"].to_numpy(dtype=np.float64)
    dist = fit_distributional_mono(
        X_fit,
        R_fit,
        policy="prob_sign",
        ridge_reg=ridge_reg,
        random_state=random_state,
    )
    f_fit = dist.predict_f(X_fit)
    mult = -1.0 if float(np.mean(f_fit * R_fit)) < 0 else 1.0
    Xv = feat_main_val[FEATURE_COLUMNS_SHARPE].to_numpy(dtype=np.float64)
    Rv = feat_main_val["R"].to_numpy(dtype=np.float64)
    f_va = dist.predict_f(Xv)
    w = mult * f_va
    return float(sharpe(w * Rv))


def _fit_val_prob_sign_sharpe_fold(
    feat_fit: pd.DataFrame,
    feat_main_val: pd.DataFrame,
    *,
    ridge_reg: float,
    random_state: int,
) -> float:
    X_fit = feat_fit[FEATURE_COLUMNS_SHARPE].to_numpy(dtype=np.float64)
    R_fit = feat_fit["R"].to_numpy(dtype=np.float64)
    dist = fit_distributional_mono(
        X_fit,
        R_fit,
        policy="prob_sign_sharpe",
        ridge_reg=ridge_reg,
        random_state=random_state,
    )
    f_fit = dist.predict_f(X_fit)
    mult = -1.0 if float(np.mean(f_fit * R_fit)) < 0 else 1.0
    Xv = feat_main_val[FEATURE_COLUMNS_SHARPE].to_numpy(dtype=np.float64)
    Rv = feat_main_val["R"].to_numpy(dtype=np.float64)
    f_va = dist.predict_f(Xv)
    w = mult * f_va
    return float(sharpe(w * Rv))


def run_session_cv_triplet(
    data_dir: Path,
    *,
    ridge_reg: float = 1.0,
    l1_ratio: float = 0.0,
    mse_anchor_lambda: float = 0.0,
    sharpe_optimizer_label: SharpeOptimizerLabel = "identity",
    n_splits: int = 5,
    cv_random_state: int = 0,
    augment_test_with_proxy: bool = True,
) -> tuple[SessionCVResult, SessionCVResult, SessionCVResult]:
    """
    K-fold over **train session ids**. Val Sharpe = positions * competition **R** on held-out
    train sessions only. Augment rows (25-bar test + proxy R) are only in **fold_fit**, not val.
    """
    feat_main, feat_fit = load_training_feature_matrices(
        data_dir,
        within_session_split=False,
        augment_test_with_proxy=augment_test_with_proxy,
    )
    train_sess = np.sort(feat_main["session"].unique())
    feat_aug = feat_fit[~feat_fit["session"].isin(set(train_sess))].copy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=cv_random_state)

    folds_sh: list[float] = []
    folds_ps: list[float] = []
    folds_pss: list[float] = []
    for fold_idx, (tr_i, va_i) in enumerate(kf.split(np.arange(len(train_sess)))):
        S_train = train_sess[tr_i]
        S_val = train_sess[va_i]
        fm_tr = feat_main[feat_main["session"].isin(S_train)]
        fm_va = feat_main[feat_main["session"].isin(S_val)]
        fold_fit = fm_tr if feat_aug.empty else pd.concat([fm_tr, feat_aug], ignore_index=True)
        rs = cv_random_state + 1000 * fold_idx
        folds_sh.append(
            _fit_val_sharpe_fold(
                fold_fit,
                fm_va,
                ridge_reg=ridge_reg,
                l1_ratio=l1_ratio,
                sharpe_optimizer_label=sharpe_optimizer_label,
                mse_anchor_lambda=mse_anchor_lambda,
                random_state=rs,
            )
        )
        folds_ps.append(
            _fit_val_prob_sign_fold(
                fold_fit,
                fm_va,
                ridge_reg=ridge_reg,
                random_state=rs,
            )
        )
        folds_pss.append(
            _fit_val_prob_sign_sharpe_fold(
                fold_fit,
                fm_va,
                ridge_reg=ridge_reg,
                random_state=rs,
            )
        )

    res_sh = SessionCVResult(
        method_label="sharpe_linear",
        ridge_alpha=float(ridge_reg),
        l1_ratio=float(l1_ratio),
        mse_anchor_lambda=float(mse_anchor_lambda),
        sharpe_optimizer_label=str(sharpe_optimizer_label),
        n_splits=n_splits,
        cv_random_state=cv_random_state,
        augment_test_with_proxy=augment_test_with_proxy,
        fold_val_sharpes=folds_sh,
        mean_val_sharpe=float(np.mean(folds_sh)),
        std_val_sharpe=float(np.std(folds_sh)),
    )
    res_ps = SessionCVResult(
        method_label="distributional_mono_prob_sign",
        ridge_alpha=float(ridge_reg),
        l1_ratio=float(l1_ratio),
        mse_anchor_lambda=0.0,
        sharpe_optimizer_label="n/a",
        n_splits=n_splits,
        cv_random_state=cv_random_state,
        augment_test_with_proxy=augment_test_with_proxy,
        fold_val_sharpes=folds_ps,
        mean_val_sharpe=float(np.mean(folds_ps)),
        std_val_sharpe=float(np.std(folds_ps)),
    )
    res_pss = SessionCVResult(
        method_label="distributional_mono_prob_sign_sharpe",
        ridge_alpha=float(ridge_reg),
        l1_ratio=float(l1_ratio),
        mse_anchor_lambda=0.0,
        sharpe_optimizer_label="n/a",
        n_splits=n_splits,
        cv_random_state=cv_random_state,
        augment_test_with_proxy=augment_test_with_proxy,
        fold_val_sharpes=folds_pss,
        mean_val_sharpe=float(np.mean(folds_pss)),
        std_val_sharpe=float(np.std(folds_pss)),
    )
    return res_sh, res_ps, res_pss


def run_session_cv_pair(
    data_dir: Path,
    *,
    ridge_reg: float = 1.0,
    l1_ratio: float = 0.0,
    mse_anchor_lambda: float = 0.0,
    sharpe_optimizer_label: SharpeOptimizerLabel = "identity",
    n_splits: int = 5,
    cv_random_state: int = 0,
    augment_test_with_proxy: bool = True,
) -> tuple[SessionCVResult, SessionCVResult]:
    a, b, _ = run_session_cv_triplet(
        data_dir,
        ridge_reg=ridge_reg,
        l1_ratio=l1_ratio,
        mse_anchor_lambda=mse_anchor_lambda,
        sharpe_optimizer_label=sharpe_optimizer_label,
        n_splits=n_splits,
        cv_random_state=cv_random_state,
        augment_test_with_proxy=augment_test_with_proxy,
    )
    return a, b


__all__ = [
    "SessionCVResult",
    "run_session_cv_pair",
    "run_session_cv_triplet",
]
