"""Fit baseline-style models on the full train set; expose per-session predictions and test submission."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from datathon_baseline.io import (
    BARS_SEEN_PRIVATE_TEST,
    BARS_SEEN_PUBLIC_TEST,
    read_bars,
)
from datathon_baseline.metrics import sharpe
from datathon_baseline.predict import Method, TrainResult, _fit_linear_sharpe
from datathon_sharpe.sentiment_features import (
    FEATURE_COLUMNS_SHARPE,
    build_sharpe_session_features,
    load_sentiments_seen_test,
)
from datathon_sharpe.sharpe_label_transforms import SharpeOptimizerLabel, transform_r_for_optimizer
from datathon_sharpe.distributional_mono import fit_distributional_mono
from datathon_sharpe.training_table import load_training_feature_matrices
from datathon_sharpe.ts_cnn import apply_cnn_r_pred_to_frame, train_cnn_predict_r


def fit_full_train_and_submission(
    data_dir: Path,
    method: Method,
    *,
    ridge_reg: float = 1.0,
    l1_ratio: float = 0.0,
    random_state: int = 0,
    within_session_split: bool = False,
    augment_test_with_proxy: bool = True,
    sharpe_optimizer_label: SharpeOptimizerLabel = "identity",
    use_cnn: bool = False,
    cnn_epochs: int = 40,
    mse_anchor_lambda: float = 0.0,
    distributional_policy: str = "prob_sign",
) -> tuple[pd.DataFrame, pd.DataFrame, TrainResult]:
    """
    Single fit on labeled rows; build test submission.

    - Default ``augment_test_with_proxy=True``: fit matrix = **1000 train** (50-bar + competition ``R``)
      **plus** all test sessions (25-bar + proxy ``R``); submission still **50-bar** test.
      Set ``augment_test_with_proxy=False`` for train-only fit (like baseline).

    - ``within_session_split``: train only, 25-bar features, proxy ``R`` on seen second half.
      Incompatible with augment; use ``augment_test_with_proxy=False``.

    - If ``augment_test_with_proxy`` is True: append public/private **test** sessions with
      **25-bar** features and ``R = close_49/close_24 - 1`` to the optimizer rows.

    Returns (train_predictions, submission, train_result).
    ``train_predictions`` is **train sessions only** (1000 rows) with competition ``R``,
    so session-level CV metrics stay comparable to the baseline.

    ``sharpe_optimizer_label``: for ``sharpe_linear`` only, optionally replace ``R`` with
    a transform (e.g. ``r2_sign_100`` for ``R**2 * 100 * sign(R)``) inside
    ``_fit_linear_sharpe`` (Ridge warm-start + SLSQP). Reported train Sharpes still use
    realized ``R`` from the data table.

    ``use_cnn``: if True, train a small 1D CNN on OHLC sequences to predict ``R``, then set
    ``cnn_r_pred`` for all sessions (train/test) before Sharpe-linear; stack optimizes jointly.

    ``mse_anchor_lambda``: if > 0, Sharpe-linear minimizes
    ``-Sharpe + λ·mean((w - w_ridge)²)`` with ``w = X_design β`` and ``w_ridge`` the Ridge
    prediction of ``R`` (same warm-start). If 0, use unit-sphere Sharpe only (default).

    ``distributional_policy``: for ``distributional_mono`` only — ``prob_sign``, ``quantile_median``,
    or ``rank_score`` (see ``distributional_mono`` module).
    """
    feat_main, feat_fit = load_training_feature_matrices(
        data_dir,
        within_session_split=within_session_split,
        augment_test_with_proxy=augment_test_with_proxy,
    )

    bars_pub = read_bars(data_dir, BARS_SEEN_PUBLIC_TEST)
    bars_priv = read_bars(data_dir, BARS_SEEN_PRIVATE_TEST)
    bars_te = pd.concat([bars_pub, bars_priv], ignore_index=True)
    try:
        h_pub = pd.read_parquet(data_dir / "headlines_seen_public_test.parquet")
        h_priv = pd.read_parquet(data_dir / "headlines_seen_private_test.parquet")
        headlines_te = pd.concat([h_pub, h_priv], ignore_index=True)
    except Exception:
        headlines_te = None

    sen_te = load_sentiments_seen_test(data_dir)

    cnn_scores: dict[int, float] | None = None
    if use_cnn:
        cnn_scores = train_cnn_predict_r(
            data_dir,
            feat_main,
            epochs=cnn_epochs,
            seed=random_state,
        )
        apply_cnn_r_pred_to_frame(feat_main, cnn_scores)
        apply_cnn_r_pred_to_frame(feat_fit, cnn_scores)

    R_fit = feat_fit["R"].to_numpy(dtype=np.float64)
    R_opt = transform_r_for_optimizer(R_fit, sharpe_optimizer_label)
    X_fit = feat_fit[FEATURE_COLUMNS_SHARPE].to_numpy(dtype=np.float64)

    model = None
    scaler_sharpe = None
    beta_sharpe = None
    opt_msg: str | None = None
    dist_mono = None
    f_fit: np.ndarray

    if method == Method.constant:
        f_fit = np.ones(len(feat_fit), dtype=np.float64)
    elif method == Method.momentum:
        f_fit = feat_fit["cum_ret"].to_numpy(dtype=np.float64)
    elif method == Method.ridge:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=ridge_reg, random_state=random_state)),
            ]
        )
        model.fit(X_fit, R_fit)
        f_fit = model.predict(X_fit)
    elif method == Method.sharpe_linear:
        scaler_sharpe, beta_sharpe, opt_msg = _fit_linear_sharpe(
            X_fit,
            R_opt,
            random_state=random_state,
            ridge_alpha=ridge_reg,
            l1_ratio=l1_ratio,
            mse_anchor_lambda=mse_anchor_lambda,
        )
        Xd_fit = np.column_stack(
            [np.ones(len(feat_fit), dtype=np.float64), scaler_sharpe.transform(X_fit)]
        )
        f_fit = Xd_fit @ beta_sharpe
    elif method == Method.distributional_mono:
        dist_mono = fit_distributional_mono(
            X_fit,
            R_fit,
            policy=distributional_policy,
            ridge_reg=ridge_reg,
            random_state=random_state,
        )
        f_fit = dist_mono.predict_f(X_fit)
    else:
        raise ValueError(method)

    if method == Method.sharpe_linear:
        train_sh = sharpe(f_fit * R_fit)
        mult = 1.0
    else:
        mult = -1.0 if float(np.mean(f_fit * R_fit)) < 0 else 1.0
        train_sh = sharpe(mult * f_fit * R_fit)

    X_main = feat_main[FEATURE_COLUMNS_SHARPE].to_numpy(dtype=np.float64)
    R_main = feat_main["R"].to_numpy(dtype=np.float64)

    if method == Method.constant:
        f_main = np.ones(len(feat_main), dtype=np.float64)
    elif method == Method.momentum:
        f_main = feat_main["cum_ret"].to_numpy(dtype=np.float64)
    elif method == Method.ridge:
        assert model is not None
        f_main = model.predict(X_main)
    elif method == Method.sharpe_linear:
        assert scaler_sharpe is not None and beta_sharpe is not None
        Xd_main = np.column_stack(
            [np.ones(len(feat_main), dtype=np.float64), scaler_sharpe.transform(X_main)]
        )
        f_main = Xd_main @ beta_sharpe
    elif method == Method.distributional_mono:
        assert dist_mono is not None
        f_main = dist_mono.predict_f(X_main)
    else:
        raise ValueError(method)

    w_main = mult * f_main
    train_pred = pd.DataFrame(
        {
            "session": feat_main["session"].to_numpy(),
            "R": R_main,
            "f": f_main,
            "w": w_main,
        }
    )

    feat_te_pred = build_sharpe_session_features(
        bars_te,
        headlines_te,
        sen_te,
        first_half=within_session_split,
    )
    if use_cnn and cnn_scores is not None:
        apply_cnn_r_pred_to_frame(feat_te_pred, cnn_scores)
    X_test = feat_te_pred[FEATURE_COLUMNS_SHARPE].to_numpy(dtype=np.float64)

    if method == Method.constant:
        f_te = np.ones(len(feat_te_pred), dtype=np.float64)
    elif method == Method.momentum:
        f_te = feat_te_pred["cum_ret"].to_numpy(dtype=np.float64)
    elif method == Method.sharpe_linear:
        assert scaler_sharpe is not None and beta_sharpe is not None
        Xd_te = np.column_stack(
            [np.ones(len(feat_te_pred), dtype=np.float64), scaler_sharpe.transform(X_test)]
        )
        f_te = Xd_te @ beta_sharpe
    elif method == Method.distributional_mono:
        assert dist_mono is not None
        f_te = dist_mono.predict_f(X_test)
    else:
        assert model is not None
        f_te = model.predict(X_test)

    w_te = mult * f_te
    sub = pd.DataFrame({"session": feat_te_pred["session"].to_numpy(), "target_position": w_te})
    sub = sub.sort_values("session").reset_index(drop=True)

    result = TrainResult(
        method=method,
        train_sharpe=float(train_sh),
        ridge_alpha=ridge_reg
        if method in (Method.ridge, Method.sharpe_linear, Method.distributional_mono)
        else None,
        sharpe_opt_message=opt_msg if method == Method.sharpe_linear else None,
        l1_ratio=l1_ratio if method == Method.sharpe_linear else None,
        mse_anchor_lambda=mse_anchor_lambda if method == Method.sharpe_linear else None,
        distributional_policy=(
            distributional_policy if method == Method.distributional_mono else None
        ),
    )
    return train_pred, sub, result


def fit_full_train_predictions(
    data_dir: Path,
    method: Method,
    *,
    ridge_reg: float = 1.0,
    l1_ratio: float = 0.0,
    random_state: int = 0,
    within_session_split: bool = False,
    augment_test_with_proxy: bool = True,
    sharpe_optimizer_label: SharpeOptimizerLabel = "identity",
    use_cnn: bool = False,
    cnn_epochs: int = 40,
    mse_anchor_lambda: float = 0.0,
    distributional_policy: str = "prob_sign",
) -> pd.DataFrame:
    """Train-only table (session, R, f, w) for **train sessions**; same flags as ``fit_full_train_and_submission``."""
    train_pred, _, _ = fit_full_train_and_submission(
        data_dir,
        method,
        ridge_reg=ridge_reg,
        l1_ratio=l1_ratio,
        random_state=random_state,
        within_session_split=within_session_split,
        augment_test_with_proxy=augment_test_with_proxy,
        sharpe_optimizer_label=sharpe_optimizer_label,
        use_cnn=use_cnn,
        cnn_epochs=cnn_epochs,
        mse_anchor_lambda=mse_anchor_lambda,
        distributional_policy=distributional_policy,
    )
    return train_pred
