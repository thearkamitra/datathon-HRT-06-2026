"""Alternative submission builders: KMeans cluster one-hot + Sharpe-linear, boosting/MLP on R, ensembles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from datathon_baseline.io import BARS_SEEN_PRIVATE_TEST, BARS_SEEN_PUBLIC_TEST, read_bars
from datathon_baseline.metrics import sharpe
from datathon_baseline.predict import Method, _fit_linear_sharpe
from datathon_sharpe.sentiment_features import (
    FEATURE_COLUMNS_SHARPE,
    build_sharpe_session_features,
    load_sentiments_seen_test,
)
from datathon_sharpe.training_table import load_training_feature_matrices


def _feat_te_pred(data_dir: Path, *, within_session_split: bool) -> pd.DataFrame:
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
    return build_sharpe_session_features(
        bars_te,
        headlines_te,
        sen_te,
        first_half=within_session_split,
    )


def _kmeans_fit(
    X_main: np.ndarray,
    *,
    n_clusters: int,
    random_state: int,
) -> tuple[StandardScaler, KMeans]:
    scaler = StandardScaler()
    Xm = scaler.fit_transform(X_main)
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )
    km.fit(Xm)
    return scaler, km


def _kmeans_ohe_transform(scaler: StandardScaler, km: KMeans, X: np.ndarray) -> np.ndarray:
    z = km.predict(scaler.transform(X))
    k = int(km.n_clusters)
    ohe = np.zeros((X.shape[0], k), dtype=np.float64)
    ohe[np.arange(X.shape[0], dtype=np.int64), z.astype(np.int64)] = 1.0
    return ohe


def submission_cluster_ohe_sharpe_linear(
    data_dir: Path,
    *,
    n_clusters: int = 8,
    ridge_reg: float = 1.0,
    l1_ratio: float = 0.0,
    random_state: int = 0,
    augment_test_with_proxy: bool = True,
) -> tuple[pd.DataFrame, float]:
    """
    Append KMeans cluster one-hot columns (fit on train sessions only), then Sharpe-linear on
    extended features. Same augment protocol as the main Sharpe pipeline.
    """
    within_session_split = False
    feat_main, feat_fit = load_training_feature_matrices(
        data_dir,
        within_session_split=within_session_split,
        augment_test_with_proxy=augment_test_with_proxy,
    )
    X_main = feat_main[FEATURE_COLUMNS_SHARPE].to_numpy(dtype=np.float64)
    X_fit = feat_fit[FEATURE_COLUMNS_SHARPE].to_numpy(dtype=np.float64)
    R_fit = feat_fit["R"].to_numpy(dtype=np.float64)
    R_main = feat_main["R"].to_numpy(dtype=np.float64)

    scaler_km, km = _kmeans_fit(X_main, n_clusters=n_clusters, random_state=random_state)
    ohe_fit = _kmeans_ohe_transform(scaler_km, km, X_fit)
    ohe_main = _kmeans_ohe_transform(scaler_km, km, X_main)

    X_fit_e = np.hstack([X_fit, ohe_fit])
    scaler_s, beta, _msg = _fit_linear_sharpe(
        X_fit_e,
        R_fit,
        random_state=random_state,
        ridge_alpha=ridge_reg,
        l1_ratio=l1_ratio,
    )
    Xd_main = np.column_stack(
        [np.ones(len(feat_main), dtype=np.float64), scaler_s.transform(np.hstack([X_main, ohe_main]))]
    )
    f_main = Xd_main @ beta
    sh_main = float(sharpe(f_main * R_main))

    feat_te = _feat_te_pred(data_dir, within_session_split=within_session_split)
    X_te = feat_te[FEATURE_COLUMNS_SHARPE].to_numpy(dtype=np.float64)
    ohe_te = _kmeans_ohe_transform(scaler_km, km, X_te)
    X_te_e = np.hstack([X_te, ohe_te])
    Xd_te = np.column_stack([np.ones(len(feat_te), dtype=np.float64), scaler_s.transform(X_te_e)])
    f_te = Xd_te @ beta

    sub = pd.DataFrame({"session": feat_te["session"].to_numpy(), "target_position": f_te})
    sub = sub.sort_values("session").reset_index(drop=True)
    return sub, sh_main


def submission_hgbr_R(
    data_dir: Path,
    *,
    random_state: int = 0,
    augment_test_with_proxy: bool = True,
) -> tuple[pd.DataFrame, float]:
    """
    HistGradientBoostingRegressor for ``R`` trained on **train sessions only**.
    """
    within_session_split = False
    feat_main, _ = load_training_feature_matrices(
        data_dir,
        within_session_split=within_session_split,
        augment_test_with_proxy=augment_test_with_proxy,
    )
    X_main = feat_main[FEATURE_COLUMNS_SHARPE].to_numpy(dtype=np.float64)
    R_main = feat_main["R"].to_numpy(dtype=np.float64)

    model = HistGradientBoostingRegressor(
        max_depth=5,
        max_iter=400,
        learning_rate=0.06,
        l2_regularization=1.0,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=15,
    )
    model.fit(X_main, R_main)
    pred_main = model.predict(X_main)
    mult = -1.0 if float(np.mean(pred_main * R_main)) < 0.0 else 1.0
    sh_main = float(sharpe(mult * pred_main * R_main))

    feat_te = _feat_te_pred(data_dir, within_session_split=within_session_split)
    X_te = feat_te[FEATURE_COLUMNS_SHARPE].to_numpy(dtype=np.float64)
    pred_te = model.predict(X_te)
    w_te = mult * pred_te

    sub = pd.DataFrame({"session": feat_te["session"].to_numpy(), "target_position": w_te})
    sub = sub.sort_values("session").reset_index(drop=True)
    return sub, sh_main


def submission_mlp_R(
    data_dir: Path,
    *,
    random_state: int = 0,
    augment_test_with_proxy: bool = True,
) -> tuple[pd.DataFrame, float]:
    """
    Small MLP regressor for ``R`` (train sessions only).
    """
    within_session_split = False
    feat_main, _ = load_training_feature_matrices(
        data_dir,
        within_session_split=within_session_split,
        augment_test_with_proxy=augment_test_with_proxy,
    )
    X_main = feat_main[FEATURE_COLUMNS_SHARPE].to_numpy(dtype=np.float64)
    R_main = feat_main["R"].to_numpy(dtype=np.float64)

    model = MLPRegressor(
        hidden_layer_sizes=(96, 48),
        activation="relu",
        max_iter=800,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=random_state,
        alpha=1e-3,
    )
    model.fit(X_main, R_main)
    pred_main = model.predict(X_main)
    mult = -1.0 if float(np.mean(pred_main * R_main)) < 0.0 else 1.0
    sh_main = float(sharpe(mult * pred_main * R_main))

    feat_te = _feat_te_pred(data_dir, within_session_split=within_session_split)
    X_te = feat_te[FEATURE_COLUMNS_SHARPE].to_numpy(dtype=np.float64)
    pred_te = model.predict(X_te)
    w_te = mult * pred_te

    sub = pd.DataFrame({"session": feat_te["session"].to_numpy(), "target_position": w_te})
    sub = sub.sort_values("session").reset_index(drop=True)
    return sub, sh_main


def submission_ensemble_linear_hgbr(
    data_dir: Path,
    *,
    ridge_reg: float = 1.0,
    l1_ratio: float = 0.0,
    random_state: int = 0,
    augment_test_with_proxy: bool = True,
    linear_weight: float = 0.5,
) -> tuple[pd.DataFrame, float, float]:
    """
    Blend Sharpe-linear ``f`` with scaled HGBR predictions. Returns ``(submission, sh_train_linear, sh_train_ensemble)``.
    """
    from datathon_sharpe.train_model import fit_full_train_and_submission

    within_session_split = False
    train_pred, sub_lin, _ = fit_full_train_and_submission(
        data_dir,
        Method.sharpe_linear,
        ridge_reg=ridge_reg,
        l1_ratio=l1_ratio,
        random_state=random_state,
        within_session_split=within_session_split,
        augment_test_with_proxy=augment_test_with_proxy,
    )

    feat_main, _ = load_training_feature_matrices(
        data_dir,
        within_session_split=within_session_split,
        augment_test_with_proxy=augment_test_with_proxy,
    )
    feat_main = feat_main.sort_values("session").reset_index(drop=True)
    train_pred = train_pred.sort_values("session").reset_index(drop=True)
    X_main = feat_main[FEATURE_COLUMNS_SHARPE].to_numpy(dtype=np.float64)
    R_main = feat_main["R"].to_numpy(dtype=np.float64)
    f_lin_tr = train_pred["f"].to_numpy(dtype=np.float64)
    sh_lin = float(sharpe(f_lin_tr * R_main))

    model = HistGradientBoostingRegressor(
        max_depth=5,
        max_iter=400,
        learning_rate=0.06,
        l2_regularization=1.0,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=15,
    )
    model.fit(X_main, R_main)
    pred_main = model.predict(X_main)
    mult = -1.0 if float(np.mean(pred_main * R_main)) < 0.0 else 1.0
    std_f = float(np.std(f_lin_tr))
    std_p = float(np.std(pred_main))
    scale = std_f / (std_p + 1e-12)
    w_hgb_tr = mult * pred_main * scale
    lw = float(linear_weight)
    w_ens_tr = lw * f_lin_tr + (1.0 - lw) * w_hgb_tr
    sh_ens = float(sharpe(w_ens_tr * R_main))

    feat_te = _feat_te_pred(data_dir, within_session_split=within_session_split).sort_values("session").reset_index(drop=True)
    X_te = feat_te[FEATURE_COLUMNS_SHARPE].to_numpy(dtype=np.float64)
    pred_te = model.predict(X_te)
    w_hgb_te = mult * pred_te * scale

    lin = sub_lin.sort_values("session").reset_index(drop=True)
    assert np.array_equal(lin["session"].to_numpy(), feat_te["session"].to_numpy())
    w_lin_te = lin["target_position"].to_numpy(dtype=np.float64)
    w_ens_te = lw * w_lin_te + (1.0 - lw) * w_hgb_te

    sub = pd.DataFrame({"session": feat_te["session"].to_numpy(), "target_position": w_ens_te})
    sub = sub.sort_values("session").reset_index(drop=True)
    return sub, sh_lin, sh_ens


__all__ = [
    "submission_cluster_ohe_sharpe_linear",
    "submission_ensemble_linear_hgbr",
    "submission_hgbr_R",
    "submission_mlp_R",
]
