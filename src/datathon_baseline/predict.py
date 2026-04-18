"""Fit baselines and build submission targets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from datathon_baseline.features import FEATURE_COLUMNS, build_session_features
from datathon_baseline.io import (
    BARS_SEEN_PRIVATE_TEST,
    BARS_SEEN_PUBLIC_TEST,
    BARS_SEEN_TRAIN,
    read_bars,
)
from datathon_baseline.labels import train_realized_returns
from datathon_baseline.metrics import neg_sharpe_linear, sharpe


class Method(str, Enum):
    sharpe_linear = "sharpe_linear"
    ridge = "ridge"
    momentum = "momentum"
    constant = "constant"


@dataclass
class TrainResult:
    method: Method
    train_sharpe: float
    ridge_alpha: float | None
    sharpe_opt_message: str | None = None


def _fit_linear_sharpe(
    X_raw: np.ndarray,
    R: np.ndarray,
    *,
    random_state: int,
) -> tuple[StandardScaler, np.ndarray, str]:
    """
    Maximize train Sharpe with w_i = (X_design @ beta)_i, X_design = [1 | X_scaled],
    subject to ||beta||_2 = 1 (otherwise Sharpe is scale-invariant along rays).
    """
    rng = np.random.default_rng(random_state)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_raw)
    n, d = Xs.shape
    Xd = np.column_stack([np.ones(n, dtype=np.float64), Xs])

    ridge = Ridge(alpha=1.0, random_state=random_state)
    ridge.fit(Xs, R)
    beta0 = np.concatenate([[ridge.intercept_], ridge.coef_])
    nrm = float(np.linalg.norm(beta0))
    if nrm < 1e-12:
        beta0 = rng.standard_normal(d + 1)
        nrm = float(np.linalg.norm(beta0))
    beta0 = beta0 / nrm

    res = minimize(
        neg_sharpe_linear,
        beta0,
        args=(Xd, R),
        method="SLSQP",
        constraints={"type": "eq", "fun": lambda b: float(np.dot(b, b) - 1.0)},
        options={"maxiter": 2000, "ftol": 1e-10},
    )
    beta = res.x.astype(np.float64)
    beta = beta / (float(np.linalg.norm(beta)) + 1e-15)
    return scaler, beta, res.message


def fit_and_predict(
    data_dir: Path,
    method: Method,
    ridge_reg: float = 1.0,
    random_state: int = 0,
) -> tuple[pd.DataFrame, TrainResult]:
    """
    Train on seen+unseen-derived labels; predict positions for public+private test sessions.
    Returns submission DataFrame and training diagnostics.
    """
    labels = train_realized_returns(data_dir)
    bars_tr = read_bars(data_dir, BARS_SEEN_TRAIN)

    # Load train headlines
    try:
        headlines_tr = pd.read_parquet(data_dir / "headlines_seen_train.parquet")
    except Exception:
        headlines_tr = None

    feat_tr = build_session_features(bars_tr, headlines_tr)
    feat_tr = feat_tr.merge(labels[["session", "R"]], on="session", how="inner")

    if len(feat_tr) != len(labels):
        raise RuntimeError("Feature / label session alignment failed.")

    feat_tr = feat_tr.sort_values("session").reset_index(drop=True)
    R = feat_tr["R"].to_numpy(dtype=np.float64)
    X_train = feat_tr[FEATURE_COLUMNS].to_numpy(dtype=np.float64)

    model = None
    ra: float | None = None
    scaler_sharpe: StandardScaler | None = None
    beta_sharpe: np.ndarray | None = None
    opt_msg: str | None = None
    f_tr: np.ndarray

    if method == Method.constant:
        f_tr = np.ones(len(feat_tr), dtype=np.float64)
    elif method == Method.momentum:
        f_tr = feat_tr["cum_ret"].to_numpy(dtype=np.float64)
    elif method == Method.ridge:
        y = feat_tr["R"].to_numpy(dtype=np.float64)
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=ridge_reg, random_state=random_state)),
            ]
        )
        model.fit(X_train, y)
        f_tr = model.predict(X_train)
        ra = ridge_reg
    elif method == Method.sharpe_linear:
        scaler_sharpe, beta_sharpe, opt_msg = _fit_linear_sharpe(
            X_train, R, random_state=random_state
        )
        Xd_tr = np.column_stack(
            [np.ones(len(feat_tr), dtype=np.float64), scaler_sharpe.transform(X_train)]
        )
        f_tr = Xd_tr @ beta_sharpe
    else:
        raise ValueError(method)

    if method == Method.sharpe_linear:
        train_sh = sharpe(f_tr * R)
        mult = 1.0
    else:
        mult = -1.0 if float(np.mean(f_tr * R)) < 0 else 1.0
        train_sh = sharpe(mult * f_tr * R)

    bars_pub = read_bars(data_dir, BARS_SEEN_PUBLIC_TEST)
    bars_priv = read_bars(data_dir, BARS_SEEN_PRIVATE_TEST)
    bars_te = pd.concat([bars_pub, bars_priv], ignore_index=True)
    try:
        h_pub = pd.read_parquet(data_dir / "headlines_seen_public_test.parquet")
        h_priv = pd.read_parquet(data_dir / "headlines_seen_private_test.parquet")
        headlines_te = pd.concat([h_pub, h_priv], ignore_index=True)
    except Exception:
        headlines_te = None

    feat_te = build_session_features(bars_te, headlines_te)
    X_test = feat_te[FEATURE_COLUMNS].to_numpy(dtype=np.float64)

    if method == Method.constant:
        f_te = np.ones(len(feat_te), dtype=np.float64)
    elif method == Method.momentum:
        f_te = feat_te["cum_ret"].to_numpy(dtype=np.float64)
    elif method == Method.sharpe_linear:
        assert scaler_sharpe is not None and beta_sharpe is not None
        Xd_te = np.column_stack(
            [np.ones(len(feat_te), dtype=np.float64), scaler_sharpe.transform(X_test)]
        )
        f_te = Xd_te @ beta_sharpe
    else:
        f_te = model.predict(X_test)

    w_te = mult * f_te

    sub = pd.DataFrame({"session": feat_te["session"].to_numpy(), "target_position": w_te})
    sub = sub.sort_values("session").reset_index(drop=True)

    result = TrainResult(
        method=method,
        train_sharpe=float(train_sh),
        ridge_alpha=ra,
        sharpe_opt_message=opt_msg if method == Method.sharpe_linear else None,
    )
    return sub, result
