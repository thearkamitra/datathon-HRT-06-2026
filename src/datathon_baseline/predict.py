"""Fit baselines and build submission targets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
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
from datathon_baseline.metrics import sharpe


class Method(str, Enum):
    ridge = "ridge"
    momentum = "momentum"
    constant = "constant"


@dataclass
class TrainResult:
    method: Method
    train_sharpe: float
    ridge_alpha: float | None


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
    feat_tr = build_session_features(bars_tr)
    feat_tr = feat_tr.merge(labels[["session", "R"]], on="session", how="inner")

    if len(feat_tr) != len(labels):
        raise RuntimeError("Feature / label session alignment failed.")

    feat_tr = feat_tr.sort_values("session").reset_index(drop=True)
    R = feat_tr["R"].to_numpy(dtype=np.float64)

    if method == Method.constant:
        f_tr = np.ones(len(feat_tr), dtype=np.float64)
        model = None
        ra = None
    elif method == Method.momentum:
        f_tr = feat_tr["cum_ret"].to_numpy(dtype=np.float64)
        model = None
        ra = None
    elif method == Method.ridge:
        X = feat_tr[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        y = feat_tr["R"].to_numpy(dtype=np.float64)
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=ridge_reg, random_state=random_state)),
            ]
        )
        model.fit(X, y)
        f_tr = model.predict(X)
        ra = ridge_reg
    else:
        raise ValueError(method)

    mult = -1.0 if float(np.mean(f_tr * R)) < 0 else 1.0
    train_sh = sharpe(mult * f_tr * R)

    bars_pub = read_bars(data_dir, BARS_SEEN_PUBLIC_TEST)
    bars_priv = read_bars(data_dir, BARS_SEEN_PRIVATE_TEST)
    bars_te = pd.concat([bars_pub, bars_priv], ignore_index=True)
    feat_te = build_session_features(bars_te)

    if method == Method.constant:
        f_te = np.ones(len(feat_te), dtype=np.float64)
    elif method == Method.momentum:
        f_te = feat_te["cum_ret"].to_numpy(dtype=np.float64)
    else:
        X_te = feat_te[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        f_te = model.predict(X_te)

    w_te = mult * f_te

    sub = pd.DataFrame({"session": feat_te["session"].to_numpy(), "target_position": w_te})
    sub = sub.sort_values("session").reset_index(drop=True)

    result = TrainResult(method=method, train_sharpe=float(train_sh), ridge_alpha=ra)
    return sub, result
