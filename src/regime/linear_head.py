"""Linear regression head over HMM posterior features.

The brainstorming plan calls this out explicitly as likely to outperform
pure Monte-Carlo continuation on this dataset:

    "For each session, derive cluster posterior probs, state posterior probs,
    state occupancy fractions, session log-likelihood under each cluster
    model, posterior entropy, expected transition counts. Then feed those
    features into ridge regression / small LightGBM / a linear Sharpe-aware
    sizing rule. This is often the best practical version: HMMs do structure
    extraction, simple downstream model does final mapping to R."

On a dataset where simple linear models already beat rich tree stacks, the
HMM acts as a *latent-state featurizer* and a well-regularised linear head
becomes the strongest regime-derived predictor.

This module implements three heads that together match the sizer contract:

* ``mu_head``    : ``sklearn.linear_model.Ridge`` targeting ``R``.
* ``p_up_head``  : ``sklearn.linear_model.LogisticRegression`` targeting ``R > 0``.
* ``quantile``   : pinball-loss quantile regressions at ``q_lower/q_median/q_upper``
                    via ``sklearn.linear_model.QuantileRegressor``.

All heads share the same posterior-feature design matrix (output of
:func:`forecast.session_posterior_features`) and expose ``fit`` / ``predict``
the same way. The pipeline path ``--method m1-linear`` first fits the pooled
HMM on full 100-bar trajectories, then fits the linear heads on first-half
posterior features OOF, tunes the sizer on the OOF predictions, and finally
refits-on-all-train + predicts-on-test.

The heads are intentionally linear (Ridge / Logistic / Quantile): on 1000
training sessions with ~(3K + 2) features this stays high-bias / low-variance
and is the right complement to the HMM structural prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, QuantileRegressor, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class LinearHeadConfig:
    """Hyperparameters for the linear heads over HMM posterior features."""

    ridge_alpha: float = 1.0
    logistic_C: float = 1.0
    quantile_alpha: float = 0.01
    quantiles: Tuple[float, float, float] = (0.1, 0.5, 0.9)
    oof_splits: int = 5
    random_state: int = 0


@dataclass
class _FittedHeads:
    scaler: StandardScaler
    mu: Ridge
    p_up: LogisticRegression
    quantile: List[QuantileRegressor]
    columns: List[str]


def _fit_heads(X: np.ndarray, y: np.ndarray, cfg: LinearHeadConfig) -> _FittedHeads:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    mu = Ridge(alpha=cfg.ridge_alpha, random_state=cfg.random_state)
    mu.fit(Xs, y)

    up_labels = (y > 0.0).astype(int)
    if up_labels.sum() in (0, up_labels.size):
        # Degenerate: all same label -> fall back to the trivial probability.
        p_head = None
    else:
        p_head = LogisticRegression(
            C=cfg.logistic_C,
            max_iter=500,
            random_state=cfg.random_state,
            solver="lbfgs",
        )
        p_head.fit(Xs, up_labels)

    q_heads: List[QuantileRegressor] = []
    for q in cfg.quantiles:
        qh = QuantileRegressor(
            quantile=float(q),
            alpha=cfg.quantile_alpha,
            fit_intercept=True,
            solver="highs",
        )
        qh.fit(Xs, y)
        q_heads.append(qh)

    return _FittedHeads(
        scaler=scaler,
        mu=mu,
        p_up=p_head if p_head is not None else None,
        quantile=q_heads,
        columns=[],
    )


def _predict_heads(
    heads: _FittedHeads,
    X: np.ndarray,
    sessions: np.ndarray,
    cfg: LinearHeadConfig,
) -> pd.DataFrame:
    Xs = heads.scaler.transform(X)
    mu = heads.mu.predict(Xs)
    if heads.p_up is not None:
        p_up = heads.p_up.predict_proba(Xs)[:, 1]
    else:
        p_up = np.full_like(mu, 0.5)
    q_vals = np.stack([qh.predict(Xs) for qh in heads.quantile], axis=0)  # (3, N)
    u = np.maximum(q_vals[-1] - q_vals[0], 1e-6)
    return pd.DataFrame(
        {
            "session": sessions,
            "mu": mu.astype(np.float64),
            "p_up": np.clip(p_up.astype(np.float64), 0.0, 1.0),
            "q_lower": q_vals[0].astype(np.float64),
            "q_median": q_vals[len(cfg.quantiles) // 2].astype(np.float64),
            "q_upper": q_vals[-1].astype(np.float64),
            "u": u.astype(np.float64),
        }
    )


@dataclass
class LinearHeadOOFResult:
    oof: pd.DataFrame
    cfg: LinearHeadConfig
    feature_columns: List[str]


def oof_linear_heads(
    posterior_features: pd.DataFrame,
    y: np.ndarray,
    cfg: LinearHeadConfig = LinearHeadConfig(),
) -> LinearHeadOOFResult:
    """Session-level KFold OOF predictions from the linear heads.

    ``posterior_features`` must include a ``session`` column; the remaining
    columns are the posterior feature matrix. ``y`` is the realized return
    aligned with ``posterior_features['session']``.
    """
    if posterior_features.empty:
        raise ValueError("posterior_features is empty")
    sessions = posterior_features["session"].to_numpy(dtype=np.int64)
    feat_cols = [c for c in posterior_features.columns if c != "session"]
    X = posterior_features[feat_cols].to_numpy(dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.shape[0] != X.shape[0]:
        raise ValueError("y must align with posterior_features rows")

    n = X.shape[0]
    splits = (
        list(KFold(n_splits=cfg.oof_splits, shuffle=True, random_state=cfg.random_state).split(X))
        if n >= cfg.oof_splits
        else [(np.arange(n - max(1, n // 5)), np.arange(n - max(1, n // 5), n))]
    )
    oof_frames: List[pd.DataFrame] = []
    for tr_idx, va_idx in splits:
        heads = _fit_heads(X[tr_idx], y[tr_idx], cfg)
        preds = _predict_heads(heads, X[va_idx], sessions[va_idx], cfg)
        oof_frames.append(preds)
    oof = pd.concat(oof_frames, axis=0, ignore_index=True).sort_values("session").reset_index(drop=True)
    return LinearHeadOOFResult(oof=oof, cfg=cfg, feature_columns=feat_cols)


def fit_and_predict(
    train_posterior: pd.DataFrame,
    y_train: np.ndarray,
    test_posterior: pd.DataFrame,
    cfg: LinearHeadConfig = LinearHeadConfig(),
) -> pd.DataFrame:
    """Fit linear heads on all train posteriors, predict test posteriors."""
    feat_cols = [c for c in train_posterior.columns if c != "session"]
    missing = [c for c in feat_cols if c not in test_posterior.columns]
    if missing:
        raise KeyError(f"test_posterior missing feature columns: {missing[:5]}")
    X_tr = train_posterior[feat_cols].to_numpy(dtype=np.float64)
    X_te = test_posterior[feat_cols].to_numpy(dtype=np.float64)
    heads = _fit_heads(X_tr, np.asarray(y_train, dtype=np.float64), cfg)
    return _predict_heads(heads, X_te, test_posterior["session"].to_numpy(dtype=np.int64), cfg)
