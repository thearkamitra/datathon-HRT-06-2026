"""Linear probabilistic / heteroskedastic regression heads.

Implements the "best practical version" prescribed in
``plans/probablisticReg.txt``:

Model :math:`Y = R_{50\\to100}` directly with two linear heads fit on the
same standardized design matrix :math:`X`:

1.  **Mean head**: ridge regression ``mu_hat = X @ beta``.
2.  **Variance head**: ridge regression on log-squared OOF residuals,
    ``log(e_i^2 + eps) = X @ gamma``, yielding
    ``sigma2_hat = exp(X @ gamma)``.

Additional optional heads (opt-in via :class:`HeadsConfig`):

3.  **Quantile triplet** (:class:`sklearn.linear_model.QuantileRegressor`)
    at ``q10 / q50 / q90``. Kept for robustness diagnostics and to feed the
    existing ``u = q90 - q10`` sizer contract as a sanity check against the
    Gaussian projection.

The Gaussian projection of ``(mu, sigma)`` into ``(p_up, q10, q50, q90, u)``
is done in :func:`to_sizing_frame` so the shared Sharpe-aware sizer
(``src/tailored-modeler/sizing.py``) can consume the output directly.

The critical statistical hygiene point from the plan is that the variance
head must be fit on *OOF* residuals, not in-sample residuals (in-sample
residuals under ridge are biased toward zero and would under-estimate
sigma). :func:`run_heteroskedastic_cv` handles that protocol:

1.  KFold across sessions; on each fold fit the mean head and collect OOF
    ``mu_hat``.
2.  Take OOF residuals ``e_i = y_i - mu_hat_i``.
3.  KFold a second time on ``log(e_i^2 + eps)`` to get OOF ``sigma2_hat``.
4.  Refit both heads on the full train for test-time prediction.

This closes the loop described in the plan without leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import (
    ElasticNetCV,
    QuantileRegressor,
    Ridge,
    RidgeCV,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadsConfig:
    """Hyperparameters for the mean/variance/quantile heads."""

    mean_regularizer: str = "ridge"  # "ridge" | "elastic_net"
    ridge_alphas: Tuple[float, ...] = (0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
    elastic_l1_ratios: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    variance_alphas: Tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
    variance_eps: float = 1e-6
    quantiles: Tuple[float, float, float] = (0.1, 0.5, 0.9)
    quantile_alpha: float = 0.01
    enable_quantile: bool = True
    use_gaussian_quantiles: bool = True
    random_state: int = 0
    cv_splits: int = 5
    sigma_floor_quantile: float = 0.05


# ---------------------------------------------------------------------------
# Fitted-artefact container
# ---------------------------------------------------------------------------


@dataclass
class FittedHeads:
    scaler: StandardScaler
    mu_model: object
    var_model: Ridge  # ridge on log(res^2 + eps)
    quantile_models: Optional[List[QuantileRegressor]]
    sigma2_floor: float  # lower bound on sigma^2 at predict time
    feature_columns: List[str]
    cfg: HeadsConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fit_mean(X: np.ndarray, y: np.ndarray, cfg: HeadsConfig):
    if cfg.mean_regularizer == "elastic_net":
        model = ElasticNetCV(
            l1_ratio=list(cfg.elastic_l1_ratios),
            alphas=list(cfg.ridge_alphas),
            cv=min(cfg.cv_splits, max(2, X.shape[0] // 30)),
            random_state=cfg.random_state,
            max_iter=5000,
        )
    else:
        model = RidgeCV(
            alphas=list(cfg.ridge_alphas),
            cv=min(cfg.cv_splits, max(2, X.shape[0] // 30)),
        )
    model.fit(X, y)
    return model


def _fit_variance(X: np.ndarray, log_e2: np.ndarray, cfg: HeadsConfig) -> Ridge:
    # RidgeCV doesn't accept alphas of zero; we use a plain Ridge with a CV
    # sweep picked via a simple leave-one-fold-out argmin over ``alphas``
    # because log_e2 is noisy and a full GCV is unnecessary.
    alphas = cfg.variance_alphas
    if X.shape[0] < max(6, 2 * cfg.cv_splits):
        return Ridge(alpha=float(alphas[len(alphas) // 2]), random_state=cfg.random_state).fit(
            X, log_e2
        )

    kf = KFold(n_splits=min(cfg.cv_splits, X.shape[0] // 2),
               shuffle=True, random_state=cfg.random_state)
    best_alpha, best_mse = float(alphas[0]), np.inf
    for a in alphas:
        mses: List[float] = []
        for tr, va in kf.split(X):
            m = Ridge(alpha=float(a), random_state=cfg.random_state).fit(X[tr], log_e2[tr])
            mses.append(float(np.mean((log_e2[va] - m.predict(X[va])) ** 2)))
        mse = float(np.mean(mses))
        if mse < best_mse:
            best_mse, best_alpha = mse, float(a)
    return Ridge(alpha=best_alpha, random_state=cfg.random_state).fit(X, log_e2)


def _fit_quantiles(X: np.ndarray, y: np.ndarray, cfg: HeadsConfig) -> List[QuantileRegressor]:
    out: List[QuantileRegressor] = []
    for q in cfg.quantiles:
        qh = QuantileRegressor(
            quantile=float(q),
            alpha=float(cfg.quantile_alpha),
            fit_intercept=True,
            solver="highs",
        )
        qh.fit(X, y)
        out.append(qh)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_heads(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    oof_residuals_sq: Optional[np.ndarray],
    cfg: HeadsConfig = HeadsConfig(),
) -> FittedHeads:
    """Fit mean + variance (+ optional quantile) heads on the FULL train set.

    ``oof_residuals_sq`` should carry the out-of-fold squared residuals
    produced by :func:`run_heteroskedastic_cv` so that the variance target
    is honest. If ``None`` is passed we fall back to in-sample residuals and
    emit a conservative note via :class:`FittedHeads` (downstream callers
    should prefer the OOF path).
    """
    feat_cols = [c for c in X_train.columns]
    scaler = StandardScaler()
    X = scaler.fit_transform(X_train[feat_cols].to_numpy(dtype=np.float64))
    y = np.asarray(y_train, dtype=np.float64)

    mu_model = _fit_mean(X, y, cfg)

    if oof_residuals_sq is None:
        residuals = y - mu_model.predict(X)
        e2 = residuals * residuals
    else:
        e2 = np.asarray(oof_residuals_sq, dtype=np.float64)

    log_e2 = np.log(np.maximum(e2, cfg.variance_eps))
    var_model = _fit_variance(X, log_e2, cfg)

    qheads = _fit_quantiles(X, y, cfg) if cfg.enable_quantile else None

    # Floor on sigma^2: pick the ``sigma_floor_quantile`` quantile of the
    # predicted sigma^2 on train so a handful of tiny-variance cells do not
    # dominate the sizer. Mirrors the ``tau_quantile`` floor in the existing
    # sizer on ``u``.
    log_var_train = var_model.predict(X)
    sigma2_train = np.exp(log_var_train)
    q = float(max(0.0, min(1.0, cfg.sigma_floor_quantile)))
    floor = float(max(np.quantile(sigma2_train, q), 1e-12))

    return FittedHeads(
        scaler=scaler,
        mu_model=mu_model,
        var_model=var_model,
        quantile_models=qheads,
        sigma2_floor=floor,
        feature_columns=feat_cols,
        cfg=cfg,
    )


def predict_heads(
    heads: FittedHeads,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Return a predictions frame with ``mu``, ``sigma``, ``sigma2``."""
    Xs = heads.scaler.transform(X[heads.feature_columns].to_numpy(dtype=np.float64))
    mu = heads.mu_model.predict(Xs).astype(np.float64)
    sigma2 = np.maximum(np.exp(heads.var_model.predict(Xs)), heads.sigma2_floor)
    sigma = np.sqrt(sigma2)
    out = pd.DataFrame({"mu": mu, "sigma": sigma, "sigma2": sigma2})

    if heads.quantile_models is not None:
        q_preds = [qh.predict(Xs) for qh in heads.quantile_models]
        names = [f"q{int(100 * q)}" for q in heads.cfg.quantiles]
        for n, arr in zip(names, q_preds):
            out[n] = arr.astype(np.float64)
    return out


def to_sizing_frame(
    sessions: np.ndarray,
    preds: pd.DataFrame,
    cfg: HeadsConfig = HeadsConfig(),
) -> pd.DataFrame:
    """Project ``(mu, sigma)`` into the ``(mu, p_up, q_lower/median/upper, u)``
    contract consumed by ``tailored-modeler/sizing.py``.

    Two projection modes are supported:

    * ``use_gaussian_quantiles = True`` (default): assume ``R | X ~ N(mu, sigma^2)``
      and derive ``p_up = Phi(mu / sigma)``, ``q_k = mu + sigma * Phi^{-1}(k)``.
    * ``use_gaussian_quantiles = False``: use the empirical quantile heads
      (``q10 / q50 / q90``) when available; fall back to the Gaussian case
      otherwise.

    This keeps the sizer's ``u = q90 - q10`` uncertainty definition intact.
    """
    mu = preds["mu"].to_numpy(dtype=np.float64)
    sigma = np.maximum(preds["sigma"].to_numpy(dtype=np.float64), 1e-9)

    if (not cfg.use_gaussian_quantiles) and ("q10" in preds and "q90" in preds):
        q_lower = preds["q10"].to_numpy(dtype=np.float64)
        q_median = preds.get("q50", pd.Series(mu)).to_numpy(dtype=np.float64)
        q_upper = preds["q90"].to_numpy(dtype=np.float64)
    else:
        z_lo = float(norm.ppf(cfg.quantiles[0]))
        z_mid = float(norm.ppf(cfg.quantiles[len(cfg.quantiles) // 2]))
        z_hi = float(norm.ppf(cfg.quantiles[-1]))
        q_lower = mu + sigma * z_lo
        q_median = mu + sigma * z_mid
        q_upper = mu + sigma * z_hi

    u = np.maximum(q_upper - q_lower, 1e-6)
    p_up = np.clip(norm.cdf(mu / sigma), 0.0, 1.0)

    return pd.DataFrame(
        {
            "session": np.asarray(sessions, dtype=np.int64),
            "mu": mu,
            "p_up": p_up,
            "q_lower": q_lower,
            "q_median": q_median,
            "q_upper": q_upper,
            "u": u,
            "sigma": sigma,
            "sigma2": sigma * sigma,
        }
    )


# ---------------------------------------------------------------------------
# OOF protocol
# ---------------------------------------------------------------------------


@dataclass
class OOFResult:
    oof_preds: pd.DataFrame           # columns: session, mu, sigma, sigma2 (+q*)
    oof_sizing_frame: pd.DataFrame    # session, mu, p_up, q_lower/median/upper, u
    residuals: np.ndarray
    oof_mse: float
    oof_mean_sharpe_edge: float       # diagnostic: sharpe of sign(mu)*y


def run_heteroskedastic_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    sessions: np.ndarray,
    cfg: HeadsConfig = HeadsConfig(),
) -> OOFResult:
    """Run the two-pass OOF heteroskedastic protocol.

    Pass 1: KFold mean regression on standardized X -> ``oof_mu``.
    Pass 2: KFold variance regression on ``log((y - oof_mu)^2 + eps)``.

    Quantile OOF predictions are computed per fold too when enabled. All
    predictions are returned on the *standardized* feature space that the
    subsequent full-train fit in :func:`fit_heads` will also use.
    """
    feat_cols = list(X.columns)
    y = np.asarray(y, dtype=np.float64)
    sessions = np.asarray(sessions, dtype=np.int64)
    n = len(y)
    n_splits = min(cfg.cv_splits, max(2, n // 2))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)

    oof_mu = np.zeros(n, dtype=np.float64)
    oof_q = (
        {q: np.zeros(n, dtype=np.float64) for q in cfg.quantiles}
        if cfg.enable_quantile else None
    )

    # Pass 1: mean head (and quantile heads) OOF
    for tr, va in kf.split(np.arange(n)):
        scaler = StandardScaler().fit(X.iloc[tr][feat_cols].to_numpy(dtype=np.float64))
        Xtr = scaler.transform(X.iloc[tr][feat_cols].to_numpy(dtype=np.float64))
        Xva = scaler.transform(X.iloc[va][feat_cols].to_numpy(dtype=np.float64))
        m = _fit_mean(Xtr, y[tr], cfg)
        oof_mu[va] = m.predict(Xva)
        if oof_q is not None:
            qheads = _fit_quantiles(Xtr, y[tr], cfg)
            for q, qh in zip(cfg.quantiles, qheads):
                oof_q[q][va] = qh.predict(Xva)

    residuals = y - oof_mu
    log_e2 = np.log(np.maximum(residuals * residuals, cfg.variance_eps))
    oof_log_var = np.zeros(n, dtype=np.float64)

    # Pass 2: variance head OOF on log_e2
    for tr, va in kf.split(np.arange(n)):
        scaler = StandardScaler().fit(X.iloc[tr][feat_cols].to_numpy(dtype=np.float64))
        Xtr = scaler.transform(X.iloc[tr][feat_cols].to_numpy(dtype=np.float64))
        Xva = scaler.transform(X.iloc[va][feat_cols].to_numpy(dtype=np.float64))
        vm = _fit_variance(Xtr, log_e2[tr], cfg)
        oof_log_var[va] = vm.predict(Xva)

    sigma2 = np.exp(oof_log_var)
    # Floor sigma2 against its own OOF low-quantile to avoid div-by-eps blow ups.
    q = float(max(0.0, min(1.0, cfg.sigma_floor_quantile)))
    floor = float(max(np.quantile(sigma2, q), 1e-12))
    sigma2 = np.maximum(sigma2, floor)
    sigma = np.sqrt(sigma2)

    oof_df = pd.DataFrame({
        "session": sessions,
        "mu": oof_mu,
        "sigma": sigma,
        "sigma2": sigma2,
    })
    if oof_q is not None:
        for q_k, arr in oof_q.items():
            oof_df[f"q{int(100 * q_k)}"] = arr

    sizing_frame = to_sizing_frame(sessions, oof_df, cfg=cfg)

    mse = float(np.mean(residuals * residuals))
    # Sign-edge sharpe diagnostic.
    pnl = np.sign(oof_mu) * y
    sd = float(np.std(pnl, ddof=0))
    sign_sharpe = 16.0 * float(np.mean(pnl)) / sd if sd > 1e-12 else 0.0

    return OOFResult(
        oof_preds=oof_df,
        oof_sizing_frame=sizing_frame,
        residuals=residuals,
        oof_mse=mse,
        oof_mean_sharpe_edge=sign_sharpe,
    )
