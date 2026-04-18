"""Distributional targets + monotone maps to positions in [-1, 1].

Policies (``distributional_policy``):

- ``prob_sign``: logistic regression on ``1[R > 0]``, then ``f = 2p - 1`` (monotone in ``p``).
- ``quantile_median``: linear quantile regression at τ=0.5, then ``f = tanh(m / σ)`` with
  ``σ = std(m)`` on the fit rows (monotone in predicted median).
- ``rank_score``: Ridge score on ``R``, map to train empirical rank percentiles, ``f = 2·pct - 1``
  (monotone in score).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression, QuantileRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DistributionalPolicy = Literal["prob_sign", "quantile_median", "rank_score"]


@dataclass
class DistributionalMonoPredictor:
    """Holds fitted objects; ``predict_f`` returns unconstrained scores in roughly [-1, 1]."""

    policy: DistributionalPolicy
    _prob_pipe: Pipeline | None = None
    _quantile_pipe: Pipeline | None = None
    _quantile_sigma: float = 1.0
    _rank_scaler: StandardScaler | None = None
    _rank_ridge: Ridge | None = None
    _rank_s_sorted: np.ndarray | None = None

    def predict_prob_positive(self, X: np.ndarray) -> np.ndarray:
        """For ``prob_sign`` only: return ``P(R > 0 | X)`` per row."""
        if self.policy != "prob_sign" or self._prob_pipe is None:
            raise ValueError("predict_prob_positive is only defined for policy prob_sign")
        X = np.asarray(X, dtype=np.float64)
        return self._prob_pipe.predict_proba(X)[:, 1].astype(np.float64)

    def predict_f(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.policy == "prob_sign":
            assert self._prob_pipe is not None
            p = self._prob_pipe.predict_proba(X)[:, 1].astype(np.float64)
            return 2.0 * p - 1.0
        if self.policy == "quantile_median":
            assert self._quantile_pipe is not None
            m = self._quantile_pipe.predict(X).astype(np.float64)
            sig = max(float(self._quantile_sigma), 1e-12)
            return np.tanh(m / sig)
        if self.policy == "rank_score":
            assert (
                self._rank_scaler is not None
                and self._rank_ridge is not None
                and self._rank_s_sorted is not None
            )
            Xs = self._rank_scaler.transform(X)
            s = self._rank_ridge.predict(Xs).astype(np.float64)
            s_sorted = self._rank_s_sorted
            n = len(s_sorted)
            idx = np.searchsorted(s_sorted, s, side="right")
            pct = idx.astype(np.float64) / float(max(n, 1))
            return 2.0 * pct - 1.0
        raise ValueError(self.policy)


def fit_distributional_mono(
    X_raw: np.ndarray,
    R: np.ndarray,
    *,
    policy: str,
    ridge_reg: float,
    random_state: int,
) -> DistributionalMonoPredictor:
    """
    ``ridge_reg`` is:

    - inverse L2 strength for ``LogisticRegression`` (``C = 1 / ridge_reg``) in ``prob_sign``;
    - ``alpha`` for ``QuantileRegressor`` / ``Ridge`` in other policies.
    """
    allowed = ("prob_sign", "quantile_median", "rank_score")
    if policy not in allowed:
        raise ValueError(f"policy must be one of {allowed}, got {policy!r}")
    policy_t: DistributionalPolicy = policy  # type: ignore[assignment]

    X_raw = np.asarray(X_raw, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    if policy_t == "prob_sign":
        y = (R > 0.0).astype(np.int32)
        C = 1.0 / max(float(ridge_reg), 1e-12)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=5000,
                        random_state=random_state,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        pipe.fit(X_raw, y)
        return DistributionalMonoPredictor(policy=policy_t, _prob_pipe=pipe)

    if policy_t == "quantile_median":
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "qr",
                    QuantileRegressor(
                        quantile=0.5,
                        alpha=float(ridge_reg),
                        solver="highs",
                    ),
                ),
            ]
        )
        pipe.fit(X_raw, R)
        m_tr = pipe.predict(X_raw).astype(np.float64)
        sigma = float(np.std(m_tr))
        if sigma < 1e-12:
            sigma = 1e-12
        return DistributionalMonoPredictor(
            policy=policy_t,
            _quantile_pipe=pipe,
            _quantile_sigma=sigma,
        )

    if policy_t == "rank_score":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_raw)
        ridge = Ridge(alpha=float(ridge_reg), random_state=random_state)
        ridge.fit(Xs, R)
        s_tr = ridge.predict(Xs).astype(np.float64)
        s_sorted = np.sort(s_tr)
        return DistributionalMonoPredictor(
            policy=policy_t,
            _rank_scaler=scaler,
            _rank_ridge=ridge,
            _rank_s_sorted=s_sorted,
        )

    raise ValueError(f"unknown policy: {policy_t!r}")


def shap_linear_parts(
    pred: DistributionalMonoPredictor,
    X_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scaled inputs and linear coefficients for the same independent linear SHAP convention
    as Ridge / Sharpe-linear: ``phi_ij = coef_j * (x_ij - mean_j)`` on scaled columns.
    """
    X_raw = np.asarray(X_raw, dtype=np.float64)
    if pred.policy == "prob_sign":
        assert pred._prob_pipe is not None
        scaler = pred._prob_pipe.named_steps["scaler"]
        clf = pred._prob_pipe.named_steps["clf"]
        Xs = scaler.transform(X_raw)
        coef = clf.coef_.ravel().astype(np.float64)
        return Xs, coef
    if pred.policy == "quantile_median":
        assert pred._quantile_pipe is not None
        scaler = pred._quantile_pipe.named_steps["scaler"]
        qr = pred._quantile_pipe.named_steps["qr"]
        Xs = scaler.transform(X_raw)
        coef = qr.coef_.ravel().astype(np.float64)
        return Xs, coef
    if pred.policy == "rank_score":
        assert pred._rank_scaler is not None and pred._rank_ridge is not None
        Xs = pred._rank_scaler.transform(X_raw)
        coef = pred._rank_ridge.coef_.ravel().astype(np.float64)
        return Xs, coef
    raise ValueError(pred.policy)


def binary_entropy_nats(p: np.ndarray) -> np.ndarray:
    """Binary entropy -p log p - (1-p) log(1-p) with natural log (nats); p clipped to (1e-15, 1-1e-15)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


__all__ = [
    "DistributionalMonoPredictor",
    "DistributionalPolicy",
    "fit_distributional_mono",
    "shap_linear_parts",
    "binary_entropy_nats",
]
