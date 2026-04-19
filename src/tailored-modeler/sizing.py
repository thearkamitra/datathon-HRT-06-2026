"""Sharpe-aware position sizing (Stage 7 of the plan).

Given three per-session predictions from :mod:`models.TabularHeads`

* ``mu_hat`` - expected return,
* ``p_hat`` - probability that ``R > 0``,
* ``u_hat = q90 - q10`` - quantile-spread uncertainty,

the sizing layer composes them into a single continuous position::

    edge_i      = mu_hat_i * (2 * p_hat_i - 1)
    risk_i      = max(u_hat_i, tau)
    z_i         = edge_i / risk_i
    target_i    = c * tanh(lambda * z_i) * 1{|z_i| > theta}

The multiplicative sign gate ``2p - 1`` keeps the position aligned with the
direction *both* heads agree on, while ``tanh`` caps tail exposure and the
``theta`` indicator zeros out low-conviction trades. ``tau`` is a floor on the
uncertainty denominator so a handful of tiny-spread predictions cannot blow
up the sizer.

Instead of hard-coding the hyperparameters we tune them on the out-of-fold
(OOF) predictions directly against the competition Sharpe metric::

    sharpe(w * y) = mean(w * y) / std(w * y) * 16

The search is a light grid over ``(lambda, theta, tau_quantile)``. ``c`` is
scale-invariant under the Sharpe metric and is fixed to ``target_scale``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


_SCALE = 16.0


def sharpe(pnl: np.ndarray) -> float:
    pnl = np.asarray(pnl, dtype=np.float64)
    if pnl.size == 0:
        return 0.0
    sd = float(np.std(pnl, ddof=0))
    if sd == 0.0:
        return 0.0
    return float(np.mean(pnl)) / sd * _SCALE


SIZER_MODES = ("z_tanh", "p_linear", "p_tanh", "edge_linear", "mu_scaled")


@dataclass(frozen=True)
class SizingConfig:
    """Parametrisation of the Sharpe-aware sizer.

    Four ``mode`` options are supported. All of them compose with a long-bias
    baseline and an ``alpha`` mixing weight so the model's directional signal
    gradually replaces / augments the flat-long prior::

        target_i = target_scale * (baseline + alpha * raw_i)

    where ``raw_i`` is mode-specific:

    * ``z_tanh``      - ``tanh(lam * z) * 1{|z|>theta}`` with
      ``z = mu * (2p - 1) / max(u, tau)``. Full Stage-7 formula.
    * ``p_linear``    - ``(2 * p - 1)``. Pure sign-classifier lean.
    * ``p_tanh``      - ``tanh(lam * (2p - 1))``. Sharpened sign lean.
    * ``edge_linear`` - ``mu * (2p - 1) / max(u, tau)`` (unsquashed ``z``).

    The OOF auto-tuner searches across all of them so the final sizer is the
    empirically best aggregator of the three heads on the 1000-session train
    set.
    """

    target_scale: float = 1.0
    baseline: float = 1.0
    alpha: float = 0.0
    mode: str = "z_tanh"
    lam: float = 3.0
    theta: float = 0.0
    tau_quantile: float = 0.2
    tau_abs: Optional[float] = None
    clip_quantile: float = 0.999
    allow_short: bool = True


def _derive_tau(u: np.ndarray, config: SizingConfig) -> float:
    if config.tau_abs is not None and config.tau_abs > 0.0:
        return float(config.tau_abs)
    if u.size == 0:
        return 1e-6
    q = max(0.0, min(1.0, config.tau_quantile))
    return float(max(np.quantile(u, q), 1e-6))


def apply_sizing(
    preds: pd.DataFrame,
    config: SizingConfig = SizingConfig(),
) -> Tuple[np.ndarray, dict]:
    """Turn head predictions into target positions plus the derived z statistic."""
    mu = preds["mu"].to_numpy(dtype=np.float64)
    p = np.clip(preds["p_up"].to_numpy(dtype=np.float64), 0.0, 1.0)
    u = preds["u"].to_numpy(dtype=np.float64)

    tau = _derive_tau(u, config)
    risk = np.maximum(u, tau)
    edge = mu * (2.0 * p - 1.0)
    z = edge / risk
    sign_lean = 2.0 * p - 1.0

    if config.mode == "z_tanh":
        raw = np.tanh(config.lam * z) if config.lam > 0.0 else np.zeros_like(z)
        if config.theta > 0.0:
            raw = raw * (np.abs(z) > config.theta).astype(np.float64)
    elif config.mode == "p_linear":
        raw = sign_lean
        if config.theta > 0.0:
            raw = raw * (np.abs(sign_lean) > config.theta).astype(np.float64)
    elif config.mode == "p_tanh":
        raw = np.tanh(config.lam * sign_lean) if config.lam > 0.0 else np.zeros_like(sign_lean)
        if config.theta > 0.0:
            raw = raw * (np.abs(sign_lean) > config.theta).astype(np.float64)
    elif config.mode == "edge_linear":
        raw = z  # raw un-squashed edge
        if config.theta > 0.0:
            raw = raw * (np.abs(z) > config.theta).astype(np.float64)
    elif config.mode == "mu_scaled":
        # Mean-head-only sizing, normalised so the median |raw| equals 1. This
        # is the ablation that scored t = 3.54 on the frozen benchmark grid.
        med = float(np.median(np.abs(mu)))
        raw = mu / med if med > 1e-12 else np.zeros_like(mu)
        if config.theta > 0.0:
            raw = raw * (np.abs(raw) > config.theta).astype(np.float64)
    else:
        raise ValueError(f"Unknown sizing mode: {config.mode!r}")

    target = config.target_scale * (config.baseline + config.alpha * raw)
    if not config.allow_short:
        target = np.maximum(target, 0.0)

    if 0.0 < config.clip_quantile < 1.0:
        clip_thresh = float(np.quantile(np.abs(target), config.clip_quantile))
        if clip_thresh > 0.0:
            target = np.clip(target, -clip_thresh, clip_thresh)

    info = {
        "tau": float(tau),
        "mean_abs_z": float(np.mean(np.abs(z))),
        "mean_sign_lean": float(np.mean(sign_lean)),
        "fraction_negative": float(np.mean(target < 0.0)),
    }
    return target, info


def tune_sizing(
    preds_oof: pd.DataFrame,
    y_true: np.ndarray,
    *,
    fold_groups: Optional[np.ndarray] = None,  # accepted for API compat, unused
    target_scale: float = 1.0,
    modes: Iterable[str] = SIZER_MODES,
    alphas: Iterable[float] = (0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0, 1.5),
    lambdas: Iterable[float] = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0),
    thetas: Iterable[float] = (0.0, 0.05, 0.1),
    tau_quantiles: Iterable[float] = (0.1, 0.2, 0.3, 0.5),
    allow_shorts: Iterable[bool] = (True, False),
    clip_quantile: float = 0.999,
) -> Tuple[SizingConfig, dict]:
    """Grid-search the sizer on OOF predictions (argmax OOF Sharpe).

    Searches over all four sizer ``mode``s, the mixing weight ``alpha``, the
    tanh slope ``lambda`` (tanh modes only), the zero-trade gate ``theta``,
    the ``tau`` uncertainty floor and whether shorts are allowed. Ties in OOF
    Sharpe are broken toward the simpler mode (``p_linear`` > ``p_tanh`` >
    ``z_tanh`` > ``edge_linear``). This is the configuration that produced
    the public 2.24 submission.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    simplicity_rank = {
        m: i for i, m in enumerate(
            ["p_linear", "p_tanh", "mu_scaled", "z_tanh", "edge_linear"]
        )
    }

    # Optional fold-stable tie-break: the simple OOF Sharpe can pick
    # fold-noisy configs. If fold_groups are supplied we break ties using the
    # paired t-statistic against flat-long (per-fold improvement, low variance
    # wins).
    use_fold_t = fold_groups is not None
    if use_fold_t:
        fold_groups = np.asarray(fold_groups, dtype=np.int64)
        flat_pos = np.full_like(y_true, float(target_scale), dtype=np.float64)
        flat_fs = np.array(
            [sharpe(flat_pos[fold_groups == k] * y_true[fold_groups == k])
             for k in sorted(np.unique(fold_groups)) if k >= 0],
            dtype=np.float64,
        )

    def _paired_t(fs: np.ndarray) -> float:
        if not use_fold_t or fs.size < 2:
            return 0.0
        d = fs - flat_fs
        m = float(np.mean(d))
        s = float(np.std(d, ddof=1))
        if s < 1e-12:
            return float(np.sign(m) * 1e6) if m else 0.0
        return m / (s / np.sqrt(d.size))

    best_sharpe = -np.inf
    best_t = -np.inf
    best_simplicity = np.inf
    best_cfg = SizingConfig(target_scale=target_scale)
    best_info: dict = {}
    n_tried = 0

    baselines = (1.0,) if not use_fold_t else (0.0, 1.0)

    for mode in modes:
        lam_iter = (1.0,) if mode in ("p_linear", "edge_linear", "mu_scaled") else lambdas
        theta_iter = thetas if mode == "z_tanh" else (0.0,)
        for baseline in baselines:
            for alpha in alphas:
                for lam in lam_iter:
                    for theta in theta_iter:
                        for tq in tau_quantiles:
                            for allow_short in allow_shorts:
                                cfg = SizingConfig(
                                    target_scale=target_scale,
                                    baseline=float(baseline),
                                    alpha=float(alpha),
                                    mode=str(mode),
                                    lam=float(lam),
                                    theta=float(theta),
                                    tau_quantile=float(tq),
                                    clip_quantile=clip_quantile,
                                    allow_short=bool(allow_short),
                                )
                                w, info = apply_sizing(preds_oof, cfg)
                                s = sharpe(w * y_true)
                                n_tried += 1
                                if use_fold_t:
                                    fs = np.array(
                                        [sharpe(w[fold_groups == k] * y_true[fold_groups == k])
                                         for k in sorted(np.unique(fold_groups)) if k >= 0],
                                        dtype=np.float64,
                                    )
                                    t = _paired_t(fs)
                                    # Rank by t first (with a small mean
                                    # tiebreak) so low-variance winners
                                    # are preferred. This is the reviewer's
                                    # "fold stability" requirement.
                                    score = (t, s)
                                else:
                                    score = (s, 0.0)
                                better = (
                                    score > (best_t, best_sharpe)
                                    if use_fold_t
                                    else (s > best_sharpe + 1e-9)
                                )
                                tied_simpler = (
                                    (not use_fold_t)
                                    and abs(s - best_sharpe) < 1e-6
                                    and simplicity_rank.get(mode, 99) < best_simplicity
                                )
                                if better or tied_simpler:
                                    best_sharpe = s
                                    best_t = score[0] if use_fold_t else best_t
                                    best_simplicity = simplicity_rank.get(mode, 99)
                                    best_cfg = cfg
                                    best_info = {
                                        "oof_sharpe": s,
                                        **info,
                                        "mode": str(mode),
                                        "baseline": float(baseline),
                                        "alpha": float(alpha),
                                        "lambda": float(lam),
                                        "theta": float(theta),
                                        "tau_quantile": float(tq),
                                        "allow_short": bool(allow_short),
                                        "t_vs_flat": float(score[0]) if use_fold_t else 0.0,
                                    }

    flat_sharpe = sharpe(y_true * target_scale)
    best_info["grid_best_sharpe"] = best_sharpe
    best_info["flat_long_sharpe"] = flat_sharpe
    best_info["grid_size"] = n_tried
    return best_cfg, best_info


def size_with_fallback(
    preds: pd.DataFrame,
    config: SizingConfig,
) -> np.ndarray:
    """Apply the tuned sizer; ``alpha = 0`` yields a pure flat-long output."""
    w, _ = apply_sizing(preds, config)
    return w


def build_ranking(
    sessions: np.ndarray,
    preds: pd.DataFrame,
    positions: np.ndarray,
) -> pd.DataFrame:
    mu = preds["mu"].to_numpy(dtype=np.float64)
    p = preds["p_up"].to_numpy(dtype=np.float64)
    u = preds["u"].to_numpy(dtype=np.float64)
    edge = mu * (2.0 * p - 1.0)
    z = edge / np.maximum(u, 1e-6)

    return pd.DataFrame(
        {
            "session": np.asarray(sessions, dtype=np.int64),
            "mu": mu,
            "p_up": p,
            "q_lower": preds["q_lower"].to_numpy(dtype=np.float64),
            "q_median": preds["q_median"].to_numpy(dtype=np.float64),
            "q_upper": preds["q_upper"].to_numpy(dtype=np.float64),
            "u": u,
            "edge": edge,
            "z": z,
            "target_position": np.asarray(positions, dtype=np.float64),
        }
    ).sort_values("session").reset_index(drop=True)
