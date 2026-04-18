"""Sharpe-aware position sizing for the regime pipeline.

The forecaster (:mod:`forecast`) produces a per-session prediction frame with
columns ``mu / p_up / q_lower / q_median / q_upper / u``. Given those three
primary signals (plus the quantile-spread uncertainty ``u = q_upper - q_lower``),
this module composes them into a continuous position::

    edge_i   = mu_i * (2 * p_up_i - 1)
    risk_i   = max(u_i, tau)
    z_i      = edge_i / risk_i
    target_i = target_scale * (baseline + alpha * f_mode(z, p, ...))

The sizer is identical in spirit to the one used by ``tailored-modeler``
(it needs to match the evaluator's Sharpe definition), but lives in the
regime module so the pipeline has no cross-dependency.
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


SIZER_MODES = ("z_tanh", "p_linear", "p_tanh", "edge_linear")


@dataclass(frozen=True)
class SizingConfig:
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
        raw = z
        if config.theta > 0.0:
            raw = raw * (np.abs(z) > config.theta).astype(np.float64)
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
    target_scale: float = 1.0,
    modes: Iterable[str] = SIZER_MODES,
    alphas: Iterable[float] = (0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0, 1.5),
    lambdas: Iterable[float] = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0),
    thetas: Iterable[float] = (0.0, 0.05, 0.1),
    tau_quantiles: Iterable[float] = (0.1, 0.2, 0.3, 0.5),
    allow_shorts: Iterable[bool] = (True, False),
    clip_quantile: float = 0.999,
) -> Tuple[SizingConfig, dict]:
    """Grid-search the sizer for max OOF Sharpe (tie-break toward simpler modes)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    simplicity_rank = {m: i for i, m in enumerate(["p_linear", "p_tanh", "z_tanh", "edge_linear"])}

    best_sharpe = -np.inf
    best_simplicity = np.inf
    best_cfg = SizingConfig(target_scale=target_scale)
    best_info: dict = {}
    n_tried = 0

    for mode in modes:
        lam_iter = (1.0,) if mode in ("p_linear", "edge_linear") else lambdas
        theta_iter = thetas if mode == "z_tanh" else (0.0,)
        for alpha in alphas:
            for lam in lam_iter:
                for theta in theta_iter:
                    for tq in tau_quantiles:
                        for allow_short in allow_shorts:
                            cfg = SizingConfig(
                                target_scale=target_scale,
                                baseline=1.0,
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
                            if s > best_sharpe or (
                                abs(s - best_sharpe) < 1e-6
                                and simplicity_rank.get(mode, 99) < best_simplicity
                            ):
                                best_sharpe = s
                                best_simplicity = simplicity_rank.get(mode, 99)
                                best_cfg = cfg
                                best_info = {
                                    "oof_sharpe": s,
                                    **info,
                                    "mode": str(mode),
                                    "alpha": float(alpha),
                                    "lambda": float(lam),
                                    "theta": float(theta),
                                    "tau_quantile": float(tq),
                                    "allow_short": bool(allow_short),
                                }

    flat_sharpe = sharpe(y_true * target_scale)
    best_info["grid_best_sharpe"] = best_sharpe
    best_info["flat_long_sharpe"] = flat_sharpe
    best_info["grid_size"] = n_tried
    return best_cfg, best_info


def size_with_fallback(preds: pd.DataFrame, config: SizingConfig) -> np.ndarray:
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
