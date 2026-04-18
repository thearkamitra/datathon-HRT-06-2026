"""Competition metric: PnL and Sharpe (README definition)."""

from __future__ import annotations

import numpy as np


def pnl_from_positions(positions: np.ndarray, realized_returns: np.ndarray) -> np.ndarray:
    """pnl_i = w_i * R_i where R_i = close_end/close_half - 1."""
    return np.asarray(positions, dtype=np.float64) * np.asarray(realized_returns, dtype=np.float64)


def sharpe(pnl: np.ndarray) -> float:
    """sharpe = mean(pnl) / std(pnl) * 16 (population std, ddof=0)."""
    x = np.asarray(pnl, dtype=np.float64)
    if x.size == 0:
        return 0.0
    m = float(np.mean(x))
    s = float(np.std(x, ddof=0))
    if s == 0.0:
        return 0.0
    return m / s * 16.0


def sharpe_for_scalar_alpha(
    alpha: float,
    signal: np.ndarray,
    realized_returns: np.ndarray,
) -> float:
    """Train-time Sharpe when w_i = alpha * f_i."""
    w = alpha * np.asarray(signal, dtype=np.float64)
    return sharpe(pnl_from_positions(w, realized_returns))


def neg_sharpe_linear(
    beta: np.ndarray,
    X_design: np.ndarray,
    R: np.ndarray,
) -> float:
    """Loss = minus train Sharpe for pnl_i = R_i * (X_design @ beta)_i."""
    w = X_design @ beta
    pnl = pnl_from_positions(w, R)
    return -float(sharpe(pnl))
