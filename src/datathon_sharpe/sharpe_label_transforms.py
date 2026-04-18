"""Optional transforms of realized return ``R`` used only inside the Sharpe-linear optimizer."""

from __future__ import annotations

from typing import Literal

import numpy as np

SharpeOptimizerLabel = Literal["identity", "r2_sign_100"]


def r_squared_times_100_signed(r: np.ndarray) -> np.ndarray:
    """
    Map R to ``R**2 * 100 * sign(R)`` (preserves sign, amplifies larger moves; zero -> zero).
    """
    r = np.asarray(r, dtype=np.float64)
    return (r**2) * 100.0 * np.sign(r)


def transform_r_for_optimizer(
    r: np.ndarray,
    label: SharpeOptimizerLabel,
) -> np.ndarray:
    """Map raw realized returns to the vector passed to ``_fit_linear_sharpe`` (warm-start + SLSQP)."""
    if label == "identity":
        return np.asarray(r, dtype=np.float64)
    if label == "r2_sign_100":
        return r_squared_times_100_signed(r)
    raise ValueError(f"Unknown sharpe optimizer label: {label!r}")


__all__ = [
    "SharpeOptimizerLabel",
    "r_squared_times_100_signed",
    "transform_r_for_optimizer",
]
