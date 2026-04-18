"""Extended OHLC path statistics for Sharpe training (merged on top of baseline session features)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from datathon_baseline.features import FEATURE_COLUMNS, build_session_features
from datathon_sharpe.features_seen_split import (
    FIRST_HALF_LAST_BAR_IX,
    build_session_features_first_half,
)

# Baseline columns used for Sharpe models (exclude redundant / weak features per analysis).
BASELINE_COLUMNS_SHARPE: list[str] = [c for c in FEATURE_COLUMNS if c != "open_first"]

EPS = 1e-12

# Dropped vs full path set: ret_last_10, ret_last_25, parkinson_var, ret_kurt (low linear attribution).
PATH_EXTRA_COLUMNS: list[str] = [
    "ret_last_5",
    "up_bar_frac",
    "up_minus_down",
    "max_drawdown",
    "p95_abs_bar_ret",
    "vol_first_half",
    "vol_second_half",
    "log_close_slope",
    "log_close_r2",
    "ret_skew",
]

# Path-only Sharpe columns (baseline subset + path extras). Full training list adds sentiment in ``sentiment_features``.
FEATURE_COLUMNS_PATH_SHARPE: list[str] = BASELINE_COLUMNS_SHARPE + PATH_EXTRA_COLUMNS


def compute_extended_path_features(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
) -> dict[str, float]:
    """
    Path statistics for one session's OHLC (sorted by bar_ix, single session only).
    Missing history uses 0.0 for stability with sklearn.
    """
    n = int(close.shape[0])
    z = {c: 0.0 for c in PATH_EXTRA_COLUMNS}
    if n < 1:
        return z

    close = np.asarray(close, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)

    # pct_change alignment with baseline: length n, first NaN -> 0
    rets = pd.Series(close).pct_change().fillna(0.0).to_numpy(dtype=np.float64)
    r_body = rets[1:] if len(rets) > 1 else np.array([], dtype=np.float64)

    def ret_last(k: int) -> float:
        if n > k:
            return float(close[-1] / max(close[-1 - k], EPS) - 1.0)
        return 0.0

    z["ret_last_5"] = ret_last(5)

    if r_body.size > 0:
        z["up_bar_frac"] = float(np.mean(r_body > 0.0))
        z["up_minus_down"] = float(np.sum(r_body > 0.0) - np.sum(r_body < 0.0))
    else:
        z["up_bar_frac"] = 0.0
        z["up_minus_down"] = 0.0

    run_max = np.maximum.accumulate(close)
    dd = (run_max - close) / (run_max + EPS)
    z["max_drawdown"] = float(np.max(dd)) if n else 0.0

    if r_body.size > 0:
        z["p95_abs_bar_ret"] = float(np.percentile(np.abs(r_body), 95.0))
    else:
        z["p95_abs_bar_ret"] = 0.0

    if len(rets) >= 2:
        mid = len(rets) // 2
        a, b = rets[:mid], rets[mid:]
        z["vol_first_half"] = float(a.std(ddof=0)) if a.size else 0.0
        z["vol_second_half"] = float(b.std(ddof=0)) if b.size else 0.0
    else:
        z["vol_first_half"] = 0.0
        z["vol_second_half"] = 0.0

    x = np.arange(n, dtype=np.float64)
    y = np.log(np.maximum(close, EPS))
    if n >= 2:
        coef = np.polyfit(x, y, 1)
        z["log_close_slope"] = float(coef[0])
        y_hat = coef[0] * x + coef[1]
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        z["log_close_r2"] = float(1.0 - ss_res / (ss_tot + EPS)) if ss_tot > EPS else 0.0
    else:
        z["log_close_slope"] = 0.0
        z["log_close_r2"] = 0.0

    if len(r_body) >= 3:
        rs = pd.Series(r_body)
        z["ret_skew"] = float(rs.skew())
    else:
        z["ret_skew"] = 0.0
    return z


def path_features_by_session(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per session: ``session`` plus ``PATH_EXTRA_COLUMNS`` from bars (pre-filtered window)."""
    if bars.empty:
        return pd.DataFrame(columns=["session"] + PATH_EXTRA_COLUMNS)

    rows: list[dict] = []
    for session, g in bars.groupby("session", sort=False):
        g = g.sort_values("bar_ix")
        close = g["close"].to_numpy(dtype=np.float64)
        high = g["high"].to_numpy(dtype=np.float64)
        low = g["low"].to_numpy(dtype=np.float64)
        feats = compute_extended_path_features(close, high, low)
        row = {"session": int(session), **feats}
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["session"] + PATH_EXTRA_COLUMNS)
    return out.sort_values("session").reset_index(drop=True)


def merge_sharpe_path_features(base: pd.DataFrame, bars_for_path: pd.DataFrame) -> pd.DataFrame:
    """Left-merge path columns onto baseline session rows; same ``bars_for_path`` as used for baseline."""
    extra = path_features_by_session(bars_for_path)
    if extra.empty:
        out = base.copy()
        for c in PATH_EXTRA_COLUMNS:
            out[c] = 0.0
        return out
    out = base.merge(extra, on="session", how="left")
    for c in PATH_EXTRA_COLUMNS:
        out[c] = out[c].fillna(0.0)
    return out


def build_session_features_with_path(
    bars: pd.DataFrame,
    headlines: pd.DataFrame | None = None,
    *,
    first_half: bool = False,
) -> pd.DataFrame:
    """
    Baseline session features plus ``PATH_EXTRA_COLUMNS``.

    When ``first_half`` is True, uses bars ``bar_ix <= FIRST_HALF_LAST_BAR_IX`` for both
    baseline and path (matches ``build_session_features_first_half``).
    """
    if first_half:
        base = build_session_features_first_half(bars, headlines)
        bpath = bars.loc[bars["bar_ix"] <= FIRST_HALF_LAST_BAR_IX]
    else:
        base = build_session_features(bars, headlines)
        bpath = bars
    return merge_sharpe_path_features(base, bpath)


__all__ = [
    "BASELINE_COLUMNS_SHARPE",
    "EPS",
    "FEATURE_COLUMNS_PATH_SHARPE",
    "PATH_EXTRA_COLUMNS",
    "build_session_features_with_path",
    "compute_extended_path_features",
    "merge_sharpe_path_features",
    "path_features_by_session",
]
