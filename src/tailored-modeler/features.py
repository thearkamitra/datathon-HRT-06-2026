"""Rich OHLC feature engineering for the first half of each session.

We assume the input DataFrame only contains bars that are visible at decision
time (``bar_ix`` 0..49 for this competition). One row is emitted per session.
Every feature here is computable on both train and test without leakage.

The feature groups mirror what the boosted-tree method needs to estimate a
predictive distribution of the next-half return, with enough signal diversity
for XGBoost to carve out mu and sigma:

* Trend / drift features
* Return-distribution moments (mean, std, skew, kurt, extremes)
* Realized-volatility estimators (Parkinson, Garman-Klass, Rogers-Satchell)
* Candlestick shape statistics (bodies, wicks, up-bar ratio)
* Momentum / mean-reversion across multiple windows
* Drawdown / run-up and extreme timing features
* Moving-average deviations and regime descriptors
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

_EPS = 1e-12


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(b) > _EPS else 0.0


def _autocorr(x: np.ndarray, lag: int) -> float:
    if x.size <= lag + 1:
        return 0.0
    a = x[:-lag]
    b = x[lag:]
    s1 = float(np.std(a, ddof=0))
    s2 = float(np.std(b, ddof=0))
    if s1 < _EPS or s2 < _EPS:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (s1 * s2))


def _skew(x: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    m = float(x.mean())
    s = float(x.std(ddof=0))
    if s < _EPS:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    if x.size < 4:
        return 0.0
    m = float(x.mean())
    s = float(x.std(ddof=0))
    if s < _EPS:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def _max_drawdown(close: np.ndarray) -> float:
    if close.size == 0:
        return 0.0
    peak = np.maximum.accumulate(close)
    dd = close / peak - 1.0
    return float(dd.min())


def _max_runup(close: np.ndarray) -> float:
    if close.size == 0:
        return 0.0
    trough = np.minimum.accumulate(close)
    ru = close / trough - 1.0
    return float(ru.max())


def _parkinson(high: np.ndarray, low: np.ndarray) -> float:
    if high.size == 0:
        return 0.0
    r = np.log(np.maximum(high, _EPS) / np.maximum(low, _EPS))
    return float(np.sqrt(np.mean(r * r) / (4.0 * np.log(2.0))))


def _garman_klass(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> float:
    if o.size == 0:
        return 0.0
    hl = np.log(np.maximum(h, _EPS) / np.maximum(l, _EPS))
    co = np.log(np.maximum(c, _EPS) / np.maximum(o, _EPS))
    val = 0.5 * hl * hl - (2.0 * np.log(2.0) - 1.0) * co * co
    val = np.maximum(val, 0.0)
    return float(np.sqrt(np.mean(val)))


def _rogers_satchell(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> float:
    if o.size == 0:
        return 0.0
    ho = np.log(np.maximum(h, _EPS) / np.maximum(o, _EPS))
    hc = np.log(np.maximum(h, _EPS) / np.maximum(c, _EPS))
    lo = np.log(np.maximum(l, _EPS) / np.maximum(o, _EPS))
    lc = np.log(np.maximum(l, _EPS) / np.maximum(c, _EPS))
    val = ho * hc + lo * lc
    val = np.maximum(val, 0.0)
    return float(np.sqrt(np.mean(val)))


def _slope(y: np.ndarray) -> float:
    n = y.size
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    x -= x.mean()
    yy = y - y.mean()
    denom = float(np.sum(x * x))
    if denom < _EPS:
        return 0.0
    return float(np.sum(x * yy) / denom)


def _ewm_std(x: np.ndarray, span: int) -> float:
    if x.size == 0:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    w = (1.0 - alpha) ** np.arange(x.size - 1, -1, -1)
    w /= w.sum() + _EPS
    mu = float(np.sum(w * x))
    var = float(np.sum(w * (x - mu) ** 2))
    return float(np.sqrt(max(var, 0.0)))


def _window_ret(close: np.ndarray, k: int) -> float:
    if close.size <= k:
        return 0.0
    return _safe_div(close[-1], close[-(k + 1)]) - 1.0


def _ma_dev(close: np.ndarray, k: int) -> float:
    if close.size < k or k <= 0:
        return 0.0
    ma = float(np.mean(close[-k:]))
    return _safe_div(close[-1], ma) - 1.0


_PATH_ANCHORS = (0.25, 0.50, 0.75)
_FACTOR_LAGS = (1, 3, 5, 10, 20)


def _rolling_extrema_ratio(close: np.ndarray, k: int, fn: str) -> float:
    if close.size < k or k <= 0:
        return 0.0
    window = close[-k:]
    ref = float(window.max()) if fn == "max" else float(window.min())
    return _safe_div(close[-1], ref) - 1.0


def _path_value(close: np.ndarray, frac: float) -> float:
    if close.size == 0:
        return 0.0
    idx = int(round(frac * (close.size - 1)))
    idx = max(0, min(close.size - 1, idx))
    return _safe_div(close[idx], close[0]) - 1.0


def _quarter_slope(close: np.ndarray, start_frac: float, end_frac: float) -> float:
    n = close.size
    if n < 3:
        return 0.0
    s = int(round(start_frac * (n - 1)))
    e = int(round(end_frac * (n - 1)))
    s = max(0, min(n - 1, s))
    e = max(0, min(n - 1, e))
    if e <= s + 1:
        return 0.0
    segment = np.log(np.maximum(close[s : e + 1], _EPS))
    return _slope(segment) * (e - s)


def _session_row(session: int, g: pd.DataFrame) -> dict:
    g = g.sort_values("bar_ix")
    o = g["open"].to_numpy(dtype=np.float64)
    h = g["high"].to_numpy(dtype=np.float64)
    l = g["low"].to_numpy(dtype=np.float64)
    c = g["close"].to_numpy(dtype=np.float64)
    n = c.size
    if n == 0:
        return {"session": int(session), "n_bars": 0}

    log_c = np.log(np.maximum(c, _EPS))
    log_ret = np.diff(log_c) if n >= 2 else np.zeros(0)

    o0 = float(o[0])
    c_last = float(c[-1])
    h_max = float(h.max())
    l_min = float(l.min())
    c_mean = float(c.mean())
    c_median = float(np.median(c))

    body = np.abs(c - o)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    bar_range = np.maximum(h - l, 0.0)

    first_half = c[: n // 2] if n // 2 > 0 else c[:1]
    second_half = c[n // 2 :] if n - n // 2 > 0 else c[-1:]

    return {
        "session": int(session),
        "n_bars": int(n),
        # Trend / drift
        "open_first": o0,
        "close_last": c_last,
        "cum_ret": _safe_div(c_last, o0) - 1.0,
        "log_cum_ret": float(log_c[-1] - log_c[0]),
        "close_vs_mean": _safe_div(c_last, c_mean) - 1.0,
        "close_vs_median": _safe_div(c_last, c_median) - 1.0,
        "trend_slope": _slope(log_c) * max(n - 1, 1),
        # Return distribution
        "ret_mean": float(log_ret.mean()) if log_ret.size else 0.0,
        "ret_std": float(log_ret.std(ddof=0)) if log_ret.size else 0.0,
        "ret_skew": _skew(log_ret),
        "ret_kurt": _kurtosis(log_ret),
        "ret_min": float(log_ret.min()) if log_ret.size else 0.0,
        "ret_max": float(log_ret.max()) if log_ret.size else 0.0,
        "ret_abs_mean": float(np.mean(np.abs(log_ret))) if log_ret.size else 0.0,
        # Volatility estimators
        "parkinson_vol": _parkinson(h, l),
        "garman_klass_vol": _garman_klass(o, h, l, c),
        "rogers_satchell_vol": _rogers_satchell(o, h, l, c),
        "ewm_vol_10": _ewm_std(log_ret, span=10),
        "ewm_vol_5": _ewm_std(log_ret[-10:] if log_ret.size > 10 else log_ret, span=5),
        # Range features
        "range_hl": h_max - l_min,
        "range_hl_norm": _safe_div(h_max - l_min, o0),
        "close_pos_in_range": _safe_div(c_last - l_min, h_max - l_min),
        "bar_range_mean": float(bar_range.mean()),
        "bar_range_std": float(bar_range.std(ddof=0)) if n > 1 else 0.0,
        # Candle-shape statistics
        "body_mean": float(body.mean()),
        "body_std": float(body.std(ddof=0)) if n > 1 else 0.0,
        "upper_wick_mean": float(upper_wick.mean()),
        "lower_wick_mean": float(lower_wick.mean()),
        "wick_asymmetry": float(upper_wick.mean() - lower_wick.mean()),
        "up_bar_ratio": float(np.mean(c > o)),
        # Serial dependence
        "ret_ac1": _autocorr(log_ret, 1),
        "ret_ac2": _autocorr(log_ret, 2),
        "ret_ac3": _autocorr(log_ret, 3),
        # Drawdown / run-up
        "max_drawdown": _max_drawdown(c),
        "max_runup": _max_runup(c),
        # Extreme-timing features
        "argmax_pos": _safe_div(float(np.argmax(c)), max(n - 1, 1)),
        "argmin_pos": _safe_div(float(np.argmin(c)), max(n - 1, 1)),
        # Windowed momentum
        "ret_last_5": _window_ret(c, 5),
        "ret_last_10": _window_ret(c, 10),
        "ret_last_20": _window_ret(c, 20),
        "ret_first_10": _safe_div(c[min(10, n - 1)], c[0]) - 1.0 if n > 1 else 0.0,
        # Moving-average deviations
        "ma_dev_5": _ma_dev(c, 5),
        "ma_dev_10": _ma_dev(c, 10),
        "ma_dev_20": _ma_dev(c, 20),
        # Regime descriptors (first half vs second half of seen window)
        "half_ratio": _safe_div(float(second_half.mean()), float(first_half.mean())) - 1.0,
        "half_vol_ratio": _safe_div(
            float(second_half.std(ddof=0)),
            float(first_half.std(ddof=0)) + _EPS,
        ),
        # Trend strength vs noise (|drift| / vol) -- discriminates trending vs chop regimes.
        "trend_strength": _safe_div(abs(float(log_c[-1] - log_c[0])), float(log_ret.std(ddof=0)) + _EPS),
        # Breakout flags: is the last close pressing against the seen-half extremes?
        "breakout_up": float(_safe_div(c_last - h_max, abs(h_max) + _EPS)),
        "breakout_dn": float(_safe_div(c_last - l_min, abs(l_min) + _EPS)),
        # Path-shape anchors at 25 / 50 / 75 percent of the seen window.
        **{
            f"path_ret_q{int(100 * f)}": _path_value(c, f) for f in _PATH_ANCHORS
        },
        # Quarter-wise log-price slopes (early / middle / late sub-trends).
        "slope_q1": _quarter_slope(c, 0.0, 0.25),
        "slope_q2": _quarter_slope(c, 0.25, 0.5),
        "slope_q3": _quarter_slope(c, 0.5, 0.75),
        "slope_q4": _quarter_slope(c, 0.75, 1.0),
        # Rolling-extrema ratios (pressure against local highs / lows).
        "close_vs_max_5": _rolling_extrema_ratio(c, 5, "max"),
        "close_vs_min_5": _rolling_extrema_ratio(c, 5, "min"),
        "close_vs_max_20": _rolling_extrema_ratio(c, 20, "max"),
        "close_vs_min_20": _rolling_extrema_ratio(c, 20, "min"),
        # OHLC factor-mining style features: multi-lag differences / ratios.
        **{f"close_ret_lag_{k}": _window_ret(c, k) for k in _FACTOR_LAGS},
        **{f"high_minus_low_lag_{k}":
           _safe_div(h[-1] - l[max(0, n - 1 - k)], o0) for k in _FACTOR_LAGS},
        # Last-window vs first-window vol ratio (volatility regime shift).
        "vol_ratio_last10_first10": _safe_div(
            float(np.std(log_ret[-10:], ddof=0)) if log_ret.size >= 10 else 0.0,
            float(np.std(log_ret[:10], ddof=0)) + _EPS if log_ret.size >= 10 else _EPS,
        ),
        # Recency-weighted mean return (EMA-style, gives more weight to later bars).
        "recency_weighted_ret": (
            float(np.dot(log_ret, np.exp(np.linspace(-2.0, 0.0, log_ret.size))) /
                  np.exp(np.linspace(-2.0, 0.0, log_ret.size)).sum())
            if log_ret.size else 0.0
        ),
    }


def build_session_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Build one feature row per session from OHLC bars.

    Parameters
    ----------
    bars:
        DataFrame with columns ``session, bar_ix, open, high, low, close``
        restricted to bars visible at decision time.
    """
    if bars.empty:
        return pd.DataFrame()
    rows = [_session_row(int(sess), g) for sess, g in bars.groupby("session", sort=False)]
    out = pd.DataFrame(rows).sort_values("session").reset_index(drop=True)
    return out


def feature_columns(df: pd.DataFrame, extra_drop: Iterable[str] | None = None) -> list[str]:
    """Return model-usable feature columns (exclude ids/labels)."""
    drop = {"session", "R", "close_half", "close_end"}
    if extra_drop:
        drop |= set(extra_drop)
    return [c for c in df.columns if c not in drop]
