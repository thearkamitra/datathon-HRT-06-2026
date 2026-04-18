"""Bar-level OHLC -> HMM emission vectors.

Method 1 of the regime plan asks for a compact per-bar emission that gives the
HMM just enough information to discriminate a handful of shared latent states
(trend / mean-revert / chop / shock) without drowning in redundant signal.

For each session we turn the ``T`` seen bars into a ``(T, F)`` matrix whose
columns are:

* ``r``          - close-to-close log return ``log(C_t / C_{t-1})`` (0 at t=0)
* ``oc``         - intra-bar drift ``(C_t - O_t) / O_t``
* ``hl``         - log high/low range ``log(H_t / L_t)``
* ``co_level``   - level vs session open ``log(C_t / C_0)``
* ``rvol_3``     - rolling realized vol of ``r`` over the last 3 bars
* ``rvol_5``     - rolling realized vol of ``r`` over the last 5 bars
* ``upper_wick`` - ``(H_t - max(O_t, C_t)) / range_t``, 0 if range is 0
* ``lower_wick`` - ``(min(O_t, C_t) - L_t) / range_t``
* ``body_frac``  - ``|C_t - O_t| / range_t``
* ``range_pos``  - close position in rolling 5-bar range (0 at low, 1 at high)

These are all scale-free (returns, ratios, fractions) so that the HMM can be
fit on the concatenation of all 1000 training sessions without leaking a
per-session scale. :func:`build_emission_matrix` returns the dense matrix plus
``lengths`` array expected by :class:`hmmlearn.hmm.GaussianHMM`.

We also emit :func:`build_seen_bars_returns` which returns just the per-bar
log-return array per session: this is the quantity we simulate to forecast the
second-half return at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd


_EPS = 1e-12


DEFAULT_EMISSION_COLUMNS: Tuple[str, ...] = (
    "r",
    "oc",
    "hl",
    "co_level",
    "rvol_3",
    "rvol_5",
    "upper_wick",
    "lower_wick",
    "body_frac",
    "range_pos",
)


@dataclass(frozen=True)
class EmissionConfig:
    """Runtime knobs for the bar-level emission featurizer."""

    columns: Tuple[str, ...] = DEFAULT_EMISSION_COLUMNS
    rvol_windows: Tuple[int, ...] = (3, 5)
    range_pos_window: int = 5
    # Column index of ``r`` inside the emission vector: used by the Monte-Carlo
    # continuation forecaster to accumulate simulated log-returns.
    return_column: str = "r"

    def return_index(self) -> int:
        return self.columns.index(self.return_column)


@dataclass
class SessionEmissions:
    """Per-session bundle of ordered bars + emission matrix."""

    session: int
    # (T,) array of log-returns at each seen bar (first element = 0).
    log_returns: np.ndarray
    # (T, F) emission matrix.
    features: np.ndarray
    # ``close`` price at the last seen bar (used downstream only for diagnostics).
    close_last: float = field(default=float("nan"))


def _rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    """Causal rolling std of ``x`` with a minimum of 2 observations.

    Returns an array of the same length as ``x`` with zeros for the first
    positions where the window is not yet filled enough.
    """
    n = x.size
    out = np.zeros(n, dtype=np.float64)
    if n == 0 or win <= 1:
        return out
    for t in range(n):
        start = max(0, t - win + 1)
        window = x[start : t + 1]
        if window.size >= 2:
            out[t] = float(window.std(ddof=0))
    return out


def _range_position(close: np.ndarray, high: np.ndarray, low: np.ndarray, win: int) -> np.ndarray:
    """Causal ``(close - roll_low) / (roll_high - roll_low)`` in [0, 1]."""
    n = close.size
    out = np.full(n, 0.5, dtype=np.float64)
    for t in range(n):
        start = max(0, t - win + 1)
        hi = float(high[start : t + 1].max())
        lo = float(low[start : t + 1].min())
        span = hi - lo
        if span > _EPS:
            out[t] = float((close[t] - lo) / span)
    return out


def _session_emission_matrix(g: pd.DataFrame, cfg: EmissionConfig) -> SessionEmissions:
    g = g.sort_values("bar_ix")
    session = int(g["session"].iat[0])
    o = g["open"].to_numpy(dtype=np.float64)
    h = g["high"].to_numpy(dtype=np.float64)
    l = g["low"].to_numpy(dtype=np.float64)
    c = g["close"].to_numpy(dtype=np.float64)

    n = c.size
    log_c = np.log(np.maximum(c, _EPS))
    r = np.zeros(n, dtype=np.float64)
    if n >= 2:
        r[1:] = np.diff(log_c)

    bar_range = np.maximum(h - l, 0.0)
    safe_range = np.where(bar_range > _EPS, bar_range, 1.0)
    upper_wick = (h - np.maximum(o, c)) / safe_range
    lower_wick = (np.minimum(o, c) - l) / safe_range
    body_frac = np.abs(c - o) / safe_range
    upper_wick = np.where(bar_range > _EPS, upper_wick, 0.0)
    lower_wick = np.where(bar_range > _EPS, lower_wick, 0.0)
    body_frac = np.where(bar_range > _EPS, body_frac, 0.0)

    oc = (c - o) / np.where(np.abs(o) > _EPS, o, 1.0)
    hl = np.log(np.maximum(h, _EPS) / np.maximum(l, _EPS))
    co_level = log_c - log_c[0]

    rvol_cols = {f"rvol_{w}": _rolling_std(r, w) for w in cfg.rvol_windows}
    range_pos = _range_position(c, h, l, cfg.range_pos_window)

    pool = {
        "r": r,
        "oc": oc,
        "hl": hl,
        "co_level": co_level,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "body_frac": body_frac,
        "range_pos": range_pos,
        **rvol_cols,
    }
    missing = [col for col in cfg.columns if col not in pool]
    if missing:
        raise KeyError(f"Unknown emission columns: {missing}")
    mat = np.column_stack([pool[col] for col in cfg.columns]).astype(np.float64)

    return SessionEmissions(
        session=session,
        log_returns=r,
        features=mat,
        close_last=float(c[-1]) if n > 0 else float("nan"),
    )


@dataclass
class EmissionBundle:
    """Dense HMM training bundle concatenating multiple sessions.

    ``X`` has shape ``(sum(lengths), F)``, ``lengths`` is the ``(n_sessions,)``
    array required by :mod:`hmmlearn`, and ``sessions`` is the parallel session
    identifier array. :attr:`per_session` lists individual
    :class:`SessionEmissions` for downstream per-session work (forecasting,
    clustering).
    """

    X: np.ndarray
    lengths: np.ndarray
    sessions: np.ndarray
    columns: Tuple[str, ...]
    per_session: List[SessionEmissions]

    def return_index(self, config: EmissionConfig) -> int:
        return config.return_index()

    def split_features(self) -> List[np.ndarray]:
        """Return the list of ``(T, F)`` feature matrices, one per session.

        This is cheaper and cleaner than re-slicing :attr:`X` + :attr:`lengths`
        every time we need per-session access (Monte-Carlo forecasting loops
        hit this path for every test session).
        """
        return [s.features for s in self.per_session]


def build_session_emissions(
    bars: pd.DataFrame,
    config: EmissionConfig = EmissionConfig(),
) -> List[SessionEmissions]:
    """Return per-session emission bundles, sorted by session id."""
    if bars.empty:
        return []
    out: List[SessionEmissions] = []
    for sess, g in bars.groupby("session", sort=True):
        _ = sess  # sort order already guaranteed by groupby(sort=True)
        out.append(_session_emission_matrix(g, config))
    return out


def build_emission_bundle(
    bars: pd.DataFrame,
    config: EmissionConfig = EmissionConfig(),
) -> EmissionBundle:
    """Concatenate the per-session emission matrices for the hmmlearn API.

    hmmlearn's multi-sequence interface expects ``X`` as the row-wise
    concatenation of all sequences and ``lengths`` as the per-sequence length.
    """
    sessions = build_session_emissions(bars, config)
    if not sessions:
        F = len(config.columns)
        return EmissionBundle(
            X=np.zeros((0, F), dtype=np.float64),
            lengths=np.zeros(0, dtype=np.int64),
            sessions=np.zeros(0, dtype=np.int64),
            columns=tuple(config.columns),
            per_session=[],
        )
    X = np.concatenate([s.features for s in sessions], axis=0)
    lengths = np.array([s.features.shape[0] for s in sessions], dtype=np.int64)
    sess_ids = np.array([s.session for s in sessions], dtype=np.int64)
    return EmissionBundle(
        X=X,
        lengths=lengths,
        sessions=sess_ids,
        columns=tuple(config.columns),
        per_session=sessions,
    )


def session_summary_features(sessions: List[SessionEmissions]) -> pd.DataFrame:
    """Coarse per-session summary features for Method-2 initial clustering.

    These are the same "session archetype" features called out in the plan:
    cumulative seen-half return, realized vol, max drawdown, up-bar fraction,
    rolling vol slope, quarter slopes, skew, kurtosis, average candle body and
    wick asymmetry.
    """
    if not sessions:
        return pd.DataFrame()
    rows: List[dict] = []
    for s in sessions:
        r = s.log_returns
        feat = s.features
        close_proxy = np.exp(np.cumsum(r))  # starts at 1
        n = r.size
        std = float(r.std(ddof=0)) if n else 0.0
        q = max(n // 4, 1)

        def _slope(y: np.ndarray) -> float:
            k = y.size
            if k < 2:
                return 0.0
            x = np.arange(k, dtype=np.float64)
            x -= x.mean()
            yy = y - y.mean()
            denom = float(np.sum(x * x))
            if denom < _EPS:
                return 0.0
            return float(np.sum(x * yy) / denom)

        skew = 0.0
        kurt = 0.0
        if n >= 3 and std > _EPS:
            z = (r - r.mean()) / std
            skew = float(np.mean(z ** 3))
            kurt = float(np.mean(z ** 4) - 3.0)

        peak = np.maximum.accumulate(close_proxy) if n > 0 else np.zeros(0)
        trough = np.minimum.accumulate(close_proxy) if n > 0 else np.zeros(0)
        mdd = float((close_proxy / np.where(peak > _EPS, peak, 1.0) - 1.0).min()) if n else 0.0
        mru = float((close_proxy / np.where(trough > _EPS, trough, 1.0) - 1.0).max()) if n else 0.0

        body = float(feat[:, 8].mean()) if feat.shape[1] > 8 else 0.0
        upper_w = float(feat[:, 6].mean()) if feat.shape[1] > 6 else 0.0
        lower_w = float(feat[:, 7].mean()) if feat.shape[1] > 7 else 0.0

        rows.append(
            {
                "session": s.session,
                "cum_log_ret": float(r.sum()),
                "realized_vol": std,
                "up_bar_frac": float(np.mean(r > 0.0)) if n else 0.0,
                "ret_skew": skew,
                "ret_kurt": kurt,
                "max_drawdown": mdd,
                "max_runup": mru,
                "avg_body_frac": body,
                "wick_asymmetry": upper_w - lower_w,
                "slope_q1": _slope(np.log(np.maximum(close_proxy[:q], _EPS))) if n else 0.0,
                "slope_q2": _slope(np.log(np.maximum(close_proxy[q : 2 * q], _EPS))) if n else 0.0,
                "slope_q3": _slope(np.log(np.maximum(close_proxy[2 * q : 3 * q], _EPS))) if n else 0.0,
                "slope_q4": _slope(np.log(np.maximum(close_proxy[3 * q :], _EPS))) if n else 0.0,
                "ret_abs_mean": float(np.mean(np.abs(r))) if n else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("session").reset_index(drop=True)
