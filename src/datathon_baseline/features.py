"""Session-level OHLC features from the seen window only (bar_ix 0–49)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_session_features(bars: pd.DataFrame, headlines: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    One row per session. `bars` must contain only bars available at decision time
    (train seen or test seen: bar_ix 0..49).
    """
    rows: list[dict] = []
    
    # Pre-calculate headline counts if provided
    h_counts = {}
    if headlines is not None and not headlines.empty:
        h_counts = headlines.groupby("session").size().to_dict()

    for session, g in bars.groupby("session", sort=False):
        g = g.sort_values("bar_ix")
        if g.empty:
            continue
        o0 = float(g["open"].iloc[0])
        c_last = float(g["close"].iloc[-1])
        close = g["close"].astype(np.float64)
        rets = close.pct_change().fillna(0.0)
        vol = float(rets.std(ddof=0))
        hi = float(g["high"].max())
        lo = float(g["low"].min())
        rows.append(
            {
                "session": int(session),
                "open_first": o0,
                "close_last": c_last,
                "cum_ret": c_last / o0 - 1.0,
                "vol": vol,
                "range_hl": hi - lo,
                "max_high": hi,
                "min_low": lo,
                "mean_bar_ret": float(rets.mean()),
                "headline_count": h_counts.get(int(session), 0),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("session").reset_index(drop=True)


FEATURE_COLUMNS = [
    "open_first",
    "close_last",
    "cum_ret",
    "vol",
    "range_hl",
    "max_high",
    "min_low",
    "mean_bar_ret",
    "headline_count",
]
