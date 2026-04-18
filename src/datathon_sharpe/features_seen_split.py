"""Session features from the first 25 seen bars only (bar_ix 0–24)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Inclusive last bar index of the first half of the 50-bar seen window.
FIRST_HALF_LAST_BAR_IX = 24


def build_session_features_first_half(
    bars: pd.DataFrame,
    headlines: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    One row per session using only bars with ``bar_ix <= 24`` (first 25 bars).

    Same schema as ``datathon_baseline.features.build_session_features`` but
    restricted to the first half of the seen window. ``close_last`` is the
    close at bar 24; ``cum_ret`` is relative to the session open (bar 0).
    """
    h_counts: dict[int, int] = {}
    if headlines is not None and not headlines.empty and "bar_ix" in headlines.columns:
        h = headlines.loc[headlines["bar_ix"] <= FIRST_HALF_LAST_BAR_IX]
        h_counts = h.groupby("session").size().to_dict()
    elif headlines is not None and not headlines.empty:
        h_counts = headlines.groupby("session").size().to_dict()

    rows: list[dict] = []
    for session, g in bars.groupby("session", sort=False):
        g = g.loc[g["bar_ix"] <= FIRST_HALF_LAST_BAR_IX].sort_values("bar_ix")
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
