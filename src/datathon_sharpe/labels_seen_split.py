"""Proxy return from the *second* half of the seen window (bars 25–49), using only seen OHLC."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from datathon_baseline.io import BARS_SEEN_TRAIN, read_bars


def proxy_returns_second_seen_half_from_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Per session in ``bars``, R = close(bar 49) / close(bar 24) - 1 (seen OHLC only).
    """
    c24 = (
        bars.loc[bars["bar_ix"] == 24]
        .groupby("session", sort=False)["close"]
        .first()
        .rename("close_24")
    )
    c49 = (
        bars.loc[bars["bar_ix"] == 49]
        .groupby("session", sort=False)["close"]
        .first()
        .rename("close_49")
    )
    out = pd.concat([c24, c49], axis=1).reset_index()
    out["R"] = out["close_49"] / out["close_24"] - 1.0
    return out.sort_values("session").reset_index(drop=True)


def train_proxy_returns_second_seen_half(data_dir: Path) -> pd.DataFrame:
    """
    Same as ``proxy_returns_second_seen_half_from_bars`` for train seen bars only.

    Not the competition label (close_99 / close_49 - 1). Use with features built
    from bars 0–24 only.
    """
    seen = read_bars(data_dir, BARS_SEEN_TRAIN)
    return proxy_returns_second_seen_half_from_bars(seen)
