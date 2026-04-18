"""Training labels: realized return R from seen (half) to unseen (end)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from datathon_baseline.io import BARS_SEEN_TRAIN, BARS_UNSEEN_TRAIN, read_bars


def train_realized_returns(data_dir: Path) -> pd.DataFrame:
    """
    One row per training session:
    - close_half: close at bar_ix 49 (end of seen segment)
    - close_end: close at bar_ix 99 (end of full session in train)
    - R: close_end / close_half - 1  (same as README pnl factor when w=1)
    """
    seen = read_bars(data_dir, BARS_SEEN_TRAIN)
    unseen = read_bars(data_dir, BARS_UNSEEN_TRAIN)

    c_half = (
        seen.loc[seen["bar_ix"] == 49]
        .groupby("session", sort=False)["close"]
        .first()
        .rename("close_half")
    )
    c_end = (
        unseen.loc[unseen["bar_ix"] == 99]
        .groupby("session", sort=False)["close"]
        .first()
        .rename("close_end")
    )
    out = pd.concat([c_half, c_end], axis=1).reset_index()
    out["R"] = out["close_end"] / out["close_half"] - 1.0
    return out.sort_values("session").reset_index(drop=True)
