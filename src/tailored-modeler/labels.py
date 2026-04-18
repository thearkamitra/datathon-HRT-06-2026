"""Training labels for the tailored modeler.

The competition scores the Sharpe ratio of ``w_i * R_i`` where

    R_i = close_end_i / close_half_i - 1

and for the train set the end close is present in ``bars_unseen_train.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from paths import BARS_SEEN_TRAIN, BARS_UNSEEN_TRAIN


def train_realized_returns(data_dir: Path) -> pd.DataFrame:
    """One row per training session with ``close_half``, ``close_end`` and ``R``."""
    seen = pd.read_parquet(data_dir / BARS_SEEN_TRAIN, columns=["session", "bar_ix", "close"])
    unseen = pd.read_parquet(
        data_dir / BARS_UNSEEN_TRAIN, columns=["session", "bar_ix", "close"]
    )

    halfway_bar = int(seen["bar_ix"].max())
    end_bar = int(unseen["bar_ix"].max())

    c_half = (
        seen.loc[seen["bar_ix"] == halfway_bar]
        .groupby("session", sort=False)["close"]
        .first()
        .rename("close_half")
    )
    c_end = (
        unseen.loc[unseen["bar_ix"] == end_bar]
        .groupby("session", sort=False)["close"]
        .first()
        .rename("close_end")
    )
    out = pd.concat([c_half, c_end], axis=1).dropna().reset_index()
    out["R"] = out["close_end"] / out["close_half"] - 1.0
    return out.sort_values("session").reset_index(drop=True)
