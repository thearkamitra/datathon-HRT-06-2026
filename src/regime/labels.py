"""Training labels for the regime pipeline.

The competition Sharpe metric evaluates ``w_i * R_i`` with

    R_i = close_end_i / close_half_i - 1.

The end close lives in ``bars_unseen_train.parquet`` only for the training
split; the test splits only ship the first half.
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


def load_full_train_bars(data_dir: Path) -> pd.DataFrame:
    """Load the concatenated 100-bar training trajectories.

    Data layout in this competition:

    * ``bars_seen_train.parquet``   -> bar_ix 0..49 for every training session
    * ``bars_unseen_train.parquet`` -> bar_ix 50..99 for every training session
    * bar_ix is shared (contiguous) across the two files, so a plain ``pd.concat``
      plus per-session ``sort_values('bar_ix')`` produces clean 100-bar
      trajectories.

    This is what the HMM (or clustered HMMs) is fit on so the generative model
    actually learns the latent-state dynamics of the full session, including
    the second half we are asked to forecast. Test-time inference still filters
    to the first 50 bars -- see ``forecast.forecast_sessions_mc`` and its
    ``seen_bars`` argument.
    """
    seen = pd.read_parquet(data_dir / BARS_SEEN_TRAIN)
    unseen = pd.read_parquet(data_dir / BARS_UNSEEN_TRAIN)
    return pd.concat([seen, unseen], ignore_index=True)
