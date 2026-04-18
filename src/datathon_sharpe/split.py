"""Session pools and deterministic 25+25 splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def sessions_from_bars_csv(path: Path) -> np.ndarray:
    """Unique session ids from a bars CSV (same columns as parquet export)."""
    df = pd.read_csv(path, usecols=["session"])
    return np.sort(df["session"].unique())


def merge_public_private_test_sessions(
    public_csv: Path | None = None,
    private_csv: Path | None = None,
) -> np.ndarray:
    """All session ids appearing in public + private seen-test bar exports."""
    from datathon_sharpe.paths import (
        default_private_test_bars_csv,
        default_public_test_bars_csv,
    )

    pub = public_csv or default_public_test_bars_csv()
    priv = private_csv or default_private_test_bars_csv()
    s1 = sessions_from_bars_csv(pub)
    s2 = sessions_from_bars_csv(priv)
    return np.sort(np.unique(np.concatenate([s1, s2])))


def train_session_pool(data_dir: Path) -> np.ndarray:
    """All training session ids (same ids as `bars_seen_train.parquet`)."""
    from datathon_baseline.io import BARS_SEEN_TRAIN, read_bars

    s = read_bars(data_dir, BARS_SEEN_TRAIN, columns=["session"])
    return np.sort(s["session"].unique())


def split_25_25(pool: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw 50 distinct sessions from `pool`, split into two blocks of 25.

    Returns (sessions_train_block, sessions_label_block) — names reflect a
    monitoring split; both blocks use the same fitted model in our CV helper.
    """
    if len(pool) < 50:
        raise ValueError(f"pool must have at least 50 sessions, got {len(pool)}")
    rng = np.random.default_rng(random_state)
    chosen = rng.choice(pool, size=50, replace=False)
    chosen.sort()
    return chosen[:25].copy(), chosen[25:].copy()
