"""Load parquet tables and enumerate test sessions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BARS_SEEN_TRAIN = "bars_seen_train.parquet"
BARS_UNSEEN_TRAIN = "bars_unseen_train.parquet"
BARS_SEEN_PUBLIC_TEST = "bars_seen_public_test.parquet"
BARS_SEEN_PRIVATE_TEST = "bars_seen_private_test.parquet"


def read_bars(data_dir: Path, name: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(data_dir / name, columns=columns)


def list_test_sessions(data_dir: Path) -> list[int]:
    """Union of session ids from public and private seen test bar files."""
    pub = read_bars(data_dir, BARS_SEEN_PUBLIC_TEST, columns=["session"])
    priv = read_bars(data_dir, BARS_SEEN_PRIVATE_TEST, columns=["session"])
    ids = sorted(set(pub["session"].unique()) | set(priv["session"].unique()))
    return [int(x) for x in ids]


