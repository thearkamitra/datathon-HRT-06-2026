"""Repository and data paths for the probabilistic-regression pipeline."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def data_dir() -> Path:
    return REPO_ROOT / "data"


def submissions_dir() -> Path:
    return REPO_ROOT / "Submissions"


BARS_SEEN_TRAIN = "bars_seen_train.parquet"
BARS_UNSEEN_TRAIN = "bars_unseen_train.parquet"
BARS_SEEN_PUBLIC_TEST = "bars_seen_public_test.parquet"
BARS_SEEN_PRIVATE_TEST = "bars_seen_private_test.parquet"

HEADLINES_SEEN_TRAIN = "headlines_seen_train.parquet"
HEADLINES_UNSEEN_TRAIN = "headlines_unseen_train.parquet"
HEADLINES_SEEN_PUBLIC_TEST = "headlines_seen_public_test.parquet"
HEADLINES_SEEN_PRIVATE_TEST = "headlines_seen_private_test.parquet"

SENTIMENTS_SEEN_TRAIN = "sentiments_seen_train.csv"
SENTIMENTS_SEEN_PUBLIC_TEST = "sentiments_seen_public_test.csv"
SENTIMENTS_SEEN_PRIVATE_TEST = "sentiments_seen_private_test.csv"
