"""Repository paths for `datathon_sharpe`."""

from __future__ import annotations

from pathlib import Path

# .../datathon-HRT-06-2026/src/datathon_sharpe/paths.py -> repo root
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def readable_export_dir() -> Path:
    return REPO_ROOT / "data" / "readable_export"


def default_public_test_bars_csv() -> Path:
    return readable_export_dir() / "bars_seen_public_test.csv"


def default_private_test_bars_csv() -> Path:
    return readable_export_dir() / "bars_seen_private_test.csv"


def default_train_bars_csv() -> Path:
    return readable_export_dir() / "bars_seen_train.csv"
