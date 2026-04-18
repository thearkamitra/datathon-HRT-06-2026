"""Repository and data paths."""

from pathlib import Path

# .../datathon-HRT-06-2026/src/datathon_baseline/paths.py -> repo root
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def data_dir() -> Path:
    return REPO_ROOT / "data"
