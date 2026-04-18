#!/usr/bin/env python3
"""Thin wrapper: same as `uv run train-predict-headlines` (see datathon_headline_pipeline.cli)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from datathon_headline_pipeline.cli import main

if __name__ == "__main__":
    main()
