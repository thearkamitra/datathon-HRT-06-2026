#!/usr/bin/env python3
"""Competition stack: extra OHLC + HGBT + Sharpe-linear ensemble → submission CSV."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from datathon_competition.cli import main

if __name__ == "__main__":
    main()
