#!/usr/bin/env python3
"""Same as: uv run generate-headline-submissions"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from datathon_headline_pipeline.all_submissions import main

if __name__ == "__main__":
    main()
