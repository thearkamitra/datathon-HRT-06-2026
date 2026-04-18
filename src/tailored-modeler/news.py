"""News-aware feature hook.

The tailored modeler is designed to run in two modes:

* ``use_news=False`` (default): a pure OHLC boosted-tree pipeline. The news
  helpers are *not* invoked and no text models, embeddings or LLM calls are
  loaded. This is the only active code path right now.
* ``use_news=True``: reserved for a later upgrade where headline-level
  sentiment / entity features will be merged into the feature table. Today,
  attempting to enable this mode raises ``NotImplementedError`` loudly so we
  never silently drop the news signal when the toggle is flipped on.

Keeping the integration point explicit now means the rest of the pipeline can
be wired for news from day one without forcing a rewrite later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class NewsConfig:
    """Configuration for the news-aware mode.

    Attributes
    ----------
    enabled:
        Master switch. When False (default) the pipeline runs OHLC-only.
    headline_file:
        Optional parquet path that will be consumed by the (future) news
        featurizer. Ignored when ``enabled`` is False.
    """

    enabled: bool = False
    headline_file: Optional[Path] = None


def build_news_features(
    sessions: pd.Series,
    config: NewsConfig,
) -> pd.DataFrame:
    """Return a per-session news feature frame.

    When ``config.enabled`` is False this returns an empty frame keyed by
    ``session`` so the caller can ``merge`` without branching. When the news
    mode is flipped on before the implementation lands we raise a clear error
    rather than silently dropping the signal.
    """
    base = pd.DataFrame({"session": pd.Series(sessions, dtype="int64").unique()})
    if not config.enabled:
        return base

    raise NotImplementedError(
        "News-aware mode is toggled on but the headline featurizer is not wired "
        "up yet. Implement `build_news_features` (FinBERT / entity-aware "
        "sentiment / LLM signals) before enabling `use_news=True`."
    )
