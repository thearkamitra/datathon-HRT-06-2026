"""News-aware feature hook (stubbed).

The regime pipeline is designed to be upgraded later with headline signal as a
*prior* over the latent regimes (or as a small additive covariate on the
downstream forecast). Until that upgrade lands we keep the toggle explicit so
the rest of the pipeline is wired for news from day one:

* ``use_news=False`` (default): pure OHLC regime pipeline. No headline
  parquets are read and no text models are loaded.
* ``use_news=True``: reserved. Any attempt to flip this on today raises
  ``NotImplementedError`` loudly so we never silently drop the signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class NewsConfig:
    """Configuration for the news-aware branch.

    Attributes
    ----------
    enabled:
        Master switch. When False (default) the regime pipeline runs OHLC-only.
    headline_file:
        Optional parquet path that will be consumed by the (future) headline
        featurizer. Ignored when ``enabled`` is False.
    regime_prior_weight:
        Reserved. Future knob for mixing a news-derived regime prior into the
        HMM state posterior before forecasting.
    """

    enabled: bool = False
    headline_file: Optional[Path] = None
    regime_prior_weight: float = 0.0


def build_news_regime_prior(
    sessions: pd.Series,
    n_states: int,
    config: NewsConfig,
) -> Optional[pd.DataFrame]:
    """Return a per-session, per-state prior over latent regimes.

    When ``config.enabled`` is False this returns ``None`` so the forecasting
    code can treat news as purely optional. Flipping the toggle on before the
    implementation lands raises a clear error.
    """
    if not config.enabled:
        return None

    raise NotImplementedError(
        "News-aware mode is toggled on but the headline -> regime prior is not "
        "wired up yet. Implement `build_news_regime_prior` (FinBERT / "
        "entity-aware sentiment / LLM signals -> per-state prior) before "
        "enabling `use_news=True`."
    )
