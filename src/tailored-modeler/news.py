"""Seen-half news / sentiment feature hook for the tailored modeler.

Two modes:

* ``NewsConfig(enabled=False)`` - OHLC-only pipeline. No sentiment data is
  loaded. Default, identical to the OHLC-only 2.24 baseline.
* ``NewsConfig(enabled=True)`` - the pipeline loads the sentiment CSVs for
  the relevant splits and merges per-session aggregates into the feature
  matrix.

Strict no-leakage contract (Stage 8 of the plan + reviewer's explicit ask):

* For train we only consume ``sentiments_seen_train.csv`` - never the
  ``sentiments_unseen_train.csv`` file. The decision bar is ``bar_ix=49``,
  which matches the seen-half cutoff in the test parquets.
* The function asserts ``bar_ix <= decision_bar`` on every event before it
  aggregates, so even if someone passes a wrong CSV the assertion fires.

Session-level feature design matches the empirical findings of
``scripts/validate-sentiment``:

* ``weighted_score`` / ``weighted_sign`` with half-life ``decay_half_life``
  bars, weighted by ``confidence``. This captures the "news impact peaks ~k=10
  ahead and decays" shape we validated (Pearson ~0.20 at the event level).
* ``late10_*`` aggregates over the last 10 bars (bar 40-49) because those
  are the events whose impact extends *past* the decision bar into the
  R = close_end / close_half - 1 window.
* Simple bookkeeping features (headline count, entity count, max |score|,
  confidence mean, last-headline sentiment).

Sessions without any seen-half sentiment event (one such session exists in
the private test) receive zero-filled features (neutral) - no NaNs are
propagated into the booster.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_DECISION_BAR = 49
DEFAULT_DECAY_HALF_LIFE = 10.0
DEFAULT_LATE_WINDOW = 10  # last 10 bars of the seen half

SENTIMENT_FILES = {
    "train_seen": "sentiments_seen_train.csv",
    "public_test": "sentiments_seen_public_test.csv",
    "private_test": "sentiments_seen_private_test.csv",
    # Explicit omission: sentiments_unseen_train.csv is NEVER exposed here
    # because it would leak the future half into training features.
}


@dataclass(frozen=True)
class NewsConfig:
    """Configuration for the seen-half sentiment branch."""

    enabled: bool = False
    decision_bar: int = DEFAULT_DECISION_BAR
    decay_half_life: float = DEFAULT_DECAY_HALF_LIFE
    late_window: int = DEFAULT_LATE_WINDOW


NEWS_FEATURE_COLUMNS: Tuple[str, ...] = (
    "news_headline_count",
    "news_entity_count",
    "news_mean_score",
    "news_mean_sign",
    "news_mean_conf",
    "news_weighted_score",
    "news_weighted_sign",
    "news_last_score",
    "news_last_sign",
    "news_max_abs_score",
    "news_high_conf_count",
    "news_late_count",
    "news_late_weighted_score",
    "news_late_weighted_sign",
    "news_late_mean_score",
    "news_late_mean_sign",
)


def _load_sentiment_splits(data_dir: Path, splits: Iterable[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for split in splits:
        if split not in SENTIMENT_FILES:
            raise ValueError(
                f"Unknown / disallowed sentiment split {split!r}. "
                f"Use one of {list(SENTIMENT_FILES)}"
            )
        path = Path(data_dir) / SENTIMENT_FILES[split]
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=["session", "bar_ix", "company", "sentiment", "sentiment_score", "confidence"]
        )
    out = pd.concat(frames, ignore_index=True)
    out["session"] = out["session"].astype("int64")
    out["bar_ix"] = out["bar_ix"].astype("int64")
    out["sentiment_score"] = out["sentiment_score"].astype("float64")
    out["confidence"] = out["confidence"].astype("float64")
    out["sign"] = np.where(
        out["sentiment"].astype(str).str.lower() == "buy", 1.0, -1.0
    ).astype("float64")
    return out


def _session_level_aggregates(
    events: pd.DataFrame, *, decision_bar: int, decay_half_life: float, late_window: int
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=("session",) + NEWS_FEATURE_COLUMNS)

    # No-leakage guard: every event must be observable at decision time.
    if (events["bar_ix"] > decision_bar).any():
        raise AssertionError(
            f"News leakage: found bar_ix > decision_bar ({decision_bar}) in "
            "sentiment events. Refusing to build features."
        )

    gap = np.maximum(decision_bar - events["bar_ix"].to_numpy(), 0)
    w_decay = np.exp(-np.log(2.0) * gap / max(decay_half_life, 1e-6))
    conf = events["confidence"].to_numpy(dtype=np.float64)
    score = events["sentiment_score"].to_numpy(dtype=np.float64)
    sign = events["sign"].to_numpy(dtype=np.float64)
    weight = w_decay * conf

    e = events.copy()
    e["w"] = weight
    e["score_w"] = score * weight
    e["sign_w"] = sign * weight
    e["high_conf"] = (conf >= 0.9).astype(np.int32)

    late_mask = e["bar_ix"] >= (decision_bar - late_window + 1)
    late = e[late_mask]

    agg = e.groupby("session").agg(
        news_headline_count=("sentiment_score", "size"),
        news_entity_count=("company", "nunique"),
        news_mean_score=("sentiment_score", "mean"),
        news_mean_sign=("sign", "mean"),
        news_mean_conf=("confidence", "mean"),
        news_max_abs_score=("sentiment_score", lambda v: float(np.max(np.abs(v)))),
        _sum_w=("w", "sum"),
        _sum_score_w=("score_w", "sum"),
        _sum_sign_w=("sign_w", "sum"),
        news_last_score=("sentiment_score", "last"),
        news_last_sign=("sign", "last"),
        news_high_conf_count=("high_conf", "sum"),
    ).reset_index()
    agg["news_weighted_score"] = agg["_sum_score_w"] / np.maximum(agg["_sum_w"], 1e-12)
    agg["news_weighted_sign"] = agg["_sum_sign_w"] / np.maximum(agg["_sum_w"], 1e-12)
    agg = agg.drop(columns=["_sum_w", "_sum_score_w", "_sum_sign_w"])

    if late.empty:
        late_agg = pd.DataFrame(
            {
                "session": pd.Series(dtype="int64"),
                "news_late_count": pd.Series(dtype="int64"),
                "news_late_weighted_score": pd.Series(dtype="float64"),
                "news_late_weighted_sign": pd.Series(dtype="float64"),
                "news_late_mean_score": pd.Series(dtype="float64"),
                "news_late_mean_sign": pd.Series(dtype="float64"),
            }
        )
    else:
        late_agg = late.groupby("session").agg(
            news_late_count=("sentiment_score", "size"),
            _sum_w=("w", "sum"),
            _sum_score_w=("score_w", "sum"),
            _sum_sign_w=("sign_w", "sum"),
            news_late_mean_score=("sentiment_score", "mean"),
            news_late_mean_sign=("sign", "mean"),
        ).reset_index()
        late_agg["news_late_weighted_score"] = late_agg["_sum_score_w"] / np.maximum(
            late_agg["_sum_w"], 1e-12
        )
        late_agg["news_late_weighted_sign"] = late_agg["_sum_sign_w"] / np.maximum(
            late_agg["_sum_w"], 1e-12
        )
        late_agg = late_agg.drop(columns=["_sum_w", "_sum_score_w", "_sum_sign_w"])

    out = agg.merge(late_agg, on="session", how="left")
    out[
        [
            "news_late_count",
            "news_late_weighted_score",
            "news_late_weighted_sign",
            "news_late_mean_score",
            "news_late_mean_sign",
        ]
    ] = out[
        [
            "news_late_count",
            "news_late_weighted_score",
            "news_late_weighted_sign",
            "news_late_mean_score",
            "news_late_mean_sign",
        ]
    ].fillna(0.0)
    return out[["session"] + list(NEWS_FEATURE_COLUMNS)]


def build_news_features(
    sessions: pd.Series,
    config: NewsConfig,
    *,
    data_dir: Optional[Path] = None,
    splits: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return per-session sentiment-derived features (seen-half only).

    Parameters
    ----------
    sessions:
        The list of sessions we want rows for. Sessions that have no seen-half
        headline are emitted with zero-filled features.
    config:
        ``NewsConfig``. When ``enabled=False`` we short-circuit to an
        all-zero frame (preserves the OHLC-only path bit-for-bit).
    data_dir, splits:
        Where and which sentiment CSVs to load. Required when
        ``config.enabled`` is True.
    """
    base_sessions = pd.Series(sessions, dtype="int64").astype("int64").unique()
    base = pd.DataFrame({"session": base_sessions})

    zero_frame = base.copy()
    for col in NEWS_FEATURE_COLUMNS:
        zero_frame[col] = 0.0

    if not config.enabled:
        return zero_frame

    if data_dir is None or splits is None:
        raise ValueError(
            "build_news_features requires data_dir and splits when enabled=True"
        )
    events = _load_sentiment_splits(Path(data_dir), list(splits))
    # Seen-half cut-off: drop anything past the decision bar (belt & braces).
    events = events[events["bar_ix"] <= config.decision_bar]
    feats = _session_level_aggregates(
        events,
        decision_bar=config.decision_bar,
        decay_half_life=config.decay_half_life,
        late_window=config.late_window,
    )
    merged = base.merge(feats, on="session", how="left")
    for col in NEWS_FEATURE_COLUMNS:
        if col not in merged.columns:
            merged[col] = 0.0
    merged[list(NEWS_FEATURE_COLUMNS)] = merged[list(NEWS_FEATURE_COLUMNS)].fillna(0.0)
    return merged[["session"] + list(NEWS_FEATURE_COLUMNS)]
