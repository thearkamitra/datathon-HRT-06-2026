"""Seen-half headline + sentiment features for the tailored modeler.

The tailored pipeline selects its three-head booster and Sharpe sizer via
repeated-KFold out-of-fold predictions. For that reason, the news block avoids
any train-wide fitted text encoder that would leak fold-level text statistics
into validation rows before they are scored.

Instead, the feature family is deliberately **stateless**:

* decay-weighted sentiment aggregates,
* recent-window sentiment aggregates,
* temporal news-arrival profile,
* entity / sector concentration descriptors,
* fixed-width hashed headline fingerprints.

`HashingVectorizer` is used for headline text so there is no fit step and the
same transformation is valid for train, public test, private test, and
augmented-train rows without leaking vocabulary / IDF information.

Strict no-leakage contract:

* Only seen-half sentiment files are loaded.
* `sentiments_unseen_train.csv` is intentionally excluded.
* Any `bar_ix > decision_bar` row raises immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer


DEFAULT_DECISION_BAR = 49
DEFAULT_DECAY_HALF_LIFE = 10.0
DEFAULT_LATE_WINDOW = 10  # last 10 bars of the seen half
DEFAULT_HIGH_CONF_THRESHOLD = 0.9
DEFAULT_HEADLINE_HASH_FEATURES = 32
DEFAULT_HEADLINE_NGRAM_MAX = 2

SENTIMENT_FILES = {
    "train_seen": "sentiments_seen_train.csv",
    "public_test": "sentiments_seen_public_test.csv",
    "private_test": "sentiments_seen_private_test.csv",
    # Explicit omission: sentiments_unseen_train.csv is NEVER exposed here
    # because it would leak the future half into training features.
}


@dataclass(frozen=True)
class NewsConfig:
    """Configuration for the seen-half tailored news branch."""

    enabled: bool = False
    decision_bar: int = DEFAULT_DECISION_BAR
    decay_half_life: float = DEFAULT_DECAY_HALF_LIFE
    late_window: int = DEFAULT_LATE_WINDOW
    high_conf_threshold: float = DEFAULT_HIGH_CONF_THRESHOLD
    headline_hash_features: int = DEFAULT_HEADLINE_HASH_FEATURES
    headline_ngram_max: int = DEFAULT_HEADLINE_NGRAM_MAX


_STATIC_NEWS_FEATURE_COLUMNS: Tuple[str, ...] = (
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
    "news_frac_q1",
    "news_frac_q2",
    "news_frac_q3",
    "news_frac_q4",
    "news_mean_bar_ix",
    "news_entity_entropy",
    "news_top_entity_share",
    "news_top_entity_weighted_sent",
    "news_sector_entropy",
    "news_sector_hhi",
    "news_sector_num_unique",
    "news_sector_max_share",
    "news_granular_sector_entropy",
)


def news_feature_columns(config: NewsConfig) -> list[str]:
    cols = list(_STATIC_NEWS_FEATURE_COLUMNS)
    cols.extend(
        f"news_hash_{i}" for i in range(max(int(config.headline_hash_features), 0))
    )
    return cols


def _zero_frame(sessions: pd.Series, config: NewsConfig) -> pd.DataFrame:
    base_sessions = pd.Series(sessions, dtype="int64").astype("int64").unique()
    out = pd.DataFrame({"session": base_sessions})
    for col in news_feature_columns(config):
        out[col] = 0.0
    return out


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
    for col in ("headline", "company", "sector", "granular_sector"):
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    out["sign"] = np.where(
        out["sentiment"].astype(str).str.lower() == "buy", 1.0, -1.0
    ).astype("float64")
    return out


def _prepare_events(events: pd.DataFrame, config: NewsConfig) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    if (events["bar_ix"] > config.decision_bar).any():
        raise AssertionError(
            f"News leakage: found bar_ix > decision_bar ({config.decision_bar}) in "
            "seen-half sentiment events."
        )
    e = events.loc[events["bar_ix"] <= config.decision_bar].copy()
    gap = np.maximum(config.decision_bar - e["bar_ix"].to_numpy(), 0.0)
    w_decay = np.exp(-np.log(2.0) * gap / max(config.decay_half_life, 1e-6))
    conf = e["confidence"].to_numpy(dtype=np.float64)
    score = e["sentiment_score"].to_numpy(dtype=np.float64)
    sign = e["sign"].to_numpy(dtype=np.float64)
    e["w"] = w_decay * conf
    e["score_w"] = score * e["w"]
    e["sign_w"] = sign * e["w"]
    e["high_conf"] = (
        conf >= float(config.high_conf_threshold)
    ).astype(np.int32)
    return e


def _sentiment_aggregates(e: pd.DataFrame, config: NewsConfig) -> pd.DataFrame:
    if e.empty:
        return pd.DataFrame(
            columns=[
                "session",
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
            ]
        )

    late_mask = e["bar_ix"] >= (config.decision_bar - int(config.late_window) + 1)
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
    return out[
        [
            "session",
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
        ]
    ]


def _temporal_profile(e: pd.DataFrame, config: NewsConfig) -> pd.DataFrame:
    if e.empty:
        return pd.DataFrame(
            columns=[
                "session",
                "news_frac_q1",
                "news_frac_q2",
                "news_frac_q3",
                "news_frac_q4",
                "news_mean_bar_ix",
            ]
        )
    bins = np.asarray([0, 12, 25, 37, config.decision_bar + 1], dtype=np.int64)
    q = np.digitize(e["bar_ix"].to_numpy(dtype=np.int64), bins[1:-1]) + 1
    h = e[["session", "bar_ix"]].copy()
    h["_quartile"] = np.clip(q, 1, 4)
    counts = h.groupby(["session", "_quartile"]).size().unstack(fill_value=0)
    for c in (1, 2, 3, 4):
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[[1, 2, 3, 4]]
    totals = counts.sum(axis=1).replace(0, 1)
    fracs = counts.div(totals, axis=0)
    fracs.columns = ["news_frac_q1", "news_frac_q2", "news_frac_q3", "news_frac_q4"]
    mean_bar = h.groupby("session")["bar_ix"].mean().rename("news_mean_bar_ix")
    return fracs.merge(mean_bar, left_index=True, right_index=True).reset_index()


def _entity_concentration(e: pd.DataFrame) -> pd.DataFrame:
    if e.empty:
        return pd.DataFrame(
            columns=[
                "session",
                "news_entity_entropy",
                "news_top_entity_share",
                "news_top_entity_weighted_sent",
            ]
        )
    rows: List[dict] = []
    for sess, g in e.groupby("session", sort=False):
        counts = g["company"].astype(str).value_counts()
        counts = counts[counts.index != ""]
        total = int(counts.sum())
        if total == 0:
            rows.append(
                {
                    "session": int(sess),
                    "news_entity_entropy": 0.0,
                    "news_top_entity_share": 0.0,
                    "news_top_entity_weighted_sent": 0.0,
                }
            )
            continue
        probs = (counts / float(total)).to_numpy(dtype=np.float64)
        top_entity = str(counts.idxmax())
        top = g[g["company"].astype(str) == top_entity]
        top_w = float(top["w"].sum())
        top_sent = float(
            np.sum(top["sentiment_score"].to_numpy(dtype=np.float64) * top["w"].to_numpy(dtype=np.float64))
            / max(top_w, 1e-12)
        )
        rows.append(
            {
                "session": int(sess),
                "news_entity_entropy": float(
                    -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)))
                ),
                "news_top_entity_share": float(probs.max()) if probs.size else 0.0,
                "news_top_entity_weighted_sent": top_sent,
            }
        )
    return pd.DataFrame(rows)


def _sector_concentration(e: pd.DataFrame) -> pd.DataFrame:
    if e.empty:
        return pd.DataFrame(
            columns=[
                "session",
                "news_sector_entropy",
                "news_sector_hhi",
                "news_sector_num_unique",
                "news_sector_max_share",
                "news_granular_sector_entropy",
            ]
        )
    rows: List[dict] = []
    for sess, g in e.groupby("session", sort=False):
        sector = g["sector"].astype(str)
        vc = sector.value_counts()
        n = float(len(sector))
        p = (vc / n).to_numpy(dtype=np.float64) if n > 0 else np.zeros(0, dtype=np.float64)
        granular = g["granular_sector"].astype(str)
        gvc = granular.value_counts()
        p_g = (
            (gvc / float(len(granular))).to_numpy(dtype=np.float64)
            if len(granular) > 0
            else np.zeros(0, dtype=np.float64)
        )
        rows.append(
            {
                "session": int(sess),
                "news_sector_entropy": float(-np.sum(p * np.log(p + 1e-12))) if p.size else 0.0,
                "news_sector_hhi": float(np.sum(p**2)) if p.size else 0.0,
                "news_sector_num_unique": float(len(vc)),
                "news_sector_max_share": float(p.max()) if p.size else 0.0,
                "news_granular_sector_entropy": float(-np.sum(p_g * np.log(p_g + 1e-12)))
                if p_g.size
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _headline_hash_features(e: pd.DataFrame, config: NewsConfig) -> pd.DataFrame:
    columns = [f"news_hash_{i}" for i in range(max(int(config.headline_hash_features), 0))]
    if e.empty or not columns:
        return pd.DataFrame(columns=["session", *columns])

    docs = (
        e.groupby("session", sort=False)["headline"]
        .apply(lambda xs: " ".join(str(x) for x in xs.astype(str) if str(x).strip()))
    )
    if docs.empty:
        return pd.DataFrame(columns=["session", *columns])

    vectorizer = HashingVectorizer(
        n_features=int(config.headline_hash_features),
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, max(1, int(config.headline_ngram_max))),
        lowercase=True,
        strip_accents="unicode",
    )
    X = vectorizer.transform(docs.fillna(""))
    dense = X.toarray().astype(np.float64, copy=False)
    return pd.DataFrame(dense, index=docs.index, columns=columns).reset_index()


def build_news_features(
    sessions: pd.Series,
    config: NewsConfig,
    *,
    data_dir: Optional[Path] = None,
    splits: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return per-session seen-half news features.

    Parameters
    ----------
    sessions:
        The list of sessions we want rows for. Sessions with no seen-half news
        are emitted with zero-filled features.
    config:
        ``NewsConfig``. When ``enabled=False`` we short-circuit to an
        all-zero frame (preserves the OHLC-only path bit-for-bit).
    data_dir, splits:
        Where and which seen-half sentiment CSVs to load. Required when
        ``config.enabled`` is True.
    """
    zero_frame = _zero_frame(pd.Series(sessions), config)

    if not config.enabled:
        return zero_frame

    if data_dir is None or splits is None:
        raise ValueError(
            "build_news_features requires data_dir and splits when enabled=True"
        )
    events = _prepare_events(_load_sentiment_splits(Path(data_dir), list(splits)), config)
    if events.empty:
        return zero_frame

    merged = zero_frame[["session"]].copy()
    for frame in (
        _sentiment_aggregates(events, config),
        _temporal_profile(events, config),
        _entity_concentration(events),
        _sector_concentration(events),
        _headline_hash_features(events, config),
    ):
        merged = merged.merge(frame, on="session", how="left")

    feature_cols = news_feature_columns(config)
    for col in feature_cols:
        if col not in merged.columns:
            merged[col] = 0.0
    merged[feature_cols] = merged[feature_cols].fillna(0.0)
    return merged[["session"] + feature_cols]
