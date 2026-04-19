"""First-half-only news feature engineering for the regime pipeline.

Design principle (from the user): the news block should encode *session
identity*, not just a directional aggregate. Two sessions that talk about
the same entities, use the same headline language, or have the same news
flow profile should end up close together in feature space so the linear
head can borrow predictive strength across similar-looking sessions.

Competition constraint: only the seen-half news (``bar_ix <= 49``) may be
consumed at inference time. ``sentiments_unseen_*`` is strictly an offline
analysis resource and is never loaded here.

The featurizer produces one per-session row with five complementary blocks:

1. **Directional sentiment aggregates** (13 features). Decay-weighted
   mean / sum / last / std of the ``sentiment_score`` channel, plus the
   buy-sell balance and confidence-weighted flavors. Reuses the design
   that ``src/tailored-modeler/sentiment.py`` validated against forward
   returns, so we do not re-derive a known-good aggregator.

2. **Temporal profile** (5 features). Fraction of seen-half headlines in
   each quartile of the seen window plus the headline count; captures
   early/mid/late news-flow intensity.

3. **Entity concentration** (4 features). Unique entity count, Shannon
   entropy over the entity distribution, top-entity share, and a dominant
   entity's weighted sentiment. Distinguishes sessions dominated by one
   company vs many.

4. **Entity footprint** (``top_entities`` * 2 features). For the ``N``
   most-frequent companies on the train split we record per-session
   ``mention_count`` and mean-sentiment-score. Companies appearing in
   test but not in the train top-N are absorbed into an ``other`` bucket
   so dimensionality stays fixed. This is the identity signal that lets
   "similar-company sessions look similar".

5. **Topic fingerprint** (``svd_components`` features). TF-IDF over the
   concatenated first-half headlines per session, reduced to a dense
   embedding via TruncatedSVD. Fit on train, transform on test. Two
   sessions that use the same headline vocabulary sit close here even
   without any explicit entity match.

Sessions with zero headlines get zeros for every feature (honest absence
signal; all-zeros sits in the middle of the standardized feature space
used by the linear head, which is the right default).

The featurizer exposes ``fit`` / ``transform`` / ``fit_transform``; the
pipeline calls ``fit_transform`` on train headlines + sentiments and then
``transform`` on the combined public+private test data. The session index
passed in guarantees the output frame has one row per requested session
regardless of whether that session had any headlines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from paths import (
    HEADLINES_SEEN_PRIVATE_TEST,
    HEADLINES_SEEN_PUBLIC_TEST,
    HEADLINES_SEEN_TRAIN,
)


SENTIMENTS_SEEN_TRAIN = "sentiments_seen_train.csv"
SENTIMENTS_SEEN_PUBLIC_TEST = "sentiments_seen_public_test.csv"
SENTIMENTS_SEEN_PRIVATE_TEST = "sentiments_seen_private_test.csv"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NewsConfig:
    """Top-level switches for the news feature block.

    Attributes
    ----------
    enabled:
        Master toggle. The pipeline runs OHLC-only when ``False``.
    decision_bar:
        Last first-half bar index (49 for this competition). Headlines and
        sentiments with ``bar_ix > decision_bar`` are dropped before any
        aggregate is computed, no matter what file they were loaded from.
    decay_half_life:
        Bars. Sentiment aggregates weight each headline by
        ``exp(-ln(2) * (decision_bar - bar_ix) / decay_half_life) * confidence``
        so that headlines closer to the decision point dominate the signal.
    svd_components:
        TruncatedSVD rank for the topic fingerprint. Sampled from train
        headlines only. Set to ``0`` to disable the topic block entirely.
    top_entities:
        Number of most-frequent train companies tracked as explicit
        footprint features. Everything else gets bucketed into ``other``.
    tfidf_min_df / tfidf_max_df:
        Term-frequency pruning passed to ``TfidfVectorizer``. The defaults
        (``>= 2 sessions``, ``<= 95%`` of sessions) strip rare typos and
        ubiquitous boilerplate.
    ngram_max:
        Upper ``ngram_range`` bound. The default ``2`` captures bigrams
        (e.g. "contract award", "new office") which usually carry more
        signal than single tokens.
    include_summary_features:
        When True, the pipeline also merges the coarse per-session OHLC
        summary features (from ``emissions.session_summary_features``) onto
        the linear-head input. That feature block is OHLC-derived, not news,
        but gets activated alongside news so the linear head sees the full
        "identity" stack the review asked for.
    """

    enabled: bool = False
    decision_bar: int = 49
    decay_half_life: float = 10.0
    svd_components: int = 16
    top_entities: int = 10
    tfidf_min_df: float = 2.0  # passed as int -> absolute doc-count threshold
    tfidf_max_df: float = 0.95
    ngram_max: int = 2
    include_summary_features: bool = True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_train_news(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train first-half headlines + sentiments. Unseen files are ignored."""
    data_dir = Path(data_dir)
    headlines = pd.read_parquet(data_dir / HEADLINES_SEEN_TRAIN)
    sentiments = pd.read_csv(data_dir / SENTIMENTS_SEEN_TRAIN)
    return headlines, sentiments


def load_test_news(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load test first-half headlines + sentiments (public + private)."""
    data_dir = Path(data_dir)
    h_pub = pd.read_parquet(data_dir / HEADLINES_SEEN_PUBLIC_TEST)
    h_priv = pd.read_parquet(data_dir / HEADLINES_SEEN_PRIVATE_TEST)
    s_pub = pd.read_csv(data_dir / SENTIMENTS_SEEN_PUBLIC_TEST)
    s_priv = pd.read_csv(data_dir / SENTIMENTS_SEEN_PRIVATE_TEST)
    return (
        pd.concat([h_pub, h_priv], ignore_index=True),
        pd.concat([s_pub, s_priv], ignore_index=True),
    )


# ---------------------------------------------------------------------------
# Sentiment aggregate block (Block 1)
# ---------------------------------------------------------------------------


def _sentiment_aggregates(
    sentiments: pd.DataFrame,
    *,
    decision_bar: int,
    decay_half_life: float,
) -> pd.DataFrame:
    """Decay-weighted per-session sentiment aggregates.

    Logic is ported from ``src/tailored-modeler/sentiment.py`` so the regime
    pipeline stays self-contained. We filter to ``bar_ix <= decision_bar``
    defensively even though seen-half CSVs already respect that bound.
    """
    if sentiments.empty:
        return pd.DataFrame(
            columns=[
                "session",
                "news_headline_count",
                "news_mean_score",
                "news_std_score",
                "news_mean_sign",
                "news_mean_conf",
                "news_max_abs_score",
                "news_last_score",
                "news_last_sign",
                "news_entity_count",
                "news_weighted_score",
                "news_weighted_sign",
                "news_buy_sell_balance",
            ]
        )

    s = sentiments.loc[sentiments["bar_ix"] <= decision_bar].copy()
    s["sign"] = np.where(
        s["sentiment"].astype(str).str.lower() == "buy", 1.0, -1.0
    ).astype(np.float64)
    gap = np.maximum(decision_bar - s["bar_ix"].astype(np.float64), 0.0)
    s["w_decay"] = np.exp(-np.log(2.0) * gap / max(decay_half_life, 1e-6))
    s["w"] = s["w_decay"] * s["confidence"].astype(np.float64)
    s["score_w"] = s["sentiment_score"].astype(np.float64) * s["w"]
    s["sign_w"] = s["sign"] * s["w"]

    agg = (
        s.groupby("session")
        .agg(
            news_headline_count=("sentiment_score", "size"),
            _sum_w=("w", "sum"),
            _sum_score_w=("score_w", "sum"),
            _sum_sign_w=("sign_w", "sum"),
            news_mean_score=("sentiment_score", "mean"),
            news_std_score=("sentiment_score", "std"),
            news_mean_sign=("sign", "mean"),
            news_mean_conf=("confidence", "mean"),
            news_max_abs_score=("sentiment_score", lambda v: float(np.max(np.abs(v)))),
            news_last_score=("sentiment_score", "last"),
            news_last_sign=("sign", "last"),
            news_entity_count=("company", "nunique"),
        )
        .reset_index()
    )
    agg["news_weighted_score"] = agg["_sum_score_w"] / np.maximum(agg["_sum_w"], 1e-12)
    agg["news_weighted_sign"] = agg["_sum_sign_w"] / np.maximum(agg["_sum_w"], 1e-12)
    agg["news_buy_sell_balance"] = agg["news_mean_sign"]
    agg["news_std_score"] = agg["news_std_score"].fillna(0.0)
    return agg.drop(columns=["_sum_w", "_sum_score_w", "_sum_sign_w"])


# ---------------------------------------------------------------------------
# Temporal profile block (Block 2)
# ---------------------------------------------------------------------------


def _temporal_profile(
    headlines: pd.DataFrame,
    *,
    decision_bar: int,
) -> pd.DataFrame:
    """Fraction of headlines in each quartile of the seen window."""
    if headlines.empty:
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
    h = headlines.loc[headlines["bar_ix"] <= decision_bar].copy()
    bins = np.asarray([0, 12, 25, 37, decision_bar + 1], dtype=np.int64)
    # Digitize assigns 1..4 for the four quartile buckets.
    q = np.digitize(h["bar_ix"].to_numpy(dtype=np.int64), bins[1:-1]) + 1
    h["_quartile"] = np.clip(q, 1, 4)
    counts = (
        h.groupby(["session", "_quartile"]).size().unstack(fill_value=0)
    )
    for c in (1, 2, 3, 4):
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[[1, 2, 3, 4]]
    totals = counts.sum(axis=1).replace(0, 1)
    fracs = counts.div(totals, axis=0)
    fracs.columns = ["news_frac_q1", "news_frac_q2", "news_frac_q3", "news_frac_q4"]
    mean_bar = h.groupby("session")["bar_ix"].mean().rename("news_mean_bar_ix")
    return fracs.merge(mean_bar, left_index=True, right_index=True).reset_index()


# ---------------------------------------------------------------------------
# Entity concentration (Block 3)
# ---------------------------------------------------------------------------


def _entity_concentration(
    sentiments: pd.DataFrame,
    *,
    decision_bar: int,
) -> pd.DataFrame:
    if sentiments.empty:
        return pd.DataFrame(
            columns=[
                "session",
                "news_entity_entropy",
                "news_top_entity_share",
                "news_unique_entities",
                "news_top_entity_weighted_sent",
            ]
        )
    s = sentiments.loc[sentiments["bar_ix"] <= decision_bar].copy()
    gap = np.maximum(decision_bar - s["bar_ix"].astype(np.float64), 0.0)
    w_decay = np.exp(-np.log(2.0) * gap / 10.0)  # local fixed half-life here
    s["_w_event"] = w_decay * s["confidence"].astype(np.float64)
    rows: List[dict] = []
    for sess, g in s.groupby("session"):
        counts = g["company"].value_counts()
        totals = counts.sum()
        probs = (counts / totals).to_numpy()
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
        top_share = float(probs.max()) if probs.size else 0.0
        top_entity = counts.idxmax() if counts.size else None
        if top_entity is not None:
            sub = g[g["company"] == top_entity]
            top_w = float(sub["_w_event"].sum())
            top_sent = float(
                np.sum(sub["sentiment_score"] * sub["_w_event"])
                / max(top_w, 1e-12)
            )
        else:
            top_sent = 0.0
        rows.append(
            {
                "session": int(sess),
                "news_entity_entropy": entropy,
                "news_top_entity_share": top_share,
                "news_unique_entities": int(counts.size),
                "news_top_entity_weighted_sent": top_sent,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entity footprint (Block 4)
# ---------------------------------------------------------------------------


def _entity_footprint(
    sentiments: pd.DataFrame,
    *,
    decision_bar: int,
    top_entities: Sequence[str],
) -> pd.DataFrame:
    """For each top-K company: per-session mention count + mean sentiment."""
    columns = ["session"]
    for e in top_entities:
        columns.extend([f"news_ent_{e}_cnt", f"news_ent_{e}_sent"])
    columns.extend(["news_ent_other_cnt", "news_ent_other_sent"])
    if sentiments.empty:
        return pd.DataFrame(columns=columns)

    s = sentiments.loc[sentiments["bar_ix"] <= decision_bar].copy()
    if s.empty:
        return pd.DataFrame(columns=columns)

    top_set = set(top_entities)
    s["_bucket"] = np.where(
        s["company"].isin(top_set), s["company"], "other"
    )
    grouped = (
        s.groupby(["session", "_bucket"])["sentiment_score"]
        .agg(["count", "mean"])
        .reset_index()
    )
    cnt = grouped.pivot(index="session", columns="_bucket", values="count").fillna(0.0)
    sent = grouped.pivot(index="session", columns="_bucket", values="mean").fillna(0.0)

    for e in top_entities:
        if e not in cnt.columns:
            cnt[e] = 0.0
            sent[e] = 0.0
    if "other" not in cnt.columns:
        cnt["other"] = 0.0
        sent["other"] = 0.0
    cnt = cnt[list(top_entities) + ["other"]]
    sent = sent[list(top_entities) + ["other"]]

    cnt.columns = [f"news_ent_{c}_cnt" for c in cnt.columns]
    sent.columns = [f"news_ent_{c}_sent" for c in sent.columns]
    return cnt.merge(sent, left_index=True, right_index=True).reset_index()


# ---------------------------------------------------------------------------
# Topic fingerprint (Block 5) -- fitted on train
# ---------------------------------------------------------------------------


def _session_docs(headlines: pd.DataFrame, decision_bar: int) -> pd.Series:
    if headlines.empty:
        return pd.Series([], dtype="object", name="headline")
    h = headlines.loc[headlines["bar_ix"] <= decision_bar]
    return h.groupby("session")["headline"].apply(lambda xs: " ".join(xs.astype(str)))


# ---------------------------------------------------------------------------
# Featurizer
# ---------------------------------------------------------------------------


class NewsFeaturizer:
    """Fit TF-IDF / SVD / top-entity list on train; transform train & test.

    Usage::

        fz = NewsFeaturizer(config).fit(train_headlines, train_sentiments)
        news_tr = fz.transform(train_headlines, train_sentiments, train_sessions)
        news_te = fz.transform(test_headlines,  test_sentiments,  test_sessions)

    ``train_sessions`` / ``test_sessions`` are the *expected* session ids for
    the output frame: sessions with zero headlines will still appear (with
    zeros across every feature), so the join on the HMM posterior table is
    safe.
    """

    def __init__(self, config: NewsConfig):
        self.config = config
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None
        self.top_entities_: List[str] = []

    # -- fit ----------------------------------------------------------------

    def fit(
        self,
        headlines: pd.DataFrame,
        sentiments: pd.DataFrame,
    ) -> "NewsFeaturizer":
        cfg = self.config

        # Top entities on the train split (by headline frequency within the
        # first half).
        if not sentiments.empty:
            s = sentiments.loc[sentiments["bar_ix"] <= cfg.decision_bar]
            counts = s["company"].value_counts()
            self.top_entities_ = counts.head(cfg.top_entities).index.tolist()
        else:
            self.top_entities_ = []

        if cfg.svd_components and not headlines.empty:
            docs = _session_docs(headlines, cfg.decision_bar)
            if len(docs) >= max(5, cfg.svd_components + 2):
                min_df = max(1, int(cfg.tfidf_min_df)) if cfg.tfidf_min_df >= 1 else cfg.tfidf_min_df
                self.vectorizer = TfidfVectorizer(
                    min_df=min_df,
                    max_df=cfg.tfidf_max_df,
                    ngram_range=(1, int(cfg.ngram_max)),
                    lowercase=True,
                    strip_accents="unicode",
                )
                X = self.vectorizer.fit_transform(docs)
                n_comp = int(min(cfg.svd_components, max(1, X.shape[1] - 1)))
                self.svd = TruncatedSVD(n_components=n_comp, random_state=0)
                self.svd.fit(X)
        return self

    # -- transform ----------------------------------------------------------

    def transform(
        self,
        headlines: pd.DataFrame,
        sentiments: pd.DataFrame,
        sessions: Iterable[int],
    ) -> pd.DataFrame:
        cfg = self.config
        sessions = pd.Index(pd.Series(sessions, dtype=np.int64).unique(), name="session")
        base = pd.DataFrame({"session": sessions.to_numpy()})

        agg = _sentiment_aggregates(
            sentiments,
            decision_bar=cfg.decision_bar,
            decay_half_life=cfg.decay_half_life,
        )
        tp = _temporal_profile(headlines, decision_bar=cfg.decision_bar)
        ec = _entity_concentration(sentiments, decision_bar=cfg.decision_bar)
        ef = _entity_footprint(
            sentiments,
            decision_bar=cfg.decision_bar,
            top_entities=self.top_entities_,
        )
        out = (
            base.merge(agg, on="session", how="left")
            .merge(tp, on="session", how="left")
            .merge(ec, on="session", how="left")
            .merge(ef, on="session", how="left")
        )

        # Topic fingerprint (SVD over TF-IDF). Fit on train only; transform
        # both train and test. Sessions with no headlines get zeros.
        if self.vectorizer is not None and self.svd is not None:
            docs = _session_docs(headlines, cfg.decision_bar)
            svd_cols = [f"news_topic_{i}" for i in range(self.svd.n_components)]
            if not docs.empty:
                X = self.vectorizer.transform(docs)
                Z = self.svd.transform(X)
                topic_df = pd.DataFrame(Z, index=docs.index, columns=svd_cols).reset_index()
            else:
                topic_df = pd.DataFrame(columns=["session", *svd_cols])
            out = out.merge(topic_df, on="session", how="left")
            for c in svd_cols:
                if c not in out.columns:
                    out[c] = 0.0

        # Fill every news-originated feature with 0 where the session had no
        # headlines (absence signal is encoded in ``news_headline_count``).
        out = out.fillna(0.0)
        return out.sort_values("session").reset_index(drop=True)

    # -- convenience --------------------------------------------------------

    def fit_transform(
        self,
        headlines: pd.DataFrame,
        sentiments: pd.DataFrame,
        sessions: Iterable[int],
    ) -> pd.DataFrame:
        return self.fit(headlines, sentiments).transform(headlines, sentiments, sessions)


# ---------------------------------------------------------------------------
# Legacy stub (kept so older callers fail loudly rather than silently)
# ---------------------------------------------------------------------------


def build_news_regime_prior(
    sessions: pd.Series,
    n_states: int,
    config: NewsConfig,
) -> Optional[pd.DataFrame]:
    """Reserved hook for news -> per-state prior integration.

    The current pipeline feeds news into the linear head (Method ``m1-linear``)
    rather than into the HMM posterior. If a caller asks for a regime-level
    prior with ``config.enabled=True``, we raise so we never silently drop
    the signal.
    """
    if not config.enabled:
        return None
    raise NotImplementedError(
        "News-as-regime-prior is not wired yet. The --use-news path merges "
        "news features into the linear head (see pipeline.py). Future work: "
        "convert per-session news features into a per-state prior to bias "
        "the HMM posterior at bar 49."
    )
