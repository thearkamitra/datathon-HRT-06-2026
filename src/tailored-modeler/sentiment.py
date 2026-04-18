"""Sentiment loading, event-level alignment, and validation utilities.

The sentiment files have the schema

    session, bar_ix, headline, company, sentiment, sentiment_score, confidence

This module does two things:

1. Aligns each sentiment *event* (a row) with the forward log-returns of the
   session's OHLC bars at multiple horizons (k = 1, 2, 3, 5, 10, 20, ...). The
   forward return is computed as ``log(close_{bar_ix+k}) - log(close_{bar_ix})``
   so we can test the "news affects later, nearby bars more than far bars"
   hypothesis directly.
2. Produces a validation report with Pearson / Spearman correlations, sign
   hit-rate (agreement between ``sentiment`` direction and forward-return
   sign), confidence-stratified versions, and bootstrap confidence intervals.

The downstream integration into the pipeline will consume the session-level
aggregator exposed at the bottom (``build_session_sentiment_features``). It
is kept deliberately separate from the validator so the modelling code does
not need to re-run any expensive diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -- Canonical file layout ---------------------------------------------------

SENTIMENT_FILES = {
    "train_seen": "sentiments_seen_train.csv",
    "train_unseen": "sentiments_unseen_train.csv",
    "public_test": "sentiments_seen_public_test.csv",
    "private_test": "sentiments_seen_private_test.csv",
}

BAR_FILES = {
    "train_seen": "bars_seen_train.parquet",
    "train_unseen": "bars_unseen_train.parquet",
    "public_test": "bars_seen_public_test.parquet",
    "private_test": "bars_seen_private_test.parquet",
}


# -- Loading -----------------------------------------------------------------


def load_sentiments(data_dir: Path, split: str) -> pd.DataFrame:
    """Load one of the four sentiment CSVs by split name."""
    path = Path(data_dir) / SENTIMENT_FILES[split]
    df = pd.read_csv(path)
    df["session"] = df["session"].astype("int64")
    df["bar_ix"] = df["bar_ix"].astype("int64")
    df["sentiment_score"] = df["sentiment_score"].astype("float64")
    df["confidence"] = df["confidence"].astype("float64")
    df["sign"] = np.where(df["sentiment"].str.lower() == "buy", 1.0, -1.0).astype(
        "float64"
    )
    return df


def load_bars(data_dir: Path, split: str) -> pd.DataFrame:
    path = Path(data_dir) / BAR_FILES[split]
    return pd.read_parquet(path)


def load_combined_bars(data_dir: Path, splits: Iterable[str]) -> pd.DataFrame:
    """Concatenate bars for one or more splits, keeping per-session ordering."""
    frames = [load_bars(data_dir, s) for s in splits]
    return pd.concat(frames, ignore_index=True)


def load_combined_sentiments(data_dir: Path, splits: Iterable[str]) -> pd.DataFrame:
    frames = [load_sentiments(data_dir, s) for s in splits]
    return pd.concat(frames, ignore_index=True)


# -- Event-level alignment ---------------------------------------------------


def _session_bar_index(bars: pd.DataFrame) -> Dict[int, pd.Series]:
    """Return a dict session -> (bar_ix -> close) for O(1) lookup."""
    bars = bars.sort_values(["session", "bar_ix"])
    out: Dict[int, pd.Series] = {}
    for sess, g in bars.groupby("session", sort=False):
        out[int(sess)] = pd.Series(
            g["close"].to_numpy(dtype=np.float64), index=g["bar_ix"].to_numpy()
        )
    return out


def align_events_with_forward_returns(
    sentiments: pd.DataFrame,
    bars: pd.DataFrame,
    horizons: Iterable[int] = (1, 2, 3, 5, 10, 20),
) -> pd.DataFrame:
    """For every sentiment event, attach forward log-returns at each horizon.

    A horizon ``k`` is only considered valid for an event at ``bar_ix = b`` if
    the session has an observed close at ``bar_ix = b + k`` (the test splits
    only contain ``bar_ix <= 49`` so we filter honestly rather than extrapolate).

    Output columns:
      session, bar_ix, company, sentiment, sentiment_score, confidence, sign,
      close_ref, fwd_ret_<k>, valid_<k>
    """
    horizons = sorted(set(int(k) for k in horizons))
    closes = _session_bar_index(bars)

    sess = sentiments["session"].to_numpy(dtype=np.int64)
    bix = sentiments["bar_ix"].to_numpy(dtype=np.int64)
    n = len(sentiments)

    close_ref = np.full(n, np.nan)
    fwd = {k: np.full(n, np.nan) for k in horizons}

    for i in range(n):
        s = closes.get(int(sess[i]))
        if s is None:
            continue
        b = int(bix[i])
        if b not in s.index:
            continue
        c_ref = float(s.loc[b])
        if c_ref <= 0:
            continue
        close_ref[i] = c_ref
        log_ref = np.log(c_ref)
        for k in horizons:
            if (b + k) in s.index:
                c_k = float(s.loc[b + k])
                if c_k > 0:
                    fwd[k][i] = np.log(c_k) - log_ref

    out = sentiments.copy()
    out["close_ref"] = close_ref
    for k in horizons:
        out[f"fwd_ret_{k}"] = fwd[k]
        out[f"valid_{k}"] = np.isfinite(fwd[k])
    return out


# -- Correlation / validation metrics ----------------------------------------


def _pearson(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if w is not None:
        mask &= np.isfinite(w)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return float("nan")
    if w is None:
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])
    w_ = w[mask]
    w_ = w_ / w_.sum()
    mx = np.sum(w_ * x)
    my = np.sum(w_ * y)
    vx = np.sum(w_ * (x - mx) ** 2)
    vy = np.sum(w_ * (y - my) ** 2)
    if vx < 1e-12 or vy < 1e-12:
        return float("nan")
    cov = np.sum(w_ * (x - mx) * (y - my))
    return float(cov / np.sqrt(vx * vy))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return _pearson(rx, ry)


def _sign_hit_rate(signed_signal: np.ndarray, fwd_ret: np.ndarray) -> float:
    mask = (
        np.isfinite(signed_signal)
        & np.isfinite(fwd_ret)
        & (np.abs(signed_signal) > 1e-12)
        & (np.abs(fwd_ret) > 1e-12)
    )
    if mask.sum() < 5:
        return float("nan")
    return float(np.mean(np.sign(signed_signal[mask]) == np.sign(fwd_ret[mask])))


def _bootstrap_ci(
    fn, x: np.ndarray, y: np.ndarray, *, w: Optional[np.ndarray] = None,
    n_boot: int = 200, seed: int = 0, q: Tuple[float, float] = (0.025, 0.975),
) -> Tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if w is not None:
        mask &= np.isfinite(w)
    x, y = x[mask], y[mask]
    if x.size < 5:
        return (float("nan"), float("nan"))
    w_ = w[mask] if w is not None else None
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    n = x.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if w_ is None:
            out[i] = fn(x[idx], y[idx])
        else:
            out[i] = fn(x[idx], y[idx], w_[idx])
    out = out[np.isfinite(out)]
    if out.size == 0:
        return (float("nan"), float("nan"))
    return float(np.quantile(out, q[0])), float(np.quantile(out, q[1]))


@dataclass
class HorizonReport:
    horizon: int
    n: int
    pearson: float
    pearson_ci: Tuple[float, float]
    pearson_weighted: float  # confidence-weighted
    spearman: float
    sign_hit_rate_buy_sell: float  # agreement of sentiment.sign with fwd sign
    sign_hit_rate_score: float  # agreement of sign(sentiment_score) with fwd sign
    mean_ret_buy: float
    mean_ret_sell: float
    mean_ret_diff: float
    pearson_high_conf: float  # confidence >= 0.9
    pearson_low_conf: float  # confidence < 0.9


def validate_event_level(
    events: pd.DataFrame,
    horizons: Iterable[int],
    n_boot: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    rows: List[dict] = []
    for k in sorted(horizons):
        fwd_col = f"fwd_ret_{k}"
        if fwd_col not in events.columns:
            continue
        sub = events.loc[np.isfinite(events[fwd_col])]
        if len(sub) < 10:
            continue
        score = sub["sentiment_score"].to_numpy(dtype=np.float64)
        sign = sub["sign"].to_numpy(dtype=np.float64)
        fwd = sub[fwd_col].to_numpy(dtype=np.float64)
        conf = sub["confidence"].to_numpy(dtype=np.float64)

        p = _pearson(score, fwd)
        ci = _bootstrap_ci(_pearson, score, fwd, n_boot=n_boot, seed=seed)
        pw = _pearson(score, fwd, w=conf)
        sp = _spearman(score, fwd)
        hit_bs = _sign_hit_rate(sign, fwd)
        hit_sc = _sign_hit_rate(score, fwd)

        buy_mask = sub["sentiment"].str.lower().to_numpy() == "buy"
        sell_mask = sub["sentiment"].str.lower().to_numpy() == "sell"
        mr_buy = float(np.mean(fwd[buy_mask])) if buy_mask.any() else float("nan")
        mr_sell = float(np.mean(fwd[sell_mask])) if sell_mask.any() else float("nan")

        hi = conf >= 0.9
        lo = ~hi
        p_hi = _pearson(score[hi], fwd[hi]) if hi.sum() >= 10 else float("nan")
        p_lo = _pearson(score[lo], fwd[lo]) if lo.sum() >= 10 else float("nan")

        rows.append(
            dict(
                horizon=k,
                n=int(len(sub)),
                pearson=p,
                pearson_ci_lo=ci[0],
                pearson_ci_hi=ci[1],
                pearson_weighted=pw,
                spearman=sp,
                sign_hit_buy_sell=hit_bs,
                sign_hit_score=hit_sc,
                mean_ret_buy=mr_buy,
                mean_ret_sell=mr_sell,
                mean_ret_diff=mr_buy - mr_sell,
                pearson_high_conf=p_hi,
                pearson_low_conf=p_lo,
            )
        )
    return pd.DataFrame(rows)


# -- Session-level aggregates (train-only full label) ------------------------


def session_label_R(bars_seen: pd.DataFrame, bars_unseen: pd.DataFrame) -> pd.DataFrame:
    """Return per-session label R = close_end / close_half - 1 (train only)."""
    halfway = int(bars_seen["bar_ix"].max())
    end = int(bars_unseen["bar_ix"].max())
    c_half = (
        bars_seen.loc[bars_seen["bar_ix"] == halfway]
        .groupby("session", sort=False)["close"].first()
        .rename("close_half")
    )
    c_end = (
        bars_unseen.loc[bars_unseen["bar_ix"] == end]
        .groupby("session", sort=False)["close"].first()
        .rename("close_end")
    )
    out = pd.concat([c_half, c_end], axis=1).dropna().reset_index()
    out["R"] = out["close_end"] / out["close_half"] - 1.0
    return out


def build_session_sentiment_features(
    sentiments: pd.DataFrame,
    *,
    decision_bar: int = 49,
    decay_half_life: float = 10.0,
) -> pd.DataFrame:
    """One row per session with decay-weighted sentiment aggregates.

    The weights shrink geometrically with the gap ``decision_bar - bar_ix`` so
    that headlines close to the decision point count more than early ones.
    This matches the "news impact decays into the future" prior and is also
    what we use to feed the downstream tailored-modeler pipeline if the
    validator returns a strong enough signal.
    """
    s = sentiments[sentiments["bar_ix"] <= decision_bar].copy()
    s["w_decay"] = np.exp(-np.log(2) * np.maximum(decision_bar - s["bar_ix"], 0) / decay_half_life)
    s["w"] = s["w_decay"] * s["confidence"]
    s["score_w"] = s["sentiment_score"] * s["w"]
    s["sign_w"] = s["sign"] * s["w"]

    agg = s.groupby("session").agg(
        headline_count=("sentiment_score", "size"),
        sum_w=("w", "sum"),
        sum_score_w=("score_w", "sum"),
        sum_sign_w=("sign_w", "sum"),
        mean_score=("sentiment_score", "mean"),
        std_score=("sentiment_score", "std"),
        mean_sign=("sign", "mean"),
        mean_conf=("confidence", "mean"),
        max_abs_score=("sentiment_score", lambda v: float(np.max(np.abs(v)))),
        last_score=("sentiment_score", "last"),
        last_sign=("sign", "last"),
        entity_count=("company", "nunique"),
    ).reset_index()

    agg["weighted_score"] = agg["sum_score_w"] / np.maximum(agg["sum_w"], 1e-12)
    agg["weighted_sign"] = agg["sum_sign_w"] / np.maximum(agg["sum_w"], 1e-12)
    agg["buy_sell_balance"] = agg["mean_sign"]
    agg["std_score"] = agg["std_score"].fillna(0.0)
    return agg[
        [
            "session", "headline_count", "mean_score", "std_score", "mean_sign",
            "mean_conf", "max_abs_score", "last_score", "last_sign", "entity_count",
            "weighted_score", "weighted_sign", "buy_sell_balance",
        ]
    ]


def session_level_correlations(
    session_features: pd.DataFrame,
    labels_R: pd.DataFrame,
    feature_cols: Iterable[str] = (
        "weighted_score", "weighted_sign", "mean_score",
        "mean_sign", "last_score", "last_sign",
    ),
    n_boot: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    merged = session_features.merge(labels_R[["session", "R"]], on="session", how="inner")
    R = merged["R"].to_numpy(dtype=np.float64)
    rows = []
    for col in feature_cols:
        x = merged[col].to_numpy(dtype=np.float64)
        p = _pearson(x, R)
        ci = _bootstrap_ci(_pearson, x, R, n_boot=n_boot, seed=seed)
        sp = _spearman(x, R)
        hit = _sign_hit_rate(x, R)
        rows.append(
            dict(feature=col, n=int(len(merged)), pearson=p,
                 pearson_ci_lo=ci[0], pearson_ci_hi=ci[1], spearman=sp,
                 sign_hit=hit)
        )
    return pd.DataFrame(rows)
