"""Session-level sentiment features (seen bars only) merged on Sharpe path features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from datathon_sharpe.features_seen_split import FIRST_HALF_LAST_BAR_IX
from datathon_sharpe.path_features import (
    FEATURE_COLUMNS_PATH_SHARPE,
    build_session_features_with_path,
)
from datathon_sharpe.ts_cnn import CNN_EXTRA_COLUMNS

EPS = 1e-12

SENTIMENTS_SEEN_TRAIN = "sentiments_seen_train.csv"
SENTIMENTS_SEEN_PUBLIC_TEST = "sentiments_seen_public_test.csv"
SENTIMENTS_SEEN_PRIVATE_TEST = "sentiments_seen_private_test.csv"

SENTIMENT_EXTRA_COLUMNS: list[str] = [
    "sentiment_ret_corr",
    "sentiment_sum",
    "sentiment_mean",
    "headline_sentiment_count",
    "sector_entropy",
    "sector_hhi",
    "sector_num_unique",
    "sector_max_share",
    # Extended (confidence, dispersion, buy/sell, time split, granular sectors)
    "sentiment_weighted_mean",
    "confidence_mean",
    "sentiment_std",
    "buy_frac",
    "sentiment_mean_early",
    "sentiment_mean_late",
    "granular_sector_entropy",
]

FEATURE_COLUMNS_SHARPE: list[str] = (
    list(FEATURE_COLUMNS_PATH_SHARPE) + SENTIMENT_EXTRA_COLUMNS + CNN_EXTRA_COLUMNS
)


def load_sentiments_seen_train(data_dir: Path) -> pd.DataFrame | None:
    p = data_dir / SENTIMENTS_SEEN_TRAIN
    if not p.is_file():
        return None
    return pd.read_csv(p)


def load_sentiments_seen_test(data_dir: Path) -> pd.DataFrame | None:
    pub = data_dir / SENTIMENTS_SEEN_PUBLIC_TEST
    priv = data_dir / SENTIMENTS_SEEN_PRIVATE_TEST
    if not pub.is_file() or not priv.is_file():
        return None
    return pd.concat(
        [pd.read_csv(pub), pd.read_csv(priv)],
        ignore_index=True,
    )


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return 0.0
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return 0.0
    r = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(r):
        return 0.0
    return r


def compute_sentiment_session_features(
    session_bars: pd.DataFrame,
    session_sentiments: pd.DataFrame,
    *,
    last_bar_ix: int,
) -> dict[str, float]:
    z = {c: 0.0 for c in SENTIMENT_EXTRA_COLUMNS}
    z["headline_sentiment_count"] = 0.0

    if session_sentiments is None or session_sentiments.empty:
        return z

    g = session_sentiments.loc[
        (session_sentiments["bar_ix"] >= 0) & (session_sentiments["bar_ix"] <= last_bar_ix)
    ]
    if g.empty:
        return z

    sb = session_bars.sort_values("bar_ix")
    if sb.empty:
        return z

    closes = sb.set_index("bar_ix")["close"].astype(np.float64)

    scores: list[float] = []
    rets: list[float] = []
    for _, row in g.iterrows():
        j = int(row["bar_ix"])
        if j <= 0:
            continue
        if j not in closes.index or (j - 1) not in closes.index:
            continue
        r_bar = float(closes.loc[j] / max(float(closes.loc[j - 1]), EPS) - 1.0)
        scores.append(float(row["sentiment_score"]))
        rets.append(r_bar)

    z["sentiment_ret_corr"] = _pearson_corr(np.array(scores), np.array(rets)) if scores else 0.0

    z["sentiment_sum"] = float(g["sentiment_score"].sum())
    z["sentiment_mean"] = float(g["sentiment_score"].mean())
    z["headline_sentiment_count"] = float(len(g))

    sect = g["sector"].astype(str)
    vc = sect.value_counts()
    n = float(len(g))
    if n < 1:
        return z
    p = (vc / n).to_numpy(dtype=np.float64)
    z["sector_num_unique"] = float(len(vc))
    z["sector_max_share"] = float(p.max()) if p.size else 0.0
    z["sector_hhi"] = float(np.sum(p**2))
    z["sector_entropy"] = float(-np.sum(p * np.log(p + EPS)))

    if "confidence" in g.columns:
        c = g["confidence"].astype(np.float64).clip(lower=0.0)
        s_sc = g["sentiment_score"].astype(np.float64)
        sw = float(c.sum())
        if sw > EPS:
            z["sentiment_weighted_mean"] = float((s_sc * c).sum() / sw)
        else:
            z["sentiment_weighted_mean"] = z["sentiment_mean"]
        z["confidence_mean"] = float(c.mean())
    else:
        z["sentiment_weighted_mean"] = z["sentiment_mean"]
        z["confidence_mean"] = 0.0

    z["sentiment_std"] = float(g["sentiment_score"].std(ddof=0)) if len(g) > 1 else 0.0

    if "sentiment" in g.columns:
        z["buy_frac"] = float(
            (g["sentiment"].astype(str).str.lower().str.strip() == "buy").mean()
        )
    else:
        z["buy_frac"] = 0.0

    if last_bar_ix >= 25:
        ge = g.loc[g["bar_ix"] <= 24]
        gl = g.loc[g["bar_ix"] >= 25]
    else:
        ge = g.loc[g["bar_ix"] <= 11]
        gl = g.loc[g["bar_ix"] >= 12]
    z["sentiment_mean_early"] = float(ge["sentiment_score"].mean()) if len(ge) else 0.0
    z["sentiment_mean_late"] = float(gl["sentiment_score"].mean()) if len(gl) else 0.0

    if "granular_sector" in g.columns:
        gs = g["granular_sector"].astype(str)
        vc_g = gs.value_counts()
        p_g = (vc_g / len(g)).to_numpy(dtype=np.float64)
        z["granular_sector_entropy"] = float(-np.sum(p_g * np.log(p_g + EPS)))
    else:
        z["granular_sector_entropy"] = 0.0

    return z


def sentiment_features_by_session(
    bars: pd.DataFrame,
    sentiments: pd.DataFrame | None,
    *,
    last_bar_ix: int,
) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(columns=["session"] + SENTIMENT_EXTRA_COLUMNS)

    rows: list[dict] = []
    if sentiments is None or sentiments.empty:
        for session in bars["session"].unique():
            rows.append({"session": int(session), **{c: 0.0 for c in SENTIMENT_EXTRA_COLUMNS}})
        out = pd.DataFrame(rows)
        return out.sort_values("session").reset_index(drop=True)

    for session, g_b in bars.groupby("session", sort=False):
        g_s = sentiments.loc[sentiments["session"] == session] if "session" in sentiments.columns else pd.DataFrame()
        feats = compute_sentiment_session_features(g_b, g_s, last_bar_ix=last_bar_ix)
        rows.append({"session": int(session), **feats})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["session"] + SENTIMENT_EXTRA_COLUMNS)
    return out.sort_values("session").reset_index(drop=True)


def merge_sharpe_sentiment_features(
    base: pd.DataFrame,
    bars_for_path: pd.DataFrame,
    sentiments: pd.DataFrame | None,
    *,
    last_bar_ix: int,
) -> pd.DataFrame:
    extra = sentiment_features_by_session(bars_for_path, sentiments, last_bar_ix=last_bar_ix)
    if extra.empty:
        out = base.copy()
        for c in SENTIMENT_EXTRA_COLUMNS:
            out[c] = 0.0
        return out
    out = base.merge(extra, on="session", how="left")
    for c in SENTIMENT_EXTRA_COLUMNS:
        out[c] = out[c].fillna(0.0)
    return out


def build_sharpe_session_features(
    bars: pd.DataFrame,
    headlines: pd.DataFrame | None = None,
    sentiments: pd.DataFrame | None = None,
    *,
    first_half: bool = False,
) -> pd.DataFrame:
    """
    Baseline + path + sentiment session features.

    ``sentiments`` must be the seen-window CSV for the same universe as ``bars``
    (train or test); use ``None`` to zero-fill sentiment columns.
    """
    base = build_session_features_with_path(bars, headlines, first_half=first_half)
    last_bar = FIRST_HALF_LAST_BAR_IX if first_half else 49
    bpath = bars.loc[bars["bar_ix"] <= FIRST_HALF_LAST_BAR_IX] if first_half else bars
    out = merge_sharpe_sentiment_features(base, bpath, sentiments, last_bar_ix=last_bar)
    out["cnn_r_pred"] = 0.0
    return out


__all__ = [
    "CNN_EXTRA_COLUMNS",
    "FEATURE_COLUMNS_SHARPE",
    "FEATURE_COLUMNS_PATH_SHARPE",
    "SENTIMENT_EXTRA_COLUMNS",
    "SENTIMENTS_SEEN_PRIVATE_TEST",
    "SENTIMENTS_SEEN_PUBLIC_TEST",
    "SENTIMENTS_SEEN_TRAIN",
    "build_sharpe_session_features",
    "compute_sentiment_session_features",
    "load_sentiments_seen_test",
    "load_sentiments_seen_train",
    "merge_sharpe_sentiment_features",
    "sentiment_features_by_session",
]
