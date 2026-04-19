"""
Interactive explorer for Zurich Datathon 2026 market data (OHLC + headlines).
Run: streamlit run app.py
"""

from __future__ import annotations

import functools
import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Default: repo root data/ (parquet files)
_DATA_FALLBACK = Path(__file__).resolve().parent.parent / "data"

BARS_SEEN_TRAIN = "bars_seen_train.parquet"
BARS_UNSEEN_TRAIN = "bars_unseen_train.parquet"

SENTIMENTS_SEEN_TRAIN_CSV = "sentiments_seen_train.csv"
SENTIMENTS_UNSEEN_TRAIN_CSV = "sentiments_unseen_train.csv"

COMBINED_TRAIN = "Train — combined (seen + unseen)"
TRAIN_CHOICES = frozenset(
    {
        COMBINED_TRAIN,
        "Train — bars seen",
        "Train — bars unseen",
    }
)

DATASET_FILES: dict[str, tuple[str, str]] = {
    "Train — bars seen": ("bars_seen_train.parquet", "headlines_seen_train.parquet"),
    "Train — bars unseen": ("bars_unseen_train.parquet", "headlines_unseen_train.parquet"),
    "Public test — seen": ("bars_seen_public_test.parquet", "headlines_seen_public_test.parquet"),
    "Private test — seen": ("bars_seen_private_test.parquet", "headlines_seen_private_test.parquet"),
}

DATASET_OPTIONS: list[str] = [COMBINED_TRAIN, *DATASET_FILES.keys()]

# Reference layout (matches released parquet layout)
SPLIT_REFERENCE = pd.DataFrame(
    [
        {
            "Split": "Train — seen",
            "Sessions": 1000,
            "Session IDs": "0 – 999",
            "bar_ix": "0 – 49",
            "Bars / session": 50,
            "Files": "bars_seen_train + headlines_seen_train",
            "Note": "First 50 bars of the full training path (same id as unseen).",
        },
        {
            "Split": "Train — unseen",
            "Sessions": 1000,
            "Session IDs": "0 – 999",
            "bar_ix": "50 – 99",
            "Bars / session": 50,
            "Files": "bars_unseen_train + headlines_unseen_train",
            "Note": "Next 50 bars; stack after seen for a 100-bar session.",
        },
        {
            "Split": "Train — combined (this app)",
            "Sessions": 1000,
            "Session IDs": "0 – 999",
            "bar_ix": "0 – 99",
            "Bars / session": 100,
            "Files": "seen + unseen train files",
            "Note": "Concatenate by bar_ix; boundary at 49 | 50.",
        },
        {
            "Split": "Public test — seen",
            "Sessions": 10000,
            "Session IDs": "1000 – 10999",
            "bar_ix": "0 – 49",
            "Bars / session": 50,
            "Files": "bars_seen_public_test + headlines_seen_public_test",
            "Note": "Submission row per id; leaderboard often public only.",
        },
        {
            "Split": "Private test — seen",
            "Sessions": 10000,
            "Session IDs": "11000 – 20999",
            "bar_ix": "0 – 49",
            "Bars / session": 50,
            "Files": "bars_seen_private_test + headlines_seen_private_test",
            "Note": "Include in CSV with public (20 000 rows total).",
        },
    ]
)


@functools.lru_cache(maxsize=16)
def _read_parquet(resolved_dir: str, filename: str) -> pd.DataFrame:
    return pd.read_parquet(Path(resolved_dir) / filename)


def load_bars(data_dir: str, bars_file: str) -> pd.DataFrame:
    return _read_parquet(str(Path(data_dir).resolve()), bars_file)


def load_headlines(data_dir: str, headlines_file: str) -> pd.DataFrame:
    return _read_parquet(str(Path(data_dir).resolve()), headlines_file)


def session_bars(bars: pd.DataFrame, session: int) -> pd.DataFrame:
    return bars[bars["session"] == session].sort_values("bar_ix").reset_index(drop=True)


def session_headlines(headlines: pd.DataFrame, session: int) -> pd.DataFrame:
    if headlines.empty:
        return pd.DataFrame(columns=headlines.columns)
    return headlines[headlines["session"] == session].sort_values("bar_ix").reset_index(drop=True)


def halfway_bar_ix(n_bars: int) -> int | None:
    """Last bar index of the first half (0-based). None if not even split."""
    if n_bars < 2 or n_bars % 2 != 0:
        return None
    return n_bars // 2 - 1


def combine_train_session(
    bars_seen: pd.DataFrame,
    bars_unseen: pd.DataFrame,
    headlines_seen: pd.DataFrame,
    headlines_unseen: pd.DataFrame,
    session: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate seen (bar_ix 0–49) and unseen (50–99) for the same session id."""
    bs = session_bars(bars_seen, session)
    bu = session_bars(bars_unseen, session)
    bars = pd.concat([bs, bu], ignore_index=True).sort_values("bar_ix").reset_index(drop=True)
    hs = session_headlines(headlines_seen, session)
    hu = session_headlines(headlines_unseen, session)
    parts: list[pd.DataFrame] = []
    if not hs.empty:
        parts.append(hs.assign(split="seen"))
    if not hu.empty:
        parts.append(hu.assign(split="unseen"))
    if not parts:
        headlines = pd.DataFrame(columns=["session", "headline", "bar_ix", "split"])
    else:
        headlines = pd.concat(parts, ignore_index=True).sort_values("bar_ix").reset_index(drop=True)
    return bars, headlines


def per_session_combined_train(
    bars_seen: pd.DataFrame,
    bars_unseen: pd.DataFrame,
    headlines_seen: pd.DataFrame,
    headlines_unseen: pd.DataFrame,
) -> pd.DataFrame:
    s_first = (
        bars_seen.sort_values(["session", "bar_ix"])
        .groupby("session", sort=False)
        .first()
        .reset_index()[["session", "open"]]
        .rename(columns={"open": "first_open"})
    )
    u_last = (
        bars_unseen.sort_values(["session", "bar_ix"])
        .groupby("session", sort=False)
        .last()
        .reset_index()[["session", "close"]]
        .rename(columns={"close": "last_close"})
    )
    per = s_first.merge(u_last, on="session")
    per["total_return"] = per["last_close"] / per["first_open"] - 1.0
    hs = headlines_seen.groupby("session").size()
    hu = headlines_unseen.groupby("session").size()
    per["headline_count"] = per["session"].apply(lambda x: int(hs.get(x, 0)) + int(hu.get(x, 0)))
    per["bars"] = 100
    return per


def _segment_ohlc_stats(g: pd.DataFrame, prefix: str) -> dict[str, float]:
    """Aggregate OHLC stats for one session segment (one contiguous bar_ix range)."""
    g = g.sort_values("bar_ix")
    if g.empty:
        return {}
    o0 = float(g["open"].iloc[0])
    c_last = float(g["close"].iloc[-1])
    close = g["close"].astype(np.float64)
    rets = close.pct_change().fillna(0.0)
    hi = float(g["high"].max())
    lo = float(g["low"].min())
    return {
        f"{prefix}open_first": o0,
        f"{prefix}close_last": c_last,
        f"{prefix}return": c_last / o0 - 1.0,
        f"{prefix}vol": float(rets.std(ddof=0)),
        f"{prefix}range_hl": hi - lo,
        f"{prefix}mean_bar_ret": float(rets.mean()),
    }


def train_bars_session_table(bars_seen: pd.DataFrame, bars_unseen: pd.DataFrame) -> pd.DataFrame:
    """
    One row per training session from `bars_seen_train` + `bars_unseen_train` only.

    **Seen (bar_ix 0–49)** → *features* observable before the single decision at bar 50
    (`feat_seen_*`).

    **Unseen (bar_ix 50–99)** → *label period*; the main outcome to predict is the return on
    that segment, **`label_unseen_return`** (last close / first open of the unseen window).

    **R** is the README-style second-half return:
    $$R = \\mathrm{close}_{99}/\\mathrm{close}_{49} - 1$$ (close-to-close from end of seen to end of session).
    """
    rows: list[dict] = []
    for session in sorted(bars_seen["session"].unique()):
        bs = bars_seen[bars_seen["session"] == session].sort_values("bar_ix")
        bu = bars_unseen[bars_unseen["session"] == session].sort_values("bar_ix")
        if bs.empty or bu.empty:
            continue
        c49 = bs.loc[bs["bar_ix"] == 49, "close"]
        c99 = bu.loc[bu["bar_ix"] == 99, "close"]
        if len(c49) != 1 or len(c99) != 1:
            continue
        close_half = float(c49.iloc[0])
        close_end = float(c99.iloc[0])
        row: dict = {"session": int(session)}
        row.update(_segment_ohlc_stats(bs, "feat_seen_"))
        row.update(_segment_ohlc_stats(bu, "label_unseen_"))
        row["ref_close_bar49"] = close_half
        row["ref_close_bar99"] = close_end
        row["R"] = close_end / close_half - 1.0
        o0 = float(bs["open"].iloc[0])
        row["full_session_return"] = close_end / o0 - 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def _corr_numeric_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Drop missing / constant columns so correlation is well-defined."""
    out: list[str] = []
    for c in candidates:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() < 2:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        out.append(c)
    return out


def train_feature_column_names(all_cols: list[str]) -> list[str]:
    """Columns derived from the seen window only (observable before bar 50)."""
    return sorted(c for c in all_cols if c.startswith("feat_seen_"))


def train_label_column_names(all_cols: list[str]) -> list[str]:
    """
    Label-period and outcome columns: unseen-segment stats, README R, and full-session return.
    Primary outcome for “what happens after you act” is **label_unseen_return**.

    Raw unseen open/close *prices* are omitted from this block so the heatmap focuses on
    returns and distributional stats (price levels often correlate mechanically with seen path).
    """
    skip_prices = frozenset({"label_unseen_open_first", "label_unseen_close_last"})
    out: list[str] = []
    for c in sorted(all_cols):
        if c.startswith("label_unseen_") and c not in skip_prices:
            out.append(c)
    for extra in ("R", "full_session_return"):
        if extra in all_cols and extra not in out:
            out.append(extra)
    return out


def feature_vs_label_corr(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_cols: list[str],
    *,
    method: str,
) -> pd.DataFrame:
    """Pairwise correlation between each feature column and each label column (cross-block only)."""
    fc = _corr_numeric_columns(df, feature_cols)
    lc = _corr_numeric_columns(df, label_cols)
    out = pd.DataFrame(index=fc, columns=lc, dtype=float)
    for f in fc:
        for ell in lc:
            out.loc[f, ell] = df[f].corr(df[ell], method=method)
    return out


def fig_cross_corr_heatmap(cross: pd.DataFrame, *, title: str) -> go.Figure:
    """Rows = features, columns = labels; values in $$[-1,1]$$."""
    row_labels = list(cross.index)
    col_labels = list(cross.columns)
    z = cross.to_numpy(dtype=float)
    text = np.round(z, 3).astype(object)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=col_labels,
            y=row_labels,
            colorscale="RdBu",
            zmid=0.0,
            zmin=-1.0,
            zmax=1.0,
            text=text,
            texttemplate="%{text}",
            hovertemplate="feature %{y}<br>label %{x}<br>r = %{z:.4f}<extra></extra>",
        )
    )
    h = 120 + len(row_labels) * 36
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        height=min(h, 720),
        margin=dict(l=160, r=48, t=56, b=120),
        xaxis=dict(side="bottom", tickangle=-35),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def fig_correlation_heatmap(corr: pd.DataFrame, *, title: str) -> go.Figure:
    labels = list(corr.columns)
    z = corr.to_numpy(dtype=float)
    text = np.round(z, 3).astype(object)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmid=0.0,
            zmin=-1.0,
            zmax=1.0,
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}<br>r = %{z:.4f}<extra></extra>",
        )
    )
    h = 480 + max(0, len(labels) - 8) * 14
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        height=min(h, 900),
        margin=dict(l=120, r=48, t=56, b=120),
        xaxis=dict(side="bottom", tickangle=-40),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def fig_correlation_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str,
    method: str = "pearson",
) -> go.Figure:
    sub = df[[x, y]].dropna()
    fig = go.Figure(
        data=go.Scatter(
            x=sub[x],
            y=sub[y],
            mode="markers",
            marker=dict(size=6, color="#90caf9", opacity=0.65),
            name="sessions",
        )
    )
    if len(sub) >= 2:
        r = sub[x].corr(sub[y], method=method)
        label = "ρ" if method == "spearman" else "r"
        fig.update_layout(title=f"{title} ({label} = {r:.3f})")
    else:
        fig.update_layout(title=title)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        height=400,
        xaxis_title=x,
        yaxis_title=y,
        margin=dict(l=48, r=24, t=48, b=48),
    )
    return fig


def fig_candlestick_with_news(
    bars_s: pd.DataFrame,
    headlines_s: pd.DataFrame,
    session: int,
    *,
    vertical_markers: list[tuple[float, str]] | None = None,
    subtitle: str | None = None,
) -> go.Figure:
    n = len(bars_s)
    mid = halfway_bar_ix(n)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.62, 0.38],
    )

    fig.add_trace(
        go.Candlestick(
            x=bars_s["bar_ix"],
            open=bars_s["open"],
            high=bars_s["high"],
            low=bars_s["low"],
            close=bars_s["close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    markers = vertical_markers
    if markers is None and mid is not None:
        markers = [(mid + 0.5, "Halfway (decision bar)")]
    if markers:
        for x, label in markers:
            fig.add_vline(
                x=x,
                line_dash="dash",
                line_color="rgba(255,255,255,0.45)",
                annotation_text=label,
                annotation_position="top",
                row=1,
                col=1,
            )

    if not headlines_s.empty:
        merged = headlines_s.merge(
            bars_s[["bar_ix", "high", "close"]],
            on="bar_ix",
            how="left",
        )
        y_mark = merged["high"] * 1.002
        if "split" in merged.columns:
            hover = (
                "bar %{x}<br>%{customdata[0]}<br><span style='text-align:left'>%{customdata[1]}</span><extra></extra>"
            )
            custom = merged[["split", "headline"]].values
        else:
            hover = "bar %{x}<br>%{text}<extra></extra>"
            custom = None
        fig.add_trace(
            go.Scatter(
                x=merged["bar_ix"],
                y=y_mark,
                mode="markers",
                marker=dict(size=8, color="#ffb74d", symbol="diamond", line=dict(width=1, color="#fff")),
                name="Headline",
                text=merged["headline"] if custom is None else None,
                customdata=custom,
                hovertemplate=hover,
            ),
            row=1,
            col=1,
        )

    rets = bars_s["close"].pct_change().fillna(0)
    colors = ["#26a69a" if r >= 0 else "#ef5350" for r in rets]
    fig.add_trace(
        go.Bar(
            x=bars_s["bar_ix"],
            y=rets,
            name="Bar return",
            marker_color=colors,
            opacity=0.85,
        ),
        row=2,
        col=1,
    )

    title = f"Session {session}"
    if subtitle:
        title = f"{title} — {subtitle}"
    fig.update_layout(
        title=title,
        height=640,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=48, r=24, t=56, b=48),
    )
    fig.update_yaxes(range=[0.93, 1.07], title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Return", row=2, col=1, tickformat=".2%")
    fig.update_xaxes(title_text="bar_ix", row=2, col=1)
    return fig


def train_session_sectors_from_sentiment_csv(
    data_dir: Path,
    train_session_ids: list[int] | np.ndarray,
) -> pd.Series:
    """
    Index: train session id; value: mode of `sector` across sentiments_seen + sentiments_unseen rows.

    If CSVs are missing or invalid, every session is labeled ``Unknown``.
    """
    ids = sorted({int(x) for x in train_session_ids})
    p_seen = data_dir / SENTIMENTS_SEEN_TRAIN_CSV
    p_unseen = data_dir / SENTIMENTS_UNSEEN_TRAIN_CSV
    if not p_seen.is_file() or not p_unseen.is_file():
        return pd.Series("Unknown", index=ids, dtype=object)
    seen = pd.read_csv(p_seen)
    unseen = pd.read_csv(p_unseen)
    if "sector" not in seen.columns or "sector" not in unseen.columns:
        return pd.Series("Unknown", index=ids, dtype=object)
    h = pd.concat([seen, unseen], ignore_index=True)

    def _mode_sector(s: pd.Series) -> str:
        s = s.dropna()
        if s.empty:
            return "Unknown"
        return str(s.value_counts().index[0])

    modes = h.groupby("session", sort=True)["sector"].apply(_mode_sector)
    return modes.reindex(ids).fillna("Unknown")


def _train_session_seen_return_through_bar49(bars_seen: pd.DataFrame) -> pd.Series:
    """
    Return from session start through end of seen window:
    close(bar_ix 49) / open(bar_ix 0) - 1 on `bars_seen_train` only.
    """
    out: dict[int, float] = {}
    for session in sorted(bars_seen["session"].unique()):
        sid = int(session)
        bs = bars_seen[bars_seen["session"] == sid].sort_values("bar_ix")
        if bs.empty:
            continue
        o0 = bs.loc[bs["bar_ix"] == 0, "open"]
        c49 = bs.loc[bs["bar_ix"] == 49, "close"]
        if len(o0) != 1 or len(c49) != 1:
            continue
        out[sid] = float(c49.iloc[0]) / float(o0.iloc[0]) - 1.0
    return pd.Series(out, dtype=np.float64)


def _train_session_unseen_R(
    bars_seen: pd.DataFrame,
    bars_unseen: pd.DataFrame,
) -> pd.Series:
    """
    README-style label per session: R = close(bar_ix 99) / close(bar_ix 49) - 1
    (return from end of seen window to bar 99 — the “50–99” / decision-relevant leg).
    """
    out: dict[int, float] = {}
    for session in sorted(bars_seen["session"].unique()):
        sid = int(session)
        bs = bars_seen[bars_seen["session"] == sid].sort_values("bar_ix")
        bu = bars_unseen[bars_unseen["session"] == sid].sort_values("bar_ix")
        if bs.empty or bu.empty:
            continue
        c49 = bs.loc[bs["bar_ix"] == 49, "close"]
        c99 = bu.loc[bu["bar_ix"] == 99, "close"]
        if len(c49) != 1 or len(c99) != 1:
            continue
        out[sid] = float(c99.iloc[0]) / float(c49.iloc[0]) - 1.0
    return pd.Series(out, dtype=np.float64)


# Path colors: 4 quadrants = (seen to bar49 up?, unseen R up?) with R = close99/close49 − 1.
_QUAD_PATH_SOLID: dict[tuple[bool, bool], str] = {
    (True, True): "#26a69a",
    (True, False): "#42a5f5",
    (False, True): "#ffca28",
    (False, False): "#ef5350",
}
_QUAD_PATH_RGBA: dict[tuple[bool, bool], str] = {
    (True, True): "rgba(38, 166, 154, 0.22)",
    (True, False): "rgba(66, 165, 245, 0.22)",
    (False, True): "rgba(255, 202, 40, 0.22)",
    (False, False): "rgba(239, 83, 80, 0.22)",
}
_QUAD_LEGEND: dict[tuple[bool, bool], str] = {
    (True, True): "seen↑ R↑",
    (True, False): "seen↑ R↓",
    (False, True): "seen↓ R↑",
    (False, False): "seen↓ R↓",
}


def _train_normalized_paths_long(
    bars_seen: pd.DataFrame,
    bars_unseen: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (session, bar_ix): normalized close = close / first open of that session."""
    full = pd.concat([bars_seen, bars_unseen], ignore_index=True).sort_values(
        ["session", "bar_ix"]
    )
    parts: list[pd.DataFrame] = []
    for session, g in full.groupby("session", sort=True):
        g = g.sort_values("bar_ix")
        if g.empty:
            continue
        o0 = float(g["open"].iloc[0])
        if o0 == 0.0:
            continue
        parts.append(
            g.assign(
                norm_close=g["close"].astype(np.float64) / o0,
                session=int(session),
            )[["session", "bar_ix", "norm_close"]]
        )
    if not parts:
        return pd.DataFrame(columns=["session", "bar_ix", "norm_close"])
    return pd.concat(parts, ignore_index=True)


def fig_train_combined_all_sessions(
    bars_seen: pd.DataFrame,
    bars_unseen: pd.DataFrame,
    *,
    show_spaghetti: bool = False,
    sample_paths: int = 28,
    y_axis_mode: str = "auto",
    allowed_sessions: set[int] | None = None,
    chart_title_note: str = "",
) -> go.Figure:
    """
    Cross-sectional distribution of normalized paths (readable default), optional spaghetti.

    `y_axis_mode`: "auto" uses percentile span + padding; "fixed" uses 0.93–1.07.
    `allowed_sessions`: if set, keep only these session ids (e.g. headline sector filter).
    """
    long_df = _train_normalized_paths_long(bars_seen, bars_unseen)
    if allowed_sessions is not None:
        long_df = long_df[long_df["session"].isin(allowed_sessions)]
    if long_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No paths to plot",
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            height=400,
        )
        return fig

    n_sessions = int(long_df["session"].nunique())

    seen_ret_by_session = _train_session_seen_return_through_bar49(bars_seen)
    R_by_session = _train_session_unseen_R(bars_seen, bars_unseen)

    def _seen_val(sid: int) -> float:
        if sid not in seen_ret_by_session.index:
            return float("nan")
        return float(seen_ret_by_session.loc[sid])

    def _r_val(sid: int) -> float:
        if sid not in R_by_session.index:
            return float("nan")
        return float(R_by_session.loc[sid])

    def _quadrant(sid: int) -> tuple[bool, bool]:
        """(seen to bar49 return > 0, unseen R > 0). Missing data → (False, False)."""
        sv, rv = _seen_val(sid), _r_val(sid)
        if np.isnan(sv) or np.isnan(rv):
            return (False, False)
        return (sv > 0.0, rv > 0.0)

    stats = (
        long_df.groupby("bar_ix", sort=True)["norm_close"]
        .agg(
            p05=lambda s: float(s.quantile(0.05)),
            p25=lambda s: float(s.quantile(0.25)),
            p50=lambda s: float(s.quantile(0.50)),
            p75=lambda s: float(s.quantile(0.75)),
            p95=lambda s: float(s.quantile(0.95)),
            mean="mean",
        )
        .reset_index()
    )
    bx = stats["bar_ix"].to_numpy(dtype=np.float64)

    fig = go.Figure()

    if show_spaghetti:

        def _spaghetti_xy_for_sessions(sessions: list[int]) -> tuple[list[float | None], list[float | None]]:
            xs: list[float | None] = []
            ys: list[float | None] = []
            for sid in sessions:
                g = long_df[long_df["session"] == sid].sort_values("bar_ix")
                if g.empty:
                    continue
                xs.extend(g["bar_ix"].tolist())
                xs.append(None)
                ys.extend(g["norm_close"].tolist())
                ys.append(None)
            return xs, ys

        sess_list = sorted(int(s) for s in long_df["session"].unique().tolist())
        buckets: dict[tuple[bool, bool], list[int]] = {k: [] for k in _QUAD_PATH_SOLID}
        for sid in sess_list:
            buckets[_quadrant(sid)].append(sid)
        for qk, sessions in buckets.items():
            if not sessions:
                continue
            xs, ys = _spaghetti_xy_for_sessions(sessions)
            if not xs:
                continue
            nq = len(sessions)
            fig.add_trace(
                go.Scattergl(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(width=0.5, color=_QUAD_PATH_RGBA[qk]),
                    name=f"All paths · {_QUAD_LEGEND[qk]} (n={nq})",
                    legendgroup=f"spaghetti_{qk}",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

    def _band(upper: np.ndarray, lower: np.ndarray, name: str, fillcolor: str) -> go.Scatter:
        x_poly = np.concatenate([bx, bx[::-1]])
        y_poly = np.concatenate([upper, lower[::-1]])
        return go.Scatter(
            x=x_poly,
            y=y_poly,
            fill="toself",
            fillcolor=fillcolor,
            line=dict(width=0, color="rgba(0,0,0,0)"),
            name=name,
            hoverinfo="skip",
            showlegend=True,
        )

    fig.add_trace(
        _band(
            stats["p95"].to_numpy(),
            stats["p05"].to_numpy(),
            "5th–95th percentile",
            "rgba(100, 181, 246, 0.14)",
        )
    )
    fig.add_trace(
        _band(
            stats["p75"].to_numpy(),
            stats["p25"].to_numpy(),
            "25th–75th percentile (IQR)",
            "rgba(100, 181, 246, 0.28)",
        )
    )

    rng = np.random.default_rng(42)
    sessions_unique = long_df["session"].unique()
    if sample_paths > 0 and len(sessions_unique) > 0:
        n_pick = min(sample_paths, len(sessions_unique))
        picked = rng.choice(sessions_unique, size=n_pick, replace=False)
        picked_sorted = sorted(int(s) for s in picked)
        leg_done: dict[tuple[bool, bool], bool] = {k: False for k in _QUAD_PATH_SOLID}
        for sid in picked_sorted:
            g = long_df[long_df["session"] == sid].sort_values("bar_ix")
            qk = _quadrant(sid)
            c = _QUAD_PATH_SOLID[qk]
            sv, rv = _seen_val(sid), _r_val(sid)
            s_txt = f"{sv:.6f}" if not np.isnan(sv) else "n/a"
            r_txt = f"{rv:.6f}" if not np.isnan(rv) else "n/a"
            sl = not leg_done[qk]
            leg_done[qk] = True
            nm = f"Sample · {_QUAD_LEGEND[qk]}" if sl else ""
            fig.add_trace(
                go.Scatter(
                    x=g["bar_ix"],
                    y=g["norm_close"],
                    mode="lines",
                    line=dict(width=1.2, color=c),
                    name=nm or "Sample paths",
                    legendgroup=f"sample_{qk}",
                    showlegend=sl,
                    hovertemplate=(
                        f"session {sid}<br>seen to 49={s_txt} (close49/open0-1)<br>"
                        f"unseen R={r_txt} (close99/close49-1)<br>"
                        "bar_ix=%{x}<br>norm close=%{y:.4f}<extra></extra>"
                    ),
                )
            )

    fig.add_trace(
        go.Scatter(
            x=stats["bar_ix"],
            y=stats["p50"],
            mode="lines",
            name="Median",
            line=dict(width=2.8, color="#eceff1"),
            hovertemplate="bar %{x}<br>median %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=stats["bar_ix"],
            y=stats["mean"],
            mode="lines",
            name="Mean",
            line=dict(width=1.8, color="#ffb74d", dash="dot"),
            hovertemplate="bar %{x}<br>mean %{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="rgba(255, 255, 255, 0.25)",
        annotation_text="Start (1.0)",
        annotation_position="left",
    )

    fig.add_vline(
        x=49.5,
        line_dash="dash",
        line_color="rgba(255, 255, 255, 0.45)",
        annotation_text="Seen | Unseen",
        annotation_position="top",
    )
    fig.update_layout(
        title=(
            f"Train combined — {n_sessions} sessions — "
            f"distribution of normalized close paths (close ÷ session first open){chart_title_note}"
        ),
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        height=560,
        margin=dict(l=48, r=24, t=56, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
        xaxis_title="bar_ix",
        yaxis_title="Normalized close",
    )
    if y_axis_mode == "fixed":
        fig.update_yaxes(range=[0.93, 1.07])
    else:
        pad = 0.004
        y_lo = float(stats["p05"].min()) - pad
        y_hi = float(stats["p95"].max()) + pad
        y_lo = max(y_lo, 0.88)
        y_hi = min(y_hi, 1.12)
        fig.update_yaxes(range=[y_lo, y_hi])
    return fig


def main() -> None:
    st.set_page_config(page_title="Datathon market data", layout="wide")
    data_dir = st.sidebar.text_input(
        "Data folder",
        value=str(_DATA_FALLBACK),
        help="Path to the folder containing parquet files.",
    )
    data_path = Path(data_dir)
    if not data_path.is_dir():
        st.error(f"Data folder not found: {data_path}")
        st.stop()

    choice = st.sidebar.selectbox("Dataset", DATASET_OPTIONS)

    bars_seen = bars_unseen = None
    h_seen = h_unseen = None
    bars: pd.DataFrame | None = None
    headlines: pd.DataFrame | None = None

    if choice == COMBINED_TRAIN:
        with st.spinner("Loading parquet…"):
            bars_seen = load_bars(str(data_path), "bars_seen_train.parquet")
            bars_unseen = load_bars(str(data_path), "bars_unseen_train.parquet")
            h_seen = load_headlines(str(data_path), "headlines_seen_train.parquet")
            h_unseen = load_headlines(str(data_path), "headlines_unseen_train.parquet")
        sessions = sorted(bars_seen["session"].unique().tolist())
    else:
        bars_file, headlines_file = DATASET_FILES[choice]
        with st.spinner("Loading parquet…"):
            bars = load_bars(str(data_path), bars_file)
            headlines = load_headlines(str(data_path), headlines_file)
        assert bars is not None and headlines is not None
        sessions = sorted(bars["session"].unique().tolist())

    ds_key = f"sid_{choice}"
    if ds_key not in st.session_state:
        st.session_state[ds_key] = sessions[0]
    lo, hi = min(sessions), max(sessions)
    if st.session_state[ds_key] not in sessions:
        st.session_state[ds_key] = sessions[0]
    if st.sidebar.button("Random session"):
        st.session_state[ds_key] = random.choice(sessions)
    sid = st.sidebar.number_input(
        "Session",
        min_value=lo,
        max_value=hi,
        value=int(st.session_state[ds_key]),
        step=1,
        help="Train ids are 0–999; public test 1000–10999; private 11000–20999.",
    )
    st.session_state[ds_key] = sid

    tab_session, tab_overview = st.tabs(["Session", "Dataset overview"])

    if choice == COMBINED_TRAIN:
        assert bars_seen is not None and bars_unseen is not None
        assert h_seen is not None and h_unseen is not None
        b_s, h_s = combine_train_session(bars_seen, bars_unseen, h_seen, h_unseen, sid)
    else:
        assert bars is not None and headlines is not None
        b_s = session_bars(bars, sid)
        h_s = session_headlines(headlines, sid)

    with tab_session:
        if b_s.empty:
            st.warning("No rows for this session.")
        else:
            first_open = float(b_s["open"].iloc[0])
            last_close = float(b_s["close"].iloc[-1])
            cum_ret = last_close / first_open - 1.0
            n_h = len(h_s)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Bars", len(b_s))
            c2.metric("Headlines", n_h)
            c3.metric("Session return", f"{cum_ret * 100:.2f}%")
            c4.metric("Last close", f"{last_close:.4f}")

            if choice == COMBINED_TRAIN:
                fig = fig_candlestick_with_news(
                    b_s,
                    h_s,
                    sid,
                    vertical_markers=[
                        (24.5, "Halfway (50-bar decision)"),
                        (49.5, "Seen | Unseen"),
                    ],
                    subtitle="full train path (100 bars)",
                )
            else:
                fig = fig_candlestick_with_news(b_s, h_s, sid)

            st.plotly_chart(fig, use_container_width=True, key=f"session_candle_{choice}_{sid}")

            st.subheader("Headlines for this session")
            if h_s.empty:
                st.caption("No headlines in this session.")
            else:
                cols = ["bar_ix", "headline"]
                if "split" in h_s.columns:
                    cols = ["bar_ix", "split", "headline"]
                show = h_s[cols].copy()
                st.dataframe(show, use_container_width=True, hide_index=True)

    with tab_overview:
        st.subheader("Splits & submission scope")
        st.dataframe(SPLIT_REFERENCE, use_container_width=True, hide_index=True)
        st.caption(
            "Submission CSV: `session`, `target_position`. Full test submission is typically **20 000** rows "
            "(**1000–10999** public + **11000–20999** private), unless the platform says otherwise."
        )

        lu = st.number_input("Which split contains session id…", 0, 20999, 1000, 1)
        hits: list[str] = []
        if 0 <= lu <= 999:
            hits.append("**Train** — seen, unseen, and combined (ids 0–999)")
        if 1000 <= lu <= 10999:
            hits.append("**Public test** — seen bars only")
        if 11000 <= lu <= 20999:
            hits.append("**Private test** — seen bars only")
        if hits:
            st.markdown("— " + "\n\n— ".join(hits))
        else:
            st.info("This id is not used in the released data (gaps exist only outside 0–20999).")

        st.subheader(f"Distributions — {choice}")

        if choice == COMBINED_TRAIN:
            assert bars_seen is not None and bars_unseen is not None
            assert h_seen is not None and h_unseen is not None
            per_session = per_session_combined_train(bars_seen, bars_unseen, h_seen, h_unseen)
        else:
            assert bars is not None and headlines is not None
            per_session = (
                bars.sort_values(["session", "bar_ix"])
                .groupby("session", sort=False)
                .agg(
                    bars=("bar_ix", "count"),
                    first_open=("open", "first"),
                    last_close=("close", "last"),
                )
                .reset_index()
            )
            per_session["total_return"] = per_session["last_close"] / per_session["first_open"] - 1.0
            hc = headlines.groupby("session").size().rename("headline_count").reset_index()
            per_session = per_session.merge(hc, on="session", how="left")
            per_session["headline_count"] = per_session["headline_count"].fillna(0).astype(int)

        m1, m2, m3 = st.columns(3)
        m1.metric("Sessions", f"{len(per_session):,}")
        m2.metric("Bars / session", f"{int(per_session['bars'].iloc[0])}")
        m3.metric("Mean headlines / session", f"{per_session['headline_count'].mean():.2f}")

        c1, c2 = st.columns(2)
        with c1:
            fig_r = go.Figure()
            fig_r.add_trace(
                go.Histogram(
                    x=per_session["total_return"],
                    nbinsx=50,
                    marker_color="#64b5f6",
                    opacity=0.9,
                )
            )
            fig_r.update_layout(
                title=f"Session total return ({choice})",
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#161b22",
                xaxis_title="last_close / first_open − 1",
                yaxis_title="Count",
                height=360,
                margin=dict(l=48, r=24, t=48, b=48),
            )
            st.plotly_chart(fig_r, use_container_width=True, key=f"overview_hist_return_{choice}")

        with c2:
            fig_h = go.Figure()
            hmax = int(per_session["headline_count"].max())
            nb = min(30, max(5, hmax + 1)) if hmax >= 0 else 5
            fig_h.add_trace(
                go.Histogram(
                    x=per_session["headline_count"],
                    nbinsx=nb,
                    marker_color="#ffb74d",
                    opacity=0.9,
                )
            )
            fig_h.update_layout(
                title=f"Headlines per session ({choice})",
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#161b22",
                xaxis_title="Headline count",
                yaxis_title="Sessions",
                height=360,
                margin=dict(l=48, r=24, t=48, b=48),
            )
            st.plotly_chart(fig_h, use_container_width=True, key=f"overview_hist_headlines_{choice}")

        if choice == COMBINED_TRAIN:
            assert bars_seen is not None and bars_unseen is not None
            st.subheader("All sessions on one chart (train combined)")
            st.caption(
                "Default view: **cross-sectional percentiles** at each bar (across the 1000 sessions) — "
                "much easier to read than 1000 overlapping lines. "
                "Normalized close = **close ÷ first open** of that session; bars 0–49 seen, 50–99 unseen."
            )
            if not (data_path / SENTIMENTS_SEEN_TRAIN_CSV).is_file() or not (
                data_path / SENTIMENTS_UNSEEN_TRAIN_CSV
            ).is_file():
                st.warning(
                    f"Add `{SENTIMENTS_SEEN_TRAIN_CSV}` and `{SENTIMENTS_UNSEEN_TRAIN_CSV}` next to your "
                    "parquet files to enable sector filtering (otherwise every session is **Unknown**)."
                )
            sector_by_session = train_session_sectors_from_sentiment_csv(
                data_path, bars_seen["session"].unique()
            )
            sector_options = sorted(sector_by_session.unique().tolist())
            sector_pick = st.multiselect(
                "Filter by sector (from sentiment CSVs)",
                options=sector_options,
                default=sector_options,
                key="overview_train_sectors",
                help=(
                    "Uses the **`sector`** column in `sentiments_seen_train.csv` and "
                    "`sentiments_unseen_train.csv`. Per session, the label is the **mode** (most frequent) "
                    "sector across all headline rows in seen + unseen."
                ),
            )
            if len(sector_pick) == 0:
                st.warning("Select at least one sector. Showing all categories until you choose at least one.")
                sector_pick = list(sector_options)
            if set(sector_pick) == set(sector_options):
                allowed_sess: set[int] | None = None
                chart_title_note = ""
            else:
                allowed_sess = set(
                    int(x) for x in sector_by_session[sector_by_session.isin(sector_pick)].index.tolist()
                )
                chart_title_note = " — sector filter (sentiment CSV)"
                st.caption(
                    f"**Sessions in view:** {len(allowed_sess)} of {len(sector_by_session)} "
                    "(after sector filter)."
                )
            oc1, oc2, oc3 = st.columns([1, 1, 1])
            with oc1:
                show_spaghetti = st.checkbox(
                    "Show all 1000 paths (spaghetti, faint)",
                    value=False,
                    key="overview_train_spaghetti",
                    help=(
                        "Behind the bands. Four colors: seen window to bar 49 up/down "
                        "(close49/open0−1) × unseen leg up/down (close99/close49−1, README R)."
                    ),
                )
            with oc2:
                n_sample = st.slider(
                    "Sample paths (random)",
                    min_value=0,
                    max_value=80,
                    value=28,
                    step=1,
                    key="overview_train_sample_n",
                    help=(
                        "Paths on top of bands; 0 hides them. Same 4-way color key as spaghetti "
                        "(seen→49 return vs unseen R)."
                    ),
                )
            with oc3:
                y_mode = st.radio(
                    "Y-axis",
                    ("Auto (percentile span)", "Fixed 0.93 – 1.07"),
                    horizontal=True,
                    key="overview_train_yaxis",
                )
            fig_all = fig_train_combined_all_sessions(
                bars_seen,
                bars_unseen,
                show_spaghetti=show_spaghetti,
                sample_paths=n_sample,
                y_axis_mode="fixed" if y_mode.startswith("Fixed") else "auto",
                allowed_sessions=allowed_sess,
                chart_title_note=chart_title_note,
            )
            st.plotly_chart(
                fig_all,
                use_container_width=True,
                key=(
                    f"overview_all_sessions_overlay_{choice}_{show_spaghetti}_{n_sample}_{y_mode}_"
                    f"{','.join(sorted(sector_pick))}"
                ),
            )

        if choice in TRAIN_CHOICES:
            st.subheader("Train: features (seen) vs labels (unseen)")
            st.caption(
                "You only choose an action once, at **bar 50**. Everything in **`bars_seen_train`** "
                "(bar_ix 0–49) is *known before* that choice — treat these as **features** (`feat_seen_*`). "
                "The **unseen** path (**`bars_unseen_train`**, bar_ix 50–99) is what you do *not* know at decision time; "
                "its total return **`label_unseen_return`** (last close / first open on that segment) is the main **label** "
                "to relate to features. **R** is $$\\mathrm{close}_{99}/\\mathrm{close}_{49}-1$$ (README second-half return)."
            )
            with st.spinner("Aggregating train bars…"):
                bs_tr = load_bars(str(data_path), BARS_SEEN_TRAIN)
                bu_tr = load_bars(str(data_path), BARS_UNSEEN_TRAIN)
                merged = train_bars_session_table(bs_tr, bu_tr)
            cand = [c for c in merged.columns if c != "session"]
            use_cols = _corr_numeric_columns(merged, cand)
            feat_names = train_feature_column_names(use_cols)
            label_names = train_label_column_names(use_cols)
            if len(use_cols) < 2:
                st.warning("Not enough varying numeric columns to compute correlations.")
            else:
                cm1, cm2 = st.columns([1, 2])
                with cm1:
                    corr_method = st.radio(
                        "Correlation",
                        ("Pearson", "Spearman"),
                        horizontal=False,
                        key=f"corr_method_{choice}",
                        help="Pearson: linear. Spearman: monotonic (rank-based), more robust to outliers.",
                    )
                method = "pearson" if corr_method == "Pearson" else "spearman"
                with cm2:
                    dropped = set(cand) - set(use_cols)
                    st.caption(
                        f"**{len(merged)}** sessions · dropped (constant): {dropped or 'none'}"
                    )

                if "feat_seen_return" in merged.columns and "label_unseen_return" in merged.columns:
                    st.markdown("**Primary view — seen return vs unseen return (label)**")
                    fig_main = fig_correlation_scatter(
                        merged,
                        "feat_seen_return",
                        "label_unseen_return",
                        title="label_unseen_return vs feat_seen_return",
                        method=method,
                    )
                    st.plotly_chart(
                        fig_main,
                        use_container_width=True,
                        key=f"train_scatter_feat_vs_label_{choice}",
                    )
                else:
                    st.info("Missing `feat_seen_return` or `label_unseen_return`; check train parquet layout.")

                if feat_names and label_names:
                    cross = feature_vs_label_corr(
                        merged, feat_names, label_names, method=method
                    )
                    if not cross.empty and cross.shape[0] > 0 and cross.shape[1] > 0:
                        st.markdown(
                            f"**Feature × label correlations** ({corr_method}) — rows = seen features, columns = unseen / outcomes"
                        )
                        fig_cross = fig_cross_corr_heatmap(
                            cross,
                            title=f"{corr_method}: feat(seen) vs label(unseen + R)",
                        )
                        st.plotly_chart(
                            fig_cross,
                            use_container_width=True,
                            key=f"train_cross_corr_heatmap_{choice}",
                        )

                st.markdown("**Full numeric correlation matrix** (all columns)")
                sub = merged[use_cols]
                corr_mat = sub.corr(method=method, numeric_only=True)
                title = f"{corr_method} · all train bar stats"
                fig_corr = fig_correlation_heatmap(corr_mat, title=title)
                st.plotly_chart(
                    fig_corr,
                    use_container_width=True,
                    key=f"train_full_corr_matrix_{choice}",
                )

                st.markdown("**Inspect any pair**")
                ax1, ax2 = st.columns(2)
                default_x = "feat_seen_return" if "feat_seen_return" in use_cols else use_cols[0]
                default_y = "label_unseen_return" if "label_unseen_return" in use_cols else use_cols[-1]
                with ax1:
                    cx = st.selectbox(
                        "X",
                        use_cols,
                        index=use_cols.index(default_x) if default_x in use_cols else 0,
                        key=f"cx_{choice}",
                    )
                with ax2:
                    cy = st.selectbox(
                        "Y",
                        use_cols,
                        index=use_cols.index(default_y) if default_y in use_cols else min(1, len(use_cols) - 1),
                        key=f"cy_{choice}",
                    )
                fig_sc = fig_correlation_scatter(
                    merged, cx, cy, title=f"{cy} vs {cx}", method=method
                )
                st.plotly_chart(
                    fig_sc,
                    use_container_width=True,
                    key=f"train_scatter_pair_{choice}_{cx}_{cy}",
                )

                with st.expander("Correlation matrix (CSV-style)"):
                    st.dataframe(corr_mat, use_container_width=True)

        st.caption(
            "Prices start near 1. For 50-bar views, the dashed line is the competition halfway decision bar. "
            "Combined train adds a second line at the seen/unseen data boundary (after bar 49)."
        )


if __name__ == "__main__":
    # Streamlit also runs this file as __main__ on every rerun. Only subprocess when
    # launched via `python app.py` (e.g. IDE Run); otherwise we'd spawn endless servers/tabs.
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        get_script_run_ctx = lambda: None  # type: ignore[assignment]

    if get_script_run_ctx() is not None:
        main()
    else:
        import subprocess
        import sys

        script = Path(__file__).resolve()
        raise SystemExit(
            subprocess.call([sys.executable, "-m", "streamlit", "run", str(script)])
        )
