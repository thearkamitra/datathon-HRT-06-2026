"""
Interactive explorer for Zurich Datathon 2026 market data (OHLC + headlines).
Run: streamlit run app.py
"""

from __future__ import annotations

import functools
import random
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Default: repo root data/ (parquet files)
_DATA_FALLBACK = Path(__file__).resolve().parent.parent / "data"

COMBINED_TRAIN = "Train — combined (seen + unseen)"

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
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Return", row=2, col=1, tickformat=".2%")
    fig.update_xaxes(title_text="bar_ix", row=2, col=1)
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

            st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(fig_r, use_container_width=True)

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
            st.plotly_chart(fig_h, use_container_width=True)

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
