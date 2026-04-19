# ---- Plotting -------------------------------------------------------------

HEADLINE_WRAP = 72          # chars per line when wrapping a full headline for hover
VLINE_SAMPLES = 24          # sample points per vertical line so hover registers anywhere along it
MAX_HOVER_HEADLINES = 6     # beyond this, truncate hover with a "+N more" footer


def _wrap_hover(text: str) -> str:
    return '<br>'.join(wrap(text, HEADLINE_WRAP)) or str(text)


def _bar_hover_text(bar_ix: int, bar_headlines) -> str:
    """Hover for a bar's vertical line: count + full (wrapped) text of every headline at that bar."""
    n = len(bar_headlines)
    header = f"<b>bar {bar_ix}</b> &middot; {n} headline{'s' if n != 1 else ''}"
    shown = bar_headlines.head(MAX_HOVER_HEADLINES)
    lines = [f"&bull; {_wrap_hover(row['headline'])}" for _, row in shown.iterrows()]
    if n > MAX_HOVER_HEADLINES:
        lines.append(f"<i>&hellip; +{n - MAX_HOVER_HEADLINES} more (see table below)</i>")
    return header + '<br>' + '<br>'.join(lines)


def _count_bar_hover_text(bar_ix: int, bar_headlines) -> str:
    """Hover for the counts-subplot bars: same info as the vertical-line hover."""
    return _bar_hover_text(bar_ix, bar_headlines)


def plot_session(kind: str, session_id: int) -> go.Figure:
    bars, headlines, halfway = session_frames(kind, session_id)
    if bars.empty:
        raise ValueError(f'No bars for session {session_id} in {kind}')

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.78, 0.22],
        subplot_titles=('OHLC (candles) + close line  \u00b7  vertical lines mark headlines',
                        'Headline count per bar  \u00b7  hover for exact headlines'),
    )

    # --- candlestick ---
    fig.add_trace(
        go.Candlestick(
            x=bars['bar_ix'], open=bars['open'], high=bars['high'],
            low=bars['low'], close=bars['close'],
            name='OHLC', increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
            showlegend=False,
        ), row=1, col=1)

    # --- close price line (per-phase) ---
    for phase, color in (('seen', '#1f77b4'), ('unseen', '#9467bd')):
        ph = bars[bars['phase'] == phase]
        if ph.empty:
            continue
        fig.add_trace(
            go.Scatter(x=ph['bar_ix'], y=ph['close'], mode='lines',
                       line=dict(color=color, width=1.6),
                       name=f'close ({phase})',
                       hovertemplate='bar %{x}<br>close %{y:.4f}<extra></extra>'),
            row=1, col=1)

    # --- semi-transparent vertical lines at every bar that has headlines ---
    if not headlines.empty:
        grouped = headlines.groupby('bar_ix')
        phase_by_bar = dict(zip(bars['bar_ix'], bars['phase']))

        # Pre-compute hover text for every bar that has headlines
        hover_by_bar = {int(bar_ix): _bar_hover_text(int(bar_ix), g) for bar_ix, g in grouped}

        # y extent for the vertical lines -- span the full price range plus a small pad
        y_lo = float(bars['low'].min())
        y_hi = float(bars['high'].max())
        pad = (y_hi - y_lo) * 0.03 if y_hi > y_lo else max(abs(y_hi), 1e-6) * 0.03
        y_lo -= pad
        y_hi += pad
        ys_sample = [y_lo + (y_hi - y_lo) * i / (VLINE_SAMPLES - 1)
                     for i in range(VLINE_SAMPLES)]

        # One trace per phase; each vertical line is VLINE_SAMPLES points stacked
        # at the same x, segments separated by None so they render independently.
        series = {'seen':   {'x': [], 'y': [], 'text': []},
                  'unseen': {'x': [], 'y': [], 'text': []}}

        for bar_ix, group in grouped:
            hover = hover_by_bar[int(bar_ix)]
            phase = phase_by_bar.get(bar_ix, 'seen')
            s = series[phase]
            for y in ys_sample:
                s['x'].append(bar_ix)
                s['y'].append(y)
                s['text'].append(hover)
            s['x'].append(None); s['y'].append(None); s['text'].append('')

        for phase, color in (('seen',   'rgba(44,160,44,0.35)'),
                             ('unseen', 'rgba(255,127,14,0.40)')):
            s = series[phase]
            if not s['x']:
                continue
            fig.add_trace(
                go.Scatter(
                    x=s['x'], y=s['y'], mode='lines',
                    line=dict(color=color, width=2),
                    name=f'headline ({phase})',
                    text=s['text'],
                    hovertemplate='%{text}<extra></extra>',
                    hoverlabel=dict(bgcolor='white', font=dict(size=11), align='left'),
                    connectgaps=False,
                ), row=1, col=1)

        # --- headline count bar chart (row 2) --------------------------------
        # Hover on each count-bar also shows exact headlines for that bar.
        counts = (headlines.groupby(['bar_ix', 'phase']).size()
                  .unstack(fill_value=0))
        for phase, color in (('seen', '#2ca02c'), ('unseen', '#ff7f0e')):
            if phase not in counts.columns:
                continue
            sub = counts[counts[phase] > 0]
            xs = list(sub.index)
            ys = [int(v) for v in sub[phase].values]
            texts = [hover_by_bar.get(int(x), f"<b>bar {int(x)}</b>") for x in xs]
            fig.add_trace(
                go.Bar(x=xs, y=ys, name=f'#headlines ({phase})',
                       marker_color=color, opacity=0.85, showlegend=False,
                       text=texts,
                       hovertemplate='%{text}<extra></extra>',
                       hoverlabel=dict(bgcolor='white', font=dict(size=11), align='left')),
                row=2, col=1)
    else:
        fig.add_annotation(text='(no headlines for this session)', showarrow=False,
                           xref='x2 domain', yref='y2 domain', x=0.5, y=0.5,
                           font=dict(color='#888'))

    # --- shade unseen region and mark halfway ---
    if halfway is not None and (bars['phase'] == 'unseen').any():
        fig.add_vrect(x0=halfway + 0.5, x1=bars['bar_ix'].max() + 0.5,
                      fillcolor='#9467bd', opacity=0.07, line_width=0,
                      row='all', col=1)
    halfway_close = None
    if halfway is not None:
        halfway_close = float(bars.loc[bars['bar_ix'] == halfway, 'close'].iloc[0])
        fig.add_vline(x=halfway + 0.5, line=dict(color='#555', width=1, dash='dash'),
                      row='all', col=1)
        fig.add_annotation(x=halfway + 0.5, y=halfway_close,
                           text=f'half-way close = {halfway_close:.4f}',
                           showarrow=True, arrowhead=2, ax=40, ay=-30,
                           row=1, col=1)

    final_close = float(bars['close'].iloc[-1])
    ret = ((final_close / halfway_close - 1) * 100
           if halfway_close is not None and (bars['phase'] == 'unseen').any() else None)
    ret_txt = f" \u2014 return {ret:+.2f}%" if ret is not None else ''

    # Summary line shown above the chart and mirrored in the whole-session hover.
    total_headlines = len(headlines)
    bars_with_news = 0 if headlines.empty else headlines['bar_ix'].nunique()

    fig.update_layout(
        title=(f"{kind} \u2014 session {session_id}  "
               f"({len(bars)} bars, {total_headlines} headlines across "
               f"{bars_with_news} bars){ret_txt}"),
        height=680, xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=20, t=70, b=40),
        hovermode='closest',
    )
    fig.update_xaxes(title_text='bar_ix', row=2, col=1)
    fig.update_yaxes(title_text='price', row=1, col=1)
    fig.update_yaxes(title_text='#headlines', row=2, col=1)
    return fig
