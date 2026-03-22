"""
charts.py
---------
Plotly chart factory functions for the Streamlit dashboard.

All functions return :class:`plotly.graph_objects.Figure` objects ready for
``st.plotly_chart(fig, use_container_width=True)``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def candlestick_chart(
    df: pd.DataFrame,
    ticker: str,
    show_volume: bool = True,
    show_ma: bool = True,
    show_bb: bool = False,
) -> go.Figure:
    """
    Create an interactive candlestick chart with optional overlays.

    Args:
        df:          OHLCV DataFrame with indicator columns appended.
        ticker:      Ticker symbol used as chart title.
        show_volume: Add volume bars on a secondary sub-plot.
        show_ma:     Overlay SMA_50 and SMA_200 lines.
        show_bb:     Overlay Bollinger Bands.

    Returns:
        Plotly Figure.
    """
    rows = 2 if show_volume else 1
    row_heights = [0.75, 0.25] if show_volume else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=ticker,
            increasing_line_color="#00C853",
            decreasing_line_color="#D50000",
        ),
        row=1, col=1,
    )

    if show_ma:
        for col_name, color, label in [
            ("SMA_50", "#2196F3", "SMA 50"),
            ("SMA_200", "#FF9800", "SMA 200"),
        ]:
            if col_name in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df[col_name],
                        mode="lines", name=label,
                        line=dict(color=color, width=1.5),
                    ),
                    row=1, col=1,
                )

    if show_bb:
        for col_name, label, fill in [
            ("BBU_20_2.0_2.0", "BB Upper", None),
            ("BBM_20_2.0_2.0", "BB Mid", None),
            ("BBL_20_2.0_2.0", "BB Lower", "tonexty"),
        ]:
            if col_name in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df[col_name],
                        mode="lines", name=label,
                        line=dict(color="#9C27B0", width=1, dash="dot"),
                        fill=fill,
                        fillcolor="rgba(156,39,176,0.05)",
                    ),
                    row=1, col=1,
                )

    if show_volume and "Volume" in df.columns:
        colors = [
            "#00C853" if c >= o else "#D50000"
            for c, o in zip(df["Close"], df["Open"])
        ]
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors),
            row=2, col=1,
        )

    fig.update_layout(
        title=f"{ticker} — Price Chart",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=40, r=40, t=50, b=20),
        legend=dict(orientation="h", y=1.02),
    )
    return fig


def rsi_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create an RSI indicator chart with overbought/oversold zones.

    Args:
        df: DataFrame with ``RSI_14`` column.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    if "RSI_14" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["RSI_14"],
                mode="lines", name="RSI(14)",
                line=dict(color="#2196F3", width=2),
            )
        )

    # Reference lines
    for level, color, label in [(70, "#D50000", "Overbought 70"), (30, "#00C853", "Oversold 30")]:
        fig.add_hline(y=level, line_dash="dash", line_color=color, annotation_text=label)

    fig.update_layout(
        title="RSI (14)",
        template="plotly_dark",
        height=250,
        yaxis=dict(range=[0, 100]),
        margin=dict(l=40, r=40, t=40, b=20),
    )
    return fig


def macd_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a MACD chart with signal line and histogram.

    Args:
        df: DataFrame with MACD indicator columns.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    if "MACD_12_26_9" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["MACD_12_26_9"],
                mode="lines", name="MACD",
                line=dict(color="#2196F3", width=2),
            )
        )

    if "MACDs_12_26_9" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["MACDs_12_26_9"],
                mode="lines", name="Signal",
                line=dict(color="#FF9800", width=1.5),
            )
        )

    if "MACDh_12_26_9" in df.columns:
        hist = df["MACDh_12_26_9"]
        colors = ["#00C853" if v >= 0 else "#D50000" for v in hist]
        fig.add_trace(
            go.Bar(x=df.index, y=hist, name="Histogram", marker_color=colors, opacity=0.6)
        )

    fig.add_hline(y=0, line_dash="solid", line_color="gray")

    fig.update_layout(
        title="MACD (12, 26, 9)",
        template="plotly_dark",
        height=250,
        margin=dict(l=40, r=40, t=40, b=20),
    )
    return fig


def equity_curve_chart(
    equity_curve: List[float],
    ticker: str,
    strategy_name: str = "Strategy",
) -> go.Figure:
    """
    Plot a strategy equity curve from backtest results.

    Args:
        equity_curve:  List of portfolio values over time.
        ticker:        Ticker symbol.
        strategy_name: Name of the strategy.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=equity_curve,
            mode="lines",
            name=strategy_name,
            line=dict(color="#2196F3", width=2),
            fill="tozeroy",
            fillcolor="rgba(33,150,243,0.1)",
        )
    )

    fig.add_hline(y=equity_curve[0], line_dash="dash", line_color="gray", annotation_text="Initial Capital")

    fig.update_layout(
        title=f"{ticker} — {strategy_name} Equity Curve",
        template="plotly_dark",
        height=400,
        yaxis_title="Portfolio Value ($)",
        xaxis_title="Trading Day",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def opportunity_bar_chart(opportunities: List[Dict[str, Any]]) -> go.Figure:
    """
    Horizontal bar chart ranking opportunities by confidence score.

    Args:
        opportunities: List of opportunity dicts with ticker and confidence_score.

    Returns:
        Plotly Figure.
    """
    if not opportunities:
        return go.Figure()

    tickers = [o["ticker"] for o in opportunities]
    scores = [o["confidence_score"] for o in opportunities]
    recs = [o.get("recommendation", "HOLD") for o in opportunities]

    color_map = {
        "STRONG BUY": "#00C853",
        "BUY": "#69F0AE",
        "HOLD": "#FFD740",
        "SELL": "#FF6D00",
        "STRONG SELL": "#D50000",
    }
    colors = [color_map.get(r, "#FFFFFF") for r in recs]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=tickers,
            orientation="h",
            marker_color=colors,
            text=[f"{s:.0f}" for s in scores],
            textposition="outside",
        )
    )

    fig.add_vline(x=50, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="Market Opportunities — Confidence Score Ranking",
        template="plotly_dark",
        height=max(300, len(tickers) * 45),
        xaxis=dict(range=[0, 105], title="Confidence Score"),
        margin=dict(l=80, r=80, t=50, b=20),
    )
    return fig


def correlation_heatmap(corr_matrix: Dict[str, Dict[str, float]]) -> go.Figure:
    """
    Render a portfolio correlation heatmap.

    Args:
        corr_matrix: Nested dict {ticker: {ticker: correlation}}.

    Returns:
        Plotly Figure.
    """
    if not corr_matrix:
        return go.Figure()

    df_corr = pd.DataFrame(corr_matrix)
    fig = go.Figure(
        go.Heatmap(
            z=df_corr.values,
            x=df_corr.columns.tolist(),
            y=df_corr.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=df_corr.round(2).values,
            texttemplate="%{text}",
            colorbar=dict(title="Correlation"),
        )
    )
    fig.update_layout(
        title="Portfolio Correlation Matrix",
        template="plotly_dark",
        height=450,
        margin=dict(l=60, r=40, t=50, b=60),
    )
    return fig


def sector_pie_chart(sector_exposure: Dict[str, float]) -> go.Figure:
    """
    Pie chart of portfolio sector exposure.

    Args:
        sector_exposure: Dict {sector_name: weight}.

    Returns:
        Plotly Figure.
    """
    if not sector_exposure:
        return go.Figure()

    fig = go.Figure(
        go.Pie(
            labels=list(sector_exposure.keys()),
            values=list(sector_exposure.values()),
            hole=0.4,
            textinfo="label+percent",
        )
    )
    fig.update_layout(
        title="Sector Exposure",
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
