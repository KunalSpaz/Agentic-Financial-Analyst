"""
metrics.py
----------
Streamlit metric and gauge helper functions for the dashboard.
"""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go


def confidence_gauge(score: float, recommendation: str, color: str = "#2196F3") -> go.Figure:
    """
    Render a Plotly gauge chart for a confidence score.

    Args:
        score:          Score between 0 and 100.
        recommendation: Label text shown in the centre.
        color:          Hex color for the gauge bar.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": recommendation, "font": {"size": 20, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 34],  "color": "#D50000"},
                    {"range": [34, 49], "color": "#FF6D00"},
                    {"range": [49, 64], "color": "#FFD740"},
                    {"range": [64, 79], "color": "#69F0AE"},
                    {"range": [79, 100], "color": "#00C853"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": score,
                },
            },
            number={"font": {"size": 40}},
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


def metric_card(label: str, value: str, delta: str | None = None) -> None:
    """
    Render a styled metric card using Streamlit's native metric widget.

    Args:
        label: Metric label.
        value: Current value string.
        delta: Optional delta string (positive = green, negative = red).
    """
    st.metric(label=label, value=value, delta=delta)


def recommendation_badge(recommendation: str, color: str) -> None:
    """
    Display a large, coloured recommendation badge in the UI.

    Args:
        recommendation: Text label.
        color:          Hex color string.
    """
    st.markdown(
        f"""
        <div style="
            background-color: {color};
            padding: 12px 24px;
            border-radius: 8px;
            text-align: center;
            font-size: 1.4rem;
            font-weight: bold;
            color: white;
            margin: 8px 0;
        ">
            {recommendation}
        </div>
        """,
        unsafe_allow_html=True,
    )
