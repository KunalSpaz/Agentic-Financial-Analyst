"""
app.py
------
Streamlit application entrypoint.

Renders the main landing page with a dashboard overview and navigation
instructions. Multi-page navigation is handled by the /pages directory.

Run with:
    streamlit run frontend/app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Autonomous Financial Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from components.sidebar import render_sidebar

render_sidebar()

st.title("📈 Autonomous Financial Analyst")
st.markdown(
    """
    ### AI-Powered Financial Intelligence Platform

    This platform uses a **12-agent CrewAI system** to analyse financial markets,
    generate investment recommendations, and provide deep portfolio insights.

    ---

    #### Available Modules

    | Page | Description |
    |---|---|
    | 🌍 **Market Overview** | Live market indices, top news, and overall sentiment |
    | 🔍 **Stock Analysis** | Deep AI analysis with technical indicators + confidence score |
    | 🎯 **Opportunity Scanner** | Ranked investment opportunities across the stock universe |
    | 📋 **Daily AI Report** | AI-generated daily market briefing |
    | 💼 **Portfolio Advisor** | Risk metrics, correlation matrix, and sector exposure |
    | 📊 **Strategy Backtesting** | Historical strategy simulation with equity curve |
    | ⚙️ **Strategy Optimization** | Automatic parameter tuning via grid search |

    ---

    > **Getting started:** Select a page from the left sidebar to begin your analysis.
    """
)

st.info(
    "💡 Make sure the FastAPI backend is running: `uvicorn backend.api.main:app --reload`",
    icon="ℹ️",
)
