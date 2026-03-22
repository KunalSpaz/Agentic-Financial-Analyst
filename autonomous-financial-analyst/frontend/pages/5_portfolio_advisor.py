"""
5_portfolio_advisor.py
-----------------------
Portfolio Advisor page — risk metrics, correlation heatmap, and sector exposure.
"""
from __future__ import annotations

import httpx
import streamlit as st

from components.charts import correlation_heatmap, sector_pie_chart
from components.sidebar import render_sidebar

st.set_page_config(page_title="Portfolio Advisor", page_icon="💼", layout="wide")
api_url = render_sidebar()

st.title("💼 Portfolio Advisor")
st.markdown("Enter your portfolio holdings to receive AI-powered risk analysis.")

st.subheader("📋 Portfolio Builder")

default_portfolio = {"AAPL": 0.25, "MSFT": 0.25, "NVDA": 0.20, "AMZN": 0.15, "TSLA": 0.15}
holdings: dict = {}

for i, col in enumerate(st.columns(5)):
    with col:
        ticker = st.text_input(f"Ticker {i+1}", value=list(default_portfolio.keys())[i] if i < len(default_portfolio) else "")
        weight = st.number_input(f"Weight {i+1}", min_value=0.0, max_value=1.0, value=list(default_portfolio.values())[i] if i < len(default_portfolio) else 0.0, step=0.05)
        if ticker and weight > 0:
            holdings[ticker.upper()] = weight

period = st.selectbox("Historical Period", ["6mo", "1y", "2y"], index=1)

if st.button("📊 Analyse Portfolio Risk", type="primary"):
    if not holdings:
        st.warning("Please enter at least one holding.")
    else:
        with st.spinner("Computing portfolio risk metrics…"):
            try:
                resp = httpx.post(
                    f"{api_url}/portfolio-analysis",
                    json={"holdings": holdings, "period": period},
                    timeout=120,
                )
                if resp.status_code == 200:
                    risk = resp.json()
                    st.session_state["portfolio_risk"] = risk
                else:
                    st.error(f"API error: {resp.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

if "portfolio_risk" in st.session_state:
    risk = st.session_state["portfolio_risk"]

    st.divider()
    st.subheader("📊 Risk Metrics")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Annualised Volatility", f"{risk.get('portfolio_volatility', 0):.2%}")
    c2.metric("Beta (vs SPY)", f"{risk.get('portfolio_beta', 1):.2f}")
    c3.metric("Sharpe Ratio", f"{risk.get('sharpe_ratio', 0):.2f}")
    c4.metric("Max Drawdown", f"{risk.get('max_drawdown', 0):.2%}")
    c5.metric("Daily VaR (95%)", f"{risk.get('var_95_daily', 0):.2%}")

    st.divider()

    col_corr, col_sector = st.columns(2)
    with col_corr:
        corr = risk.get("correlation_matrix", {})
        if corr:
            st.plotly_chart(correlation_heatmap(corr), use_container_width=True)

    with col_sector:
        sector = risk.get("sector_exposure", {})
        if sector:
            st.plotly_chart(sector_pie_chart(sector), use_container_width=True)
