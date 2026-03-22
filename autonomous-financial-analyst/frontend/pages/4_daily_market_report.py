"""
4_daily_market_report.py
------------------------
Daily AI Market Report page — full narrative briefing with top picks and indices.
"""
from __future__ import annotations

import httpx
import streamlit as st

from components.sidebar import render_sidebar

st.set_page_config(page_title="Daily AI Market Report", page_icon="📋", layout="wide")
api_url = render_sidebar()

st.title("📋 Daily AI Market Report")
st.markdown("AI-generated daily market briefing covering opportunities, sentiment, and recommended actions.")

if st.button("🔄 Refresh Report", type="primary"):
    st.cache_data.clear()

with st.spinner("Generating AI market briefing…"):
    try:
        resp = httpx.get(f"{api_url}/market-report", timeout=60)
        report = resp.json() if resp.status_code == 200 else {}
    except Exception as e:
        st.error(f"Connection error: {e}")
        report = {}

if report:
    col_date, col_sentiment = st.columns([2, 1])
    with col_date:
        st.markdown(f"**📅 Report Date:** {report.get('date', 'N/A')}")
    with col_sentiment:
        sentiment = report.get("overall_sentiment", "NEUTRAL")
        color = {"POSITIVE": "#00C853", "NEUTRAL": "#FFD740", "NEGATIVE": "#D50000"}.get(sentiment, "#FFFFFF")
        st.markdown(
            f'<div style="background:{color};padding:8px;border-radius:6px;text-align:center;color:white;font-weight:bold">'
            f'Market Sentiment: {sentiment}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    col_report, col_picks = st.columns([2, 1])

    with col_report:
        st.subheader("📊 Market Briefing")
        st.markdown(report.get("narrative", "No briefing available."))

    with col_picks:
        st.subheader("🎯 Top AI Picks")
        for i, pick in enumerate(report.get("top_picks", []), 1):
            st.markdown(f"**{i}.** `{pick}`")

        st.divider()
        st.subheader("📈 Market Indices")
        for sym, data in report.get("market_indices", {}).items():
            price = data.get("price", "N/A")
            chg = (data.get("change_pct") or 0) * 100
            sign = "▲" if chg >= 0 else "▼"
            color = "#00C853" if chg >= 0 else "#D50000"
            st.markdown(
                f'**{sym}** — ${price} '
                f'<span style="color:{color}">{sign} {abs(chg):.2f}%</span>',
                unsafe_allow_html=True,
            )
else:
    st.warning("No market report available. Make sure the backend is running.")
