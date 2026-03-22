"""
1_market_overview.py
--------------------
Market Overview page — live market indices, top news, and sentiment summary.
"""
from __future__ import annotations

import httpx
import streamlit as st

from components.sidebar import render_sidebar

st.set_page_config(page_title="Market Overview", page_icon="🌍", layout="wide")
api_url = render_sidebar()

st.title("🌍 Market Overview")
st.markdown("Live market snapshot — indices, top financial headlines, and overall sentiment.")

col1, col2, col3, col4 = st.columns(4)

# ── Market Report ─────────────────────────────────────────────────────────────
with st.spinner("Loading market data…"):
    try:
        resp = httpx.get(f"{api_url}/market-report", timeout=30)
        report = resp.json() if resp.status_code == 200 else {}
    except Exception as e:
        report = {}
        st.error(f"Cannot connect to backend: {e}")

indices = report.get("market_indices", {})

with col1:
    spx = indices.get("SPY", {})
    st.metric("S&P 500 (SPY)", f"${spx.get('price', 'N/A')}", f"{(spx.get('change_pct') or 0):.2f}%")

with col2:
    qqq = indices.get("QQQ", {})
    st.metric("NASDAQ 100 (QQQ)", f"${qqq.get('price', 'N/A')}", f"{(qqq.get('change_pct') or 0):.2f}%")

with col3:
    dia = indices.get("DIA", {})
    st.metric("Dow Jones (DIA)", f"${dia.get('price', 'N/A')}", f"{(dia.get('change_pct') or 0):.2f}%")

with col4:
    vix = indices.get("VIX", {})
    st.metric("Volatility Index (VIX)", f"${vix.get('price', 'N/A')}", f"{(vix.get('change_pct') or 0):.2f}%")

st.divider()

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📰 Daily AI Market Briefing")
    narrative = report.get("narrative", "Run a market scan to generate the daily briefing.")
    st.markdown(narrative)

with col_right:
    st.subheader("📊 Overall Sentiment")
    sentiment = report.get("overall_sentiment", "NEUTRAL")
    color_map = {"POSITIVE": "#00C853", "NEUTRAL": "#FFD740", "NEGATIVE": "#D50000"}
    color = color_map.get(sentiment, "#FFFFFF")
    st.markdown(
        f'<div style="background:{color};padding:16px;border-radius:8px;text-align:center;font-size:1.5rem;font-weight:bold;color:white">{sentiment}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("🎯 Top Picks")
    for pick in report.get("top_picks", []):
        st.badge(pick)

st.divider()
st.subheader("📰 Top Financial News")

with st.spinner("Loading headlines…"):
    try:
        news_resp = httpx.get(f"{api_url}/top-news?limit=10", timeout=20)
        news = news_resp.json() if news_resp.status_code == 200 else []
    except Exception:
        news = []

for article in news[:8]:
    with st.expander(article.get("title", "No title")[:100]):
        st.caption(f"📅 {article.get('published_at', '')} | 📰 {article.get('source', '')}")
        st.write(article.get("description", "No description available."))
        if article.get("url"):
            st.link_button("Read full article →", article["url"])
