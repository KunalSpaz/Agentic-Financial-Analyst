"""
3_opportunity_scanner.py
------------------------
Opportunity Scanner page — ranked investment opportunities across the stock universe.
"""
from __future__ import annotations

import httpx
import streamlit as st

from components.charts import opportunity_bar_chart
from components.sidebar import render_sidebar

st.set_page_config(page_title="Opportunity Scanner", page_icon="🎯", layout="wide")
api_url = render_sidebar()

st.title("🎯 Market Opportunity Scanner")
st.markdown(
    "Scans AAPL, MSFT, NVDA, TSLA, AMZN, META, GOOGL, AMD, NFLX, INTC "
    "and ranks them by AI confidence score."
)

col_refresh, col_info = st.columns([1, 3])
with col_refresh:
    refresh = st.button("🔄 Run Full Scan (takes ~5 min)", type="primary")
with col_info:
    st.info("Click 'Run Full Scan' to re-score all tickers, or view the latest cached results below.")

with st.spinner("Loading opportunities…"):
    url = f"{api_url}/market-opportunities{'?refresh=true' if refresh else ''}"
    try:
        resp = httpx.get(url, timeout=600)
        opportunities = resp.json() if resp.status_code == 200 else []
    except Exception as e:
        st.error(f"Connection error: {e}")
        opportunities = []

if opportunities:
    st.plotly_chart(opportunity_bar_chart(opportunities), use_container_width=True)

    st.divider()
    st.subheader("📊 Opportunity Rankings Table")

    for opp in opportunities:
        rank = opp.get("rank", "?")
        ticker = opp.get("ticker", "")
        rec = opp.get("recommendation", "HOLD")
        score = opp.get("confidence_score", 0)
        price = opp.get("current_price")
        sector = opp.get("sector", "N/A")

        color_map = {
            "STRONG BUY": "#00C853", "BUY": "#69F0AE",
            "HOLD": "#FFD740", "SELL": "#FF6D00", "STRONG SELL": "#D50000",
        }
        color = color_map.get(rec, "#FFFFFF")

        with st.expander(f"#{rank} {ticker} — {rec} ({score:.0f}/100)"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ticker", ticker)
            c2.metric("Confidence", f"{score:.0f}/100")
            c3.metric("Price", f"${price:.2f}" if price else "N/A")
            c4.metric("Sector", sector or "N/A")
            st.markdown(
                f'<span style="background:{color};padding:4px 12px;border-radius:4px;color:white;font-weight:bold">{rec}</span>',
                unsafe_allow_html=True,
            )
            if opp.get("rationale"):
                st.markdown(f"**Rationale:** {opp['rationale']}")
else:
    st.warning("No opportunities found. Run a scan or check that the backend is running.")
