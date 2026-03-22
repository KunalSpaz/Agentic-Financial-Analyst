"""
2_stock_analysis.py
-------------------
Stock Analysis page — deep AI analysis with technical indicators,
sentiment, confidence score gauge, and narrative report.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path so backend packages are importable from the frontend
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
import streamlit as st

from components.charts import candlestick_chart, rsi_chart, macd_chart
from components.metrics import confidence_gauge, recommendation_badge
from components.sidebar import render_sidebar
from backend.services.market_data_service import MarketDataService
from backend.services.technical_analysis_service import TechnicalAnalysisService

st.set_page_config(page_title="Stock Analysis", page_icon="🔍", layout="wide")
api_url = render_sidebar()

st.title("🔍 Stock Analysis")
st.markdown("AI-powered deep analysis combining technical indicators, news sentiment, and RAG document intelligence.")

ticker = st.text_input("Enter Ticker Symbol", value="AAPL", placeholder="e.g. AAPL, MSFT, NVDA").upper()
period = st.selectbox("Historical Period", ["6mo", "1y", "2y", "5y"], index=1)

col_analyze, _ = st.columns([1, 3])

with col_analyze:
    run_analysis = st.button("🤖 Run AI Analysis", type="primary", use_container_width=True)

if run_analysis and ticker:
    with st.spinner(f"Running 8-agent CrewAI analysis for {ticker}… (this may take 1–2 minutes)"):
        try:
            resp = httpx.post(
                f"{api_url}/analyze-stock",
                json={"ticker": ticker, "period": period},
                timeout=180,
            )
            if resp.status_code == 200:
                result = resp.json()
                st.session_state["analysis_result"] = result
                st.session_state["analysis_ticker"] = ticker
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

if "analysis_result" in st.session_state and st.session_state.get("analysis_ticker") == ticker:
    result = st.session_state["analysis_result"]

    # ── Metrics row ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Ticker", result.get("ticker", ticker))
    with c2:
        price = result.get("current_price")
        st.metric("Price", f"${price:.2f}" if price else "N/A")
    with c3:
        st.metric("RSI", f"{result.get('technical_signals', {}).get('rsi', 'N/A'):.1f}" if result.get('technical_signals', {}).get('rsi') else "N/A")
    with c4:
        st.metric("Sentiment", result.get("sentiment", {}).get("label", "N/A"))
    with c5:
        st.metric("Articles Analysed", result.get("sentiment", {}).get("article_count", 0))

    st.divider()

    col_gauge, col_signals = st.columns([1, 2])

    with col_gauge:
        st.subheader("Confidence Score")
        fig_gauge = confidence_gauge(
            score=result["confidence_score"],
            recommendation=result["recommendation"],
            color=result.get("confidence_color", "#2196F3"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        recommendation_badge(result["recommendation"], result.get("confidence_color", "#2196F3"))

    with col_signals:
        st.subheader("Technical Signals")
        signals = result.get("technical_signals", {})
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.write("✅" if signals.get("golden_cross") else "❌", " **Golden Cross** (SMA50 > SMA200)")
            st.write("✅" if signals.get("macd_bullish_crossover") else "❌", " **MACD Bullish Crossover**")
            st.write("✅" if signals.get("rsi_oversold") else "❌", " **RSI Oversold** (< 35)")
            st.write("✅" if signals.get("high_volume") else "❌", " **High Volume** (> 1.5× avg)")
        with s_col2:
            st.write("⚠️" if signals.get("rsi_overbought") else "✅", " **RSI Overbought** (> 70)")
            st.write("⚠️" if signals.get("death_cross") else "✅", " **Death Cross** (SMA50 < SMA200)")
            st.write("⚠️" if signals.get("above_bb_upper") else "✅", " **Above BB Upper**")
            st.write("✅" if signals.get("below_bb_lower") else "❌", " **Below BB Lower**")

    st.divider()
    st.subheader("📈 Price Chart")

    # Load live chart data directly
    try:
        mds = MarketDataService()
        tas = TechnicalAnalysisService()
        df = mds.get_historical_data(ticker, period=period)
        if not df.empty:
            df = tas.compute_indicators(df)
            show_bb = st.checkbox("Show Bollinger Bands", value=False)
            st.plotly_chart(
                candlestick_chart(df, ticker, show_volume=True, show_ma=True, show_bb=show_bb),
                use_container_width=True,
            )
            c_rsi, c_macd = st.columns(2)
            with c_rsi:
                st.plotly_chart(rsi_chart(df), use_container_width=True)
            with c_macd:
                st.plotly_chart(macd_chart(df), use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load chart data: {e}")

    st.divider()
    st.subheader("📄 AI Analysis Report")
    narrative = result.get("narrative", "No report generated.")
    st.markdown(narrative)
