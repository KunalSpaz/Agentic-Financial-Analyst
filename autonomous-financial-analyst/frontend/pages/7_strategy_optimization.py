"""
7_strategy_optimization.py
---------------------------
Strategy Optimization page — automated grid-search parameter tuning.
"""
from __future__ import annotations

import httpx
import pandas as pd
import streamlit as st

from components.sidebar import render_sidebar

st.set_page_config(page_title="Strategy Optimization", page_icon="⚙️", layout="wide")
api_url = render_sidebar()

st.title("⚙️ Strategy Optimization")
st.markdown(
    "Automatically tune strategy parameters using grid search to maximize "
    "return, Sharpe ratio, or minimize drawdown."
)

col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Ticker", value="AAPL").upper()
with col2:
    objective = st.selectbox(
        "Optimization Objective",
        ["maximize_return", "maximize_sharpe", "minimize_drawdown"],
        format_func=lambda x: x.replace("_", " ").title(),
    )
with col3:
    period = st.selectbox("Historical Period", ["1y", "2y", "3y"], index=1)
    sentiment = st.selectbox("Sentiment Bias", ["POSITIVE", "NEUTRAL", "NEGATIVE"], index=1)

st.info(
    f"Grid search will test **{4 * 4 * 2 * 2 - 8} combinations** of RSI thresholds, "
    "MACD confirmation, and MA filter settings."
)

if st.button("🔍 Run Optimization", type="primary"):
    with st.spinner(f"Running grid-search optimization for {ticker}… (may take several minutes)"):
        try:
            resp = httpx.post(
                f"{api_url}/optimize-strategy",
                json={
                    "ticker": ticker,
                    "objective": objective,
                    "period": period,
                    "sentiment_label": sentiment,
                },
                timeout=600,
            )
            if resp.status_code == 200:
                st.session_state["opt_result"] = resp.json()
            else:
                st.error(f"API error: {resp.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

if "opt_result" in st.session_state:
    opt = st.session_state["opt_result"]

    st.divider()
    st.subheader("🏆 Best Configuration")

    best = opt.get("best_parameters", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Return", f"{opt.get('best_return', 0):.2%}")
    c2.metric("Best Sharpe", f"{opt.get('best_sharpe', 0):.2f}")
    c3.metric("Best Max Drawdown", f"{opt.get('best_drawdown', 0):.2%}")

    st.subheader("⚙️ Optimal Parameters")
    cols = st.columns(4)
    cols[0].metric("RSI Buy", best.get("rsi_buy_threshold", "N/A"))
    cols[1].metric("RSI Sell", best.get("rsi_sell_threshold", "N/A"))
    cols[2].metric("MACD Confirm", "Yes" if best.get("macd_confirmation") else "No")
    cols[3].metric("MA Filter", "Yes" if best.get("ma_filter") else "No")

    st.divider()
    st.subheader(f"📊 All Results ({opt.get('iterations', 0)} iterations)")

    all_results = opt.get("all_results", [])
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results["total_return"] = df_results["total_return"].map(lambda x: f"{x:.2%}")
        df_results["sharpe_ratio"] = df_results["sharpe_ratio"].map(lambda x: f"{x:.3f}")
        df_results["max_drawdown"] = df_results["max_drawdown"].map(lambda x: f"{x:.2%}")
        df_results["win_rate"] = df_results["win_rate"].map(lambda x: f"{x:.2%}")
        st.dataframe(df_results, use_container_width=True)
