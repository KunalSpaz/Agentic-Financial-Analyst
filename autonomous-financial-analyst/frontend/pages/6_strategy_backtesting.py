"""
6_strategy_backtesting.py
--------------------------
Strategy Backtesting page — simulate trading strategies and view equity curves.
"""
from __future__ import annotations

import httpx
import streamlit as st

from components.charts import equity_curve_chart
from components.sidebar import render_sidebar

st.set_page_config(page_title="Strategy Backtesting", page_icon="📊", layout="wide")
api_url = render_sidebar()

st.title("📊 Strategy Backtesting")
st.markdown(
    "Simulate the RSI + MACD + Sentiment trading strategy on historical data and evaluate performance."
)

st.subheader("⚙️ Strategy Parameters")

col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Ticker", value="AAPL").upper()
    period = st.selectbox("Historical Period", ["1y", "2y", "3y", "5y"], index=1)
    initial_capital = st.number_input("Initial Capital ($)", value=10_000, min_value=1_000, step=1_000)
with col2:
    rsi_buy = st.slider("RSI Buy Threshold", min_value=10, max_value=50, value=35)
    rsi_sell = st.slider("RSI Sell Threshold", min_value=50, max_value=90, value=70)
    sentiment = st.selectbox("Sentiment Bias", ["POSITIVE", "NEUTRAL", "NEGATIVE"], index=1)
with col3:
    macd_confirm = st.checkbox("Require MACD Confirmation", value=True)
    ma_filter = st.checkbox("MA200 Filter (only buy above SMA200)", value=True)

if st.button("▶ Run Backtest", type="primary"):
    with st.spinner(f"Running backtest for {ticker}…"):
        try:
            resp = httpx.post(
                f"{api_url}/backtest",
                json={
                    "ticker": ticker,
                    "period": period,
                    "rsi_buy_threshold": rsi_buy,
                    "rsi_sell_threshold": rsi_sell,
                    "macd_confirmation": macd_confirm,
                    "ma_filter": ma_filter,
                    "initial_capital": float(initial_capital),
                    "sentiment_label": sentiment,
                },
                timeout=120,
            )
            if resp.status_code == 200:
                st.session_state["backtest_result"] = resp.json()
            else:
                st.error(f"API error: {resp.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

if "backtest_result" in st.session_state:
    bt = st.session_state["backtest_result"]
    metrics = bt.get("metrics", {})

    st.divider()
    st.subheader("📊 Performance Metrics")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Return", metrics.get("total_return_pct", "N/A"),
              delta=metrics.get("total_return"))
    c2.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
    c3.metric("Max Drawdown", metrics.get("max_drawdown_pct", "N/A"))
    c4.metric("Win Rate", metrics.get("win_rate_pct", "N/A"))
    c5.metric("Total Trades", metrics.get("num_trades", 0))

    st.divider()
    st.subheader("📈 Equity Curve")
    equity = bt.get("equity_curve", [])
    if equity:
        st.plotly_chart(
            equity_curve_chart(equity, bt.get("ticker", ticker), "RSI+MACD+Sentiment"),
            use_container_width=True,
        )

    st.divider()
    col_params, col_trades = st.columns([1, 2])
    with col_params:
        st.subheader("⚙️ Parameters Used")
        st.json(bt.get("parameters", {}))
    with col_trades:
        st.subheader("📋 Trade Log (last 20)")
        trade_log = bt.get("trade_log", [])[-20:]
        if trade_log:
            import pandas as pd
            st.dataframe(pd.DataFrame(trade_log), use_container_width=True)
