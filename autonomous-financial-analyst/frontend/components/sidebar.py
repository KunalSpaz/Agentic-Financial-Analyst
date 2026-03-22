"""
sidebar.py
----------
Global Streamlit sidebar configuration and controls.
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def _check_backend(api_url: str) -> bool:
    try:
        resp = httpx.get(f"{api_url}/", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def render_sidebar() -> str:
    """
    Render the global sidebar and return the API base URL.

    Returns:
        API base URL string.
    """
    with st.sidebar:
        # ── Branding ──────────────────────────────────────────────────
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                border-radius: 12px;
                padding: 20px 16px;
                margin-bottom: 8px;
                text-align: center;
                border: 1px solid rgba(99,179,237,0.2);
            ">
                <div style="font-size: 2.2rem; margin-bottom: 6px;">📈</div>
                <div style="
                    font-size: 1.1rem;
                    font-weight: 700;
                    color: #e2e8f0;
                    letter-spacing: 0.5px;
                ">Financial Analyst AI</div>
                <div style="
                    font-size: 0.72rem;
                    color: #63b3ed;
                    margin-top: 4px;
                    letter-spacing: 1px;
                    text-transform: uppercase;
                ">Powered by CrewAI · GPT-4o</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── API Config ────────────────────────────────────────────────
        st.markdown("#### ⚙️ Configuration")
        api_url = st.text_input(
            "Backend URL",
            value=API_BASE_URL,
            label_visibility="collapsed",
            placeholder="http://localhost:8000",
        )

        # Backend status pill
        online = _check_backend(api_url)
        if online:
            st.markdown(
                '<div style="background:#1a4731;border:1px solid #276749;border-radius:20px;'
                'padding:4px 12px;display:inline-block;font-size:0.75rem;color:#68d391;">'
                '● Backend Online</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#3b1111;border:1px solid #742a2a;border-radius:20px;'
                'padding:4px 12px;display:inline-block;font-size:0.75rem;color:#fc8181;">'
                '● Backend Offline</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Stock Universe ────────────────────────────────────────────
        st.markdown("#### 🎯 Stock Universe")
        tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD", "NFLX", "INTC"]
        cols = st.columns(2)
        for i, ticker in enumerate(tickers):
            with cols[i % 2]:
                st.markdown(
                    f'<div style="background:#1e2a3a;border:1px solid #2d3748;border-radius:6px;'
                    f'padding:4px 8px;margin:2px 0;font-size:0.75rem;font-weight:600;'
                    f'color:#90cdf4;text-align:center;">{ticker}</div>',
                    unsafe_allow_html=True,
                )

        # ── Hide default Streamlit nav, replace with styled page links ──
        st.markdown(
            """
            <style>
            [data-testid="stSidebarNav"] { display: none; }
            div[data-testid="stPageLink"] > a {
                background: #1e2a3a;
                border: 1px solid #2d3748;
                border-radius: 8px;
                padding: 8px 12px;
                margin: 2px 0;
                color: #e2e8f0 !important;
                font-size: 0.85rem;
                font-weight: 500;
                text-decoration: none;
                display: flex;
                align-items: center;
                gap: 8px;
                transition: background 0.15s;
            }
            div[data-testid="stPageLink"] > a:hover {
                background: #2d3748;
                border-color: #4a5568;
                color: #90cdf4 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("#### 🧭 Navigation")

        pages = [
            ("app.py",                              "🏠  Home"),
            ("pages/1_market_overview.py",         "🌍  Market Overview"),
            ("pages/2_stock_analysis.py",          "🔍  Stock Analysis"),
            ("pages/3_opportunity_scanner.py",     "🎯  Opportunity Scanner"),
            ("pages/4_daily_market_report.py",     "📋  Daily AI Report"),
            ("pages/5_portfolio_advisor.py",       "💼  Portfolio Advisor"),
            ("pages/6_strategy_backtesting.py",    "📊  Strategy Backtesting"),
            ("pages/7_strategy_optimization.py",   "⚙️  Strategy Optimization"),
            ("pages/8_document_library.py",        "📚  Document Library"),
            ("pages/9_ai_chatbot.py",              "🤖  AI Chatbot"),
        ]
        for path, label in pages:
            st.page_link(path, label=label)

        st.divider()
        st.markdown(
            '<div style="text-align:center;font-size:0.7rem;color:#4a5568;">© 2025 Autonomous Financial Analyst</div>',
            unsafe_allow_html=True,
        )

    return api_url
