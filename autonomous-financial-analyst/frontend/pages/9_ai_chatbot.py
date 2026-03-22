"""
9_ai_chatbot.py
---------------
AI Financial Analyst Chatbot — conversational interface powered by GPT-4o
with RAG document context, live market data, and news integration.
"""
from __future__ import annotations

import httpx
import streamlit as st

from components.sidebar import render_sidebar

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
api_url = render_sidebar()

st.title("🤖 AI Financial Analyst Chatbot")
st.markdown(
    "Ask anything about markets, stocks, or financial strategy. "
    "The AI automatically pulls in live quotes, recent news, and your uploaded documents."
)

# ── Session state init ────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.subheader("⚙️ Chat Settings")
    focus_ticker = st.text_input(
        "Focus Ticker (optional)",
        placeholder="e.g. AAPL",
        help="Attach live quote + news for this stock to every message.",
        key="chat_ticker",
    ).upper().strip()

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()

    st.divider()
    st.caption("💡 **Example questions:**")
    example_qs = [
        "What's the outlook for NVDA?",
        "Explain what RSI oversold means",
        "Compare AAPL and MSFT momentum",
        "What are the key risks in tech right now?",
        "Should I buy or wait on TSLA?",
        "What does a golden cross signal?",
    ]
    for q in example_qs:
        if st.button(q, key=f"eq_{q[:20]}", use_container_width=True):
            st.session_state["pending_message"] = q
            st.rerun()

# ── Render chat history ───────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state["chat_history"]:
        st.markdown(
            """
            <div style="text-align:center; padding: 60px 20px; color: #888;">
                <div style="font-size: 3rem;">🤖</div>
                <div style="font-size: 1.2rem; margin-top: 12px;">
                    Hi! I'm your AI Financial Analyst.<br>
                    Ask me about any stock, strategy, or market concept.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    source_labels = {
                        "knowledge_base": "📚 Documents",
                        "live_market_data": "📈 Live Quote",
                        "news": "📰 News",
                    }
                    badge_str = "  ".join(
                        f"`{source_labels.get(s, s)}`"
                        for s in msg["sources"]
                    )
                    if badge_str:
                        st.caption(f"Sources: {badge_str}")

# ── Chat input ────────────────────────────────────────────────────────────────
# Handle example question button clicks
prefill = st.session_state.pop("pending_message", "")

user_input = st.chat_input(
    placeholder="Ask about stocks, strategies, market conditions…",
)

# Use either the typed input or the clicked example
message = user_input or prefill

if message:
    # Add user message to history immediately
    st.session_state["chat_history"].append({"role": "user", "content": message})

    # Show user message right away
    with st.chat_message("user", avatar="🧑"):
        st.markdown(message)

    # Call backend
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            try:
                # Build history payload (exclude the message we just added)
                history_payload = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state["chat_history"][:-1]
                ]

                resp = httpx.post(
                    f"{api_url}/chat",
                    json={
                        "message": message,
                        "ticker": focus_ticker or None,
                        "history": history_payload,
                    },
                    timeout=60,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    reply = data["response"]
                    sources = data.get("sources_used", [])

                    st.markdown(reply)

                    # Show source badges
                    source_labels = {
                        "knowledge_base": "📚 Documents",
                        "live_market_data": "📈 Live Quote",
                        "news": "📰 News",
                    }
                    if sources:
                        badge_str = "  ".join(f"`{source_labels.get(s, s)}`" for s in sources)
                        st.caption(f"Sources: {badge_str}")

                    # RAG index size hint
                    rag_count = data.get("rag_vectors", 0)
                    if rag_count == 0:
                        st.caption("💡 No documents in knowledge base yet — upload some in **📚 Document Library** to improve answers.")

                    # Persist in history
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": reply,
                        "sources": sources,
                    })

                elif resp.status_code == 401:
                    err = "❌ Invalid OpenAI API key. Check your `.env` file — the key must start with `sk-`."
                    st.error(err)
                    st.session_state["chat_history"].append({"role": "assistant", "content": err, "sources": []})

                elif resp.status_code == 429:
                    err = "⏳ OpenAI rate limit hit. Please wait a few seconds and try again."
                    st.warning(err)
                    st.session_state["chat_history"].append({"role": "assistant", "content": err, "sources": []})

                else:
                    detail = resp.json().get("detail", resp.text)
                    err = f"❌ API error {resp.status_code}: {detail}"
                    st.error(err)
                    st.session_state["chat_history"].append({"role": "assistant", "content": err, "sources": []})

            except httpx.ConnectError:
                err = "❌ Cannot reach the backend. Make sure the API server is running on port 8000."
                st.error(err)
                st.session_state["chat_history"].append({"role": "assistant", "content": err, "sources": []})
            except Exception as e:
                err = f"❌ Unexpected error: {e}"
                st.error(err)
                st.session_state["chat_history"].append({"role": "assistant", "content": err, "sources": []})
