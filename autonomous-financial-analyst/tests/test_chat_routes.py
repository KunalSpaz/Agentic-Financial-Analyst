"""
test_chat_routes.py
-------------------
Unit tests for the POST /chat endpoint.

All external dependencies (OpenAI, FAISS, NewsAPI, yfinance) are mocked
so tests run without network access or API keys.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import sys

sys.modules.setdefault("faiss",        MagicMock())
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("torch",        MagicMock())
sys.modules.setdefault("newsapi",      MagicMock())
sys.modules.setdefault("crewai",       MagicMock())

from fastapi.testclient import TestClient  # noqa: E402

from backend.api.main import create_app  # noqa: E402
from backend.database.connection import Base, engine  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with patch("backend.api.main.run_migrations"), \
         patch("backend.api.main.start_scheduler"), \
         patch("backend.api.main.shutdown_scheduler"), \
         patch("backend.api.main._prewarm_finbert"):
        app = create_app()
        Base.metadata.create_all(bind=engine)
        with TestClient(app) as c:
            yield c
        Base.metadata.drop_all(bind=engine)


def _openai_mock(content: str = "Looks bullish. Consider buying AAPL."):
    """Build a minimal mock that mimics openai.AsyncOpenAI().chat.completions.create()."""
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
    return mock_client


# ─────────────────────────────────────────────────────────────────────────────
# POST /chat — success cases
# ─────────────────────────────────────────────────────────────────────────────

class TestChatSuccess:
    def test_returns_200(self, client):
        with patch("backend.api.routes.chat_routes.openai.AsyncOpenAI", return_value=_openai_mock()), \
             patch("backend.api.routes.chat_routes._rag") as mock_rag, \
             patch("backend.api.routes.chat_routes._ns") as mock_ns:
            mock_rag._index = None
            mock_ns.get_top_financial_news.return_value = []
            resp = client.post("/chat", json={"message": "What is the market outlook?"})
        assert resp.status_code == 200

    def test_response_has_required_keys(self, client):
        with patch("backend.api.routes.chat_routes.openai.AsyncOpenAI", return_value=_openai_mock()), \
             patch("backend.api.routes.chat_routes._rag") as mock_rag, \
             patch("backend.api.routes.chat_routes._ns") as mock_ns:
            mock_rag._index = None
            mock_ns.get_top_financial_news.return_value = []
            resp = client.post("/chat", json={"message": "Is AAPL a buy?"})
        data = resp.json()
        assert "response" in data
        assert "sources_used" in data
        assert "context_available" in data
        assert "rag_vectors" in data

    def test_reply_text_is_returned(self, client):
        expected = "AAPL looks technically strong with RSI at 45."
        with patch("backend.api.routes.chat_routes.openai.AsyncOpenAI", return_value=_openai_mock(expected)), \
             patch("backend.api.routes.chat_routes._rag") as mock_rag, \
             patch("backend.api.routes.chat_routes._ns") as mock_ns:
            mock_rag._index = None
            mock_ns.get_top_financial_news.return_value = []
            resp = client.post("/chat", json={"message": "Analyse AAPL"})
        assert resp.json()["response"] == expected

    def test_ticker_added_to_context(self, client):
        """When ticker is provided, market data source should appear in sources_used."""
        mock_quote = {"ticker": "AAPL", "price": 175.0, "company_name": "Apple Inc.",
                      "change_pct": 1.2, "market_cap": 2_700_000_000_000,
                      "52w_high": 199.0, "52w_low": 124.0, "sector": "Technology"}
        with patch("backend.api.routes.chat_routes.openai.AsyncOpenAI", return_value=_openai_mock()), \
             patch("backend.api.routes.chat_routes._rag") as mock_rag, \
             patch("backend.api.routes.chat_routes._mds") as mock_mds, \
             patch("backend.api.routes.chat_routes._ns") as mock_ns:
            mock_rag._index = None
            mock_mds.get_quote.return_value = mock_quote
            mock_ns.get_stock_news.return_value = []
            resp = client.post("/chat", json={"message": "Analyse AAPL", "ticker": "AAPL"})
        assert "live_market_data" in resp.json()["sources_used"]

    def test_news_added_to_sources(self, client):
        """When news is available, 'news' should appear in sources_used."""
        articles = [{"title": "Apple hits record high", "source": "Reuters",
                     "description": "", "content": "", "url": ""}]
        with patch("backend.api.routes.chat_routes.openai.AsyncOpenAI", return_value=_openai_mock()), \
             patch("backend.api.routes.chat_routes._rag") as mock_rag, \
             patch("backend.api.routes.chat_routes._ns") as mock_ns:
            mock_rag._index = None
            mock_ns.get_top_financial_news.return_value = articles
            resp = client.post("/chat", json={"message": "Any market news?"})
        assert "news" in resp.json()["sources_used"]

    def test_rag_context_used_when_index_has_vectors(self, client):
        """RAG source included when FAISS index has content."""
        with patch("backend.api.routes.chat_routes.openai.AsyncOpenAI", return_value=_openai_mock()), \
             patch("backend.api.routes.chat_routes._rag") as mock_rag, \
             patch("backend.api.routes.chat_routes._ns") as mock_ns:
            mock_index = MagicMock()
            mock_index.ntotal = 10
            mock_rag._index = mock_index
            mock_rag.retrieve.return_value = "Earnings were exceptionally strong with record revenue of $120B beating all analyst expectations."
            mock_ns.get_top_financial_news.return_value = []
            resp = client.post("/chat", json={"message": "What were recent earnings?"})
        assert "knowledge_base" in resp.json()["sources_used"]

    def test_conversation_history_accepted(self, client):
        """Prior history turns are accepted without error."""
        history = [
            {"role": "user", "content": "Tell me about NVDA"},
            {"role": "assistant", "content": "NVDA is a chip maker."},
        ]
        with patch("backend.api.routes.chat_routes.openai.AsyncOpenAI", return_value=_openai_mock()), \
             patch("backend.api.routes.chat_routes._rag") as mock_rag, \
             patch("backend.api.routes.chat_routes._ns") as mock_ns:
            mock_rag._index = None
            mock_ns.get_top_financial_news.return_value = []
            resp = client.post("/chat", json={"message": "What about its margins?", "history": history})
        assert resp.status_code == 200

    def test_ticker_uppercased(self, client):
        """Lowercase ticker is normalised to uppercase."""
        with patch("backend.api.routes.chat_routes.openai.AsyncOpenAI", return_value=_openai_mock()), \
             patch("backend.api.routes.chat_routes._rag") as mock_rag, \
             patch("backend.api.routes.chat_routes._mds") as mock_mds, \
             patch("backend.api.routes.chat_routes._ns") as mock_ns:
            mock_rag._index = None
            mock_mds.get_quote.return_value = None
            mock_ns.get_stock_news.return_value = []
            resp = client.post("/chat", json={"message": "Should I buy?", "ticker": "aapl"})
        assert resp.json()["ticker"] == "AAPL"


# ─────────────────────────────────────────────────────────────────────────────
# POST /chat — error cases
# ─────────────────────────────────────────────────────────────────────────────

class TestChatErrors:
    def test_empty_message_returns_422(self, client):
        resp = client.post("/chat", json={"message": ""})
        assert resp.status_code == 422

    def test_missing_message_returns_422(self, client):
        resp = client.post("/chat", json={})
        assert resp.status_code == 422

    def test_openai_auth_error_returns_401(self, client):
        import openai as _openai
        with patch("backend.api.routes.chat_routes.openai.AsyncOpenAI") as mock_ctor, \
             patch("backend.api.routes.chat_routes._rag") as mock_rag, \
             patch("backend.api.routes.chat_routes._ns") as mock_ns:
            mock_rag._index = None
            mock_ns.get_top_financial_news.return_value = []
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=_openai.AuthenticationError("bad key", response=MagicMock(status_code=401), body={})
            )
            mock_ctor.return_value = mock_client
            resp = client.post("/chat", json={"message": "Hello"})
        assert resp.status_code == 401

    def test_openai_rate_limit_returns_429(self, client):
        import openai as _openai
        with patch("backend.api.routes.chat_routes.openai.AsyncOpenAI") as mock_ctor, \
             patch("backend.api.routes.chat_routes._rag") as mock_rag, \
             patch("backend.api.routes.chat_routes._ns") as mock_ns:
            mock_rag._index = None
            mock_ns.get_top_financial_news.return_value = []
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=_openai.RateLimitError("rate limit", response=MagicMock(status_code=429), body={})
            )
            mock_ctor.return_value = mock_client
            resp = client.post("/chat", json={"message": "Hello"})
        assert resp.status_code == 429

    def test_generic_openai_error_returns_500(self, client):
        with patch("backend.api.routes.chat_routes.openai.AsyncOpenAI") as mock_ctor, \
             patch("backend.api.routes.chat_routes._rag") as mock_rag, \
             patch("backend.api.routes.chat_routes._ns") as mock_ns:
            mock_rag._index = None
            mock_ns.get_top_financial_news.return_value = []
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=Exception("unexpected error"))
            mock_ctor.return_value = mock_client
            resp = client.post("/chat", json={"message": "Hello"})
        assert resp.status_code == 500

    def test_market_data_failure_does_not_crash(self, client):
        """Market data error is caught; chat still returns 200."""
        with patch("backend.api.routes.chat_routes.openai.AsyncOpenAI", return_value=_openai_mock()), \
             patch("backend.api.routes.chat_routes._rag") as mock_rag, \
             patch("backend.api.routes.chat_routes._mds") as mock_mds, \
             patch("backend.api.routes.chat_routes._ns") as mock_ns:
            mock_rag._index = None
            mock_mds.get_quote.side_effect = RuntimeError("yfinance down")
            mock_ns.get_stock_news.return_value = []
            resp = client.post("/chat", json={"message": "Analyse TSLA", "ticker": "TSLA"})
        assert resp.status_code == 200
        assert "live_market_data" not in resp.json()["sources_used"]
