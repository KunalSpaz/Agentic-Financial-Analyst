"""
test_news_service.py
---------------------
Unit tests for NewsService.

Covers:
  - get_stock_news returns structured list of article dicts
  - get_stock_news handles NewsAPI errors gracefully
  - get_stock_news respects max_articles limit
  - get_top_financial_news returns articles
  - Both methods return [] when api key is absent
  - Cache prevents duplicate API calls within TTL
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import backend.services.news_service as _ns_module
from backend.services.news_service import NewsService


EXPECTED_ARTICLE_KEYS = {"ticker", "title", "description", "content", "url", "source", "author", "published_at"}


def _make_newsapi_article(title: str = "Test headline") -> dict:
    return {
        "title": title,
        "description": "A description.",
        "content": "Article content.",
        "url": "https://example.com/article",
        "source": {"name": "Reuters"},
        "author": "Jane Doe",
        "publishedAt": "2024-06-01T09:00:00Z",
    }


@pytest.fixture(autouse=True)
def clear_news_cache():
    _ns_module._cache.clear()
    yield
    _ns_module._cache.clear()


@pytest.fixture
def service_with_key(monkeypatch) -> NewsService:
    monkeypatch.setattr("backend.services.news_service.settings", MagicMock(news_api_key="fake_key"))
    return NewsService()


@pytest.fixture
def service_no_key(monkeypatch) -> NewsService:
    monkeypatch.setattr("backend.services.news_service.settings", MagicMock(news_api_key=""))
    return NewsService()


# ─────────────────────────────────────────────────────────────────────────────
# get_stock_news
# ─────────────────────────────────────────────────────────────────────────────

class TestGetStockNews:
    def test_returns_list_of_dicts(self, service_with_key):
        mock_response = {"articles": [_make_newsapi_article("AAPL hits record"), _make_newsapi_article("Apple beats estimates")]}
        service_with_key._client = MagicMock()
        service_with_key._client.get_everything.return_value = mock_response

        articles = service_with_key.get_stock_news("AAPL")
        assert isinstance(articles, list)
        assert len(articles) == 2

    def test_article_has_required_keys(self, service_with_key):
        mock_response = {"articles": [_make_newsapi_article()]}
        service_with_key._client = MagicMock()
        service_with_key._client.get_everything.return_value = mock_response

        articles = service_with_key.get_stock_news("MSFT")
        assert EXPECTED_ARTICLE_KEYS.issubset(set(articles[0].keys()))

    def test_ticker_is_set_on_each_article(self, service_with_key):
        mock_response = {"articles": [_make_newsapi_article(), _make_newsapi_article()]}
        service_with_key._client = MagicMock()
        service_with_key._client.get_everything.return_value = mock_response

        articles = service_with_key.get_stock_news("NVDA")
        assert all(a["ticker"] == "NVDA" for a in articles)

    def test_returns_empty_list_on_api_error(self, service_with_key):
        service_with_key._client = MagicMock()
        service_with_key._client.get_everything.side_effect = Exception("API down")

        articles = service_with_key.get_stock_news("TSLA")
        assert articles == []

    def test_returns_empty_list_without_api_key(self, service_no_key):
        articles = service_no_key.get_stock_news("AAPL")
        assert articles == []

    def test_cache_prevents_second_api_call(self, service_with_key):
        mock_response = {"articles": [_make_newsapi_article()]}
        service_with_key._client = MagicMock()
        service_with_key._client.get_everything.return_value = mock_response

        service_with_key.get_stock_news("AAPL", max_articles=5)
        service_with_key.get_stock_news("AAPL", max_articles=5)
        assert service_with_key._client.get_everything.call_count == 1

    def test_max_articles_respected(self, service_with_key):
        # API returns 5 articles but we request max 2
        mock_response = {"articles": [_make_newsapi_article(f"Article {i}") for i in range(5)]}
        service_with_key._client = MagicMock()
        service_with_key._client.get_everything.return_value = mock_response

        articles = service_with_key.get_stock_news("AMD", max_articles=2)
        # The service passes page_size to the API; the returned slice depends on API
        # but we verify the call was made with the correct page_size
        call_kwargs = service_with_key._client.get_everything.call_args.kwargs
        assert call_kwargs["page_size"] == 2

    def test_uses_company_name_as_query_when_provided(self, service_with_key):
        service_with_key._client = MagicMock()
        service_with_key._client.get_everything.return_value = {"articles": []}

        service_with_key.get_stock_news("AAPL", company_name="Apple Inc.")
        call_kwargs = service_with_key._client.get_everything.call_args.kwargs
        assert call_kwargs["q"] == "Apple Inc."


# ─────────────────────────────────────────────────────────────────────────────
# get_top_financial_news
# ─────────────────────────────────────────────────────────────────────────────

class TestGetTopFinancialNews:
    def test_returns_list(self, service_with_key):
        mock_response = {"articles": [_make_newsapi_article("Top headline")]}
        service_with_key._client = MagicMock()
        service_with_key._client.get_top_headlines.return_value = mock_response

        articles = service_with_key.get_top_financial_news()
        assert isinstance(articles, list)
        assert len(articles) == 1

    def test_ticker_is_none_for_top_news(self, service_with_key):
        mock_response = {"articles": [_make_newsapi_article()]}
        service_with_key._client = MagicMock()
        service_with_key._client.get_top_headlines.return_value = mock_response

        articles = service_with_key.get_top_financial_news()
        assert articles[0]["ticker"] is None

    def test_returns_empty_on_error(self, service_with_key):
        service_with_key._client = MagicMock()
        service_with_key._client.get_top_headlines.side_effect = Exception("timeout")

        articles = service_with_key.get_top_financial_news()
        assert articles == []

    def test_returns_empty_without_key(self, service_no_key):
        articles = service_no_key.get_top_financial_news()
        assert articles == []
