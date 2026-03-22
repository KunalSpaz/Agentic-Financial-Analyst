"""
news_service.py
---------------
Retrieves financial news articles from NewsAPI.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from newsapi import NewsApiClient

from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_cache: Dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 600  # 10 minutes


def _cache_get(key: str) -> Optional[Any]:
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < _CACHE_TTL:
        return entry[1]
    return None


def _cache_set(key: str, value: Any) -> None:
    _cache[key] = (time.time(), value)


class NewsService:
    """
    Wrapper around NewsAPI for fetching financial headlines and articles.
    """

    def __init__(self) -> None:
        if not settings.news_api_key:
            logger.warning("NEWS_API_KEY not set — news features will be limited.")
            self._client: Optional[NewsApiClient] = None
        else:
            self._client = NewsApiClient(api_key=settings.news_api_key)

    def get_stock_news(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        max_articles: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent news articles related to *ticker*.

        Args:
            ticker:       Stock symbol used as a query term.
            company_name: Optional full company name for a richer query.
            max_articles: Maximum number of articles to return.

        Returns:
            List of article dicts with keys: title, description, content,
            url, source, author, published_at.
        """
        cache_key = f"news:{ticker}:{max_articles}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        query = company_name if company_name else ticker

        if self._client is None:
            logger.warning("NewsAPI client not initialised; returning empty list.")
            return []

        try:
            response = self._client.get_everything(
                q=query,
                language="en",
                sort_by="publishedAt",
                page_size=min(max_articles, 100),
            )
            articles = []
            for art in response.get("articles", []):
                articles.append(
                    {
                        "ticker": ticker,
                        "title": art.get("title", ""),
                        "description": art.get("description", ""),
                        "content": art.get("content", ""),
                        "url": art.get("url", ""),
                        "source": art.get("source", {}).get("name", ""),
                        "author": art.get("author", ""),
                        "published_at": art.get("publishedAt", ""),
                    }
                )
            _cache_set(cache_key, articles)
            logger.info("Fetched %d articles for %s", len(articles), ticker)
            return articles
        except Exception as exc:
            logger.error("NewsAPI error for %s: %s", ticker, exc)
            return []

    def get_top_financial_news(self, max_articles: int = 30) -> List[Dict[str, Any]]:
        """
        Fetch top financial / business headlines (not ticker-specific).

        Args:
            max_articles: Maximum number of articles to return.

        Returns:
            List of article dicts.
        """
        cache_key = f"top_news:{max_articles}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        if self._client is None:
            return []

        try:
            response = self._client.get_top_headlines(
                category="business",
                language="en",
                page_size=min(max_articles, 100),
            )
            articles = [
                {
                    "ticker": None,
                    "title": art.get("title", ""),
                    "description": art.get("description", ""),
                    "content": art.get("content", ""),
                    "url": art.get("url", ""),
                    "source": art.get("source", {}).get("name", ""),
                    "author": art.get("author", ""),
                    "published_at": art.get("publishedAt", ""),
                }
                for art in response.get("articles", [])
            ]
            _cache_set(cache_key, articles)
            return articles
        except Exception as exc:
            logger.error("NewsAPI top headlines error: %s", exc)
            return []
