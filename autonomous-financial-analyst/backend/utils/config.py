"""
config.py
---------
Centralised application configuration loaded from environment variables
via pydantic-settings.  All modules import `settings` from here.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings resolved from .env or environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────
    app_env: str = Field(default="development")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json", description="'json' for structured output, 'text' for human-readable")
    allowed_origins: str = Field(default="http://localhost:8501")
    api_secret_key: str = Field(default="")  # If set, all requests must include X-API-Key header

    # ── OpenAI ─────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o")
    openai_embedding_model: str = Field(default="text-embedding-3-small")

    # ── NewsAPI ────────────────────────────────────────────────────────────
    news_api_key: str = Field(default="")

    # ── Database ───────────────────────────────────────────────────────────
    database_url: str = Field(default="sqlite:///./data/financial_analyst.db")

    # ── Scheduler ──────────────────────────────────────────────────────────
    scheduler_enabled: bool = Field(default=True)
    daily_report_hour: int = Field(default=8)
    daily_report_minute: int = Field(default=0)

    # ── Market Data ────────────────────────────────────────────────────────
    market_data_cache_ttl: int = Field(default=300)
    stock_universe: str = Field(
        default="AAPL,MSFT,NVDA,TSLA,AMZN,META,GOOGL,AMD,NFLX,INTC"
    )

    # ── FinBERT ────────────────────────────────────────────────────────────
    finbert_model: str = Field(default="ProsusAI/finbert")

    # ── RAG ────────────────────────────────────────────────────────────────
    faiss_index_path: str = Field(default="./data/faiss_index")
    rag_top_k: int = Field(default=5)

    @property
    def stock_universe_list(self) -> List[str]:
        """Return the stock universe as a list of ticker strings."""
        return [t.strip().upper() for t in self.stock_universe.split(",") if t.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()


# Module-level convenience alias
settings = get_settings()
