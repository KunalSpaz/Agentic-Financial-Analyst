"""
migrations.py
-------------
Programmatic schema creation / upgrade.  Imports all ORM models to ensure
they are registered with ``Base.metadata`` before ``create_all`` is invoked.
"""

from __future__ import annotations

from backend.utils.logger import get_logger

logger = get_logger(__name__)


def run_migrations() -> None:
    """
    Create all tables that do not yet exist in the target database.

    This is intentionally simple (create-if-not-exists) for SQLite.
    For production PostgreSQL migrations use Alembic instead.
    """
    # Import models so they register themselves with Base.metadata
    import backend.models.stock  # noqa: F401
    import backend.models.news_article  # noqa: F401
    import backend.models.sentiment_score  # noqa: F401
    import backend.models.analysis_report  # noqa: F401
    import backend.models.market_opportunity  # noqa: F401
    import backend.models.backtest_result  # noqa: F401
    import backend.models.strategy_optimization  # noqa: F401
    import backend.models.portfolio_risk_report  # noqa: F401
    import backend.models.financial_document  # noqa: F401
    import backend.models.user_query  # noqa: F401

    from backend.database.connection import Base, engine

    logger.info("Running database migrations (create_all)…")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema is up to date.")
