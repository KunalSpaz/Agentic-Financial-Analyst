"""
conftest.py
-----------
Shared pytest fixtures.

Disables slowapi rate limiting for the whole test session: TestClient reuses
a single fake remote address for every request, so route test classes that
issue more than a handful of requests to the same endpoint (e.g.
TestGetMarketOpportunities, TestIngestDocument) would otherwise start
receiving 429s once real per-route limits are enforced. Rate limiting itself
is not what these tests are verifying.
"""
from __future__ import annotations

import pytest

from backend.api.rate_limit import limiter


@pytest.fixture(autouse=True, scope="session")
def _disable_rate_limiting():
    limiter.enabled = False
    yield
