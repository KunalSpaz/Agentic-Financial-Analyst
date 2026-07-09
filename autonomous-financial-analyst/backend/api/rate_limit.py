"""
rate_limit.py
-------------
Single shared slowapi Limiter instance used by every route module and by
main.py. Route modules previously each instantiated their own `Limiter`,
which meant `app.state.limiter` (used to build 429 response headers) was a
different object from whichever instance actually recorded the hit, and the
app carried N redundant in-memory rate-limit stores instead of one.
"""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
