"""
logger.py
---------
Structured logging configuration for the entire application.
All modules should obtain their logger via ``get_logger(__name__)``.

Two output formats are supported, controlled by the ``LOG_FORMAT`` env var:

* ``json``  (default) — one JSON object per line, suitable for log aggregators
  (Datadog, Loki, CloudWatch, etc.).  Every record includes ``timestamp``,
  ``level``, ``logger``, ``message``, and any ``extra`` fields passed by the
  caller.

* ``text`` — human-readable ``asctime | LEVEL | name | message`` format,
  useful during local development (set ``LOG_FORMAT=text`` in ``.env``).
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from typing import Optional

from backend.utils.config import settings

# Per-request correlation ID — set by the request-ID middleware in main.py
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)

# Fields that are part of the LogRecord internals — excluded from extra output
_RESERVED = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName", "asctime",
})

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


class _JsonFormatter(logging.Formatter):
    """
    Emit each log record as a single JSON line.

    Output schema::

        {
          "timestamp": "2026-03-20T10:30:00.123Z",
          "level":     "INFO",
          "logger":    "backend.api.main",
          "message":   "API started",
          // ...any extra fields passed via logger.info(..., extra={...})
          "exc":       "Traceback ..."   // only present when an exception is logged
        }
    """

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()

        ts = datetime.fromtimestamp(record.created, tz=timezone.utc)
        payload: dict = {
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
        }

        # Attach request correlation ID when inside a request context
        rid = request_id_ctx.get("")
        if rid:
            payload["request_id"] = rid

        # Attach any caller-supplied extra fields
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                try:
                    json.dumps(value)  # only include JSON-serialisable values
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = str(value)

        # Attach exception info when present
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        elif record.exc_text:
            payload["exc"] = record.exc_text

        return json.dumps(payload, ensure_ascii=False)


def _configure_root_logger() -> None:
    """Configure the root logger once on first call."""
    global _configured
    if _configured:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if settings.log_format.lower() == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "transformers", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        Configured :class:`logging.Logger` instance.

    Example::

        logger = get_logger(__name__)
        logger.info("Analysis complete", extra={"ticker": "AAPL", "score": 72.5})
    """
    _configure_root_logger()
    return logging.getLogger(name or "autonomous-financial-analyst")
