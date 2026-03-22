"""
test_logger.py
--------------
Unit tests for the structured logging configuration.

Covers:
  - _JsonFormatter: valid JSON, required fields, extra fields, exception info,
                    non-serialisable values, asctime excluded
  - get_logger: returns Logger, same name → same instance, default name
  - Format selection: JSON vs text based on settings.log_format
  - Noisy loggers silenced: httpx, urllib3, transformers, yfinance
"""
from __future__ import annotations

import json
import logging
import sys
from io import StringIO
from unittest.mock import patch

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reset_logger_module() -> None:
    """Reset the module-level _configured flag and strip all root handlers."""
    import backend.utils.logger as log_mod
    log_mod._configured = False
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)


@pytest.fixture(autouse=True)
def reset_logger():
    """Ensure each test starts with a clean logging state."""
    _reset_logger_module()
    yield
    _reset_logger_module()


def _capture_json(name: str, message: str, **extra) -> dict:
    """Log *message* with _JsonFormatter and return the parsed JSON dict."""
    from backend.utils.logger import _JsonFormatter
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(_JsonFormatter())
    logger = logging.getLogger(name)
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if extra:
        logger.info(message, extra=extra)
    else:
        logger.info(message)
    logger.removeHandler(handler)
    return json.loads(stream.getvalue().strip())


# ─────────────────────────────────────────────────────────────────────────────
# _JsonFormatter
# ─────────────────────────────────────────────────────────────────────────────

class TestJsonFormatter:
    def test_output_is_valid_json(self):
        parsed = _capture_json("test.valid", "hello")
        assert isinstance(parsed, dict)

    def test_timestamp_field_present(self):
        parsed = _capture_json("test.ts", "msg")
        assert "timestamp" in parsed

    def test_timestamp_is_utc_iso_format(self):
        parsed = _capture_json("test.ts2", "msg")
        ts = parsed["timestamp"]
        assert ts.endswith("Z")
        assert "T" in ts

    def test_level_field_is_info(self):
        parsed = _capture_json("test.level", "msg")
        assert parsed["level"] == "INFO"

    def test_logger_field_matches_name(self):
        parsed = _capture_json("my.special.logger", "msg")
        assert parsed["logger"] == "my.special.logger"

    def test_message_field_matches_message(self):
        parsed = _capture_json("test.msg", "the quick brown fox")
        assert parsed["message"] == "the quick brown fox"

    def test_extra_fields_included_in_output(self):
        parsed = _capture_json("test.extra", "with extra", ticker="AAPL", score=72.5)
        assert parsed.get("ticker") == "AAPL"
        assert parsed.get("score") == 72.5

    def test_asctime_not_in_json_output(self):
        """asctime is in _RESERVED and must never appear in JSON payload."""
        from backend.utils.logger import _JsonFormatter
        stream = StringIO()
        json_handler = logging.StreamHandler(stream)
        json_handler.setFormatter(_JsonFormatter())
        # Text handler runs first, stamping record.asctime
        text_handler = logging.StreamHandler(StringIO())
        text_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger = logging.getLogger("test.asctime.leak")
        logger.handlers = []
        logger.addHandler(text_handler)
        logger.addHandler(json_handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.info("check asctime")
        logger.removeHandler(text_handler)
        logger.removeHandler(json_handler)
        parsed = json.loads(stream.getvalue().strip())
        assert "asctime" not in parsed

    def test_exception_info_in_exc_field(self):
        from backend.utils.logger import _JsonFormatter
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(_JsonFormatter())
        logger = logging.getLogger("test.exc")
        logger.handlers = []
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        try:
            raise ValueError("something went wrong")
        except ValueError:
            logger.exception("caught error")
        logger.removeHandler(handler)
        parsed = json.loads(stream.getvalue().strip())
        assert "exc" in parsed
        assert "ValueError" in parsed["exc"]
        assert "something went wrong" in parsed["exc"]

    def test_non_serialisable_extra_converted_to_string(self):
        """Extra values that can't be JSON-serialised should become strings."""
        from backend.utils.logger import _JsonFormatter
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(_JsonFormatter())
        logger = logging.getLogger("test.serial")
        logger.handlers = []
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.info("with object", extra={"obj": object()})
        logger.removeHandler(handler)
        parsed = json.loads(stream.getvalue().strip())  # must not raise
        assert "obj" in parsed
        assert isinstance(parsed["obj"], str)

    def test_internal_log_record_fields_not_in_output(self):
        """Standard LogRecord internals should not appear in JSON payload."""
        parsed = _capture_json("test.internals", "msg")
        for field in ("msg", "args", "levelno", "pathname", "lineno",
                      "funcName", "msecs", "thread", "processName"):
            assert field not in parsed, f"Internal field {field!r} leaked into JSON"


# ─────────────────────────────────────────────────────────────────────────────
# get_logger
# ─────────────────────────────────────────────────────────────────────────────

class TestGetLogger:
    def test_returns_logger_instance(self):
        from backend.utils.logger import get_logger
        assert isinstance(get_logger("test.module"), logging.Logger)

    def test_same_name_returns_same_instance(self):
        from backend.utils.logger import get_logger
        assert get_logger("same.name") is get_logger("same.name")

    def test_default_name(self):
        from backend.utils.logger import get_logger
        logger = get_logger()
        assert logger.name == "autonomous-financial-analyst"

    def test_different_names_return_different_loggers(self):
        from backend.utils.logger import get_logger
        assert get_logger("a.b") is not get_logger("a.c")

    def test_configured_only_once(self):
        """Calling get_logger multiple times must not add duplicate handlers."""
        from backend.utils.logger import get_logger
        get_logger("first.call")
        get_logger("second.call")
        root = logging.getLogger()
        handler_count = len(root.handlers)
        get_logger("third.call")
        assert len(root.handlers) == handler_count


# ─────────────────────────────────────────────────────────────────────────────
# Format selection
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatSelection:
    def test_json_format_uses_json_formatter(self):
        from backend.utils.logger import _JsonFormatter, _configure_root_logger
        with patch("backend.utils.logger.settings") as mock_settings:
            mock_settings.log_level = "INFO"
            mock_settings.log_format = "json"
            _configure_root_logger()
        root = logging.getLogger()
        assert any(isinstance(h.formatter, _JsonFormatter) for h in root.handlers)

    def test_text_format_uses_standard_formatter(self):
        from backend.utils.logger import _JsonFormatter, _configure_root_logger
        with patch("backend.utils.logger.settings") as mock_settings:
            mock_settings.log_level = "INFO"
            mock_settings.log_format = "text"
            _configure_root_logger()
        root = logging.getLogger()
        assert any(
            not isinstance(h.formatter, _JsonFormatter)
            for h in root.handlers
        )


# ─────────────────────────────────────────────────────────────────────────────
# Noisy logger silencing
# ─────────────────────────────────────────────────────────────────────────────

class TestNoisyLoggerSilencing:
    @pytest.mark.parametrize("noisy_name", ["httpx", "httpcore", "urllib3", "transformers", "yfinance"])
    def test_noisy_logger_set_to_warning(self, noisy_name):
        from backend.utils.logger import get_logger
        get_logger("trigger.configure")
        assert logging.getLogger(noisy_name).level == logging.WARNING
