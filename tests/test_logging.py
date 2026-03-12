import logging
import os

import pytest

from yugiquery.utils.logging import (
    ColorFormatter,
    setup_logging,
)

# Helpers to isolate tests from each other's logger state


def _reset_logger():
    """Remove all handlers and reset the yugiquery logger between tests."""
    logger = logging.getLogger("yugiquery")
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.setLevel(logging.NOTSET)


# ============== #
# setup_logging  #
# ============== #


def test_setup_logging_default_level(monkeypatch):
    _reset_logger()
    monkeypatch.delenv("YQ_LOG_LEVEL", raising=False)
    logger = setup_logging()
    assert logger.level == logging.INFO


def test_setup_logging_debug_level_arg(monkeypatch):
    _reset_logger()
    monkeypatch.delenv("YQ_LOG_LEVEL", raising=False)
    logger = setup_logging(level="DEBUG")
    assert logger.level == logging.DEBUG


def test_setup_logging_level_arg_overrides_env(monkeypatch):
    _reset_logger()
    monkeypatch.setenv("YQ_LOG_LEVEL", "WARNING")
    logger = setup_logging(level="ERROR")
    assert logger.level == logging.ERROR


def test_setup_logging_adds_handler():
    _reset_logger()
    logger = setup_logging()
    assert len(logger.handlers) == 1


def test_setup_logging_idempotent():
    """Calling setup_logging twice without force must not add duplicate handlers."""
    _reset_logger()
    setup_logging()
    setup_logging()
    assert len(logging.getLogger("yugiquery").handlers) == 1


def test_setup_logging_propagate_false():
    _reset_logger()
    logger = setup_logging()
    assert logger.propagate is False


def test_logger_setLevel_updates_format():
    _reset_logger()
    logger = setup_logging()
    # Should start with INFO and concise format
    assert logger.level == logging.INFO
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            fmt = handler.formatter._fmt
            assert fmt == "%(message)s"
    # Switch to DEBUG, should update to default format
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            fmt = handler.formatter._fmt
            assert fmt == "%(asctime)s | %(levelname)s | %(message)s"
    # Switch back to INFO, should update to concise format
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            fmt = handler.formatter._fmt
            assert fmt == "%(message)s"


# ================ #
# ColorFormatter   #
# ================ #


def test_color_formatter_non_tty_no_color(monkeypatch):
    """When stderr is not a TTY, output must be plain (no ANSI codes)."""
    monkeypatch.setattr("sys.stderr", open(os.devnull, "w"))
    fmt = ColorFormatter("%(levelname)s %(message)s")
    record = logging.LogRecord("test", logging.WARNING, "", 0, "hello", (), None)
    result = fmt.format(record)
    assert result == "WARNING hello"
    assert "\x1b" not in result


def test_color_formatter_plain_on_non_tty(monkeypatch):
    """When stderr is not a TTY, output is plain."""
    monkeypatch.setattr("sys.stderr.isatty", lambda: False, raising=False)
    fmt = ColorFormatter("%(levelname)s %(message)s")
    record = logging.LogRecord("test", logging.ERROR, "", 0, "oops", (), None)
    result = fmt.format(record)
    assert "\x1b" not in result
