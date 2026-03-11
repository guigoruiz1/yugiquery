import logging
import os

import pytest

from yugiquery.utils.logging import (
    ColorFormatter,
    TqdmLoggingHandler,
    _parse_level,
    is_debug_enabled,
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
# _parse_level   #
# ============== #


def test_parse_level_none():
    assert _parse_level(None) is None


def test_parse_level_int():
    assert _parse_level(logging.WARNING) == logging.WARNING


def test_parse_level_string_name():
    assert _parse_level("DEBUG") == logging.DEBUG
    assert _parse_level("warning") == logging.WARNING


def test_parse_level_numeric_string():
    assert _parse_level("20") == 20


def test_parse_level_unknown_string():
    assert _parse_level("NONSENSE") is None


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


def test_setup_logging_env_level(monkeypatch):
    _reset_logger()
    monkeypatch.setenv("YQ_LOG_LEVEL", "WARNING")
    logger = setup_logging()
    assert logger.level == logging.WARNING


def test_setup_logging_preserves_existing_level(monkeypatch):
    _reset_logger()
    monkeypatch.delenv("YQ_LOG_LEVEL", raising=False)
    setup_logging(level="ERROR")
    logger = setup_logging()
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


def test_setup_logging_force_resets_handlers():
    _reset_logger()
    setup_logging()
    setup_logging(force=True, level="WARNING")
    logger = logging.getLogger("yugiquery")
    assert len(logger.handlers) == 1
    assert logger.level == logging.WARNING


def test_setup_logging_log_file(tmp_path):
    _reset_logger()
    log_path = tmp_path / "test.log"
    logger = setup_logging(force=True, log_file=log_path)
    assert len(logger.handlers) == 2
    logger.warning("file test")
    assert log_path.exists()
    assert "file test" in log_path.read_text()


def test_setup_logging_propagate_false():
    _reset_logger()
    logger = setup_logging()
    assert logger.propagate is False


def test_setup_logging_switches_to_tqdm_handler():
    _reset_logger()
    logger = setup_logging(use_tqdm=True)
    assert any(isinstance(handler, TqdmLoggingHandler) for handler in logger.handlers)


def test_setup_logging_replaces_stream_handler_type():
    _reset_logger()
    logger = setup_logging()
    assert any(type(handler) is logging.StreamHandler for handler in logger.handlers)
    logger = setup_logging(use_tqdm=True)
    assert any(isinstance(handler, TqdmLoggingHandler) for handler in logger.handlers)


# ================= #
# is_debug_enabled  #
# ================= #


def test_is_debug_enabled_when_logger_at_info(monkeypatch):
    _reset_logger()
    setup_logging(level="INFO")
    assert is_debug_enabled() is False


def test_is_debug_enabled_when_logger_at_debug(monkeypatch):
    _reset_logger()
    setup_logging(level="DEBUG")
    assert is_debug_enabled() is True


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


def test_color_formatter_plain_fallback(monkeypatch):
    """When termcolor is absent, output is plain regardless of TTY."""
    import sys

    log_mod = sys.modules["yugiquery.utils.logging"]
    monkeypatch.setattr(log_mod, "_HAS_TERMCOLOR", False)
    # Pretend it's a TTY
    monkeypatch.setattr("sys.stderr.isatty", lambda: True, raising=False)
    fmt = ColorFormatter("%(levelname)s %(message)s")
    record = logging.LogRecord("test", logging.ERROR, "", 0, "oops", (), None)
    result = fmt.format(record)
    assert "\x1b" not in result
