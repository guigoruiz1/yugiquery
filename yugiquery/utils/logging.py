# yugiquery/utils/logging.py

# -*- coding: utf-8 -*-

import logging
import os
import sys
from pathlib import Path

from tqdm.auto import tqdm
from termcolor import colored as _colored

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_CONCISE_FORMAT = "%(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_LEVEL_COLORS = {
    logging.DEBUG: "cyan",
    logging.INFO: None,
    logging.WARNING: "yellow",
    logging.ERROR: "red",
    logging.CRITICAL: "red",
}


class ColorFormatter(logging.Formatter):
    """Formatter that applies ANSI colour codes by log level (TTY only)."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if not sys.stderr.isatty():
            return msg
        color = _LEVEL_COLORS.get(record.levelno)
        attrs = ["bold"] if record.levelno >= logging.CRITICAL else []
        return _colored(msg, color, attrs=attrs or None) if color or attrs else msg


class TqdmLoggingHandler(logging.StreamHandler):
    """Stream handler that writes through tqdm to preserve active progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


def _parse_level(level: str | int | None) -> int | None:
    if level is None:
        return None
    if isinstance(level, int):
        return level

    normalized = str(level).strip().upper()
    if normalized.isdigit():
        return int(normalized)

    return logging._nameToLevel.get(normalized)


def setup_logging(
    *,
    level: str | int | None = None,
    log_file: str | Path | None = None,
    force: bool = False,
    use_tqdm: bool = False,
) -> logging.Logger:
    """Configure package logging handlers and level.

    The package logger is `yugiquery` and child loggers inherit from it.
    """
    logger = logging.getLogger("yugiquery")
    env_level = _parse_level(os.environ.get("YQ_LOG_LEVEL"))
    selected_level = _parse_level(level)

    if selected_level is None and logger.handlers and logger.level != logging.NOTSET:
        selected_level = logger.level
    if selected_level is None:
        selected_level = env_level or logging.INFO

    logger.setLevel(selected_level)

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    stream_handler_class = TqdmLoggingHandler if use_tqdm else logging.StreamHandler
    stream_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
    ]

    if any(type(handler) is not stream_handler_class for handler in stream_handlers):
        for handler in stream_handlers:
            logger.removeHandler(handler)
        stream_handlers = []

    if not stream_handlers:
        stream_handler = stream_handler_class()
        stream_format = _DEFAULT_FORMAT if selected_level == logging.DEBUG else _CONCISE_FORMAT
        stream_handler.setFormatter(ColorFormatter(stream_format, datefmt=_DEFAULT_DATEFMT))
        logger.addHandler(stream_handler)

    if log_file is not None and not any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == Path(log_file)
        for handler in logger.handlers
    ):
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
