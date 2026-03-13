# yugiquery/utils/logging.py

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import logging
import os
import sys
from pathlib import Path

# --- Imports: Third-Party --- #
from termcolor import colored as _colored
import logging
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Color mapping for log levels
_LEVEL_COLORS = {
    logging.DEBUG: "cyan",
    logging.INFO: None,
    logging.WARNING: "yellow",
    logging.ERROR: "red",
    logging.CRITICAL: "red",
}


class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


# --- ColorFormatter class --- #
class ColorFormatter(logging.Formatter):
    def __init__(self, show_source=False):
        if show_source:
            fmt = "%(asctime)s | %(levelname)s | %(module)s.%(funcName)s | %(message)s"
        else:
            fmt = "%(asctime)s | %(levelname)s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        color = _LEVEL_COLORS.get(record.levelno)
        attrs = ["bold"] if record.levelno >= logging.CRITICAL else []
        return _colored(msg, color, attrs=attrs or None) if color or attrs else msg


# --- LoggerConfig class --- #
class LoggerConfig:
    _level = None
    _log_file = None
    _logger_name = None

    @classmethod
    def setup(cls, level=None, log_file=None, logger_name=None):
        cls._level = level or os.environ.get("YQ_LOG_LEVEL", "WARNING")
        cls._log_file = log_file or os.environ.get("YQ_LOG_FILE", None)
        cls._logger_name = logger_name or cls._logger_name or "yugiquery"
        logger = logging.getLogger(cls._logger_name)
        logger.setLevel(logging._nameToLevel.get(str(cls._level).upper(), logging.INFO))
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        if cls._log_file:
            handler = logging.FileHandler(cls._log_file)
            handler.setFormatter(ColorFormatter(show_source=True))
        else:
            handler = TqdmLoggingHandler()
            handler.setFormatter(ColorFormatter(show_source=False))
        logger.addHandler(handler)
        logger.propagate = False
        return logger

    @classmethod
    def get_logger(cls, logger_name="yugiquery"):
        return logging.getLogger(logger_name)

    @classmethod
    def propagate_env(cls):
        if cls._level is not None:
            os.environ["YQ_LOG_LEVEL"] = cls._level
        if cls._log_file is not None:
            os.environ["YQ_LOG_FILE"] = cls._log_file

    @classmethod
    def clean_env(cls):
        os.environ.pop("YQ_LOG_LEVEL", None)
        os.environ.pop("YQ_LOG_FILE", None)

    @classmethod
    def get_level(cls):
        """
        Returns the current logging level as a logging enum value (e.g., logging.INFO).
        If not set, defaults to logging.WARNING.
        """
        return logging._nameToLevel.get(str(cls._level).upper(), logging.WARNING)
