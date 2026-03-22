# yugiquery/utils/logging.py

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import logging
import os
from pathlib import Path

# --- Imports: Third-Party --- #
from termcolor import colored as _colored
import logging
import os
from pathlib import Path
from tqdm import tqdm

# --- Imports: Local Application --- #
from ..metadata import __title__

# Color mapping for log levels
_LEVEL_COLORS = {
    logging.DEBUG: "cyan",
    logging.INFO: None,
    logging.WARNING: "yellow",
    logging.ERROR: "red",
    logging.CRITICAL: "red",
}
"""
Mapping from logging level (int) to color name (str or None) for colored log output.

Keys are standard logging levels (e.g., logging.DEBUG), values are color names for use with termcolor.
If the value is None, no color is applied for that level.
"""


class TqdmLoggingHandler(logging.StreamHandler):
    """
    Logging handler that writes log messages using tqdm.write, so logs don't interfere with tqdm progress bars.
    """

    def emit(self, record):
        """
        Emit a record using tqdm.write to avoid breaking progress bars.
        """
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


# --- ColorFormatter class --- #
class ColorFormatter(logging.Formatter):
    """
    Formatter that adds color to log messages based on their severity level.
    Optionally includes source module and function in the log output.
    """

    def __init__(self, show_source=False):
        """
        Initialize the formatter.
        Args:
            show_source (bool): If True, include module and function name in log output.
        """
        if show_source:
            fmt = "%(asctime)s | %(levelname)s | %(module)s.%(funcName)s | %(lineno)d | %(message)s"
        else:
            fmt = "%(asctime)s | %(levelname)s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the specified record, adding color based on log level.
        Args:
            record (logging.LogRecord): The log record to format.
        Returns:
            str: The formatted log message with appropriate color.
        """
        msg = super().format(record)
        color = _LEVEL_COLORS.get(record.levelno)
        attrs = ["bold"] if record.levelno >= logging.CRITICAL else []
        return _colored(msg, color, attrs=attrs or None) if color or attrs else msg


# --- LoggerConfig class --- #
class LoggerConfig:
    """
    Centralized logger configuration utility for the Yugiquery package.
    Provides setup, environment propagation, and accessors for logger state.
    """

    _level = None
    _log_file = None
    _logger_name = __title__.lower()

    @classmethod
    def setup(cls, level=None, log_file=None, logger_name=None) -> logging.Logger:
        """
        Set up the logger with the specified level, log file, and logger name.
        Removes existing handlers and attaches a new handler (file or tqdm-based).

        Args:
            level (str|int|None): Logging level (e.g., 'INFO', logging.DEBUG). If None, uses env or WARNING.
            log_file (str|Path|None): Path to log file. If None, logs to stream.
            logger_name (str|None): Name for the logger. If None, uses previous or 'yugiquery'.

        Returns:
            logging.Logger: The configured logger instance.
        """
        cls._level = level or os.environ.get("YQ_LOG_LEVEL", "WARNING")
        cls._log_file = log_file or os.environ.get("YQ_LOG_FILE", None)
        cls._logger_name = logger_name or cls._logger_name or __title__.lower()
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
    def get_logger(cls, logger_name: str | None = None) -> logging.Logger:
        """
        Return the logger instance for the current logger name.

        Returns:
            logging.Logger: Logger instance for the configured logger name.
        """
        return logging.getLogger(logger_name or cls._logger_name or __title__.lower())

    @classmethod
    def propagate_env(cls) -> None:
        """
        Set environment variables to reflect the current logger configuration.
        """
        if cls._level is not None:
            os.environ["YQ_LOG_LEVEL"] = cls._level
        if cls._log_file is not None:
            os.environ["YQ_LOG_FILE"] = cls._log_file

    @classmethod
    def clean_env(cls) -> None:
        """
        Remove logger-related environment variables.
        """
        os.environ.pop("YQ_LOG_LEVEL", None)
        os.environ.pop("YQ_LOG_FILE", None)

    @classmethod
    def get_level(cls) -> int:
        """
        Return the current logging level as a logging enum value (e.g., logging.INFO).
        If not set, defaults to logging.WARNING.

        Returns:
            int: Logging level as an enum value (e.g., logging.INFO).
        """
        return logging._nameToLevel.get(str(cls._level).upper(), logging.WARNING)

    @classmethod
    def get_file(cls) -> Path | None:
        """
        Return the current log file as a Path object, or None if not set.

        Returns:
            Path | None: Path to the log file, or None if not set.
        """
        if cls._log_file:
            return Path(cls._log_file)
        return None
