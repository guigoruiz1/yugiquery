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

    @staticmethod
    def _level_to_int(level: str | int | None) -> int:
        """
        Convert a logging level (str/int/None) to an int. Defaults to logging.WARNING.
        """
        if level is None:
            return logging.WARNING
        if isinstance(level, int):
            if level in logging._levelToName:
                return level
            # If int but not a valid level, fallback
            return logging.WARNING
        if isinstance(level, str):
            return logging._nameToLevel.get(level.upper(), logging.WARNING)
        return logging.WARNING

    _logger_name: str = __title__.lower()

    @classmethod
    def setup(
        cls,
        level: str | int | None = None,
        log_file: str | Path | None = None,
        logger_name: str | None = None,
    ) -> logging.Logger:
        """
        Set up the logger with the specified level, log file, and logger name.
        Removes existing handlers and attaches a new handler (file or tqdm-based).

        Args:
            level (str|int|None): Logging level (e.g., 'INFO', logging.DEBUG). If None, uses env or WARNING.
            log_file (str|Path|None): Path to log file. If None, logs to stream.
            logger_name (str|None): Name for the logger. If None, uses package title.

        Returns:
            logging.Logger: The configured logger instance.
        """
        # Resolve config from arguments or environment
        resolved_level = level or os.environ.get("YQ_LOG_LEVEL") or "WARNING"
        resolved_log_file = log_file or os.environ.get("YQ_LOG_FILE")
        resolved_logger_name = logger_name or __title__.lower()

        level_int = cls._level_to_int(resolved_level)
        log_file_path = Path(resolved_log_file) if resolved_log_file else None

        logger = logging.getLogger(resolved_logger_name)
        logger.setLevel(level_int)

        # Remove all handlers
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

        # Add appropriate handler
        if log_file_path:
            handler = logging.FileHandler(log_file_path)
            handler.setFormatter(ColorFormatter(show_source=True))
        else:
            handler = TqdmLoggingHandler()
            handler.setFormatter(ColorFormatter(show_source=False))
        logger.addHandler(handler)
        logger.propagate = False
        # Store logger name for future use
        cls._logger_name = resolved_logger_name
        return logger

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """
        Return the logger instance for the configured logger name.
        """
        return logging.getLogger(cls._logger_name)

    @classmethod
    def propagate_env(
        cls,
        level: str | int | None = None,
        log_file: str | Path | None = None,
    ) -> None:
        """
        Set environment variables to reflect the given logger configuration.
        """
        if level is not None:
            os.environ["YQ_LOG_LEVEL"] = str(level)
        if log_file is not None:
            os.environ["YQ_LOG_FILE"] = str(log_file)

    @staticmethod
    def clean_env() -> None:
        """
        Remove logger-related environment variables.
        """
        os.environ.pop("YQ_LOG_LEVEL", None)
        os.environ.pop("YQ_LOG_FILE", None)

    @classmethod
    def get_level(cls) -> int:
        """
        Return the logging level of the configured logger.
        """
        logger = cls.get_logger()
        return logger.level

    @classmethod
    def get_file(cls) -> Path | None:
        """
        Return the log file path from the logger's handlers, if any.
        """
        logger = cls.get_logger()
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return Path(handler.baseFilename)
        return None
