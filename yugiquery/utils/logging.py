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

# Color mapping for log levels
_LEVEL_COLORS = {
    logging.DEBUG: "cyan",
    logging.INFO: None,
    logging.WARNING: "yellow",
    logging.ERROR: "red",
    logging.CRITICAL: "red",
}


class LoggerWriter:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message):
        # tqdm may send partial lines, so buffer until newline
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.logger.log(self.level, line)

    def flush(self):
        if self._buffer.strip():
            self.logger.log(self.level, self._buffer.strip())
        self._buffer = ""


# ColorFormatter class
class ColorFormatter(logging.Formatter):
    def __init__(self, fmt, datefmt=None, use_concise=False):
        super().__init__(fmt, datefmt)
        self.use_concise = use_concise

    def format(self, record: logging.LogRecord) -> str:
        if self.use_concise:
            fmt = "%(message)s"
            msg = logging.Formatter(fmt).format(record)
        else:
            msg = super().format(record)
        color = _LEVEL_COLORS.get(record.levelno)
        attrs = ["bold"] if record.levelno >= logging.CRITICAL else []
        return _colored(msg, color, attrs=attrs or None) if color or attrs else msg


def _parse_level(level):
    if isinstance(level, str):
        level = level.upper()
        return logging._nameToLevel.get(level, logging.INFO)
    return level


def setup_logging(
    *,
    level: str | int | None = None,
    log_file: str | Path | None = None,
) -> logging.Logger:
    logger = logging.getLogger("yugiquery")
    parsed_level = _parse_level(level)
    if parsed_level:
        logger.setLevel(parsed_level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    if log_file is not None:
        handler = logging.FileHandler(log_file)
    else:
        handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# --- Logger Accessor --- #
def get_logger():
    return logging.getLogger("yugiquery")
