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


# Custom logger that updates handler format on setLevel
class YugiqueryLogger(logging.Logger):
    _DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
    _CONCISE_FORMAT = "%(message)s"

    def setLevel(self, level):
        super().setLevel(level)
        use_concise = level == logging.INFO
        fmt = self._CONCISE_FORMAT if use_concise else self._DEFAULT_FORMAT
        for handler in self.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(ColorFormatter(fmt, datefmt="%Y-%m-%d %H:%M:%S", use_concise=use_concise))


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
    level: str | int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    logging.setLoggerClass(YugiqueryLogger)
    logger = logging.getLogger("yugiquery")
    level_int = _parse_level(level)
    logger.setLevel(level_int)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    use_concise = level_int == logging.INFO
    fmt = YugiqueryLogger._CONCISE_FORMAT if use_concise else YugiqueryLogger._DEFAULT_FORMAT
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(ColorFormatter(fmt, datefmt="%Y-%m-%d %H:%M:%S", use_concise=use_concise))
    logger.addHandler(stream_handler)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(YugiqueryLogger._DEFAULT_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger
