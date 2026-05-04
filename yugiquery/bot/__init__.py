# yugiquery/bot/__init__.py

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import importlib

# --- Imports: Local Application --- #
from .cli import set_parser, main
from .base import Base


# --- Lazy Loading of Bot Subclasses --- #
class _LazyLoader:
    def __init__(self, module_name, class_name):
        self.module_name = module_name
        self.class_name = class_name
        self._bot_class = None

    def _load_class(self):
        if self._bot_class is None:
            module = importlib.import_module(self.module_name, package=__package__)
            self._bot_class = getattr(module, self.class_name)
        return self._bot_class

    def __call__(self, *args, **kwargs):
        return self._load_class()(*args, **kwargs)


Discord = _LazyLoader(module_name=".discord", class_name="Discord")
Telegram = _LazyLoader(module_name=".telegram", class_name="Telegram")
