# yugiquery/__init__.py

# -*- coding: utf-8 -*-

# --- Imports: Submodules --- #
from . import api
from . import bot

# --- Imports: All Symbols --- #
from .core import *
from .metadata import *
from .utils import *

# --- Notebook-Specific Imports --- #

if dirs.is_notebook:
    import arrow
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from itables import init_notebook_mode
    from termcolor import cprint

    LoggerConfig.setup()

    # Default pandas display settings
    pd.set_option("display.max_columns", 40)
