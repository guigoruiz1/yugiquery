# yugiquery/__init__.py

# -*- coding: utf-8 -*-

from . import api
from . import core
from .api import *
from .core import *
from .metadata import *
from .utils import *

# Explicit Notebook-specific imports for notebook convenience
if dirs.is_notebook:
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from itables import init_notebook_mode

    # Default pandas display settings
    pd.set_option("display.max_columns", 40)
