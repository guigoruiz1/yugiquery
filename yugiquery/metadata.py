# yugiquery/metadata.py

# -*- coding: utf-8 -*-

# =============== #
# Metadata module #
# =============== #

__all__ = [
    "__title__",
    "__description__",
    "__url__",
    "__version__",
    "__author__",
    "__author_email__",
    "__license__",
    "__copyright__",
    "__maintainer__",
    "__status__",
]
__title__ = "YugiQuery"
__description__ = "Python package to query and display Yu-Gi-Oh! data built on Jupyter notebooks and Git."
__url__ = "https://github.com/guigoruiz1/yugiquery"
__author__ = "Guilherme Ruiz"
__author_email__ = "guilherme.guigoruiz@gmail.com"
__license__ = "MIT"
__copyright__ = "2023, Guilherme Ruiz"
__maintainer__ = "Guilherme Ruiz"
__status__ = "Development"

try:
    # Check if the _version.py file exists
    from ._version import __version__, __version_tuple__

    __all__.append("__version_tuple__")
except ImportError:
    # Fallback values if _version.py is not present
    __version__ = "2.0.5"
