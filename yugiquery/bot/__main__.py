# yugiquery/bot/__main__.py

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import argparse

# --- Imports: Local Application --- #
from .base import set_parser, main
from ..cli import CustomHelpFormatter

# --- Main Execution --- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    set_parser(parser)
    args = parser.parse_args()
    main(args)
