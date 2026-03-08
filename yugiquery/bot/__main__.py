# yugiquery/bot/__main__.py

# -*- coding: utf-8 -*-

import argparse
from .base import set_parser, main
from ..cli import CustomHelpFormatter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    set_parser(parser)
    args = parser.parse_args()
    main(args)
