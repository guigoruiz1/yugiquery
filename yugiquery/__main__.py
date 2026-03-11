# yugiquery/__main__.py

# -*- coding: utf-8 -*-

# Standard library imports
import argparse
import importlib

# Local application imports
from .metadata import __title__, __version__
from .cli import CustomHelpFormatter
from .utils import dirs, setup_logging
from . import api
from . import cli
from . import bot
from .scripts import optionals


def main():
    # Create the primary parser
    parser = argparse.ArgumentParser(description="Yugiquery CLI tool", prog=__title__, formatter_class=CustomHelpFormatter)

    subparsers = parser.add_subparsers(dest="command")
    parser.add_argument("-a", "--api", action="store_true", help="Print API status and exit")
    parser.add_argument("-p", "--paths", action="store_true", help="Print YugiQuery paths and exit")
    parser.add_argument("-v", "--version", action="store_true", help="Print YugiQuery version and exit")

    # Subparser for the main yugiquery flow
    yugiquery_parser = subparsers.add_parser("run", help="Run the main Yugiquery flow", formatter_class=CustomHelpFormatter)
    cli.set_parser(yugiquery_parser)
    # Subparser for the bot mode
    bot_parser = subparsers.add_parser("bot", help="Run yugiquery bot", formatter_class=CustomHelpFormatter)
    bot.set_parser(bot_parser)
    # Subparser for the optional installation of additional components
    optionals_parser = subparsers.add_parser(
        "install", help="Install various additional components. If no flags are passed, all components will be installed"
    )
    optionals.set_parser(optionals_parser)

    # Parse initial arguments
    args = parser.parse_args()
    if args.command is None:
        setup_logging()
        if args.version:
            print(f"{__title__} {__version__}")
        if args.api:
            api.check_status()
        if args.paths:
            dirs.print()

        exit()

    else:
        print(
            "\n"
            " ██    ██ ██    ██  ██████  ██  ██████  ██    ██ ███████ ██████  ██    ██ \n"
            "  ██  ██  ██    ██ ██       ██ ██    ██ ██    ██ ██      ██   ██  ██  ██  \n"
            "   ████   ██    ██ ██   ███ ██ ██    ██ ██    ██ █████   ██████    ████   \n"
            "    ██    ██    ██ ██    ██ ██ ██ ▄▄ ██ ██    ██ ██      ██   ██    ██    \n"
            "    ██     ██████   ██████  ██  ██████   ██████  ███████ ██   ██    ██    \n"
            "                                   ▀▀                                     \n"
        )

        if args.command == "install":
            optionals.main(args)

        elif args.command == "bot":
            # Call the bot main function with parsed arguments
            bot.main(args)
        else:
            # Main Yugiquery flow
            cli.main(args)


if __name__ == "__main__":
    main()
