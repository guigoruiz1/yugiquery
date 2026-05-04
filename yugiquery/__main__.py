# yugiquery/__main__.py

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import argparse

# --- Imports: Local Application --- #
from .metadata import __title__, __version__
from .cli import CustomHelpFormatter
from .utils import dirs, LoggerConfig
from . import api
from . import cli
from . import bot
from .scripts import optionals


# --- Main Execution --- #
def main():
    """Main entry point for the YugiQuery CLI tool. Parses command-line arguments and dispatches to the appropriate handlers."""
    # Create the primary parser
    parser = argparse.ArgumentParser(
        description="Yugiquery CLI tool",
        prog=__title__,
        formatter_class=CustomHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
    )
    helper_group = parser.add_argument_group(
        "helpers",
        description="Used without a command to print helpful information then exit",
    )
    helper_group.add_argument("-a", "--api", action="store_true", help="Print API status")
    helper_group.add_argument("-p", "--paths", action="store_true", help=f"Print {__title__} paths")
    helper_group.add_argument("-v", "--version", action="store_true", help=f"Print {__title__} version")

    # Subparser for the main yugiquery flow
    cli.set_run_parser(subparsers)
    cli.set_fetch_parser(subparsers)
    cli.set_report_parser(subparsers)
    cli.set_cleanup_parser(subparsers)

    # Subparser for the bot mode
    bot.set_parser(subparsers)

    # Subparser for the optional installation of additional components
    optionals.set_parser(subparsers)

    # Parse initial arguments
    args = parser.parse_args()
    if args.command is None:
        if not (args.version or args.api or args.paths):
            parser.print_help()
        else:
            if args.version:
                print(f"{__title__} {__version__}")
            if args.api:
                LoggerConfig.setup()
                api.check_status()
            if args.paths:
                dirs.print()
        exit()

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
        bot.main(args)
    elif args.command == "run":
        cli.handle_run(args)
    elif args.command == "fetch":
        cli.handle_fetch(args)
    elif args.command == "report":
        cli.handle_report(args)
    elif args.command == "cleanup":
        cli.handle_cleanup(args)


if __name__ == "__main__":
    main()
