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
    # Create the primary parser
    parser = argparse.ArgumentParser(description="Yugiquery CLI tool", prog=__title__, formatter_class=CustomHelpFormatter)

    subparsers = parser.add_subparsers(dest="command")
    parser.add_argument("-a", "--api", action="store_true", help="Print API status and exit")
    parser.add_argument("-p", "--paths", action="store_true", help="Print YugiQuery paths and exit")
    parser.add_argument("-v", "--version", action="store_true", help="Print YugiQuery version and exit")

    # Subparser for the main yugiquery flow
    run_parser = subparsers.add_parser("run", help="Run the full Yugiquery flow", formatter_class=CustomHelpFormatter)
    cli.set_run_parser(run_parser)
    # Subparser for fetching data only
    fetch_parser = subparsers.add_parser(
        "fetch", help="Fetch/update data only (no reports)", formatter_class=CustomHelpFormatter
    )
    cli.set_fetch_parser(fetch_parser)
    # Subparser for generating reports only
    report_parser = subparsers.add_parser(
        "report", help="Generate reports only (no data update)", formatter_class=CustomHelpFormatter
    )
    cli.set_report_parser(report_parser)
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
    # No fallback: all commands are now handled explicitly


if __name__ == "__main__":
    main()
