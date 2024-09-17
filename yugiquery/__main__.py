# yugiquery/__main__.py

# -*- coding: utf-8 -*-

# Standard library imports
import argparse
import importlib

# Local application imports
from .utils import dirs
from . import yugiquery as yq
from . import bot


def main():
    # Create the primary parser
    parser = argparse.ArgumentParser(description="Yugiquery CLI tool")

    subparsers = parser.add_subparsers(dest="command")
    parser.add_argument("-p", "--paths", action="store_true", help="Print YugiQuery paths and exit")
    parser.add_argument("-v", "--version", action="store_true", help="Print YugiQuery version and exit")

    # Subparser for the main yugiquery flow
    yugiquery_parser = subparsers.add_parser("run", help="Run the main Yugiquery flow")
    yq.set_parser(yugiquery_parser)  # TODO: Make --reports positional
    # Subparser for the bot mode
    bot_parser = subparsers.add_parser("bot", help="Run yugiquery bot")
    bot.set_parser(bot_parser)

    # Subparser for the kernel installation
    try:
        spec = importlib.util.spec_from_file_location(
            name="post_install",
            location=dirs.get_asset("scripts", "post_install.py"),
        )
        post_install = importlib.util.module_from_spec(spec=spec)
        spec.loader.exec_module(post_install)

        post_install_parser = subparsers.add_parser(
            "install",
            help="Run post-install script to install various additional components. If no flags are passed, all components will be installed.",
        )
        post_install.set_parser(post_install_parser)
    except:
        pass

    # Parse initial arguments
    args = parser.parse_args()

    if args.paths or args.version:
        if args.paths:
            dirs.print()
        if args.version:
            from .metadata import __title__, __version__

            print(f"{__title__} {__version__}")

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
            post_install.main(args)

        elif args.command == "bot":
            # Call the bot main function with parsed arguments
            bot.main(args)
        else:
            # Main Yugiquery flow
            yq.main(args)


if __name__ == "__main__":
    main()
