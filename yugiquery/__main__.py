# yugiquery/__main__.py

import argparse
import importlib
from .utils import auto_or_bool, dirs


def main():
    # Create the primary parser
    parser = argparse.ArgumentParser(description="Yugiquery CLI tool")

    subparsers = parser.add_subparsers(dest="command")
    parser.add_argument(
        "-p", "--paths", action="store_true", help="Print YugiQuery paths and exit"
    )
    parser.add_argument(
        "-v", "--version", action="store_true", help="Print YugiQuery version and exit"
    )
    # Subparser for the main yugiquery flow
    yugiquery_parser = subparsers.add_parser("run", help=" Run the main Yugiquery flow")

    yugiquery_parser.add_argument(
        "-r",
        "--reports",
        nargs="+",
        dest="reports",
        default="all",
        type=str,
        required=False,
        help="The report(s) to be generated.",
    )
    yugiquery_parser.add_argument(
        "-t",
        "--telegram-token",
        dest="telegram_token",
        type=str,
        required=False,
        help="Telegram API token.",
    )
    yugiquery_parser.add_argument(
        "-d",
        "--discord-token",
        dest="discord_token",
        type=str,
        required=False,
        help="Discord API token.",
    )
    yugiquery_parser.add_argument(
        "-c",
        "--channel",
        dest="channel_id",
        type=int,
        required=False,
        help="Discord or Telegram Channel/chat ID.",
    )
    yugiquery_parser.add_argument(
        "-s",
        "--suppress-contribs",
        action="store_true",
        required=False,
        help="Disables using TQDM contribs entirely.",
    )
    yugiquery_parser.add_argument(
        "-f",
        "--telegram-first",
        action="store_true",
        required=False,
        help="Force TQDM to try using Telegram as progress bar before Discord.",
    )
    yugiquery_parser.add_argument(
        "--cleanup",
        default="auto",
        type=auto_or_bool,
        nargs="?",
        const=True,
        action="store",
        help="Wether to run the cleanup routine. Options are True, False and 'auto'. Defaults to auto.",
    )
    yugiquery_parser.add_argument(
        "--dryrun",
        action="store_true",
        required=False,
        help="Whether to dry run the cleanup routine. No effect if cleanup is False.",
    )
    yugiquery_parser.add_argument(
        "--debug",
        action="store_true",
        required=False,
        help="Enables debug flag.",
    )

    # Subparser for the bot mode
    bot_parser = subparsers.add_parser("bot", help="Run yugiquery bot")
    bot_parser.add_argument(
        "subclass",
        choices=["discord", "telegram"],
        help="Select between a Discord or a Telegram bot",
    )
    bot_parser.add_argument("-t", "--token", type=str, help="Bot API token")
    bot_parser.add_argument(
        "-c", "--channel", dest="channel_id", type=int, help="Bot responses channel id"
    )
    bot_parser.add_argument("--debug", action="store_true", help="Enable debug flag")

    # Subparser for the kernel installation
    post_install_parser = subparsers.add_parser(
        "install", help="Run post-install script"
    )
    post_install_parser.add_argument(
        "--tqdm",
        action="store_true",
        help="Install tqdm fork for Discord bot.",
    )
    post_install_parser.add_argument(
        "--kernel",
        action="store_true",
        help="Install Jupyter kernel for Yugiquery.",
    )
    post_install_parser.add_argument(
        "--nbconvert",
        action="store_true",
        help="Install nbconvert templates.",
    )
    post_install_parser.add_argument(
        "--all",
        action="store_true",
        help="Install all.",
    )

    # Parse initial arguments
    args = parser.parse_args()

    if args.paths:
        dirs.print()
    elif args.version:
        from .metadata import __version__

        print(f"Yugiquery version {__version__}")
    elif args.command == "install":
        spec = importlib.util.spec_from_file_location(
            name="post_install",
            location=dirs.ASSETS / "scripts" / "post_install.py",
        )
        post_install = importlib.util.module_from_spec(spec=spec)
        spec.loader.exec_module(post_install)
        post_install.main(args)

    elif args.command == "bot":
        # Call the bot main function with parsed arguments
        from .bot import main

        main(args)
    else:
        # Main Yugiquery flow
        from .yugiquery import main

        main(args)


if __name__ == "__main__":
    main()
