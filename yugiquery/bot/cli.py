# yugiquery/bot/cli.py

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import argparse
import sys
import multiprocessing as mp
from typing import (
    Any,
    Tuple,
)

# --- Imports: Local Application --- #
from ..utils import dirs, LoggerConfig, load_secrets
from ..cli import set_debug_args, CustomHelpFormatter

# --- Logger Config --- #
logger = LoggerConfig.get_logger()


def set_parser(target: argparse._SubParsersAction | argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Configure the argument parser for the bot mode, including required arguments for bot subclass, token, and channel ID, as well as debug arguments."""
    desc = "Run YugiQuery bot"
    if target is None:
        parser = argparse.ArgumentParser(description=desc, formatter_class=CustomHelpFormatter)
    elif isinstance(target, argparse._SubParsersAction):
        parser = target.add_parser("bot", description=desc, formatter_class=CustomHelpFormatter)
    else:
        parser = target

    parser.add_argument(
        "subclass",
        choices=["discord", "telegram"],
        help="select between a Discord or a Telegram bot",
    )
    parser.add_argument("-t", "--token", type=str, help="bot API token")
    parser.add_argument(
        "-c",
        "--channel",
        "--chat",
        dest="ch",
        type=int,
        help="bot responses Channel/Chat ID",
    )
    set_debug_args(parser, log_only=True)

    return parser


def _load_secrets_with_args(args: Any) -> Tuple[str, int | str]:
    """
    Load secrets from command-line arguments, and update them with values from
    environment variables or a .env file, placed in the `Assets` directory, if necessary.
    If the required secrets are not found, the function will exit the program.

    Args:
        args (Any): The parsed command-line arguments.

    Returns:
        (str, int): The token and channel ID.
    """
    subclass_upper = args.subclass.upper()
    if subclass_upper == "DISCORD":
        ch_key = "CHANNEL_ID"
    elif subclass_upper == "TELEGRAM":
        ch_key = "CHAT_ID"

    secrets_args = {
        f"{subclass_upper}_TOKEN": args.token,
    }
    secrets_args[f"{subclass_upper}_{ch_key}"] = args.ch

    secrets = {key: value for key, value in secrets_args.items() if value}
    missing = [key for key, value in secrets_args.items() if not value]

    if missing:
        loaded_secrets = load_secrets(
            requested_secrets=missing,
            secrets_file=dirs.secrets_file,
            required=True,
        )
        secrets.update(loaded_secrets)

    tkn = secrets[f"{subclass_upper}_TOKEN"]
    ch = secrets[f"{subclass_upper}_{ch_key}"]

    return tkn, ch


# --- Main Entry Point --- #
def main(args: argparse.Namespace) -> None:
    """Main entry point for the bot mode. Initializes the bot based on the specified subclass and runs it."""
    LoggerConfig.setup(level=args.log_level, log_file=args.log_file)
    # Set multiprocessing start method
    mp.set_start_method("spawn")

    # Make sure the data and reports directories exist
    dirs.make()

    # Load secrets
    try:
        tkn, ch = _load_secrets_with_args(args)
    except KeyError as e:
        logger.error("%s. Aborting...", e)
        sys.exit(1)

    # Handle bots based on subclass
    if args.subclass == "discord":
        # Initialize and run the Discord bot
        from .discord import Discord as Subclass

        bot = Subclass(token=tkn, channel_id=ch)

    elif args.subclass == "telegram":
        # Initialize and run the Telegram bot
        from .telegram import Telegram as Subclass

        bot = Subclass(token=tkn, chat_id=ch)

    # Run the bot subclass
    bot.run()
