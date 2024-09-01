# yugiquery/__main__.py

import argparse
from .utils import auto_or_bool


def main():
    # Create the primary parser
    parser = argparse.ArgumentParser(description="Yugiquery CLI tool")

    # Add a flag to switch to bot mode
    parser.add_argument(
        "-b", "--bot", action="store_true", help="Run bot instead of yugiquery"
    )

    # Arguments common to both yugiquery and bot can go here if any.

    # Parse initial arguments
    args, remaining_args = parser.parse_known_args()

    if args.bot:
        # Bot mode parser
        bot_parser = argparse.ArgumentParser(description="Bot commands")
        bot_parser.add_argument(
            "subclass",
            choices=["discord", "telegram"],
            help="Select between a Discord or a Telegram bot",
        )
        bot_parser.add_argument("-t", "--token", type=str, help="Bot API token")
        bot_parser.add_argument(
            "-c",
            "--channel",
            dest="channel_id",
            type=int,
            help="Bot responses channel id",
        )
        bot_parser.add_argument(
            "--debug", action="store_true", help="Enable debug flag"
        )
        bot_args = bot_parser.parse_args(remaining_args)
        print(bot_args)

        # Call the bot main function with parsed arguments
        from .bot import main

        main(bot_args)

    else:
        # Yugiquery mode parser
        yugiquery_parser = argparse.ArgumentParser(description="Yugiquery tasks")
        yugiquery_parser.add_argument(
            "-r",
            "--reports",
            nargs="+",
            dest="reports",
            default="all",
            type=str,
            help="The report(s) to be generated.",
        )
        yugiquery_parser.add_argument(
            "-t",
            "--telegram-token",
            dest="telegram_token",
            type=str,
            help="Telegram API token.",
        )
        yugiquery_parser.add_argument(
            "-d",
            "--discord-token",
            dest="discord_token",
            type=str,
            help="Discord API token.",
        )
        yugiquery_parser.add_argument(
            "-c",
            "--channel",
            dest="channel_id",
            type=int,
            help="Discord or Telegram Channel/chat ID.",
        )
        yugiquery_parser.add_argument(
            "-s",
            "--suppress-contribs",
            action="store_true",
            help="Disables using TQDM contribs entirely.",
        )
        yugiquery_parser.add_argument(
            "-f",
            "--telegram-first",
            action="store_true",
            help="Force TQDM to try using Telegram as progress bar before Discord.",
        )
        yugiquery_parser.add_argument(
            "--cleanup",
            default="auto",
            type=auto_or_bool,
            nargs="?",
            const=True,
            action="store",
            help="Whether to run the cleanup routine. Options are True, False, and 'auto'. Defaults to auto.",
        )
        yugiquery_parser.add_argument(
            "--dryrun",
            action="store_true",
            help="Whether to dry run the cleanup routine. No effect if cleanup is False.",
        )
        yugiquery_parser.add_argument(
            "--debug", action="store_true", help="Enables debug flag."
        )
        yugiquery_args = yugiquery_parser.parse_args(remaining_args)

        # Call the yugiquery main function with parsed arguments
        from .yugiquery import main

        main(yugiquery_args)


if __name__ == "__main__":
    main()
