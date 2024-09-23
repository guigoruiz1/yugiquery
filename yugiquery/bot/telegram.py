#!/usr/bin/env python3

# yugiquery/bot/telegram.py

# -*- coding: utf-8 -*-

# ======= #
# Imports #
# ======= #

# Standard library packages
import platform

# Third-party imports
import arrow
from termcolor import cprint

# Local application imports
from ..metadata import __version__
from ..utils import get_granularity, escape_chars
from .base import Bot

# Telegram
try:
    import telegram
    from telegram import Update
    from telegram.ext import (
        ApplicationBuilder,
        CommandHandler,
        filters,
        CallbackContext,
    )

except ImportError:
    raise RuntimeError(
        'Missing bot Telegram bot package. Please install the required packages with "pip install yugiquery[telegram]".'
    )

# ===================== #
# Telegram Bot Subclass #
# ===================== #


class Telegram(Bot):
    """
    Telegram bot subclass. Inherits from Bot class.

    Args:
        token (str): The token for the Telegram bot.
        chat_id (str | int): The Telegram chat ID.

    Attributes:
        application (telegram.Application): The Telegram Application bot instance.
        Bot attributes
    """

    def __init__(self, token: str, chat_id: str | int):
        """
        Initialize the Telegram Bot subclass.

        Args:
            token (str): The token for the Telegram bot.
            chat_id (str | int): The chat ID for the Telegram bot.
        """
        from tqdm.contrib.telegram import tqdm as telegram_pbar

        self.telegram_pbar = telegram_pbar
        Bot.__init__(self)
        self.token = token
        self.chat_id = int(chat_id)
        # Initialize the Telegram bot
        self.application = ApplicationBuilder().token(token).build()
        self.register_commands()
        self.register_events()

    def run(self) -> None:
        """
        Start running the Telegram bot.
        """
        cprint(text="Running Telegram bot...", color="green")
        self.application.run_polling(stop_signals=None)

    # ======== #
    # Commands #
    # ======== #

    def register_commands(self) -> None:
        """
        Register command handlers for the Telegram bot.

        Command descriptions to pass to BotFather:

            abort - Aborts a running YugiQuery flow by terminating the process.

            battle - Simulate a battle of all monster cards.

            benchmark - Show average time each report takes to complete.

            data - Send latest data files.

            shutdown - Shutdown bot.

            latest - Show latest time each report was generated.

            links - Show YugiQuery links.

            ping - Test the bot connection latency.

            pull - Pull latest data files from the repository.

            push - Push latest data files to the repository.

            run - Run full YugiQuery flow.

            status - Display bot status and system information.

        """

        async def abort(update: Update, context: CallbackContext) -> None:
            """
            Aborts a running YugiQuery flow by terminating the thread.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            original_response = await context.bot.send_message(chat_id=update.effective_chat.id, text="Aborting...")
            response = self.abort()
            await original_response.edit_text(response)

        async def battle(
            update: Update,
            context: CallbackContext,
        ) -> None:
            """
            Loads the list of all Monster Cards and simulates a battle between them.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            # Create a dictionary with the provided arguments if they exist
            provided_arguments = {}
            if context.args and len(context.args) > 1:
                try:
                    provided_arguments["atk_weight"] = int(context.args[0])
                    provided_arguments["def_weight"] = int(context.args[1])
                except:
                    pass

            original_message = await context.bot.send_message(
                chat_id=update.effective_chat.id, text="Simulating a battle... ⚔️"
            )
            callback_first = None

            async def callback(first) -> None:
                nonlocal callback_first
                callback_first = first
                await original_message.edit_text(
                    escape_chars(f"*First contestant*: {first}\n\nStill battling... ⏳"),
                    parse_mode="MarkdownV2",
                )

            response = await self.battle(callback=callback, **provided_arguments)
            if "error" in response.keys():
                message = response["error"]
            else:
                winner = response["winner"]
                longest = response["longest"]
                message = (
                    f"*First contestant*: {callback_first}\n\n"
                    f"*Winner*: {winner[0]['Name']}\n"
                    f"*Wins*: {winner[1]}\n"
                    f"*Stats remaining*: ATK={winner[0]['ATK']}, DEF={winner[0]['DEF']}\n\n"
                    f"*Longest streak*: {longest[0]['Name']}\n"
                    f"*Wins*: {longest[1]}\n"
                    f"*Stats when defeated*: ATK={longest[0]['ATK']}, DEF={longest[0]['DEF']}"
                )

            message = escape_chars(message)

            await original_message.edit_text(message, parse_mode="MarkdownV2")

        async def benchmark(update: Update, context: CallbackContext) -> None:
            """
            Returns the average time each report takes to complete and the latest time for each report.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            response = self.benchmark()
            if "error" in response.keys():
                await context.bot.send_message(chat_id=update.effective_chat.id, text=response["error"])
                return

            message = f"*{response.pop('title')}*\n{response.pop('description')}\n\n"
            for key, value in response.items():
                message += f"*{key}*\n{value}\n\n"

            message = escape_chars(message)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=message, parse_mode="MarkdownV2")

        async def data(update: Update, context: CallbackContext) -> None:
            """
            Sends the latest data files available in the repository as direct download links.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """

            response = self.data()
            if "error" in response.keys():
                message = response["error"]
            else:
                message = f"*{response['title']}*\n{response['description']}\n\n"
                for field, content in response["fields"].items():
                    message += f"*{field}*:\n{content}\n"

            message = escape_chars(message)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=message, parse_mode="MarkdownV2")

        async def latest(update: Update, context: CallbackContext) -> None:
            """
            Displays the timestamp of the latest local and live reports generated.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            response = self.latest()
            message = f"*{response['title']}*\n{response['description']}\n\n*Local:*\n{response['local']}"
            if "live" in message:
                message += f"\n*Live:*\n{response['live']}"

            message = escape_chars(message)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=message, parse_mode="MarkdownV2")

        async def links(update: Update, context: CallbackContext) -> None:
            """
            Displays the links to the YugiQuery webpage, repository, and data.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            response = self.links()
            message = f"*{response['title']}*\n{response['description']}"
            message = escape_chars(message)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=message, parse_mode="MarkdownV2")

        async def ping(update: Update, context: CallbackContext) -> None:
            """
            Tests the bot's connection latency and sends the result back to the user.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            start_time = arrow.utcnow()
            original_message = await context.bot.send_message(
                chat_id=update.effective_chat.id, text="Calculating latency..."
            )
            end_time = arrow.utcnow()
            latency_ms = (end_time - start_time).total_seconds() * 1e3
            response = f"🏓 Pong! {round(latency_ms,1)}ms"
            await original_message.edit_text(response)

        async def pull(update: Update, context: CallbackContext) -> None:
            """
            Pulls the latest data files from the repository.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            passphrase = context.args[0] if context.args else ""
            response = self.pull(passphrase)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=response)

        async def push(update: Update, context: CallbackContext) -> None:
            """
            Pushes the latest data files to the repository.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            passphrase = context.args[0] if context.args else ""
            response = self.push(passphrase)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=response)

        async def run_query(
            update: Update,
            context: CallbackContext,
        ) -> None:
            """
            Runs a YugiQuery flow by launching a separate thread and monitoring its progress.
            The progress is reported back to the Telegram chat where the command was issued.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            last_run = context.user_data.get("last_run", arrow.get(0.0))
            if (arrow.utcnow() - last_run).total_seconds() < self.cooldown_limit:
                granularity = get_granularity((last_run.shift(seconds=self.cooldown_limit) - arrow.utcnow()).total_seconds())
                next_available = last_run.shift(seconds=self.cooldown_limit).humanize(
                    arrow.utcnow(), granularity=granularity
                )
                await update.effective_message.reply_text(f"You are on cooldown. Try again {next_available}")
                return

            report = (
                self.Reports[context.args[0].capitalize()]
                if context.args and context.args[0].capitalize() in self.Reports.__members__
                else self.Reports.All
            )

            original_response = await context.bot.send_message(chat_id=update.effective_chat.id, text="Initializing...")

            async def callback(content: str) -> None:
                await original_response.edit_text(content)

            response = await self.run_query(
                callback=callback,
                report=report,
                progress_bar=self.telegram_pbar,
                chat_id=update.effective_chat.id,
                token=self.token,
            )
            if "error" in response.keys():
                await context.bot.send_message(chat_id=update.effective_chat.id, text=response["error"])
            else:
                context.user_data["last_run"] = arrow.utcnow()
                await context.bot.send_message(chat_id=update.effective_chat.id, text=response["content"])

        async def status(update: Update, context: CallbackContext) -> None:
            """
            Displays information about the bot, including uptime, versions, and system details.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            app_info = await context.bot.get_me()
            bot_name = app_info.username

            message = (
                f"*Bot name*: {bot_name}\n"
                f"*Uptime*: {self.uptime()}\n"
                f"*Bot Version*: {__version__}\n"
                f"*Telegram Python library Version*: {telegram.__version__}\n"
                f"*Telegram Bot API Version*: {telegram.__bot_api_version__}\n"
                f"*Python Version*: {platform.python_version()}\n"
                f"*Operating System:*\n"
                f" • *Name:* {platform.system()}\n"
                f" • *Release:* {platform.release()}\n"
                f" • *Machine:* {platform.machine()}\n"
                f" • *Version:* {platform.version()}"
            )
            message = escape_chars(message)

            await context.bot.send_message(chat_id=update.effective_chat.id, text=message, parse_mode="MarkdownV2")

        async def shutdown(update: Update, context: CallbackContext) -> None:
            """
            Shuts down the bot gracefully by sending a message and stopping the polling.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Shutting down...")
            self.application.stop_running()

        # Register the command handlers
        self.application.add_handler(CommandHandler(command="abort", callback=abort, block=False))
        self.application.add_handler(CommandHandler(command="battle", callback=battle))
        self.application.add_handler(CommandHandler(command="benchmark", callback=benchmark))
        self.application.add_handler(CommandHandler(command="data", callback=data))
        self.application.add_handler(CommandHandler(command="latest", callback=latest))
        self.application.add_handler(CommandHandler(command="links", callback=links))
        self.application.add_handler(CommandHandler(command="ping", callback=ping))
        self.application.add_handler(CommandHandler(command="pull", callback=pull))
        self.application.add_handler(CommandHandler(command="push", callback=push))
        self.application.add_handler(CommandHandler(command="run", callback=run_query, block=False))
        self.application.add_handler(CommandHandler(command="status", callback=status))
        self.application.add_handler(
            CommandHandler(
                command="shutdown",
                callback=shutdown,
                filters=filters.Chat(chat_id=int(self.chat_id)),
            )
        )

    # ====== #
    # Events #
    # ====== #

    def register_events(self) -> None:
        """
        Register event handlers for the Telegram bot.
        """

        async def start(update: Update, context: CallbackContext) -> None:
            """
            Send a message when the command /start is issued.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            user = update.effective_user
            await update.message.reply_html(
                rf"Hi {user.mention_html()}!",
            )

        async def on_command_error(update: Update, context: CallbackContext) -> None:
            """
            Event that runs whenever a command invoked by the user results in an error.
            Sends a message to the chat indicating the type of error that occurred.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            error = str(context.error)
            print(error)
            if update is not None:
                await update.message.reply_text(error)
            else:
                await context.bot.send_message(chat_id=self.chat_id, text=error)

        self.application.add_handler(CommandHandler(command="start", callback=start))
        self.application.add_error_handler(callback=on_command_error)
