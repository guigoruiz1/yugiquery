# yugiquery/bot/telegram.py

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import platform

# --- Imports: Third-Party --- #
import arrow
from termcolor import cprint

# --- Imports: Local Application --- #
from ..metadata import __version__
from ..utils import get_ts_granularity, escape_chars, LoggerConfig
from .base import Base, GitCommands

# Telegram
try:
    import telegram
    from telegram import Update
    from telegram.ext import (
        Application,
        ApplicationBuilder,
        CommandHandler,
        filters,
        CallbackContext,
    )

except ImportError:
    raise RuntimeError(
        'Missing bot Telegram bot package. Please install the required packages with "pip install yugiquery[telegram]".'
    )


# --- Logger Setup --- #
logger = LoggerConfig.get_logger()


# --- Helper Functions --- #
def _split_run_args(args) -> tuple[list[str], list[str]]:
    """
    Splits arguments for the run command into those for data and report flows.
    Positional args go to both, --data/--report assign specifically.

    Args:
        args (list[str]): The list of arguments to split.

    Returns:
        tuple[list[str], list[str]]: (data_args, report_args) where each is a list of arguments for that

    """
    args = args or []
    data_args = []
    report_args = []
    both_args = []
    mode = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--data":
            mode = "data"
        elif arg == "--report":
            mode = "report"
        elif arg.startswith("--"):
            mode = None  # ignore unknown flags
        else:
            if mode == "data":
                data_args.append(arg)
            elif mode == "report":
                report_args.append(arg)
            else:
                both_args.append(arg)
        i += 1
    data_all_args = both_args + data_args
    report_all_args = both_args + report_args
    return data_all_args, report_all_args


def _parse_args(args, dict_obj) -> tuple[list[str], list[str]]:
    """
    Given a list of arguments and a dictionary, return a tuple:
    (list of valid enum values, list of missing args)
    Tries both key (name) and value (case-insensitive string match).

    Args:
        args (list[str]): The list of arguments to parse.
        dict_obj (dict): The dictionary to match against.

    Returns:
        tuple[list[str], list[str]]: A tuple containing a list of valid enum values and a list of missing arguments that were not recognized.
    """
    if not args:
        return ["all"], []
    found = []
    missing = []
    # If 'all' is present in args (case-insensitive), return only All
    if any(str(arg).lower() == "all" for arg in args):
        return ["all"], []
    for arg in args:
        arg_lc = str(arg).lower()
        # Try to match by enum member name (case-insensitive)
        member = next((m for n, m in dict_obj.items() if n.lower() == arg_lc), None)
        if member is not None:
            found.append(member)
            continue
        # Try to match by value (case-insensitive string match)
        value_match = next((v for v in dict_obj.values() if str(v).lower() == arg_lc), None)
        if value_match is not None:
            found.append(value_match)
        else:
            missing.append(arg)

    return found, missing


# --- Telegram Bot Class Definition --- #
class Telegram(Base):
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
        Base.__init__(self)
        self.token = token
        self.chat_id = int(chat_id)
        # Initialize the Telegram bot
        self.application = ApplicationBuilder().token(token).post_init(post_init=self.post_init_handler).build()
        self.register_commands()
        self.register_events()

    async def post_init_handler(self, application: Application) -> None:
        """
        Post-initialization handler for the Telegram bot.

        Args:
            application (telegram.ext.Application): The Telegram Application bot instance.
        """
        me = await application.bot.get_me()
        chat = await application.bot.get_chat(self.chat_id)
        username = f"{chat.first_name} {chat.last_name}" if chat.first_name and chat.last_name else chat.username
        await application.bot.send_message(chat_id=self.chat_id, text=f"Hello {username}!\n{me.first_name} bot is online.")
        cprint(text="Telegram bot initialized successfully.", color="green")
        logger.info("Telegram bot initialized successfully.")

    async def _check_cooldown(self, update, context, cooldown_key="last_run"):
        """
        Checks if the user is on cooldown for a specific command. If so, sends a cooldown message and returns True. Otherwise, returns False.
        """
        user_data = context.user_data or {}
        last = user_data.get(cooldown_key, arrow.get(0.0))
        if (arrow.utcnow() - last).total_seconds() < self.cooldown_limit:
            if update.effective_message is None:
                return True
            granularity = get_ts_granularity((last.shift(seconds=self.cooldown_limit) - arrow.utcnow()).total_seconds())
            next_available = last.shift(seconds=self.cooldown_limit).humanize(arrow.utcnow(), granularity=granularity)
            await update.effective_message.reply_text(f"You are on cooldown. Try again {next_available}")
            return True
        return False

    async def _handle_query_response(self, response, context, update, cooldown_key="last_run"):
        """
        Handles the response from a query, updating cooldown and sending the appropriate message.
        """
        user_data = context.user_data or {}
        if "error" in response.keys():
            await context.bot.send_message(chat_id=update.effective_chat.id, text=response["error"])
        else:
            user_data[cooldown_key] = arrow.utcnow()
            context.user_data = user_data
            await context.bot.send_message(chat_id=update.effective_chat.id, text=response["content"])

    def run(self) -> None:
        """
        Start running the Telegram bot.
        """
        cprint(text="Running Telegram bot...", color="green")
        logger.info("Running Telegram bot...")
        self.application.run_polling(stop_signals=None)

    def command_handler(self, command, **kwargs):
        """
        Decorator to register a command handler for the Telegram bot.

        Args:
            command (str): The command to register.
            **kwargs: Additional keyword arguments to pass to the CommandHandler.

        Returns:
            function: The decorated function.
        """

        def decorator(func):
            handler = CommandHandler(command=command, callback=func, **kwargs)
            self.application.add_handler(handler)
            return func

        return decorator

    # --- Commands --- #

    def register_commands(self) -> None:
        """
        Register command handlers for the Telegram bot.

        Command descriptions to pass to BotFather:
            abort - Aborts a running YugiQuery flow by terminating the process.
            battle - Simulate a battle of all monster cards.
            benchmark - Show average time each report takes to complete.
            data - Send latest data files.
            git - Run a Git command.
            latest - Show latest time each report was generated.
            links - Show YugiQuery links.
            ping - Test the bot connection latency.
            run - Run full YugiQuery flow.
            status - Display bot status and system information.
            shutdown - Shutdown bot.

        """

        @self.command_handler("abort", block=False, filters=filters.Chat(chat_id=int(self.chat_id)))
        async def abort(update: Update, context: CallbackContext) -> None:
            """
            Aborts a running YugiQuery flow by terminating the thread.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            if update.effective_chat is None:
                return
            original_response = await context.bot.send_message(chat_id=update.effective_chat.id, text="Aborting...")
            response = self.abort()
            await original_response.edit_text(response)

        @self.command_handler("battle")
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
            if update.effective_chat is None:
                return

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

        @self.command_handler("benchmark")
        async def benchmark(update: Update, context: CallbackContext) -> None:
            """
            Returns the average time each report takes to complete and the latest time for each report.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            if update.effective_chat is None:
                return

            response = self.benchmark()
            if "error" in response.keys():
                await context.bot.send_message(chat_id=update.effective_chat.id, text=response["error"])
                return

            message = f"*{response.pop('title')}*\n{response.pop('description')}\n\n"
            for key, value in response.items():
                message += f"*{key}*\n{value}\n\n"

            message = escape_chars(message)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=message, parse_mode="MarkdownV2")

        @self.command_handler("data")
        async def data(update: Update, context: CallbackContext) -> None:
            """
            Sends the latest data files available in the repository as direct download links.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            if update.effective_chat is None:
                return

            response = self.data()
            if "error" in response.keys():
                message = response["error"]
            else:
                message = f"*{response['title']}*\n{response['description']}\n\n"

                fields = response["fields"]
                if isinstance(fields, dict):
                    for field, content in fields.items():
                        message += f"*{field}*:\n{content}\n"

            message = escape_chars(message)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=message, parse_mode="MarkdownV2")

        @self.command_handler("git", has_args=True, filters=filters.Chat(chat_id=int(self.chat_id)))
        async def git_cmd(update: Update, context: CallbackContext) -> None:
            """
            Handles git-related commands sent via Telegram.

            This asynchronous handler function processes incoming Telegram messages that invoke git commands.
            It validates the command, extracts arguments, executes the corresponding git operation, and sends the result back to the user.

            Args:
                update (telegram.Update): The incoming Telegram update containing the message and chat information.
                context (telegram.ext.CallbackContext): The context object containing arguments and bot instance.
            """
            if update.effective_chat is None:
                return

            if not context.args or context.args[0] not in GitCommands.__members__:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Invalid command")
                return

            command = GitCommands[context.args[0]]
            passphrase = context.args[1] if len(context.args) > 1 else ""
            response = self.git_cmd(command=command, passphrase=passphrase)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=response)

        @self.command_handler("latest")
        async def latest(update: Update, context: CallbackContext) -> None:
            """
            Displays the timestamp of the latest local and live reports generated.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            if update.effective_chat is None:
                return

            response = self.latest()
            message = f"*{response['title']}*\n{response['description']}\n\n*Local:*\n{response['local']}"
            if "live" in message:
                message += f"\n*Live:*\n{response['live']}"

            message = escape_chars(message)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=message, parse_mode="MarkdownV2")

        @self.command_handler("links")
        async def links(update: Update, context: CallbackContext) -> None:
            """
            Displays the links to the YugiQuery webpage, repository, and data.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            if update.effective_chat is None:
                return

            response = self.links()
            message = f"*{response['title']}*\n{response['description']}"
            message = escape_chars(message)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=message, parse_mode="MarkdownV2")

        @self.command_handler("ping")
        async def ping(update: Update, context: CallbackContext) -> None:
            """
            Tests the bot's connection latency and sends the result back to the user.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            if update.effective_chat is None:
                return

            start_time = arrow.utcnow()
            original_message = await context.bot.send_message(
                chat_id=update.effective_chat.id, text="Calculating latency..."
            )
            end_time = arrow.utcnow()
            latency_ms = (end_time - start_time).total_seconds() * 1e3
            response = f"🏓 Pong! {round(latency_ms,1)}ms"
            await original_message.edit_text(response)

        @self.command_handler("run", block=False, filters=filters.Chat(chat_id=int(self.chat_id)))
        async def run_query(update: Update, context: CallbackContext) -> None:
            """
            Runs a YugiQuery flow by launching a separate thread and monitoring its progress.
            The progress is reported back to the Telegram chat where the command was issued.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            cooldown_key = "last_run"
            if await self._check_cooldown(update, context, cooldown_key):
                return
            if update.effective_chat is None:
                return

            # Use helper to split args for run
            data_all_args, report_all_args = _split_run_args(context.args)
            data_flows, missing_data = _parse_args(data_all_args, self.DataFlows)
            report_flows, missing_report = _parse_args(report_all_args, self.Reports)

            missing = []
            if missing_data:
                missing.append(f"Data: {', '.join(missing_data)}")
            if missing_report:
                missing.append(f"Report: {', '.join(missing_report)}")

            initial_message = "Initializing..."
            if missing:
                initial_message += (
                    f"\n\nNote: The following arguments were not recognized and will be ignored: {'; '.join(missing)}"
                )
            original_response = await context.bot.send_message(chat_id=update.effective_chat.id, text=initial_message)

            async def callback(content: str) -> None:
                await original_response.edit_text(content)

            response = await self.query_run(
                callback=callback,
                data=data_flows,
                report=report_flows,
                progress_bar=self.telegram_pbar,
                chat_id=update.effective_chat.id,
                token=self.token,
            )
            await self._handle_query_response(response, context, update, cooldown_key)

        @self.command_handler("fetch", block=False, filters=filters.Chat(chat_id=int(self.chat_id)))
        async def run_fetch(update: Update, context: CallbackContext) -> None:
            """
            Runs the YugiQuery data fetch operation by launching a separate thread and monitoring its progress.
            The progress is reported back to the Telegram chat where the command was issued.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            cooldown_key = "last_fetch"
            if await self._check_cooldown(update, context, cooldown_key):
                return
            if update.effective_chat is None:
                return

            flows, missing = _parse_args(context.args, self.DataFlows)

            initial_message = "Initializing..."
            if missing:
                initial_message += (
                    f"\n\nNote: The following data flows were not recognized and will be ignored: {', '.join(missing)}"
                )
            original_response = await context.bot.send_message(chat_id=update.effective_chat.id, text=initial_message)

            async def callback(content: str) -> None:
                await original_response.edit_text(content)

            response = await self.query_fetch(
                callback=callback,
                data=flows,
                progress_bar=self.telegram_pbar,
                chat_id=update.effective_chat.id,
                token=self.token,
            )

            await self._handle_query_response(response, context, update, cooldown_key)

        @self.command_handler("report", block=False, filters=filters.Chat(chat_id=int(self.chat_id)))
        async def run_report(update: Update, context: CallbackContext) -> None:
            """
            Runs the YugiQuery report generation operation by launching a separate thread and monitoring its progress.
            The progress is reported back to the Telegram chat where the command was issued.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            cooldown_key = "last_report"
            if await self._check_cooldown(update, context, cooldown_key):
                return
            if update.effective_chat is None:
                return

            report, missing = _parse_args(context.args, self.DataFlows)

            initial_message = "Initializing..."
            if missing:
                initial_message += (
                    f"\n\nNote: The following reports were not recognized and will be ignored: {', '.join(missing)}"
                )
            original_response = await context.bot.send_message(chat_id=update.effective_chat.id, text=initial_message)

            async def callback(content: str) -> None:
                await original_response.edit_text(content)

            response = await self.query_report(
                callback=callback,
                report=report,
                progress_bar=self.telegram_pbar,
                chat_id=update.effective_chat.id,
                token=self.token,
            )
            await self._handle_query_response(response, context, update, cooldown_key)

        @self.command_handler("status")
        async def status(update: Update, context: CallbackContext) -> None:
            """
            Displays information about the bot, including uptime, versions, and system details.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            if update.effective_chat is None:
                return

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

        @self.command_handler("shutdown", filters=filters.Chat(chat_id=int(self.chat_id)))
        async def shutdown(update: Update, context: CallbackContext) -> None:
            """
            Shuts down the bot gracefully by sending a message and stopping the polling.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            if update.effective_chat is None:
                return

            await context.bot.send_message(chat_id=update.effective_chat.id, text="Shutting down...")
            self.application.stop_running()

    # --- Events --- #

    def register_events(self) -> None:
        """
        Register event handlers for the Telegram bot.
        """

        async def on_command_error(update: object, context: CallbackContext) -> None:
            """
            Event that runs whenever a command invoked by the user results in an error.
            Sends a message to the chat indicating the type of error that occurred.

            Args:
                update (object): An object representing the update that caused the error.
                context (telegram.ext.CallbackContext): The callback context.
            """
            error = str(context.error)
            logger.error("%s", error)
            print(error)
            if isinstance(update, Update) and update.message is not None:
                await update.message.reply_text(error)
            else:
                await context.bot.send_message(chat_id=self.chat_id, text=error)

        self.application.add_error_handler(callback=on_command_error)
