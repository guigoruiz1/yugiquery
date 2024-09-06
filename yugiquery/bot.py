#!/usr/bin/env python3

# yugiquery/bot.py

# -*- coding: utf-8 -*-

# ======================================================================== #
#                                                                          #
# ██    ██ ██    ██  ██████  ██  ██████  ██    ██ ███████ ██████  ██    ██ #
#  ██  ██  ██    ██ ██       ██ ██    ██ ██    ██ ██      ██   ██  ██  ██  #
#   ████   ██    ██ ██   ███ ██ ██    ██ ██    ██ █████   ██████    ████   #
#    ██    ██    ██ ██    ██ ██ ██ ▄▄ ██ ██    ██ ██      ██   ██    ██    #
#    ██     ██████   ██████  ██  ██████   ██████  ███████ ██   ██    ██    #
#                                   ▀▀                                     #
# ======================================================================== #

# ======= #
# Imports #
# ======= #

import yugiquery as yq

if __package__:
    from .utils import *
else:
    from utils import *

# Native python packages
import argparse
import asyncio
from glob import glob
import io
import multiprocessing as mp
import os
import platform
import random
import subprocess
from enum import Enum

# PIP packages - installed by yugiquery
import pandas as pd

# Telegram
import telegram
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    filters,
    CallbackContext,
)

# Discord
import discord
from discord.ext import commands

# Silence discord.py pynacl optional dependency warning.
discord.VoiceClient.warn_nacl = False


# ============== #
# Helper methods #
# ============== #


# Data loaders
def load_secrets_with_args(args: Dict[str, Any]):
    """
    Load secrets from command-line arguments, and update them with values from
    environment variables or a .env file if necessary.

    Args:
        dict[str: Any]: A dictionary of secrets.

    Returns:
        Dict[str, str]: A dictionary containing the loaded secrets.

    Raises:
        KeyError: If a required secret is not found in the loaded secrets.
    """

    secrets = {key: value for key, value in args.items() if value is not None}
    missing = [key for key, value in args.items() if value is None]
    if len(missing) > 0:
        try:
            loaded_secrets = yq.load_secrets(
                requested_secrets=missing,
                secrets_file=yq.SECRETS_FILE,
                required=True,
            )
        except:
            print("Secrets not found. Exiting...")
            exit()

        secrets = secrets | loaded_secrets

    return secrets


def escape_chars(string: str, chars: List[str] = ["_", ".", "-", "+", "#", "@", "="]):
    """
    Escapes specified characters in a given string by adding a backslash before each occurrence.

    Args:
        string (str): The input string to be processed.
        chars (list, optional): A list of characters to be escaped. Default is ["_", ".", "-", "+", "#", "@", "="].

    Returns:
        str: The input string with the specified characters escaped.
    """
    for char in chars:
        string = string.replace(char, "\\" + char)
    return string


def get_humanize_granularity(seconds: int):
    """
    Humanizes a time interval given in seconds.

    Args:
        seconds (int): The time interval in seconds.

    Returns:
        list: A list of human-readable granularities for the time interval.
    """
    granularities = [
        ("year", 31536000),  # seconds in a year
        ("quarter", 7776000),  # seconds in a quarter
        ("month", 2592000),  # seconds in a month
        ("week", 604800),  # seconds in a week
        ("day", 86400),  # seconds in a day
        ("hour", 3600),  # seconds in an hour
        ("minute", 60),  # seconds in a minute
        ("second", 1),
    ]

    selected_granularity = []

    for granularity, divisor in granularities:
        value = seconds // divisor
        if value > 0:
            selected_granularity.append(granularity)
            seconds %= divisor

    return selected_granularity


# ====================== #
# Progress handler class #
# ====================== #


class ProgressHandler:
    """
    Progress handler class.

    Args:
        queue (multiprocessing.Queue): The multiprocessing queue to communicate progress status.
        progress_bar (tqdm, optional): The tqdm progress bar implementation. Defaults to None.
        pbar_kwargs (Dict[str, Any], optional): Keyword arguments to customize the progress bar. Defaults to None.

    Attributes:
        queue (multiprocessing.Queue): The multiprocessing queue to communicate progress status.
        progress_bar (tqdm, optional): The tqdm progress bar implementation.
        pbar_kwargs (Dict[str, Any], optional): Keyword arguments to customize the progress bar.
    """

    def __init__(
        self,
        queue: mp.Queue,
        progress_bar: tqdm = None,
        pbar_kwargs: Dict[str, Any] = {},
    ):
        """
        Initializes the ProgressHandler class.

        Args:
            queue (multiprocessing.Queue): The multiprocessing queue to communicate progress status.
            progress_bar (tqdm, optional): The tqdm progress bar implementation. Defaults to None.
            pbar_kwargs (Dict[str, Any], optional): Keyword arguments to customize the progress bar. Defaults to None.
        """
        self.queue = queue
        self.progress_bar = progress_bar
        self.pbar_kwargs = pbar_kwargs

    def pbar(self, iterable, **kwargs):
        """
        Initializes and returns a progress bar instance if progress_bar is not None.

        Args:
            iterable (iterable): The iterable to track progress.
            **kwargs: Additional keyword arguments for the progress bar.

        Returns:
            Progress bar instance or None: The initialized progress bar instance or None if progress_bar is None.
        """
        if self.progress_bar is None:
            return None
        else:
            return self.progress_bar(
                iterable, file=io.StringIO(), **self.pbar_kwargs, **kwargs
            )

    def exit(self, API_status: bool = True):
        """
        Puts the API status in the queue.

        Args:
            API_status (bool, optional): The status to put in the queue. Defaults to True.
        """
        self.queue.put(API_status)


# ============== #
# Bot Superclass #
# ============== #


class Bot:
    """
    Bot superclass.

    Args:
        token (str): The token for bot authentication.
        channel (int): The bot channel ID.
        **kwargs: Additional keyword arguments.

    Attributes:
        start_time (arrow.Arrow): The bot initialization timestamp.
        token (str): The token for bot authentication.
        channel (int): The bot channel ID.
        has_remote (bool): Whether the repository has a remote.
        repo (git.Repo): The git repository object, if has_remote.
        repository_api_url (str): The GitHub API URL for the remote repository, if has_remote.
        repository_url (str): The URL of the remote repository, if has_remote.
        webpage_url (str): The URL of the GitHub pages webpage for the remote repository, if has_remote.
        process (multiprocessing.Process): The variable to hold a process spawned by run_query.
        cooldown_limit (Tnt): The cooldown time in seconds to wait between consecutive calls to run_query.

    """

    def __init__(self, token: str, channel: int, **kwargs):
        """
        Initializes the Bot base class.

        Args:
            token (str): The token for the Telegram bot.
            channel (int): The Telegram channel ID.
            **kwargs: Additional keyword arguments.
        """
        self.start_time = arrow.utcnow()
        self.token = token
        self.channel = int(channel)
        self.init_reports_enum()
        self.load_repo_vars()

    # Other
    process = None
    Reports = Enum("Reports", {"All": "all"})
    cooldown_limit = 12 * 3600  # 12 hours

    # ======================== #
    # Bot Superclass Functions #
    # ======================== #

    def load_repo_vars(self):
        """
        Load repository variables including URLs and paths.
        """
        try:
            # Open the repository
            self.repo = yq.git.assure_repo()
            # Get the remote repository
            remote = self.repo.remote()
            remote_url = remote.url

            # Extract the GitHub page URL from the remote URL
            # by removing the ".git" suffix and splitting the URL
            # by the "/" character
            remote_url_parts = remote_url[:-4].split("/")

            # Repository
            (author, repo) = remote_url_parts[-2:]
            # URLs
            self.repository_api_url = f"https://api.github.com/repos/{author}/{repo}"
            self.repository_url = f"https://github.com/{author}/{repo}"
            self.webpage_url = f"https://{author}.github.io/{repo}"

            self.has_remote = True
        except:
            self.has_remote = False

    def init_reports_enum(self):
        """
        Initializes and returns an Enum object containing the available reports.
        The reports are read from the dirs.NOTEBOOKS directory, where they are expected to be Jupyter notebooks.
        The Enum object is created using the reports' file names, with the .ipynb extension removed and the first letter capitalized.

        Returns:
            Enum: An Enum object containing the available reports.
        """
        reports_dict = {"All": "all"}
        reports = sorted(glob(dirs.NOTEBOOKS / "*.ipynb"))
        for report in reports:
            reports_dict[os.path.basename(report)[:-6].capitalize()] = report

        self.Reports = Enum("Reports", reports_dict)

    def abort(self):
        """
        Aborts a running YugiQuery flow by terminating the process.

        Returns:
            str: The result message indicating whether the abortion was successful.
        """
        try:
            self.process.terminate()
            if self.has_remote:
                self.repo.git.restore(dirs.NOTEBOOKS / "*.ipynb")
            return "Aborted"
        except:
            return "Abort failed"

    async def battle(
        self, callback: Callable, atk_weight: int = 4, def_weight: int = 1
    ):
        """
        This function loads the list of all Monster Cards and simulates a battle between them. Each card is represented by its name, attack (ATK), and defense (DEF) stats. At the beginning of the battle, a random card is chosen as the initial contestant. Then, for each subsequent card, a random stat (ATK or DEF) is chosen to compare with the corresponding stat of the current winner. If the challenger's stat is higher, the challenger becomes the new winner. If the challenger's stat is lower, the current winner retains its position. If the stats are tied, the comparison is repeated with the other stat. The battle continues until there is only one card left standing.

        Args:
            callback: Callable: A callback function which receives a string argument.
            atk_weight (int, optional): The weight to use for the ATK stat when randomly choosing the monster's stat to compare. This affects the probability that ATK will be chosen over DEF. The default value is 4.
            def_weight (int, optional): The weight to use for the DEF stat when randomly choosing the monster's stat to compare. This affects the probability that DEF will be chosen over ATK. The default value is 1.

        Returns:
            dict: A dictionary containing information about the battle.
        """
        MONSTER_STATS = ["Name", "ATK", "DEF"]
        cards_files = sorted(
            glob(dirs.DATA / "cards_data_*.bz2"),
            key=os.path.getmtime,
        )
        if not cards_files:
            return {"error": "Cards data not found... Try again later."}

        cards = pd.read_csv(cards_files[0])
        weights = [atk_weight, def_weight]
        monsters = cards[
            (cards["Card type"] == "Monster Card")
            & (cards["Primary type"] != "Monster Token")
        ][MONSTER_STATS].set_index("Name")
        monsters = monsters.map(
            lambda x: x if x != "?" else random.randrange(0, 51) * 100
        )
        monsters = (
            monsters.map(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(int)
            .reset_index()
        )

        # Shuffle the monsters and select the first one as the initial winner
        monsters = monsters.sample(frac=1).reset_index(drop=True)
        winner = (monsters.iloc[0], 0)
        longest = (monsters.iloc[0], 0)

        await callback(winner[0]["Name"])

        for i in range(1, len(monsters)):
            current_winner = (winner[0].copy(), winner[1])
            next_monster = monsters.iloc[i].copy()
            chosen_stat = random.choices(MONSTER_STATS[1:], weights=weights)[0]
            not_chosen_stat = [
                stat for stat in MONSTER_STATS[1:] if stat != chosen_stat
            ][0]
            if next_monster[chosen_stat] > current_winner[0][chosen_stat]:
                next_monster[chosen_stat] -= current_winner[0][chosen_stat]
                if current_winner[1] > longest[1]:
                    longest = current_winner
                current_winner = (next_monster, 0)
            elif next_monster[chosen_stat] < current_winner[0][chosen_stat]:
                current_winner[0][chosen_stat] -= next_monster[chosen_stat]
            elif next_monster[chosen_stat] == current_winner[0][chosen_stat]:
                if next_monster[not_chosen_stat] > current_winner[0][not_chosen_stat]:
                    next_monster[not_chosen_stat] -= current_winner[0][not_chosen_stat]
                    if current_winner[1] > longest[1]:
                        longest = current_winner
                    current_winner = (next_monster, 0)
                elif next_monster[not_chosen_stat] < current_winner[0][not_chosen_stat]:
                    current_winner[0][not_chosen_stat] -= next_monster[not_chosen_stat]

            winner = (current_winner[0], current_winner[1] + 1)

        return {"winner": winner, "longest": longest}

    def benchmark(self):
        """
        Returns the average time each report takes to complete and the latest time for each report.

        Returns:
            dict: A dictionary containing benchmark information.
        """
        try:
            with open(dirs.DATA / "benchmark.json", "r") as file:
                data = json.load(file)
        except:
            return {
                "error": "Unable to find benchmark records at this time. Try again later."
            }

        response = {
            "title": "Benchmark",
            "description": "The average time each report takes to complete",
        }

        # Get benchmark
        value = ""
        for key, values in data.items():
            weighted_sum = 0
            total_weight = 0
            for entry in values:
                weighted_sum += entry["average"] * entry["weight"]
                total_weight += entry["weight"]

            avg_time = weighted_sum / total_weight
            latest_time = entry["average"]

            avg_time_str = (
                arrow.now()
                .shift(seconds=avg_time)
                .humanize(
                    granularity=get_humanize_granularity(avg_time), only_distance=True
                )
            )
            latest_time_str = (
                arrow.now()
                .shift(seconds=latest_time)
                .humanize(
                    granularity=get_humanize_granularity(latest_time),
                    only_distance=True,
                )
            )

            value = f"• Entries: {total_weight}\n• Average: {avg_time_str}\n• Latest: {latest_time_str}"
            response[key.capitalize()] = value

        return response

    def data(self):
        """
        Sends the latest data files available in the repository as direct download links.

        Returns:
            dict: A dictionary containing direct links to the latest data files.
        """
        if not self.has_remote:
            return {"error": "No github repository."}
        try:
            files = pd.read_json(f"{self.repository_api_url}/contents/data")
            files = files[
                files["name"].str.endswith(".bz2")
            ]  # Remove .json files from lists
            files[["Group", "Timestamp"]] = (
                files["name"]
                .str.extract(
                    r"(\w+_\w+)_(.*)(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})Z.bz2", expand=True
                )
                .drop(1, axis=1)
            )
            files["Timestamp"] = pd.to_datetime(files["Timestamp"], utc=True)
            index = files.groupby("Group")["Timestamp"].idxmax()
            latest_files = files.loc[index, ["name", "download_url"]]

            data_value = ""
            changelog_value = ""
            for _, file in latest_files.iterrows():
                if "changelog" in file["name"]:
                    changelog_value += f'• [{file["name"]}]({file["download_url"]})\n'
                else:
                    data_value += f'• [{file["name"]}]({file["download_url"]})\n'

            response = {
                "title": "Latest data files",
                "description": "Direct links to download files from GitHub",
                "data": data_value,
                "changelog": changelog_value,
            }
            return response
        except:
            return {
                "error": "Unable to obtain the latest files at this time. Try again later."
            }

    def latest(self):
        """
        Displays the timestamp of the latest local and live reports generated.
        Reads the report files from `dirs.REPORTS` and queries the GitHub API
        for the latest commit timestamp for each file. Returns the result as an
        message in the channel.

        Returns:
            dict: A dictionary containing information about the latest reports.
        """

        reports = sorted(glob(dirs.REPORTS / "*.html"))
        response = {
            "title": "Latest reports generated",
            "description": "The live reports may not always be up to date with the local reports",
        }

        # Get local files timestamps
        local_value = ""
        for report in reports:
            local_value += f'• {os.path.basename(report).split(".html")[0]}: {pd.to_datetime(os.path.getmtime(report),unit="s", utc=True).strftime("%d/%m/%Y %H:%M %Z")}\n'

        response["local"] = local_value

        # Get live files timestamps
        if self.has_remote:
            try:
                live_value = ""
                for report in reports:
                    result = pd.read_json(
                        f"{self.repository_api_url}/commits?path={os.path.basename(report)}"
                    )
                    timestamp = pd.DataFrame(result.loc[0, "commit"]).loc[
                        "date", "author"
                    ]
                    live_value += f'• {os.path.basename(report).split(".html")[0]}: {pd.to_datetime(timestamp, utc=True).strftime("%d/%m/%Y %H:%M %Z")}\n'

                response["live"] = live_value
            except:
                pass

        return response

    def links(self):
        """
        Displays the links to the YugiQuery webpage, repository, and data.

        Returns:
            dict: A dictionary containing links to YugiQuery resources.
        """
        if self.has_remote:
            description = f"[Webpage]({self.webpage_url}) • [Repository]({self.repository_url}) • [Data]({self.repository_url}/tree/main/data)"
        else:
            description = "No github repository."
        response = {
            "title": "YugiQuery links",
            "description": description,
        }

        return response

    async def run_query(
        self,
        callback: Callable,
        channel_id: int,
        report: Reports = Reports.All,
        progress_bar: tqdm = None,
    ):
        """
        Runs a YugiQuery flow by launching a separate thread and monitoring its progress.

        Args:
            callback (Callable): A callback function which receives a string argument.
            channel_id (int): The channel ID to display the progress bar.
            report (Reports, optional): The report to run. Defaults to All.
            progress_bar (tqdm, optional): A tqdm progress bar. Defaults to None.

        Returns:
            dict: A dictionary containing the result of the query execution.
        """
        if self.process is not None:
            return {"error": "Query already running. Try again after it has finished."}

        queue = mp.Queue()
        stderr_read, stderr_write = mp.Pipe(duplex=False)  # Create a pipe for stderr

        if isinstance(self, Discord):
            pbar_kwargs = {"channel_id": channel_id, "token": self.token}
        elif isinstance(self, Telegram):
            pbar_kwargs = {"chat_id": channel_id, "token": self.token}

        progress_handler = ProgressHandler(
            queue=queue,
            progress_bar=(
                progress_bar
                if ((channel_id != self.channel) or isinstance(self, Telegram))
                else None
            ),
            pbar_kwargs=pbar_kwargs,
        )
        try:
            self.process = mp.Process(
                target=yq.run,
                args=[report.value, progress_handler],  # isinstance(self,Telegram)
            )
            self.process.start()
            stderr_write.close()  # Close the write end in the parent process to ensure it only reads
            await callback("Running...")
        except Exception as e:
            print(e)
            await callback("Initialization failed!")

        async def await_result():
            stderr_output = ""
            while self.process.is_alive():
                await asyncio.sleep(1)
            # Read stderr from the pipe
            if stderr_read.poll():
                stderr_output = stderr_read.recv()
            stderr_read.close()
            API_error = False
            while not queue.empty():
                API_error = not queue.get()
            return self.process.exitcode, API_error, stderr_output

        exitcode, API_error, stderr_output = await await_result()
        self.process.close()
        self.process = None
        queue.close()

        if API_error:
            return {"error": "Unable to communicate with the API. Try again later."}
        else:
            if exitcode is None:
                return {"error": "Query execution failed!\n\n" + stderr_output}
            elif exitcode == 0:
                return {"content": "Query execution completed!"}
            elif exitcode == -15:
                return {"error": "Query execution aborted!"}
            else:
                return {
                    "error": f"Query execution exited with exit code: {exitcode}\n\n"
                    + stderr_output
                }

    def uptime(self):
        """
        Returns humanized bot uptime.
        """
        time_difference = (arrow.utcnow() - self.start_time).total_seconds()
        granularity = get_humanize_granularity(time_difference)
        humanized = self.start_time.humanize(
            arrow.utcnow(), only_distance=True, granularity=granularity
        )
        return humanized

    def push(self, passphrase: str = None):
        try:
            # Attempt to call yq.push and return its result if successful
            return yq.push(passphrase)
        except Exception as e:
            # If an exception occurs, return the exception message instead
            return str(e)

    def pull(self, passphrase: str = None):
        try:
            # Attempt to call yq.pull and return its result if successful
            return yq.pull(passphrase)
        except Exception as e:
            # If an exception occurs, return the exception message instead
            return str(e)


# ===================== #
# Telegram Bot Subclass #
# ===================== #


class Telegram(Bot):
    """
    Telegram bot subclass. Inherits from Bot class.

    Args:
        token (str): The token for the Telegram bot.
        channel (Union[str, int]): The Telegram channel ID.

    Attributes:
        application (telegram.Application): The Telegram Application bot instance.
        Bot attributes
    """

    def __init__(self, token: str, channel: Union[str, int]):
        """
        Initialize Telegram Bot subclass.

        Args:
            token (str): The token for the Telegram bot.
            channel (Union[str, int]): The Telegram channel ID.

        """
        from tqdm.contrib.telegram import tqdm as telegram_pbar

        self.telegram_pbar = telegram_pbar
        Bot.__init__(self, token, channel)
        # Initialize the Telegram bot
        self.application = ApplicationBuilder().token(token).build()
        self.register_commands()
        self.register_events()

    def run(self):
        """
        Start running the Telegram bot.
        """
        print("Running Telegram bot...")
        self.application.run_polling(stop_signals=None)

    # ======== #
    # Commands #
    # ======== #

    def register_commands(self):
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

            run - Run full YugiQuery flow.

            status - Display bot status and system information.

        """

        async def abort(update: Update, context: CallbackContext):
            """
            Aborts a running YugiQuery flow by terminating the thread.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            original_response = await context.bot.send_message(
                chat_id=update.effective_chat.id, text="Aborting..."
            )
            response = self.abort()
            await original_response.edit_text(response)

        async def battle(
            update: Update,
            context: CallbackContext,
        ):
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

            async def callback(first):
                nonlocal callback_first
                callback_first = first
                await original_message.edit_text(
                    escape_chars(
                        f"*First contestant*: {first}\n\nStill battling... ⏳"
                    ),
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

        async def benchmark(update: Update, context: CallbackContext):
            """
            Returns the average time each report takes to complete and the latest time for each report.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            response = self.benchmark()
            if "error" in response.keys():
                await context.bot.send_message(
                    chat_id=update.effective_chat.id, text=response["error"]
                )
                return

            message = f"*{response.pop('title')}*\n{response.pop('description')}\n\n"
            for key, value in response.items():
                message += f"*{key}*\n{value}\n\n"

            message = escape_chars(message)
            await context.bot.send_message(
                chat_id=update.effective_chat.id, text=message, parse_mode="MarkdownV2"
            )

        async def data(update: Update, context: CallbackContext):
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
                message = f"*{response['title']}*\n{response['description']}\n\nData:\n{response['data']}\nChangelog:\n{response['changelog']}"

            message = escape_chars(message)
            await context.bot.send_message(
                chat_id=update.effective_chat.id, text=message, parse_mode="MarkdownV2"
            )

        async def latest(update: Update, context: CallbackContext):
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
            await context.bot.send_message(
                chat_id=update.effective_chat.id, text=message, parse_mode="MarkdownV2"
            )

        async def links(update: Update, context: CallbackContext):
            """
            Displays the links to the YugiQuery webpage, repository, and data.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            response = self.links()
            message = f"*{response['title']}*\n{response['description']}"
            message = escape_chars(message)
            await context.bot.send_message(
                chat_id=update.effective_chat.id, text=message, parse_mode="MarkdownV2"
            )

        async def ping(update: Update, context: CallbackContext):
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

        async def pull(update: Update, context: CallbackContext):
            """
            ...
            """
            passphrase = context.args[0] if context.args else None
            response = self.pull(passphrase)
            await context.bot.send_message(
                chat_id=update.effective_chat.id, text=response
            )

        async def push(update: Update, context: CallbackContext):
            """
            ...
            """
            passphrase = context.args[0] if context.args else None
            response = self.push(passphrase)
            await context.bot.send_message(
                chat_id=update.effective_chat.id, text=response
            )

        async def run_query(
            update: Update,
            context: CallbackContext,
        ):
            """
            Runs a YugiQuery flow by launching a separate thread and monitoring its progress.
            The progress is reported back to the Telegram chat where the command was issued.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            last_run = context.user_data.get("last_run", arrow.get(0.0))
            if (arrow.utcnow() - last_run).total_seconds() < self.cooldown_limit:
                granularity = get_humanize_granularity(
                    (
                        last_run.shift(seconds=self.cooldown_limit) - arrow.utcnow()
                    ).total_seconds()
                )
                next_available = last_run.shift(seconds=self.cooldown_limit).humanize(
                    arrow.utcnow(), granularity=granularity
                )
                await update.effective_message.reply_text(
                    f"You are on cooldown. Try again {next_available}"
                )
                return

            report = (
                self.Reports[context.args[0].capitalize()]
                if context.args
                and context.args[0].capitalize() in self.Reports.__members__
                else self.Reports.All
            )

            original_response = await context.bot.send_message(
                chat_id=update.effective_chat.id, text="Initializing..."
            )

            async def callback(content: str):
                await original_response.edit_text(content)

            response = await self.run_query(
                callback=callback,
                report=report,
                channel_id=update.effective_chat.id,
                progress_bar=self.telegram_pbar,
            )
            if "error" in response.keys():
                await context.bot.send_message(
                    chat_id=update.effective_chat.id, text=response["error"]
                )
            else:
                context.user_data["last_run"] = arrow.utcnow()
                await context.bot.send_message(
                    chat_id=update.effective_chat.id, text=response["content"]
                )

        async def status(update: Update, context: CallbackContext):
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

            await context.bot.send_message(
                chat_id=update.effective_chat.id, text=message, parse_mode="MarkdownV2"
            )

        async def shutdown(update: Update, context: CallbackContext):
            """
            Shuts down the bot gracefully by sending a message and stopping the polling.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            await context.bot.send_message(
                chat_id=update.effective_chat.id, text="Shutting down..."
            )
            self.application.stop_running()

        # Register the command handlers
        self.application.add_handler(CommandHandler("abort", abort, block=False))
        self.application.add_handler(CommandHandler("battle", battle))
        self.application.add_handler(CommandHandler("benchmark", benchmark))
        self.application.add_handler(CommandHandler("data", data))
        self.application.add_handler(CommandHandler("latest", latest))
        self.application.add_handler(CommandHandler("links", links))
        self.application.add_handler(CommandHandler("ping", ping))
        self.application.add_handler(CommandHandler("run", run_query, block=False))
        self.application.add_handler(CommandHandler("status", status))
        self.application.add_handler(
            CommandHandler(
                "shutdown", shutdown, filters=filters.Chat(chat_id=int(self.channel))
            )
        )

    # ====== #
    # Events #
    # ====== #

    def register_events(self):
        """
        Register event handlers for the Telegram bot.
        """

        async def start(update: Update, context: CallbackContext):
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

        async def on_command_error(update: Update, context: CallbackContext):
            """
            Event that runs whenever a command invoked by the user results in an error.
            Sends a message to the chat indicating the type of error that occurred.

            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            error = context.error
            print(error.message)
            if update is not None:
                await update.message.reply_text(error.message)
            else:
                await context.bot.send_message(chat_id=self.channel, text=error.message)

        self.application.add_handler(CommandHandler("start", start))
        self.application.add_error_handler(on_command_error)


# ==================== #
# Discord Bot Subclass #
# ==================== #


class Discord(Bot, commands.Bot):
    """
    Discord bot subclass. Inherits from Bot class and discord.ext.commands.Bot.

    Args:
        token (str): The token for the Discord bot.
        channel (Union[str, int]): The channel for the bot.

    Attributes:
        discord.ext.commands.Bot attributes
        Bot attributes
    """

    def __init__(self, token: str, channel: Union[str, int]):
        """
        Initializes the Discord bot subclass.

        Args:
            token (str): The token for the Discord bot.
            channel (Union[str, int]): The channel for the bot.

        """

        self.discord_pbar = ensure_tqdm()
        Bot.__init__(self, token, channel)
        # Initialize the Discord bot
        intents = discord.Intents(messages=True, guilds=True, members=True)
        help_command = commands.DefaultHelpCommand(no_category="Commands")
        description = "Bot to manage YugiQuery data and execution."
        activity = discord.Activity(
            type=discord.ActivityType.watching, name="for /status"
        )
        commands.Bot.__init__(
            self,
            command_prefix="/",
            intents=intents,
            activity=activity,
            description=description,
            help_command=help_command,
        )
        self.register_commands()

    def run(self):
        """
        Starts running the discord Bot.
        """
        commands.Bot.run(self, self.token)

    # ====== #
    # Events #
    # ====== #

    async def on_ready(self):
        """
        Event callback that runs when the bot is ready to start receiving events and commands.
        Prints out the bot's username and the guilds it's connected to.
        """
        print("You are logged as {}".format(self.user))
        await self.tree.sync()

        print(f"{self.user} is connected to the following guilds:")
        for guild in self.guilds:
            print(f"{guild.name}(id: {guild.id})")
            members = "\n - ".join([member.name for member in guild.members])
            print(f"Guild Members:\n - {members}")

    async def on_message(self, message: discord.Message):
        """
        Event callback that runs whenever a message is sent in a server where the bot is present.
        Responds with a greeting to any message starting with 'hi'.

        Args:
            ctx (commands.Context): The context of the message.
            message (discord.Message): The message received.
        """
        if message.author == self.user:
            return

        await self.process_commands(message)
        if message.content.lower().startswith("hi"):
            await message.channel.send(content=f"Hello, {message.author.name}!")

    async def on_command_error(self, ctx, error: commands.CommandError):
        """
        Event callback that runs whenever a command invoked by the user results in an error.
        Sends a message to the channel indicating the type of error that occurred.

        Args:
            ctx (commands.Context): The context of the error.
            error (commands.CommandError): The error received.
        """
        print(error)
        if isinstance(error, commands.CommandOnCooldown):
            await ctx.send(content=error, ephemeral=True, delete_after=60)
        elif isinstance(error, commands.NotOwner):
            await ctx.send(content=error, ephemeral=True, delete_after=60)
        elif isinstance(error, commands.CheckFailure):
            await ctx.send(content=error, ephemeral=True, delete_after=60)

    # ======== #
    # Commands #
    # ======== #

    def register_commands(self):
        """
        Register command handlers for the Discord bot.

        Command descriptions:

            abort - Aborts a running YugiQuery flow by terminating the process.

            battle - Simulate a battle of all monster cards.

            benchmark - Show average time each report takes to complete.

            data - Send latest data files.

            shutdown - Shutdown bot

            latest - Show latest time each report was generated.

            links - Show YugiQuery links.

            ping - Test the bot connection latency.

            run - Run full YugiQuery flow.

            status - Display bot status and system information.

        """

        @self.hybrid_command(
            name="abort",
            description="Abort running YugiQuery flow.",
            with_app_command=True,
        )
        @commands.is_owner()
        async def abort(ctx):
            """
            Aborts a running YugiQuery flow by terminating the process.

            Args:
                ctx (commands.Context): The context of the command.
            """
            original_response = await ctx.send(
                content="Aborting...", ephemeral=True, delete_after=60
            )
            response = self.abort()
            await original_response.edit(content=response)

        @self.hybrid_command(
            name="battle",
            description="Simulate a battle of all monster cards.",
            with_app_command=True,
        )
        @commands.is_owner()
        async def battle(ctx, atk_weight: int = 4, def_weight: int = 1):
            """
            Loads the list of all Monster Cards and simulates a battle between them.

            Args:
                ctx (discord.ext.commands.Context): The context of the command.
                atk_weight (int): Weight for ATK stat.
                def_weight (int): Weight for DEF stat.
            """
            await ctx.defer()
            embed = discord.Embed(
                title="Battle",
                description="Simulate a battle of all monster cards",
                color=discord.Colour.purple(),
            )

            original_response = None

            async def callback(first):
                embed.add_field(name="First contestant", value=first, inline=False)
                embed.set_footer(text="Still battling... ⏳")
                nonlocal original_response
                original_response = await ctx.send(embed=embed)

            response = await self.battle(
                atk_weight=atk_weight, def_weight=def_weight, callback=callback
            )

            if "error" in response.keys():
                await ctx.send(
                    content="Cards data not found... Try again later.",
                    ephemeral=True,
                    delete_after=60,
                )
                return

            winner = response["winner"]
            longest = response["longest"]

            embed.add_field(name="Winner", value=winner[0]["Name"], inline=True)
            embed.add_field(name="Wins", value=winner[1], inline=True)
            embed.add_field(
                name="Stats remaining",
                value=f'ATK={winner[0]["ATK"]}, DEF={winner[0]["DEF"]}',
                inline=True,
            )

            embed.add_field(
                name="Longest streak", value=longest[0]["Name"], inline=True
            )
            embed.add_field(name="Wins", value=longest[1], inline=True)
            embed.add_field(
                name="Stats when defeated",
                value=f'ATK={longest[0]["ATK"]}, DEF={longest[0]["DEF"]}',
                inline=True,
            )
            embed.remove_footer()
            await original_response.edit(embed=embed)

        @self.hybrid_command(
            name="benchmark",
            description="Show average time each report takes to complete.",
            with_app_command=True,
        )
        async def benchmark(ctx):  # Improve function
            """
            Returns the average time each report takes to complete and the latest time for each report.

            Args:
                ctx (discord.ext.commands.Context): The context of the command.
            """
            await ctx.defer()
            response = self.benchmark()
            if "error" in response.keys():
                await ctx.send(response["error"])
                return
            title = response.pop("title")
            description = response.pop("description")
            embed = discord.Embed(
                title=title,
                description=description,
                color=discord.Colour.gold(),
            )

            for key, value in response.items():
                embed.add_field(name=key, value=value, inline=False)

            await ctx.send(embed=embed)

        @self.hybrid_command(
            name="data", description="Send latest data files.", with_app_command=True
        )
        async def data(ctx):
            """
            This command sends the latest data files available in the repository as direct download links.

            Parameters:
                ctx (discord.ext.commands.Context): The context of the command.
            """
            await ctx.defer()
            response = self.data()
            if "error" in response.keys():
                await ctx.send(response["error"])

            else:
                embed = discord.Embed(
                    title=response["title"],
                    description=response["description"],
                    color=discord.Colour.magenta(),
                )
                embed.add_field(name="Data", value=response["data"], inline=False)
                embed.add_field(
                    name="Changelog", value=response["changelog"], inline=False
                )
                await ctx.send(embed=embed)

        @self.hybrid_command(
            name="latest",
            description="Show latest time each report was generated.",
            with_app_command=True,
        )
        async def latest(ctx):
            """
            Displays the timestamp of the latest local and live reports generated. Reads the report files from `dirs.REPORTS` and
            queries the GitHub API for the latest commit timestamp for each file. Returns the result as an embedded message in
            the channel.

            Args:
                ctx (discord.ext.commands.Context): The context of the command.
            """
            await ctx.defer()
            response = self.latest()
            embed = discord.Embed(
                title=response["title"],
                description=response["description"],
                color=discord.Colour.orange(),
            )
            embed.add_field(name="Local", value=response["local"], inline=False)
            if "live" in response:
                embed.add_field(name="Live", value=response["live"], inline=False)

            await ctx.send(embed=embed)

        @self.hybrid_command(
            name="links", description="Show YugiQuery links.", with_app_command=True
        )
        async def links(ctx):
            """
            Displays the links to the YugiQuery webpage, repository, and data. Returns the links as an embedded message in
            the channel.

            Args:
                ctx (discord.ext.commands.Context): The context of the command.
            """
            response = self.links()
            embed = discord.Embed(
                title=response["title"],
                description=response["description"],
                color=discord.Colour.green(),
            )

            await ctx.send(embed=embed)

        @self.hybrid_command(
            name="ping",
            description="Test the bot connection latency.",
            with_app_command=True,
        )
        async def ping(ctx):
            """
            This command tests the bot's connection latency and sends the result back to the user.

            Parameters:
                ctx (discord.ext.commands.Context): The context of the command.
            """
            await ctx.send(
                content="🏓 Pong! {0}ms".format(round(self.latency * 1000, 1)),
                ephemeral=True,
                delete_after=60,
            )

        @self.hybrid_command(
            name="pull", description="Pull changes from remote.", with_app_command=True
        )
        @commands.is_owner()
        async def pull(ctx, passphrase: str = None):
            """
            ...
            """

            await ctx.defer()
            response = self.pull(passphrase)
            await ctx.send(content=response)

        @self.hybrid_command(
            name="push", description="Push repository to remote.", with_app_command=True
        )
        @commands.is_owner()
        async def push(ctx, passphrase: str = None):
            """
            ...
            """

            await ctx.defer()
            response = self.push(passphrase)
            await ctx.send(content=response)

        @self.hybrid_command(
            name="run", description="Run full YugiQuery flow.", with_app_command=True
        )
        @commands.is_owner()
        @commands.cooldown(1, self.cooldown_limit, commands.BucketType.user)
        async def run_query(ctx, report: self.Reports = self.Reports.All):
            """
            Runs a YugiQuery flow by launching a separate process and monitoring its progress.
            The progress is reported back to the Discord channel where the command was issued.
            The command has a cooldown period of 12 hours per user.

            Args:
                ctx (commands.Context): The context of the command.
                report (Reports): An Enum value indicating which YugiQuery report to run.

            Raises:
                discord.ext.commands.CommandOnCooldown: If the command is on cooldown for the user.
            """
            original_response = await ctx.send(
                content="Initializing...", ephemeral=True, delete_after=60
            )

            async def callback(content: str):
                await original_response.edit(content=content)

            response = await self.run_query(
                callback=callback,
                report=report,
                channel_id=ctx.channel.id,
                progress_bar=self.discord_pbar,
            )
            if "error" in response.keys():
                await ctx.channel.send(content=response["error"])
                # Reset cooldown in case query did not complete
                ctx.command.reset_cooldown(ctx)
            else:
                await ctx.channel.send(content=response["content"])

        @self.hybrid_command(
            name="status",
            description="Displays bot status and system information.",
            with_app_command=True,
        )
        async def status(ctx):
            """
            Displays information about the bot, including uptime, guilds, users, channels, available commands,
            bot version, discord.py version, python version, and operating system.

            Args:
                ctx (discord.ext.commands.Context): The context of the command.
            """
            appInfo = await self.application_info()
            admin = appInfo.owner
            users = 0
            channels = 0
            guilds = len(self.guilds)
            for guild in self.guilds:
                users += len(guild.members)
                channels += len(guild.channels)

            if len(self.commands):
                commandsInfo = " • `\\" + "\n • `\\".join(
                    sorted(
                        [
                            f"{i.name}`: {i.description}"
                            for i in self.commands
                            if not i.name == "help"
                        ]
                    )
                )

            embed = discord.Embed(color=ctx.me.colour)
            embed.set_footer(text="Time to duel!")
            embed.set_thumbnail(url=ctx.me.avatar)
            embed.add_field(name="Admin", value=admin, inline=False)
            embed.add_field(name="Uptime", value=self.uptime(), inline=False)
            embed.add_field(name="Guilds", value=guilds, inline=True)
            embed.add_field(name="Users", value=users, inline=True)
            embed.add_field(name="Channels", value=channels, inline=True)
            embed.add_field(name="Available Commands", value=commandsInfo, inline=False)
            embed.add_field(name="Bot Version", value=__version__, inline=True)
            embed.add_field(
                name="Discord.py Version", value=discord.__version__, inline=True
            )
            embed.add_field(
                name="Python Version", value=platform.python_version(), inline=True
            )
            embed.add_field(
                name="Operating System",
                value=f" • **Name**: {platform.system()}\n • **Release**: {platform.release()}\n • **Machine**: {platform.machine()}\n • **Version**: {platform.version()}",
                inline=False,
            )
            await ctx.send(
                "**:information_source:** Information about this bot:", embed=embed
            )

        @self.hybrid_command(
            name="shutdown", description="Shutdown bot.", with_app_command=True
        )
        @commands.is_owner()
        async def shutdown(ctx):
            """
            Shuts down the bot gracefully by sending a message and closing the connection.

            Args:
                ctx (commands.Context): The context of the command.
            """
            await ctx.send(content="Shutting down...")
            await self.close()


# ========= #
# Execution #
# ========= #


def main(args):
    # Load secrets
    secrets_args = {
        f"{args.subclass.upper()}_TOKEN": args.token,
        f"{args.subclass.upper()}_CHANNEL_ID": args.channel_id,
    }
    secrets = load_secrets_with_args(secrets_args)

    # Set multiprocessing start method
    mp.set_start_method("spawn")

    # Handle bots based on subclass
    if args.subclass == "discord":
        if "DISCORD_TOKEN" not in secrets or "DISCORD_CHANNEL_ID" not in secrets:
            raise ValueError(
                "Discord bot requires DISCORD_TOKEN and DISCORD_CHANNEL_ID in secrets."
            )
        # Initialize and run the Discord bot
        discord_bot = Discord(secrets["DISCORD_TOKEN"], secrets["DISCORD_CHANNEL_ID"])
        discord_bot.run()

    elif args.subclass == "telegram":
        if "TELEGRAM_TOKEN" not in secrets or "TELEGRAM_CHANNEL_ID" not in secrets:
            raise ValueError(
                "Telegram bot requires TELEGRAM_TOKEN and TELEGRAM_CHANNEL_ID in secrets."
            )
        # Initialize and run the Telegram bot
        telegram_bot = Telegram(
            secrets["TELEGRAM_TOKEN"], secrets["TELEGRAM_CHANNEL_ID"]
        )
        telegram_bot.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "subclass",
        choices=["discord", "telegram"],
        default="discord",
        help="Select between a Discord or a Telegram bot",
    )
    parser.add_argument("-t", "--token", type=str, help="Bot API token")
    parser.add_argument(
        "-c",
        "--channel",
        dest="channel_id",
        type=int,
        help="Bot responses channel id",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        required=False,
        help="Enable debug flag",
    )

    args = parser.parse_args()

    main(args)
