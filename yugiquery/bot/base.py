#!/usr/bin/env python3

# yugiquery/bot.py

# -*- coding: utf-8 -*-

# ======= #
# Imports #
# ======= #

# Standard library packages
import argparse
import asyncio
import multiprocessing as mp
import os
import random
from enum import Enum, StrEnum
from typing import (
    Any,
    Callable,
    Dict,
    Tuple,
)

# Third-party imports
import pandas as pd
from termcolor import cprint
from tqdm.auto import tqdm

# Local application imports
from ..utils import *
from ..yugiquery import run

# Set multiprocessing start method
mp.set_start_method("spawn")

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
        repo (git.Repo): The git repository object, if there is a repository, else None.
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
        try:
            # Open the repository
            self.repo = git.ensure_repo()
        except:
            self.repo = None

    # ======================== #
    # Bot Superclass Variables #
    # ======================== #
    process = None
    Reports = Enum("Reports", {"All": "all"})
    cooldown_limit = 12 * 3600  # 12 hours

    @property
    def URLS(self) -> StrEnum:
        """
        Property to get the URLs of the remote repository and webpage.
        """
        repository_api_url = ""
        repository_url = ""
        webpage_url = ""
        try:
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
            repository_api_url = f"https://api.github.com/repos/{author}/{repo}"
            repository_url = remote_url.split(".git")[0]
            webpage_url = f"https://{author}.github.io/{repo}"
        finally:
            return StrEnum(
                "URLS",
                {
                    "api": repository_api_url,
                    "repo": repository_url,
                    "webpage": webpage_url,
                },
            )

    # ====================== #
    # Bot Superclass Methods #
    # ====================== #

    def init_reports_enum(self) -> None:
        """
        Initializes and returns an Enum object containing the available reports.
        The reports are read from the NOTEBOOKS directories, where they are expected to be Jupyter notebooks.
        The Enum object is created using the reports' file names, with the .ipynb extension removed and the first letter capitalized.

        Returns:
            None
        """
        reports_dict = {"All": "all", "User": "user"}
        reports = sorted(dirs.NOTEBOOKS.pkg.glob("*.ipynb")) + sorted(
            dirs.NOTEBOOKS.user.glob("*.ipynb")
        )  # First user, then package
        for report in reports:
            reports_dict[report.stem.capitalize()] = report  # Will replace package by user

        self.Reports = Enum("Reports", reports_dict)

    def abort(self) -> str:
        """
        Aborts a running YugiQuery flow by terminating the process.

        Returns:
            str: The result message indicating whether the abortion was successful.
        """
        try:
            self.process.terminate()
            if self.repo is not None:
                git.restore(files=list(dirs.NOTEBOOKS.user.glob("*.ipynb")), repo=self.repo)
            return "Aborted"
        except:
            return "Abort failed"

    async def battle(self, callback: Callable, atk_weight: int = 4, def_weight: int = 1) -> dict:
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
            list(dirs.DATA.glob("cards_data_*.bz2")),
            key=os.path.getmtime,
        )
        if not cards_files:
            return {"error": "Cards data not found... Try again later."}

        cards = pd.read_csv(cards_files[0])
        weights = [atk_weight, def_weight]
        monsters = cards[(cards["Card type"] == "Monster Card") & (cards["Primary type"] != "Monster Token")][
            MONSTER_STATS
        ].set_index("Name")
        monsters = monsters.map(lambda x: x if x != "?" else random.randrange(start=0, stop=51) * 100)
        monsters = monsters.map(pd.to_numeric, errors="coerce").fillna(0).astype(int).reset_index()

        # Shuffle the monsters and select the first one as the initial winner
        monsters = monsters.sample(frac=1).reset_index(drop=True)
        winner = (monsters.iloc[0], 0)
        longest = (monsters.iloc[0], 0)

        await callback(winner[0]["Name"])

        for i in range(1, len(monsters)):
            current_winner = (winner[0].copy(), winner[1])
            next_monster = monsters.iloc[i].copy()
            chosen_stat = random.choices(MONSTER_STATS[1:], weights=weights)[0]
            not_chosen_stat = [stat for stat in MONSTER_STATS[1:] if stat != chosen_stat][0]
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

    def benchmark(self) -> Dict[str, str]:
        """
        Returns the average time each report takes to complete and the latest time for each report.

        Returns:
            dict: A dictionary containing benchmark information.
        """
        try:
            with open(dirs.DATA / "benchmark.json", "r") as file:
                data = json.load(file)
        except:
            return {"error": "Unable to find benchmark records at this time. Try again later."}

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
                arrow.now().shift(seconds=avg_time).humanize(granularity=get_granularity(avg_time), only_distance=True)
            )
            latest_time_str = (
                arrow.now()
                .shift(seconds=latest_time)
                .humanize(
                    granularity=get_granularity(latest_time),
                    only_distance=True,
                )
            )

            value = f"• Entries: {total_weight}\n• Average: {avg_time_str}\n• Latest: {latest_time_str}"
            response[key.capitalize()] = value

        return response

    def data(self) -> Dict[str, str]:
        """
        Sends the latest data files available in the repository as direct download links.

        Returns:
            dict: A dictionary containing direct links to the latest data files.
        """
        if self.URLS.api is None:
            return {"error": "No github repository."}
        try:
            files = pd.read_json(f"{self.URLS.api}/contents/data")
            files = files[files["name"].str.endswith(".bz2")]  # Remove .json files from lists
            files[["Group", "Timestamp"]] = (
                files["name"]
                .str.extract(
                    pat=r"(\w+_\w+)_(.*)(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})Z.bz2",
                    expand=True,
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
            return {"error": "Unable to obtain the latest files at this time. Try again later."}

    def latest(self) -> Dict[str, str]:
        """
        Displays the timestamp of the latest local and live reports generated.
        Reads the report files from `dirs.REPORTS` and queries the GitHub API
        for the latest commit timestamp for each file. Returns the result as an
        message in the channel.

        Returns:
            dict: A dictionary containing information about the latest reports.
        """

        reports = sorted(dirs.REPORTS.glob("*.html"))
        response = {
            "title": "Latest reports generated",
            "description": "The live reports may not always be up to date with the local reports",
        }

        # Get local files timestamps
        local_value = ""
        for report in reports:
            local_value += f'• {report.stem}: {pd.to_datetime(report.stats()._mtime,unit="s", utc=True).strftime("%d/%m/%Y %H:%M %Z")}\n'

        response["local"] = local_value

        # Get live files timestamps
        if self.URLS.api is not None:
            try:
                live_value = ""
                for report in reports:
                    result = pd.read_json(f"{self.URLS.api}/commits?path={report.name}")
                    timestamp = pd.DataFrame(result.loc[0, "commit"]).loc["date", "author"]
                    live_value += f'• {report.stem}: {pd.to_datetime(timestamp, utc=True).strftime("%d/%m/%Y %H:%M %Z")}\n'

                response["live"] = live_value
            except:
                pass

        return response

    def links(self) -> Dict[str, str]:
        """
        Displays the links to the YugiQuery webpage, repository, and data.

        Returns:
            dict: A dictionary containing links to YugiQuery resources.
        """
        if self.URLS.webpage is not None and self.URLS.repo is not None:
            description = (
                f"[Webpage]({self.URLS.webpage}) • [Repository]({self.URLS.repo}) • [Data]({self.URLS.repo}/tree/main/data)"
            )
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
    ) -> Dict[str, str]:
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

        if type(self).__name__ == "DiscordBot":
            pbar_kwargs = {"channel_id": channel_id, "token": self.token}
        elif type(self).__name__ == "TelegramBot":
            pbar_kwargs = {"chat_id": channel_id, "token": self.token}
        else:
            pbar_kwargs = {}

        progress_handler = ProgressHandler(
            queue=queue,
            progress_bar=progress_bar,
            pbar_kwargs=pbar_kwargs,
        )
        try:
            self.process = mp.Process(
                target=run,
                args=[report.value, progress_handler],
            )
            self.process.start()  # Close the write end in the parent process to ensure it only reads
            await callback("Running...")
        except Exception as e:
            print(e)
            await callback("Initialization failed!")

        async def await_result():
            while self.process.is_alive():
                await asyncio.sleep(1)

            API_error = False
            while not queue.empty():
                API_error = not queue.get()
            return self.process.exitcode, API_error

        # Wait for the process to finish and get the result
        exitcode, API_error = await await_result()
        self.process.close()
        self.process = None
        queue.close()

        if API_error:
            return {"error": "Unable to communicate with the API. Try again later."}
        else:
            if exitcode is None:
                return {"error": "Query execution failed!"}
            elif exitcode == 0:
                return {"content": "Query execution completed!"}
            elif exitcode == -15:
                return {"error": "Query execution aborted!"}
            else:
                return {"error": f"Query execution exited with exit code: {exitcode}"}

    def uptime(self):
        """
        Returns humanized bot uptime.
        """
        time_difference = (arrow.utcnow() - self.start_time).total_seconds()
        granularity = get_granularity(time_difference)
        humanized = self.start_time.humanize(arrow.utcnow(), only_distance=True, granularity=granularity)
        return humanized

    def push(self, passphrase: str = "") -> str:
        """
        Pushes the latest data files to the repository.

        Args:
            passphrase (str, optional): The passphrase to use for encryption. Defaults to empty.

        Returns:
            str: The result message indicating whether the push was successful
        """
        if not self.URLS.repo:
            return "No github repository."
        try:
            # Attempt to call git.push and return its result if successful
            return git.push(passphrase=passphrase, repo=self.repo)
        except Exception as e:
            # If an exception occurs, return the exception message instead
            return str(e)

    def pull(self, passphrase: str = "") -> str:
        """
        Pulls the latest data files from the repository.

        Args:
            passphrase (str, optional): The passphrase to use for decryption. Defaults to empty.

        Returns:
            str: The result message indicating whether the pull was successful
        """
        if not self.URLS.repo:
            return "No github repository."
        try:
            # Attempt to call git.pull and return its result if successful
            return git.pull(passphrase=passphrase, repo=self.repo)
        except Exception as e:
            # If an exception occurs, return the exception message instead
            return str(e)


# ========= #
# Execution #
# ========= #


# Helper function
def load_secrets_with_args(args: Any) -> Tuple[str, int | str]:
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
    secrets_args = {
        f"{subclass_upper}_TOKEN": args.token,
        f"{subclass_upper}_CHANNEL_ID": args.channel,
    }

    secrets = {key: value for key, value in secrets_args.items() if value}
    missing = [key for key, value in secrets_args.items() if not value]

    if missing:
        loaded_secrets = load_secrets(
            requested_secrets=missing,
            secrets_file=dirs.secrets_file,
            required=True,
        )
        secrets.update(loaded_secrets)

    return secrets[f"{subclass_upper}_TOKEN"], secrets[f"{subclass_upper}_CHANNEL_ID"]


def set_parser(parser: argparse.ArgumentParser) -> None:
    """
    Set the parser arguments for the bot mode.

    Args:
        parser (argparse.ArgumentParser): The parser object to set the arguments on.
        subclass (bool, optional): Whether to include the subclass argument. Defaults to False.
    """
    parser.add_argument(
        "subclass",
        choices=["discord", "telegram"],
        help="Select between a Discord or a Telegram bot",
    )
    parser.add_argument("-t", "--token", type=str, help="Bot API token")
    parser.add_argument(
        "-c",
        "--channel",
        dest="channel",
        type=int,
        help="Bot responses channel id",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        required=False,
        help="Enable debug flag",
    )


def main(args) -> None:
    # Make sure the data and reports directories exist
    dirs.make()

    # Load secrets
    try:
        token, channel = load_secrets_with_args(args)
    except KeyError as e:
        cprint(text=f"{e}. Aborting...", color="red")
        return

    # Handle bots based on subclass
    if args.subclass == "discord":
        # Initialize and run the Discord bot
        from .discord import Discord as Subclass

    elif args.subclass == "telegram":
        # Initialize and run the Telegram bot
        from .telegram import Telegram as Subclass

    # Run the bot subclass
    bot = Subclass(token=token, channel=channel, debug=args.debug)
    bot.run()
