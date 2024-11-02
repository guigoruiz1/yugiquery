#!/usr/bin/env python3

# yugiquery/bot.py

# -*- coding: utf-8 -*-

# ======= #
# Imports #
# ======= #

# Standard library packages
import argparse
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


# ============ #
# Enum Classes #
# ============ #


class GitCommands(StrEnum):
    """
    Enum class to represent the available git commands.
    """

    status = "status"
    log = "log"
    pull = "pull"
    push = "push"


# ============== #
# Bot Superclass #
# ============== #


class Bot:
    """
    Bot superclass.

    Args:
        **kwargs: Additional keyword arguments.

    Attributes:
        start_time (arrow.Arrow): The bot initialization timestamp.
        repo (git.Repo): The git repository object, if there is a repository, else None.
        process (multiprocessing.Process): The variable to hold a process spawned by run_query.
        cooldown_limit (Tnt): The cooldown time in seconds to wait between consecutive calls to run_query.

    """

    def __init__(self, **kwargs):
        """
        Initializes the Bot base class.

        Args:
            **kwargs: Additional keyword arguments.
        """
        self.start_time = arrow.utcnow()
        self.init_reports_enum()
        try:
            # Open the repository
            self.repo = git.ensure_repo()
        except:
            self.repo = None

    # ======================== #
    # Bot Superclass Variables #
    # ======================== #
    process: mp.Process = None
    cooldown_limit: int = 12 * 3600  # 12 hours
    # Placeholder for the Enum object
    Reports: Enum = Enum("Reports", {"All": "all", "User": "user"})

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

    async def battle(self, callback: Callable[[str], None], atk_weight: int = 4, def_weight: int = 1) -> dict:
        """
        This function loads the list of all Monster Cards and simulates a battle between them. Each card is represented by its name, attack (ATK), and defense (DEF) stats. At the beginning of the battle, a random card is chosen as the initial contestant. Then, for each subsequent card, a random stat (ATK or DEF) is chosen to compare with the corresponding stat of the current winner. If the challenger's stat is higher, the challenger becomes the new winner. If the challenger's stat is lower, the current winner retains its position. If the stats are tied, the comparison is repeated with the other stat. The battle continues until there is only one card left standing.

        Args:
            callback (Callable[[str], None]): A callback function which receives a string argument.
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
            extracted = files["name"].str.extract(pat=r"^(.+?)_(\d{8}T\d{4}Z)?_?(\d{8}T\d{4}Z)?\.bz2$", expand=True)
            extracted[1] = extracted[2].fillna(extracted[1])
            files[["Group", "Timestamp"]] = extracted.iloc[:, [0, 1]]
            files["Timestamp"] = pd.to_datetime(files["Timestamp"], utc=True)
            index = files.groupby("Group")["Timestamp"].idxmax()
            latest_files = files.loc[index, ["Group", "name", "download_url"]]
            latest_files[["g1", "g2"]] = latest_files["Group"].apply(lambda x: pd.Series(x.split("_")).str.capitalize())

            fields = {}
            for report in latest_files["g1"].unique():
                selection = latest_files[latest_files["g1"] == report]
                if not selection.empty:
                    fields[report] = ""
                    for kind in selection["g2"].unique()[::-1]:
                        fields[report] += f"• [{kind}]({selection[selection['g2'] == kind]['download_url'].values[0]}) "

            response = {
                "title": "Latest data files",
                "description": "Direct download links from GitHub",
                "fields": fields,
            }
            return response
        except:
            return {"error": "Unable to obtain the latest files at this time. Try again later."}

    def git_cmd(self, command: GitCommands, passphrase: str = "") -> str:
        """
        Executes a git command on the repository.

        Args:
            command (Git_command): The git command to execute.
            passphrase (str, optional): The passphrase to use for encryption/decryption. Defaults to empty.

        Returns:
            str: The result message of the command.
        """

        def github_cmd(cmd) -> str:
            if not self.URLS.repo:
                return "No github repository."
            try:
                return cmd(passphrase=passphrase, repo=self.repo)
            except Exception as e:
                return str(e)

        match command:
            case GitCommands.status:
                return self.repo.git.status()
            case GitCommands.log:
                return self.repo.git.log()
            case GitCommands.pull:
                return github_cmd(cmd=git.pull)
            case GitCommands.push:
                return github_cmd(cmd=git.push)
            case _:
                return "Invalid command."

    def latest(self) -> Dict[str, str]:
        """
        Displays the timestamp of the latest local and live reports generated.
        Reads the report files from `dirs.REPORTS` and queries the GitHub API
        for the latest commit timestamp for each file.

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
            local_value += f'• {report.stem}: {pd.to_datetime(report.stat().st_mtime,unit="s", utc=True).strftime("%d/%m/%Y %H:%M %Z")}\n'

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
        callback: Callable[[str], None],
        report: Enum = Reports.All,
        progress_bar: tqdm = None,
        **pbar_kwargs,
    ) -> Dict[str, str]:
        """
        Runs a YugiQuery flow by launching a separate thread and monitoring its progress.

        Args:
            callback (Callable[[str], None]): A callback function which receives a string argument.
            report (Reports, optional): The report to run. Defaults to All.
            progress_bar (tqdm, optional): A tqdm progress bar. Defaults to None.
            **pbar_kwargs: Additional Keyword arguments to customize the progress bar..

        Returns:
            dict: A dictionary containing the result of the query execution.
        """
        if self.process is not None:
            return {"error": "Query already running. Try again after it has finished."}

        progress_handler = ProgressHandler(
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
            await callback(f"Initialization failed!\n{e}")

        # Wait for the process to finish and get the result
        API_status, errors = await progress_handler.await_result(self.process)
        exitcode = self.process.exitcode
        self.process.close()
        self.process = None

        error_message = "\n".join(errors)

        if API_status is not None and not API_status:
            return {"error": f"Unable to communicate with the API. Try again later.\n{error_message}"}
        else:
            if exitcode is None:
                return {"error": f"Query execution failed!\n{error_message}"}
            elif exitcode == 0:
                return {"content": "Query execution completed!"}
            elif exitcode == -15:
                return {"error": "Query execution aborted!"}
            else:
                return {"error": f"Query execution exited with exit code: {exitcode}\n{error_message}"}

    def uptime(self):
        """
        Returns humanized bot uptime.
        """
        time_difference = (arrow.utcnow() - self.start_time).total_seconds()
        granularity = get_granularity(time_difference)
        humanized = self.start_time.humanize(arrow.utcnow(), only_distance=True, granularity=granularity)
        return humanized


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
    parser.add_argument(
        "--debug",
        action="store_true",
        required=False,
        help="run in debug mode (not implemented)",
    )


def main(args) -> None:
    # Set multiprocessing start method
    mp.set_start_method("spawn")
    
    # Make sure the data and reports directories exist
    dirs.make()

    # Load secrets
    try:
        tkn, ch = load_secrets_with_args(args)
    except KeyError as e:
        cprint(text=f"{e}. Aborting...", color="red")
        return

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
