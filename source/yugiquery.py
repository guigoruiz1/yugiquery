#!/usr/bin/env python3

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

__author__ = "Guilherme Ruiz"
__copyright__ = "2023, Guilherme Ruiz"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Guilherme Ruiz"
__email__ = "57478888+guigoruiz1@users.noreply.github.com"
__status__ = "Development"

# ======= #
# Imports #
# ======= #

# Native python packages
import argparse
import calendar
import colorsys
import glob
import hashlib
import io
import json
import logging
import os
import re
import socket
import subprocess
import time
import warnings
from enum import Enum
from textwrap import wrap
from typing import Any, Callable, Dict, List, Tuple, Union

# Shorthand variables
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

# PIP packages
loop = 0
while True:
    try:
        import asyncio
        import urllib.parse as up
        from ast import literal_eval

        import aiohttp
        import arrow
        import git
        import jupyter_client
        import matplotlib.colors as mc  # LogNorm, Normalize, ListedColormap, cnames, to_rgb
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        import nbformat
        import numpy as np
        import pandas as pd
        import papermill as pm
        import requests
        import seaborn as sns
        import wikitextparser as wtp
        from dotenv import dotenv_values
        from halo import Halo
        from ipylab import JupyterFrontEnd
        from IPython.display import Markdown, display
        from matplotlib.ticker import (
            AutoMinorLocator,
            FixedLocator,
            MaxNLocator,
            MultipleLocator,
        )
        from matplotlib_venn import venn2
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from tqdm.auto import tqdm, trange

        break

    except ImportError:
        if loop > 1:
            print("Failed to install required packages twice. Aborting...")
            quit()

        loop += 1
        print("Missing required packages. Trying to install now...")
        subprocess.call(["sh", os.path.join(SCRIPT_DIR, "./install.sh")])

# Overwrite packages with versions specific for jupyter notebook
try:
    if get_ipython() is not None:
        from itables import init_notebook_mode
        from halo import HaloNotebook as Halo
except:
    pass

# Default settings overrides
pd.set_option("display.max_columns", 40)

# ======= #
# Helpers #
# ======= #


# Variables
class CG(Enum):
    """
    Enum representing the card game formats.

    Attributes:
        CG (str): Both TCG and OCG.
        ALL (CG): Alias for CG, representing all card games.
        BOTH (CG): Alias for CG, representing both card games.
        TCG (str): The 'trading card game' type.
        OCG (str): The 'official card game' type.
    """

    CG = "CG"
    ALL = CG
    BOTH = CG
    TCG = "TCG"
    OCG = "OCG"


arrows_dict = {
    "Middle-Left": "←",
    "Middle-Right": "→",
    "Top-Left": "↖",
    "Top-Center": "↑",
    "Top-Right": "↗",
    "Bottom-Left": "↙",
    "Bottom-Center": "↓",
    "Bottom-Right": "↘",
}

# Functions

## Data loaders


def load_secrets(
    requested_secrets: List[str] = [], secrets_file: str = None, required: bool = False
):
    """
    Load secrets from environment variables and/or a .env file.

    The secrets can be specified by name using the `requested_secrets` argument, which should be a list of strings. If `requested_secrets` is not specified, all available secrets will be returned.

    The `secrets_file` argument is the path to a .env file containing additional secrets to load. If `secrets_file` is specified and the file exists, the function will load the secrets from the file and merge them with the secrets loaded from the environment variables giving priority to secrets obtained from the .env file.

    The `required` argument is a boolean or list of booleans indicating whether each requested secret is required to be present. If `required` is True, a KeyError will be raised if the secret is not found. If `required` is False or not specified, missing secrets will be skipped.

    Args:
        requested_secrets (List[str], optional): A list of names of the secrets to retrieve. If empty or not specified, all available secrets will be returned. Defaults to [].
        secrets_file (str, optional): The path to a .env file containing additional secrets to load. Defaults to None.
        required (bool or List[bool], optional): A boolean or list of booleans indicating whether each requested secret is required to be present. If True, a KeyError will be raised if the secret is not found. If False or not specified, missing secrets will be skipped. Defaults to False.

    Returns:
        Dict[str, str]: A dictionary containing the requested secrets as key-value pairs.

    Raises:
        KeyError: If a required secret is not found in the environment variables or .env file.

    """
    secrets = {
        key: os.environ.get(key, os.environ.get(f"TQDM_{key}"))
        for key in requested_secrets
    }
    if secrets_file and os.path.isfile(secrets_file):
        secrets = secrets | dotenv_values(secrets_file)

        if not requested_secrets:
            return secrets
        else:
            secrets = {
                key: secrets[key]
                for key in requested_secrets
                if key in secrets.keys() and secrets[key]
            }
    if required:
        for i, key in enumerate(requested_secrets):
            check = required if isinstance(required, bool) else required[i]
            if check and key not in secrets.keys():
                raise KeyError(f'Secret "{requested_secrets[i]}" not found')

    return secrets


def load_json(json_file: str):
    """
    Load data from a JSON file.

    Args:
        json_file (str): The file path to the JSON file.

    Returns:
        dict: A dictionary containing the data from the JSON file. If the file does not exist, an empty dictionary is returned.
    """
    try:
        with open(json_file, "r") as file:
            data = json.load(file)
            return data
    except:
        print(f"Error loading {json_file}. Ignoring...")
        return {}


## Validators


def md5(name: str):
    """
    Generate the MD5 hash of a string.

    Args:
        name (str): The string to hash.

    Returns:
        str: The MD5 hash of the string.
    """
    hash_md5 = hashlib.md5()
    hash_md5.update(name.encode())
    return hash_md5.hexdigest()


def separate_words_and_acronyms(strings: List[str]):
    """
    Separates a list of strings into words and acronyms.

    Args:
        strings (List[str]): A list of strings to be categorized.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
            - The first list contains words (strings starting with an uppercase letter followed by lowercase letters).
            - The second list contains acronyms (strings not meeting the word criteria).
    """
    words = []
    acronyms = []
    for string in strings:
        if re.match(r"^[A-Z][a-z]+$", string):
            words.append(string)
        else:
            acronyms.append(string)
    return words, acronyms


def make_filename(
    report: str, timestamp: arrow.Arrow, previous_timestamp: arrow.Arrow = None
):
    """
    Generates a standardized filename based on the provided parameters.

    Args:
        report (str): The name or identifier of the report.
        timestamp (arrow.Arrow): The timestamp to be included in the filename.
        previous_timestamp (arrow.Arrow): The previous timestamp, if applicable. Defaults to None.

    Returns:
        str: The generated filename.
    """
    if previous_timestamp is None:
        return f"{report}_data_{timestamp.isoformat(timespec='minutes').replace('+00:00', 'Z')}.bz2"
    else:
        return f"{report}_changelog_{previous_timestamp.isoformat(timespec='minutes').replace('+00:00', 'Z')}_{timestamp.isoformat(timespec='minutes').replace('+00:00', 'Z')}.bz2"


## Image handling


async def download_images(
    file_names: pd.DataFrame, save_folder: str = "../images/", max_tasks: int = 10
):
    """
    Downloads a set of images given their names and saves them to a specified folder.

    Args:
        file_names (pandas.DataFrame): A DataFrame containing the names of the image files to be downloaded.
        save_folder (str): The path to the folder where the downloaded images will be saved. Defaults to "../images/".
        max_tasks (int): The maximum number of images to download at once. Defaults to 10.

    Returns:
        None
    """
    # Prepare URL from file names
    file_names_md5 = file_names.apply(md5)
    urls = file_names_md5.apply(lambda x: f"/{x[0]}/{x[0]}{x[1]}/") + file_names

    # Download image from URL
    async def download_image(session, url, save_folder, semaphore, pbar):
        async with semaphore:
            async with session.get(url) as response:
                save_name = url.split("/")[-1]
                if response.status != 200:
                    raise ValueError(
                        f"URL {url} returned status code {response.status}"
                    )
                total_size = int(response.headers.get("Content-Length", 0))
                progress = tqdm(
                    unit="B",
                    total=total_size,
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=save_name,
                    leave=False,
                    disable=("PM_IN_EXECUTION" in os.environ),
                )
                if os.path.isfile(f"{save_folder}/{save_name}"):
                    os.remove(f"{save_folder}/{save_name}")
                while True:
                    chunk = await response.content.read(1024)
                    if not chunk:
                        break
                    progress.update(len(chunk))
                    with open(f"{save_folder}/{save_name}", "ab") as f:
                        f.write(chunk)
                progress.close()
                return save_name

    # Parallelize image downloads
    semaphore = asyncio.Semaphore(max_tasks)
    async with aiohttp.ClientSession(
        base_url="https://ms.yugipedia.com/", headers=http_headers
    ) as session:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        with tqdm(
            total=len(urls),
            unit_scale=True,
            unit="file",
            disable=("PM_IN_EXECUTION" in os.environ),
        ) as pbar:
            tasks = [
                download_image(session, url, save_folder, semaphore, pbar)
                for url in urls
            ]
            for task in asyncio.as_completed(tasks):
                pbar.update()
                await task


# =============== #
# Data management #
# =============== #


# Git management


def assure_repo():
    """
    Assures the script is inside a git repository. Initializes a repository if one is not found.

    Raises:
        Exception: For any unexpected errors.

    Returns:
        None
    """
    try:
        # Try to create a Repo object
        repo = git.Repo(PARENT_DIR)

    except git.InvalidGitRepositoryError:
        # Handle the case when the path is not a valid Git repository
        git.Repo.init(PARENT_DIR)
        print(f"Git repository initialized in {PARENT_DIR}")

    except Exception as e:
        # Handle any exceptions (e.g., invalid path)
        raise RuntimeError(f"Unable to init Git repository: {e}")


def commit(files: Union[str, List[str]], commit_message: str = None):
    """
    Commits the specified files to the git repository after staging them.

    Args:
        files (Union[str, List[str]]): A list of file paths to be committed.
        commit_message (str, optional): The commit message. If not provided, a default message will be used.

    Raises:
        git.InvalidGitRepositoryError: If the PARENT_DIR is not a git repository.
        git.GitCommandError: If an error occurs while committing the changes.
        Exception: For any other unexpected errors.

    Returns:
        None
    """
    if commit_message is None:
        commit_message = f"Commit - {arrow.utcnow().isoformat()}"
    if isinstance(files, str):
        files = [files]
    try:
        with git.Repo(SCRIPT_DIR, search_parent_directories=True) as repo:
            # Stage the files before committing
            repo.git.add(*files)
            return repo.git.commit(message=commit_message)

    except git.InvalidGitRepositoryError as e:
        raise RuntimeError(f"Unable to find a git repository: {e}")
    except git.GitCommandError as e:
        raise RuntimeError(f"Failed to commit changes: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


def pull(passphrase: str = None):
    """
    Pulls changes from the remote git repository.

    Args:
        passphrase (str, optional): The passphrase to unlock your Git credential store.

    Raises:
        git.InvalidGitRepositoryError: If the PARENT_DIR is not a git repository.
        git.GitCommandError: If an error occurs while committing the changes.
        Exception: For any other unexpected errors.

    Returns:
        None
    """
    result = subprocess.run(
        ["sh", "unlock_git.sh", passphrase if passphrase else ""],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return result.stdout.decode("utf-8")
    else:
        try:
            with git.Repo(SCRIPT_DIR, search_parent_directories=True) as repo:
                return repo.git.pull()

        except git.InvalidGitRepositoryError as e:
            raise RuntimeError(f"Unable to find a git repository: {e}")
        except git.GitCommandError as e:
            raise RuntimeError(f"Failed to push changes: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred: {e}")


def push(passphrase: str = None):
    """
    Pushes commits to the remote git repository.

    Args:
        passphrase (str, optional): The passphrase to unlock your Git credential store.

    Raises:
        git.InvalidGitRepositoryError: If the PARENT_DIR is not a git repository.
        git.GitCommandError: If an error occurs while committing the changes.
        Exception: For any other unexpected errors.

    Returns:
        None
    """
    result = subprocess.run(
        ["sh", "unlock_git.sh", passphrase if passphrase else ""],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return result.stdout.decode("utf-8")
    else:
        try:
            with git.Repo(SCRIPT_DIR, search_parent_directories=True) as repo:
                return repo.git.push()

        except git.InvalidGitRepositoryError as e:
            raise RuntimeError(f"Unable to find a git repository: {e}")
        except git.GitCommandError as e:
            raise RuntimeError(f"Failed to push changes: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred: {e}")


# Data files


def benchmark(timestamp: arrow.Arrow, report: str = None):
    """
    Records the execution time of a report and saves the data to a JSON file.

    Args:
        timestamp (arrow.Arrow): The timestamp when the report execution began.
         report (str): The name of the report being benchmarked. If None, tries obtaining report name from JPY_SESSION_NAME environment variable.

    Returns:
        None
    """
    if report is None:
        try:
            report = os.path.basename(os.environ["JPY_SESSION_NAME"]).split(".")[0]
        except:
            report = ""

    now = arrow.utcnow()
    timedelta = now - timestamp
    benchmark_file = os.path.join(PARENT_DIR, "data/benchmark.json")
    data = load_json(benchmark_file)
    # Add the new data to the existing data
    if report not in data:
        data[report] = []
    data[report].append(
        {"ts": now.isoformat(), "average": timedelta.total_seconds(), "weight": 1}
    )
    # Save new data to file
    with open(benchmark_file, "w+") as file:
        json.dump(data, file)

    result = commit(
        files=[benchmark_file],
        commit_message=f"{report} report benchmarked - {now.isoformat()}",
    )
    print(result)


def condense_changelogs(files: pd.DataFrame):
    """
    Condenses multiple changelog files into a consolidated dataframe and generates a new filename.

    Args:
        files (pd.DataFrame): A dataframe containing the changelog files.

    Returns:
        Tuple[pd.DataFrame, str]: A tuple containing the consolidated changelog dataframe and the new filename.
    """
    new_changelog = pd.DataFrame()
    changelog_name = None
    first_date = None
    last_date = None
    for file in files:
        match = re.search(
            r"(\w+)_\w+_(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})Z_(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})Z.bz2",
            os.path.basename(file),
        )
        name = match.group(1)
        from_date = match.group(2)
        to_date = match.group(3)
        if changelog_name is not None and not changelog_name == name:
            print("Names mismatch!")
        changelog_name = name
        if first_date is None or first_date > from_date:
            first_date = from_date
        if last_date is None or last_date < to_date:
            last_date = to_date
        df = pd.read_csv(file, dtype=object)
        df["Version"] = df["Version"].map({"Old": from_date, "New": to_date})
        new_changelog = pd.concat([new_changelog, df], axis=0, ignore_index=True)

    new_changelog.sort_values(
        by=[new_changelog.columns[0], "Version"],
        ascending=[True, True],
        axis=0,
        inplace=True,
    )
    new_changelog = new_changelog.drop_duplicates(keep="last").dropna(how="all", axis=0)
    index = (
        new_changelog.drop(["Modification date", "Version"], axis=1)
        .drop_duplicates(keep="last")
        .index
    )
    new_filename = os.path.join(
        os.path.dirname(file),
        make_filename(
            report=changelog_name,
            timestamp=arrow.get(last_date),
            previous_timestamp=arrow.get(first_date),
        ),
    )
    return new_changelog.loc[index], new_filename


def condense_benchmark(benchmark: dict):
    """
    Condenses a benchmark dictionary by calculating the weighted average and total weight for each key.

    Args:
        benchmark (dict): A dictionary containing benchmark data.

    Returns:
        dict: The condensed benchmark dictionary with updated entries.
    """
    now = arrow.utcnow()
    for key, pair in benchmark.items():
        for key, values in benchmark.items():
            weighted_sum = 0
            total_weight = 0
            for entry in values:
                weighted_sum += entry["average"] * entry["weight"]
                total_weight += entry["weight"]
            weighted_average = weighted_sum / total_weight
            benchmark.update(
                {
                    key: [
                        {
                            "ts": now.isoformat(),
                            "average": weighted_average,
                            "weight": total_weight,
                        }
                    ]
                }
            )

    return benchmark


def cleanup_data(dry_run=False):
    """
    Cleans up data files, keeping only the most recent file from each month and week.

    Args:
        dry_run (bool): If True, the function will only print the files that would be deleted without actually deleting them. Defaults to False.

    Returns:
        None
    """
    # Benchmark
    now = arrow.utcnow()
    benchmark_path = os.path.join(PARENT_DIR, "data/benchmark.json")
    benchmark = load_json(benchmark_path)
    new_benchmark = condense_benchmark(benchmark)
    if dry_run:
        print("Benchmark:", new_benchmark)
    else:
        with open(benchmark_path, "w+") as f:
            json.dump(new_benchmark, f)

    # Data CSV files
    file_list = glob.glob(os.path.join(PARENT_DIR, "data/*.bz2"))

    # Create a DataFrame
    df = pd.DataFrame(file_list, columns=["Name"])

    # Convert the 'Date' column to a datetime type
    df["Date"] = pd.to_datetime(df["Name"].apply(os.path.getctime), unit="s")

    # Create a new column 'Group' based on the first two elements after splitting the filename
    df["Group"] = df["Name"].apply(
        lambda x: "_".join(os.path.basename(x).split("_", 2)[:2])
    )

    # Group the DataFrame by 'Group' and 'Date' (year and month)
    grouped = df.groupby(["Group", pd.Grouper(key="Date", freq="MS")])

    # Get a list of all the files created on the same month of the same year, separated by whether they contain "changelog"
    same_month_files = {
        "changelog": [
            group[1]["Name"].tolist() for group in grouped if "changelog" in group[0][0]
        ],
        "data": [
            group[1]["Name"].tolist()
            for group in grouped
            if not "changelog" in group[0][0]
        ],
    }

    # Get a list of all the files created in the last month and split them into weeks
    last_month_files = (
        df[df["Date"] >= df["Date"].max() - pd.Timedelta("1MS")]
        .resample("W", on="Date")
        .first()
    )

    # Separate the last_month_files by whether they contain "changelog"
    last_month_files = {
        "changelog": last_month_files[
            last_month_files["Group"].str.contains("changelog")
        ]["Name"].tolist(),
        "data": last_month_files[~last_month_files["Group"].str.contains("changelog")][
            "Name"
        ].tolist(),
    }

    # Remove the last_month_files from the same_month_files
    same_month_files["changelog"] = [
        files
        for files in same_month_files["changelog"]
        if files not in last_month_files["changelog"]
    ]
    same_month_files["data"] = [
        files
        for files in same_month_files["data"]
        if files not in last_month_files["data"]
    ]

    print("\n- same month (with changelog)")
    for files in same_month_files["changelog"]:
        if len(files) > 1:
            new_changelog, new_filepath = condense_changelogs(files)
            print(f"New changelog file: {new_filepath}")
            if dry_run:
                display(new_changelog)
            else:
                new_changelog.to_csv(new_filepath)
            for file in files:
                if dry_run:
                    print("Delete", file)
                else:
                    os.remove(file)

    print("\n- same month (without changelog)")
    for files in same_month_files["data"]:
        for file in files[:-1]:
            if dry_run:
                print("Delete", file)
            else:
                os.remove(file)
        if dry_run:
            print("Keep", files[-1])

    if (files := last_month_files["changelog"]) and (len(files) > 1):
        print("\n- Last month (with changelog)")
        new_changelog, new_filepath = condense_changelogs(files)
        print(f"New changelog file: {new_filepath}")
        if dry_run:
            display(new_changelog)
        else:
            new_changelog.to_csv(new_filepath)
        for file in last_month_files["changelog"]:
            if dry_run:
                print("Delete", file)
            else:
                os.remove(file)

    if files := last_month_files["data"]:
        print("\n- Last month (without changelog)")
        for file in files[:-1]:
            if dry_run:
                print("Delete", file)
            else:
                os.remove(file)
        if dry_run:
            print("Keep", files[-1])

    if not dry_run:
        result = commit(
            files=[
                os.path.join(PARENT_DIR, "data/benchmark.json"),
                os.path.join(PARENT_DIR, "data/*bz2"),  # May not work
            ],
            commit_message=f"Data cleanup {arrow.utcnow().isoformat()}",
        )
        print(result)


def load_corrected_latest(name_pattern: str, tuple_cols: List[str] = []):
    """
    Loads the most recent data file matching the specified name pattern and applies corrections.

    Args:
        name_pattern (str): Data file name pattern to load.
        tuple_cols (List[str]): List of columns containing tuple values to apply literal_eval.

    Returns:
        Tuple[pd.DataFrame, arrow.Arrow]: A tuple containing the loaded dataframe and the timestamp of the file.
    """
    files = sorted(
        glob.glob(f"../data/{name_pattern}_data_*.bz2"),
        key=os.path.getctime,
        reverse=True,
    )

    if files:
        df = pd.read_csv(files[0], dtype=object)
        for col in tuple_cols:
            if col in df:
                df[col] = df[col].dropna().apply(literal_eval)

        for col in ["Modification date", "Release"]:
            if col in df:
                df[col] = pd.to_datetime(df[col])

        ts = arrow.get(os.path.basename(files[0]).split("_")[-1].split(".bz2")[0])
        print(f"{name_pattern} file loaded")
        return df, ts
    else:
        print(f"No {name_pattern} files")
        return None, None


# Data formating


def extract_fulltext(x: List[Union[Dict[str, Any], str]], multiple: bool = False):
    """
    Extracts fulltext from a list of dictionaries or strings.
    If multiple is True, returns a sorted tuple of all fulltexts.
    Otherwise, returns the first fulltext found, with leading/trailing whitespaces removed.
    If the input list is empty, returns np.nan.

    Args:
        x (List[Union[Dict[str, Any], str]]): A list of dictionaries or strings to extract fulltext from.
        multiple (bool): If True, return a tuple of all fulltexts. Otherwise, return the first fulltext. Default is False.

    Returns:
        str or Tuple[str] or np.nan: The extracted fulltext(s).
    """
    if len(x) > 0:
        if isinstance(x[0], int):
            return str(x[0])
        elif "fulltext" in x[0]:
            if multiple:
                return tuple(sorted([i["fulltext"] for i in x]))
            else:
                return x[0]["fulltext"].strip("\u200e")
        else:
            if multiple:
                return tuple(sorted(x))
            else:
                return x[0].strip("\u200e")
    else:
        return np.nan


def format_df(input_df: pd.DataFrame, include_all: bool = False):
    """
    Formats a dataframe containing card information.
    Returns a new dataframe with specific columns extracted and processed.

    Args:
        input_df (pd.DataFrame): The input dataframe to format.
        include_all (bool): If True, include all unspecified columns in the output dataframe. Default is False.

    Returns:
        pd.DataFrame: The formatted dataframe.
    """
    df = pd.DataFrame(index=input_df.index)

    # Column name: multiple values
    individual_cols = {
        "Name": False,
        "Password": False,
        "Card type": False,
        "Property": False,
        "Card image": False,
        "Archseries": True,
        "Misc": True,
        "Category": True,
        "Summoning": True,
        # Monster card specific columns
        "Attribute": False,
        "Primary type": True,
        "Secondary type": True,
        "Monster type": False,
        "Effect type": True,
        "DEF": False,
        "Pendulum Scale": False,
        "Link": False,
        # Skill card specific columns
        "Character": False,
        # Rush duel specific columns
        # Set specific columns
        "Set": False,
        "Card number": False,
        "Series": False,
        "Set type": False,
        "Cover card": True,
        # Bandai specific columns
        "Ability": False,
        "Rule": False,
    }
    for col, multi in individual_cols.items():
        if col in input_df.columns:
            extracted_col = input_df[col].apply(extract_fulltext, multiple=multi)
            # Primary type classification
            if col == "Primary type":
                df[col] = extracted_col.apply(extract_primary_type)
            elif col == "Misc":
                # Rush specific
                df = df.join(extracted_col.apply(extract_misc))
            else:
                df[col] = extracted_col

    # Link arrows styling
    if "Link Arrows" in input_df.columns:
        df["Link Arrows"] = input_df["Link Arrows"].apply(
            lambda x: (
                tuple([arrows_dict[i] for i in sorted(x)]) if len(x) > 0 else np.nan
            )
        )

    # Columns with matching name pattern: extraction function
    filter_cols = {
        "ATK": True,
        "Level": True,
        " status": True,
        " Material": True,
        "Page ": False,
    }
    for col, extract in filter_cols.items():
        col_matches = input_df.filter(like=col).columns
        if len(col_matches) > 0:
            extracted_cols = input_df[col_matches].map(
                extract_fulltext if extract else lambda x: x
            )
            if col == " Material":
                df["Materials"] = extracted_cols.apply(
                    lambda x: tuple(elem for tup in col for elem in tup), axis=1
                )
            else:
                df = df.join(extracted_cols)

    # Category boolean columns for merging into tuple
    category_bool_cols = {
        "Artwork": ".*[aA]rtworks$",
    }
    for col, cat in category_bool_cols.items():
        col_matches = input_df.filter(regex=cat).columns
        if len(col_matches) > 0:
            cat_bool = input_df[col_matches].map(extract_category_bool)
            # Artworks extraction
            if col == "Artwork":
                df[col] = cat_bool.apply(format_artwork, axis=1)
            else:
                df[col] = cat_bool

    # Date columns concatenation
    if len(input_df.filter(like=" date").columns) > 0:
        df = df.join(
            input_df.filter(like=" date").map(
                lambda x: (
                    pd.to_datetime(
                        pd.to_numeric(x[0]["timestamp"]), unit="s", errors="coerce"
                    )
                    if len(x) > 0
                    else np.nan
                )
            )
        )

    # Include other unspecified columns
    if include_all:
        df = df.join(
            input_df[input_df.columns.difference(df.columns)].map(
                extract_fulltext, multiple=True
            )
        )

    return df


## Cards


def extract_primary_type(x: Union[str, List[str], Tuple[str]]):
    """
    Extracts the primary type of a card.
    If the input is a list or tuple, removes "Pendulum Monster" and "Maximum Monster" from the list.
    If the input is a list or tuple with only one element, returns that element.
    If the input is a list or tuple with multiple elements, returns the first element that is not "Effect Monster".
    Otherwise, returns the input.

    Args:
        x (Union[str, List[str], Tuple[str]]): The input type(s) to extract the primary type from.

    Returns:
        Union[str, List[str]]: The extracted primary type(s).
    """
    if isinstance(x, list) or isinstance(x, tuple):
        if "Monster Token" in x:
            return "Monster Token"
        else:
            x = [z for z in x if (z != "Pendulum Monster") and (z != "Maximum Monster")]
            if len(x) == 1 and "Effect Monster" in x:
                return "Effect Monster"
            elif len(x) > 0:
                return [z for z in x if z != "Effect Monster"][0]
            else:
                return "???"

    return x


def extract_misc(x: Union[str, List[str], Tuple[str]]):
    """
    Extracts the misc properties of a card.
    Checks whether the input contains the values "Legend Card" or "Requires Maximum Mode" and creates a boolean table.

    Args:
        x (Union[str, List[str], Tuple[str]]): The Misc values to generate the boolean table from.

    Returns:
        pd.Series: A pandas Series of boolean values indicating whether "Legend Card" and "Requires Maximum Mode" are present in the input.
    """
    if isinstance(x, list) or isinstance(x, tuple):
        return pd.Series(
            [val in x for val in ["Legend Card", "Requires Maximum Mode"]],
            index=["Legend", "Maximum mode"],
        )
    else:
        return pd.Series([False, False], index=["Legend", "Maximum mode"])


def extract_category_bool(x: List[str]):
    """
    Extracts a boolean value from a list of strings that represent a boolean value.
    If the first string in the list is "t", returns True.
    If the first string in the list is "f", returns False.
    Otherwise, returns np.nan.

    Args:
        x (List[str]): The input list of strings to extract a boolean value from.

    Returns:
        Union[bool, np.nan]: The extracted boolean value.
    """
    if len(x) > 0:
        if x[0] == "f":
            return False
        elif x[0] == "t":
            return True

    return np.nan


def format_artwork(row: pd.Series):
    """
    Formats a row of a dataframe that contains "alternate artworks" and "edited artworks" columns.
    If the "alternate artworks" column(s) in the row contain at least one "True" value, adds "Alternate" to the result tuple.
    If the "edited artworks" column(s) in the row contain at least one "True" value, adds "Edited" to the result tuple.
    Returns the result tuple.

    Args:
        row (pd.Series): A row of a dataframe that contains "alternate artworks" and "edited artworks" columns.

    Returns:
        Tuple[str]: The formatted row as a tuple.
    """
    result = tuple()
    index_str = row.index.str
    if index_str.endswith("alternate artworks").any():
        matching_cols = row.index[index_str.endswith("alternate artworks")]
        if row[matching_cols].any():
            result += ("Alternate",)
    if index_str.endswith("edited artworks").any():
        matching_cols = row.index[index_str.endswith("edited artworks")]
        if row[matching_cols].any():
            result += ("Edited",)
    if result == tuple():
        return np.nan
    else:
        return result


def format_errata(row: pd.Series):
    """
    Formats errata information from a pandas Series and returns a tuple of errata types.

    Args:
        row (pd.Series): A pandas Series containing errata information for a single card.

    Returns:
        Tuple[str]: Tuple of errata types if any errata information is present in the input Series, otherwise np.nan.
    """
    result = []
    if "Cards with name errata" in row:
        if row["Cards with name errata"]:
            result.append("Name")
    if "Cards with card type errata" in row:
        if row["Cards with card type errata"]:
            result.append("Type")
    if "Card Errata" in row and not result:
        if row["Card Errata"]:
            result.append("Any")
    if result:
        return tuple(sorted(result))
    else:
        return np.nan


def merge_errata(input_df: pd.DataFrame, input_errata_df: pd.DataFrame):
    """
    Merges errata information from an input errata DataFrame into an input DataFrame based on card names.

    Args:
        input_df (pd.DataFrame): A pandas DataFrame containing card information.
        input_errata_df (pd.DataFrame): A pandas DataFrame containing errata information.

    Returns:
        pd.DataFrame: A pandas DataFrame with errata information merged into it.
    """
    if "Name" in input_df.columns:
        errata_series = input_errata_df.apply(format_errata, axis=1).rename("Errata")
        input_df = input_df.merge(
            errata_series,
            left_on="Name",
            right_index=True,
            how="left",
            suffixes=("", " errata"),
        )
    else:
        print('Error! No "Name" column to join errata')

    return input_df


## Sets


def merge_set_info(input_df: pd.DataFrame, input_info_df: pd.DataFrame):
    """
    Merges set information from an input set info DataFrame into an input set list DataFrame based on set and region.

    Args:
        input_df (pd.DataFrame): A pandas DataFrame containing set lists.
        input_info_df (pd.DataFrame): A pandas DataFrame containing set information.

    Returns:
        pd.DataFrame: A pandas DataFrame with set information merged into it.
    """
    if all([col in input_df.columns for col in ["Set", "Region"]]):
        regions_dict = load_json(os.path.join(PARENT_DIR, "assets/json/regions.json"))
        input_df["Release"] = input_df[["Set", "Region"]].apply(
            lambda x: (
                input_info_df[regions_dict[x["Region"]] + " release date"][x["Set"]]
                if (
                    x["Region"] in regions_dict.keys()
                    and x["Set"] in input_info_df.index
                )
                else np.nan
            ),
            axis=1,
        )
        input_df["Release"] = pd.to_datetime(
            input_df["Release"].astype(str), errors="coerce"
        )  # Bug fix
        input_df = input_df.merge(
            input_info_df.loc[:, :"Cover card"],
            left_on="Set",
            right_index=True,
            how="outer",
            indicator=False,
        ).reset_index(drop=True)
        print("Set properties merged")
    else:
        print('Error! No "Set" and/or "Region" column(s) to join set info')

    return input_df


# ========= #
# Changelog #
# ========= #


def generate_changelog(
    previous_df: pd.DataFrame, current_df: pd.DataFrame, col: Union[str, List[str]]
):
    """
    Generates a changelog DataFrame by comparing two DataFrames based on a specified column.

    Args:
        previous_df (pd.DataFrame): A DataFrame containing the previous version of the data.
        current_df (pd.DataFrame): A DataFrame containing the current version of the data.
        col (Union[str, List[str]]): The name of the column to compare the DataFrames on.

    Returns:
        pd.DataFrame: A DataFrame containing the changes made between the previous and current versions of the data. The DataFrame will have the following columns: the specified column name, the modified data, and the indicator for whether the data is new or modified renamed as version (either "Old" or "New"). If there are no changes, the function will return a DataFrame with no rows.
    """
    if isinstance(col, str):
        col = [col]
    changelog = (
        previous_df.merge(current_df, indicator=True, how="outer")
        .loc[lambda x: x["_merge"] != "both"]
        .sort_values(col, ignore_index=True)
    )
    changelog["_merge"] = changelog["_merge"].cat.rename_categories(
        {"left_only": "Old", "right_only": "New"}
    )
    changelog.rename(columns={"_merge": "Version"}, inplace=True)
    nunique = changelog.groupby(col).nunique(dropna=False)
    cols_to_drop = (
        nunique[nunique < 2]
        .dropna(axis=1)
        .columns.difference(["Modification date", "Version"])
    )
    changelog.drop(cols_to_drop, axis=1, inplace=True)
    changelog = changelog.set_index(col)

    if all(col in changelog.columns for col in ["Modification date", "Version"]):
        true_changes = (
            changelog.drop(["Modification date", "Version"], axis=1)[nunique > 1]
            .dropna(axis=0, how="all")
            .index
        )
        new_entries = nunique[nunique["Version"] == 1].dropna(axis=0, how="all").index
        rows_to_keep = true_changes.union(new_entries).unique()
        changelog = changelog.loc[rows_to_keep].sort_values(by=[*col, "Version"])

    if changelog.empty:
        print("No changes")

    return changelog


# =================== #
# Notebook management #
# =================== #


def save_notebook():
    """
    Save the current notebook opened in JupyterLab to disk.

    Args:
        None

    Returns:
        None
    """
    app = JupyterFrontEnd()
    app.commands.execute("docmanager:save")
    print("Notebook saved to disk")


def run_notebooks(
    reports: Union[str, List[str]] = "all",
    progress_handler: Callable = None,
    telegram_first: bool = False,
    suppress_contribs: bool = False,
    **kwargs,
):
    """
    Execute specified Jupyter notebooks in the source directory using Papermill.

    Args:
        reports (Union[str, List[str]]): List of notebooks to execute or 'all' to execute all notebooks in the source directory. Default is 'all'.
        progress_handler (callable): An optional callable to provide progress bar functionality. Default is None.
        telegram_first (bool, optional): Default is False.
        suppress_contribs (bool, optional): Default is False.
        **kwargs: Additional keyword arguments containing secrets key-value pairs to pass to TQDM contrib iterators.

    Returns:
        None
    """
    debug = kwargs.pop("debug", False)

    if reports == "all":
        # Get reports
        reports = sorted(glob.glob("*.ipynb"))
    else:
        # TODO eliminate the need to add file extension
        reports = [str(reports)] if not isinstance(reports, list) else reports

    if progress_handler:
        external_pbar = progress_handler.pbar(
            iterable=reports, desc="Completion", unit="report", unit_scale=True
        )
    else:
        external_pbar = None

    # Initialize iterators
    iterator = tqdm(
        reports,
        desc="Completion",
        unit="report",
        unit_scale=True,
        dynamic_ncols=True,
    )

    if not suppress_contribs:
        contribs = ["DISCORD", "TELEGRAM"]
        if telegram_first:
            contribs = contribs[::-1]
        secrets = {
            key: value
            for key, value in kwargs.items()
            if (value is not None) and ("TOKEN" in key) or ("CHANNEL_ID") in key
        }
        secrets_file = os.path.join(PARENT_DIR, "assets/secrets.env")
        for contrib in contribs:
            required_secrets = [
                f"{contrib}_" + key if key == "CHANNEL_ID" else key
                for key in [f"{contrib}_TOKEN", f"CHANNEL_ID"]
                if key not in secrets
            ]
            try:
                loaded_secrets = load_secrets(
                    required_secrets, secrets_file=secrets_file, required=True
                )
                secrets = secrets | loaded_secrets

                token = secrets.get(f"{contrib}_TOKEN")
                channel_id = secrets.get(
                    f"{contrib}_CHANNEL_ID", secrets.get("CHANNEL_ID")
                )
                if contrib == "DISCORD":
                    from tqdm.contrib.discord import tqdm as contrib_tqdm

                    channel_id_dict = {"channel_id": channel_id}

                elif contrib == "TELEGRAM":
                    from tqdm.contrib.telegram import tqdm as contrib_tqdm

                    channel_id_dict = {"chat_id": channel_id}

                iterator = contrib_tqdm(
                    reports,
                    desc="Completion",
                    unit="report",
                    unit_scale=True,
                    dynamic_ncols=True,
                    token=token,
                    **channel_id_dict,  # Needed to handle Telegram using chat_ID instaed of channel_ID.
                )

                break
            except:
                pass

    # Create the main logger
    logger = logging.getLogger("papermill")
    logger.setLevel(logging.INFO)

    # Create a StreamHandler and attach it to the logger
    stream_handler = logging.StreamHandler(io.StringIO())
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    stream_handler.addFilter(
        lambda record: record.getMessage().startswith("Ending Cell")
    )
    logger.addHandler(stream_handler)

    # Define a function to update the output variable
    def update_pbar():
        iterator.update((1 / cells))
        if external_pbar:
            external_pbar.update((1 / cells))

    exceptions = []
    for i, report in enumerate(iterator):
        iterator.n = i
        iterator.last_print_n = i
        iterator.refresh()
        report_name = os.path.basename(report)[:-6]

        with open(report) as f:
            nb = nbformat.read(f, nbformat.NO_CONVERT)
            cells = len(nb.cells)
            # print(f'Number of Cells: {cells}')

        # Attach the update_pbar function to the stream_handler
        stream_handler.flush = update_pbar

        # Update postfix
        tqdm.write(f"Generating {report_name} report")
        iterator.set_postfix(report=report_name)
        if external_pbar:
            external_pbar.set_postfix(report=report_name)

        # execute the notebook with papermill
        os.environ["PM_IN_EXECUTION"] = "True"
        if "yugiquery" in jupyter_client.kernelspec.find_kernel_specs():
            kernel_name = "yugiquery"
        else:
            kernel_name = "python3"

        try:
            pm.execute_notebook(
                report,
                report,
                log_output=True,
                progress_bar=True,
                kernel_name=kernel_name,
            )
        except pm.PapermillExecutionError as e:
            exceptions.append(e)
        finally:
            os.environ.pop("PM_IN_EXECUTION", None)

    # Close the iterator
    iterator.close()
    if external_pbar:
        external_pbar.close()

    # Close the stream_handler
    stream_handler.close()
    # Clear custom handler
    logger.handlers.clear()

    if exceptions:
        combined_message = "\n".join(str(e) for e in exceptions)
        raise Exception(combined_message)


# ================ #
# Markdown editing #
# ================ #


def update_index():  # Handle index and readme properly
    """
    Update the index.md and README.md files with a table of links to all HTML reports in the parent directory.
    Also update the @REPORT_|_TIMESTAMP@ and @TIMESTAMP@ placeholders in the index.md file with the latest timestamp.
    If the update is successful, commit the changes to Git with a commit message that includes the timestamp.
    If there is no index.md or README.md files in the assets directory, print an error message and abort.

    Raises:
        FileNotFoundError: If the "index.md" or "README.md" files in "assets" are not found.

    Returns:
        None
    """

    index_file_name = "index.md"
    readme_file_name = "README.md"

    index_input_path = os.path.join(PARENT_DIR, "assets/markdown", index_file_name)
    readme_input_path = os.path.join(PARENT_DIR, "assets/markdown", readme_file_name)
    index_output_path = os.path.join(PARENT_DIR, index_file_name)
    readme_output_path = os.path.join(PARENT_DIR, readme_file_name)

    timestamp = arrow.utcnow()
    try:
        with open(index_input_path) as f:
            index = f.read()

        with open(readme_input_path) as f:
            readme = f.read()
    except:
        print('Missing template files in "assets". Aborting...')

    reports = sorted(glob.glob(os.path.join(PARENT_DIR, "*.html")))
    rows = []
    for report in reports:
        rows.append(
            f"[{os.path.basename(report).split('.')[0]}]({os.path.basename(report)}) | {pd.to_datetime(os.path.getmtime(report),unit='s', utc=True).strftime('%d/%m/%Y %H:%M %Z')}"
        )
    table = " |\n| ".join(rows)

    index = index.replace(f"@REPORT_|_TIMESTAMP@", table)
    index = index.replace(f"@TIMESTAMP@", timestamp.strftime("%d/%m/%Y %H:%M %Z"))

    with open(index_output_path, "w+") as o:
        print(index, file=o)

    readme = readme.replace(f"@REPORT_|_TIMESTAMP@", table)
    readme = readme.replace(f"@TIMESTAMP@", timestamp.strftime("%d/%m/%Y %H:%M %Z"))

    with open(readme_output_path, "w+") as o:
        print(readme, file=o)

    result = commit(
        files=[index_output_path, readme_output_path],
        commit_message=f"Index and README timestamp update - {timestamp.isoformat()}",
    )
    print(result)


def header(name: str = None):
    """
    Generates a Markdown header with a timestamp and the name of the notebook (if provided).

    Args:
        name (str, optional): The name of the notebook. If None, attempts to extract the name from the environment variable JPY_SESSION_NAME. Defaults to None.

    Returns:
        Markdown: The generated Markdown header.
    """
    if name is None:
        try:
            name = os.path.basename(os.environ["JPY_SESSION_NAME"]).split(".")[0]
        except:
            name = ""

    with open(os.path.join(PARENT_DIR, "assets/markdown/header.md")) as f:
        header = f.read()
        header = header.replace(
            "@TIMESTAMP@",
            arrow.utcnow().strftime("%d/%m/%Y %H:%M %Z"),
        )
        header = header.replace("@NOTEBOOK@", name)
        return Markdown(header)


def footer(timestamp: arrow.Arrow = None):
    """
    Generates a Markdown footer with a timestamp.

    Args:
        timestamp (arrow.Arrow, optional): The timestamp to use. If None, uses the current time. Defaults to None.

    Returns:
        Markdown: The generated Markdown footer.
    """
    with open(os.path.join(PARENT_DIR, "assets/markdown/footer.md")) as f:
        footer = f.read()
        now = arrow.utcnow()
        footer = footer.replace("@TIMESTAMP@", now.strftime("%d/%m/%Y %H:%M %Z"))

        return Markdown(footer)


# ================== #
# API call functions #
# ================== #

# Variables
http_headers = {"User-Agent": "Yugiquery/1.0 - https://guigoruiz1.github.io/yugiquery/"}
base_url = "https://yugipedia.com/api.php"
media_url = "https://ws.yugipedia.com/"
revisions_query_action = (
    "?action=query&format=json&prop=revisions&rvprop=content&titles="
)
ask_query_action = "?action=ask&format=json&query="
askargs_query_action = "?action=askargs&format=json&conditions="
categorymembers_query_action = "?action=query&format=json&list=categorymembers&cmdir=desc&cmsort=timestamp&cmtitle=Category:"
redirects_query_action = "?action=query&format=json&redirects=True&titles="
backlinks_query_action = (
    "?action=query&format=json&list=backlinks&blfilterredir=redirects&bltitle="
)


# Functions
def extract_results(response: requests.Response):
    """
    Extracts the relevant data from the response object and returns it as a Pandas DataFrame.

    Args:
        response (requests.Response): The response object obtained from making a GET request to the Yu-Gi-Oh! Wiki API.

    Returns:
        pd.DataFrame: A DataFrame containing the relevant data extracted from the response object.
    """
    json = response.json()
    df = pd.DataFrame(json["query"]["results"]).transpose()
    if "printouts" in df:
        df = pd.DataFrame(df["printouts"].values.tolist(), index=df["printouts"].keys())
        page_url = (
            pd.DataFrame(json["query"]["results"])
            .transpose()["fullurl"]
            .rename("Page URL")
        )
        page_name = (
            pd.DataFrame(json["query"]["results"])
            .transpose()["fulltext"]
            .rename("Page name")
        )  # Not necessarily same as card name (Used to merge errata)
        df = pd.concat([df, page_name, page_url], axis=1)

    return df


def card_query(default: str = None, *args, **kwargs):
    """
    Builds a string of arguments to be passed to the yugipedia Wiki API for a card search query.

    Args:
        default (str, optional): The default card type to build a query string for. Can be one of {'spell', 'trap', 'st', 'monster', 'skill', 'counter', 'speed', 'rush', None}. Defaults to None.
        *args: Additional positional arguments to be passed to the API.
        **kwargs: Additional keyword arguments to be passed to the API.

    Raises:
        ValueError: If default is not a valid card type.

    Returns:
        str: A string containing the arguments to be passed to the API for the card search query.
    """
    # Default card query
    prop_bool = {
        "password": True,
        "card_type": True,
        "property": True,
        "primary": True,
        "secondary": True,
        "attribute": True,
        "monster_type": True,
        "stars": True,
        "atk": True,
        "def": True,
        "scale": True,
        "link": True,
        "arrows": True,
        "effect_type": True,
        "archseries": True,
        "alternate_artwork": True,
        "edited_artwork": True,
        "tcg": True,
        "ocg": True,
        "date": True,
    }

    if default is not None:
        default = default.lower()
    valid_default = {
        "spell",
        "trap",
        "st",
        "monster",
        "skill",
        "counter",
        "speed",
        "rush",
        None,
    }
    if default not in valid_default:
        raise ValueError("results: default must be one of %r." % valid_default)
    elif default == "monster":
        prop_bool.update({"property": False})
    elif default == "st" or default == "trap" or default == "spell":
        prop_bool.update(
            {
                "primary": False,
                "secondary": False,
                "attribute": False,
                "monster_type": False,
                "stars": False,
                "atk": False,
                "def": False,
                "scale": False,
                "link": False,
                "arrows": False,
            }
        )
    elif default == "counter":
        prop_bool.update(
            {
                "primary": False,
                "secondary": False,
                "attribute": False,
                "monster_type": False,
                "property": False,
                "stars": False,
                "atk": False,
                "def": False,
                "scale": False,
                "link": False,
                "arrows": False,
            }
        )
    elif default == "skill":
        prop_bool.update(
            {
                "password": False,
                "primary": False,
                "secondary": False,
                "attribute": False,
                "monster_type": False,
                "stars": False,
                "atk": False,
                "def": False,
                "scale": False,
                "link": False,
                "arrows": False,
                "effect_type": False,
                "edited_artwork": False,
                "alternate_artwork": False,
                "ocg": False,
                "speed": True,
                "character": True,
            }
        )
    elif default == "speed":
        prop_bool.update(
            {
                "speed": True,
                "scale": False,
                "link": False,
                "arrows": False,
            }
        )
    elif default == "rush":
        prop_bool.update(
            {
                "password": False,
                "secondary": False,
                "scale": False,
                "link": False,
                "arrows": False,
                "tcg": False,
                "ocg": False,
                "maximum_atk": True,
                "edited_artwork": False,
                "alternate_artwork": False,
                "rush_alt_artwork": True,
                "rush_edited_artwork": True,
                "misc": True,
            }
        )

    # Card properties dictionary
    prop_dict = {
        "password": "|?Password",
        "card_type": "|?Card%20type",
        "property": "|?Property",
        "primary": "|?Primary%20type",
        "secondary": "|?Secondary%20type",
        "attribute": "|?Attribute",
        "monster_type": "|?Type=Monster%20type",
        "stars": "|?Stars%20string=Level%2FRank%20",
        "atk": "|?ATK%20string=ATK",
        "def": "|?DEF%20string=DEF",
        "scale": "|?Pendulum%20Scale",
        "link": "|?Link%20Rating=Link",
        "arrows": "|?Link%20Arrows",
        "effect_type": "|?Effect%20type",
        "archseries": "|?Archseries",
        "alternate_artwork": "|?Category:OCG/TCG%20cards%20with%20alternate%20artworks",
        "edited_artwork": "|?Category:OCG/TCG%20cards%20with%20edited%20artworks",
        "tcg": "|?TCG%20status",
        "ocg": "|?OCG%20status",
        "date": "|?Modification%20date",
        "image_URL": "|?Card%20image",
        "misc": "|?Misc",
        "summoning": "|?Summoning",
        # Speed duel specific
        "speed": "|?TCG%20Speed%20Duel%20status",
        "character": "|?Character",
        # Rush duel specific
        "rush_alt_artwork": "|?Category:Rush%20Duel%20cards%20with%20alternate%20artworks",
        "rush_edited_artwork": "|?Category:Rush%20Duel%20cards%20with%20edited%20artworks",
        "maximum_atk": "|?MAXIMUM%20ATK",
        # Deprecated - Use for debuging
        "category": "|?category",
    }
    # Change default values to kwargs values
    prop_bool.update(kwargs)
    # Initialize string
    search_string = "|?English%20name=Name"
    # Iterate default plus kwargs items
    for arg, value in prop_bool.items():
        # If property is true
        if value:
            # If property in the dictionary, get its value
            if arg in prop_dict.keys():
                search_string += f"{prop_dict[arg]}"
            # If property is not in the dictionary, assume generic property
            else:
                print(f"Unrecognized property {arg}. Assuming |?{up.quote(arg)}.")
                search_string += f"|?{up.quote(arg)}"

    for arg in args:
        search_string += f"|?{up.quote(arg)}"

    return search_string


def check_API_status():
    """
    Checks if the API is running and reachable by making a query to retrieve site information. If the API is up and running, returns True. If the API is down or unreachable, returns False and prints an error message with details.

    Returns:
        bool: True if the API is up and running, False otherwise.
    """
    params = {
        "action": "query",
        "meta": "siteinfo",
        "siprop": "general",
        "format": "json",
    }

    try:
        response = requests.get(base_url, params=params, headers=http_headers)
        response.raise_for_status()
        print(
            f"{base_url} is up and running {response.json()['query']['general']['generator']}"
        )
        return True
    except requests.exceptions.RequestException as err:
        print(f"{base_url} is not alive: {err}")
        domain = up.urlparse(base_url).netloc
        port = 443

        try:
            socket.create_connection((domain, port), timeout=2)
            print(f"{domain} is reachable")
        except OSError as err:
            print(f"{domain} is not reachable: {err}")

        return False


def fetch_categorymembers(
    category: str,
    namespace: int = 0,
    step: int = 500,
    iterator=None,
    debug: bool = False,
):
    """
    Fetches members of a category from the API by making iterative requests with a specified step size until all members are retrieved.

    Args:
        category (str): The category to retrieve members for.
        namespace (int, optional): The namespace ID to filter the members by. Defaults to 0 (main namespace).
        step (int, optional): The number of members to retrieve in each request. Defaults to 500.
        iterator (tqdm.std.tqdm, optional): A tqdm iterator to display progress updates. Defaults to None.
        debug (bool, optional): If True, prints the URL of each request for debugging purposes. Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing the members of the category.
    """
    params = {"cmlimit": step, "cmnamespace": namespace}

    lastContinue = {}
    all_results = []
    i = 0
    with Halo(
        text="Fetching category members...",
        spinner="line",
        enabled=("PM_IN_EXECUTION" not in os.environ),
    ) as spinner:
        try:
            while True:
                if iterator is None:
                    spinner.text = f"Fetching category members... Iteration {i+1}"
                else:
                    iterator.set_postfix(it=i + 1)

                params = params.copy()
                params.update(lastContinue)
                response = requests.get(
                    f"{base_url}{categorymembers_query_action}{category}",
                    params=params,
                    headers=http_headers,
                )
                if debug:
                    print(response.url)
                if response.status_code != 200:
                    spinner.fail(f"HTTP error code {response.status_code}")
                    break

                result = response.json()
                if "error" in result:
                    spinner.fail(result["error"]["info"])
                    # raise Exception(result['error']['info'])
                if "warnings" in result:
                    spinner.warn(result["warnings"])
                    # print(result['warnings'])
                if "query" in result:
                    all_results += result["query"]["categorymembers"]
                if "continue" not in result:
                    spinner.succeed("Fetch completed")
                    break
                lastContinue = result["continue"]
                i += 1

            if "PM_IN_EXECUTION" not in os.environ:
                time.sleep(0.5)

        except (KeyboardInterrupt, SystemExit):
            spinner.fail("Execution interrupted.")
            if "PM_IN_EXECUTION" not in os.environ:
                time.sleep(0.5)
            raise

        spinner.output.close()

    results_df = pd.DataFrame(all_results)
    return results_df


def fetch_properties(
    condition: str,
    query: str,
    step: int = 500,
    limit: int = 5000,
    iterator=None,
    include_all: bool = False,
    debug: bool = False,
):
    """
    Fetches properties from the API by making iterative requests with a specified step size until a specified limit is reached.

    Args:
        condition (str): The query condition to filter the properties by.
        query (str): The query to retrieve the properties.
        step (int, optional): The number of properties to retrieve in each request. Defaults to 500.
        limit (int, optional): The maximum number of properties to retrieve. Defaults to 5000.
        iterator (tqdm.std.tqdm, optional): A tqdm iterator to display progress updates. Defaults to None.
        include_all (bool, optional): If True, includes all properties in the DataFrame. If False, includes only properties that have values. Defaults to False.
        debug (bool, optional): If True, prints the URL of each request for debugging purposes. Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing the properties matching the query and condition.
    """
    df = pd.DataFrame()
    i = 0
    complete = False
    with Halo(
        text="Fetching properties...",
        spinner="line",
        enabled=("PM_IN_EXECUTION" not in os.environ),
    ) as spinner:
        try:
            while not complete:
                if iterator is None:
                    # spinner.clear()
                    spinner.text = f"Fetching properties... Iteration {i+1}"
                else:
                    iterator.set_postfix(it=i + 1)

                response = requests.get(
                    f"{base_url}{ask_query_action}{condition}{query}|limit%3D{step}|offset={i*step}|order%3Dasc",
                    headers=http_headers,
                )
                if debug:
                    print(response.url)
                if response.status_code != 200:
                    spinner.fail(f"HTTP error code {response.status_code}")
                    break

                result = extract_results(response)
                formatted_df = format_df(result, include_all=include_all)
                df = pd.concat([df, formatted_df], ignore_index=True, axis=0)

                if debug:
                    tqdm.write(f"Iteration {i+1}: {len(formatted_df.index)} results")

                if len(formatted_df.index) < step or (i + 1) * step >= limit:
                    spinner.succeed("Fetch completed")
                    complete = True
                else:
                    i += 1

            if "PM_IN_EXECUTION" not in os.environ:
                time.sleep(0.5)

        except (KeyboardInterrupt, SystemExit):
            spinner.fail("Execution interrupted.")
            if "PM_IN_EXECUTION" not in os.environ:
                time.sleep(0.5)
            raise

        spinner.output.close()

    return df


def fetch_redirects(titles: List[str]):
    """
    Fetches redirects for a list of page titles.

    Args:
        titles (List[str]): A list of titles.

    Returns:
        Dict[str, str]: A dictionary mapping source titles to their corresponding redirect targets.
    """
    results = {}
    iterator = trange(
        np.ceil(len(titles) / 50).astype(int), desc="Redirects", leave=False
    )
    for i in iterator:
        first = i * 50
        last = (i + 1) * 50
        target_titles = "|".join(titles[first:last])
        response = requests.get(
            base_url + redirects_query_action + target_titles, headers=http_headers
        ).json()
        redirects = response["query"]["redirects"]
        for redirect in redirects:
            results[redirect.get("from", "")] = redirect.get("to", "")

    return results


def fetch_backlinks(titles: List[str]):
    """
    Fetches backlinks for a list of page titles.

    Args:
        titles (List[str]): A list of titles.

    Returns:
        Dict[str, str]: A dictionary mapping backlink titles to their corresponding target titles.
    """
    results = {}
    iterator = tqdm(titles, desc="Backlinks", leave=False)
    for target_title in iterator:
        iterator.set_postfix(title=target_title)
        response = requests.get(
            base_url + backlinks_query_action + target_title, headers=http_headers
        ).json()
        backlinks = response["query"]["backlinks"]
        for backlink in backlinks:
            if re.match(r"^[a-zA-Z]+$", backlink["title"]) and backlink[
                "title"
            ] not in target_title.split(" "):
                results[backlink["title"]] = target_title

    return results


## Rarities dictionary


def fetch_rarities_dict(rarities_list: List[str] = []):
    """
    Fetches backlinks and redirects for a list of rarities, including abbreviations, to generate a map of rarity abbreviations to their corresponding names.

    Args:
        rarities_list (List[str]): A list of rarities.

    Returns:
        Dict[str, str]: A dictionary mapping rarity abbreviations to their corresponding names.

    """
    words, acronyms = separate_words_and_acronyms(rarities_list)
    if len(rarities_list) > 0:
        print(f"Words: {words}")
        print(f"Acronyms: {acronyms}")

    titles = fetch_categorymembers(category="Rarities", namespace=0)["title"]
    words = words + titles.tolist()
    rarity_backlinks = fetch_backlinks(words)
    rarity_redirects = fetch_redirects(acronyms)
    rarity_dict = rarity_backlinks | rarity_redirects

    return rarity_dict


## Bandai


def fetch_bandai(limit: int = 200, *args, **kwargs):
    """
    Fetch Bandai cards.

    Args:
        limit (int, optional): An integer that represents the maximum number of results to fetch. Defaults to 200.
        *args: Additional properties to query.
        **kwargs: keyword arguments to disable specific properties from query. Remaining keword arguments are passed to fetch_properties()

    Returns:
        pandas.DataFrame: A pandas DataFrame object containing the properties of the fetched Bandai cards.
    """
    debug = kwargs.get("debug", False)
    bandai_query = "|?English%20name=Name"
    bandai_prop_dict = {
        "card_type": "|?Card%20type",
        "level": "|?Level",
        "atk": "|?ATK",
        "def": "|?DEF",
        "number": "|?Bandai%20number=Card%20number",
        "type": "|?Type=Monster%20type",
        "rule": "|?Bandai%20rule=Rule",
        "sets": "|?Sets=Set",
        "rarity": "|?Rarity",
        "ability": "|?Ability",
        "date": "|?Modification%20date",
    }
    for key, value in bandai_prop_dict.items():
        if key in kwargs:
            disable = not kwargs.pop(key)
            if disable:
                continue
        bandai_query += value

    for arg in args:
        bandai_query += f"|?{up.quote(arg)}"

    print(f"Downloading bandai cards")
    concept = "[[Medium::Bandai]]"
    bandai_df = fetch_properties(
        concept, bandai_query, step=limit, limit=limit, **kwargs
    )
    if "Monster type" in bandai_df:
        bandai_df["Monster type"] = (
            bandai_df["Monster type"].dropna().apply(lambda x: x.split("(")[0])
        )  # Temporary
    if debug:
        print("- Total")

    print(f"{len(bandai_df.index)} results\n")

    time.sleep(0.5)

    return bandai_df


## Cards


def fetch_st(
    st_query: str = None,
    st: str = "both",
    cg: CG = CG.ALL,
    step: int = 500,
    limit: int = 5000,
    **kwargs,
):
    """
    Fetch spell or trap cards based on query and properties of the cards.

    Args:
        st_query (str, optional): A string representing a SMW query to search for. Defaults to None.
        st (str, optional): A string representing the type of cards to fetch, either "spell", "trap", "both", or "all". Defaults to "both".
        cg (CG, optional): An Enum that represents the card game to fetch cards from. Defaults to CG.ALL.
        step (int, optional): An integer that represents the number of results to fetch at a time. Defaults to 500.
        limit (int, optional): An integer that represents the maximum number of results to fetch. Defaults to 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pandas.DataFrame: A pandas DataFrame object containing the properties of the fetched spell/trap cards.

    Raises:
        ValueError: Raised if the "st" argument is not one of "spell", "trap", "both", or "all".
    """
    debug = kwargs.get("debug", False)
    st = st.capitalize()
    valid_st = {"Spell", "Trap", "Both", "All"}
    valid_cg = cg.value
    concept = f"[[Concept:CG%20non-monster%20cards]]"
    if st not in valid_st:
        raise ValueError("results: st must be one of %r." % valid_st)
    elif st == "Both" or st == "All":
        st = "Spells and Trap"
    else:
        concept += f"[[Card type::{st} card]]"
    if valid_cg != "CG":
        concept += f"[[Medium::{valid_cg}]]"
    print(f"Downloading {st}s")
    if st_query is None:
        st_query = card_query(default="st")

    st_df = fetch_properties(concept, st_query, step=step, limit=limit, **kwargs)

    if debug:
        print("- Total")

    print(f"{len(st_df.index)} results\n")

    return st_df


def fetch_monster(
    monster_query: str = None,
    cg: CG = CG.ALL,
    step: int = 500,
    limit: int = 5000,
    exclude_token=True,
    **kwargs,
):
    """
    Fetch monster cards based on query and properties of the cards.

    Args:
        monster_query (str, optional): A string representing a SMW query to search for. Defaults to None.
        cg (CG, optional): An Enum that represents the card game to fetch cards from. Defaults to CG.ALL.
        step (int, optional): An integer that represents the number of results to fetch at a time. Defaults to 500.
        limit (int, optional): An integer that represents the maximum number of results to fetch. Defaults to 5000.
        exclude_token (bool, optional): A boolean that determines whether to exclude Monster Tokens or not. Defaults to True.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pandas.DataFrame: A pandas DataFrame object containing the properties of the fetched monster cards.
    """
    debug = kwargs.get("debug", False)
    valid_cg = cg.value
    attributes = ["DIVINE", "LIGHT", "DARK", "WATER", "EARTH", "FIRE", "WIND", "?"]
    print("Downloading monsters")
    if monster_query is None:
        monster_query = card_query(default="monster")

    monster_df = pd.DataFrame()
    iterator = tqdm(
        attributes,
        leave=False,
        unit="attribute",
        disable=("PM_IN_EXECUTION" in os.environ),
    )
    for att in iterator:
        iterator.set_description(att)
        if debug:
            tqdm.write(f"- {att}")

        concept = f"[[Concept:CG%20monsters]][[Attribute::{att}]]"

        if valid_cg != "CG":
            concept += f"[[Medium::{valid_cg}]]"

        temp_df = fetch_properties(
            concept, monster_query, step=step, limit=limit, iterator=iterator, **kwargs
        )
        monster_df = pd.concat([monster_df, temp_df], ignore_index=True, axis=0)

    if exclude_token and "Primary type" in monster_df:
        monster_df = monster_df[
            monster_df["Primary type"] != "Monster Token"
        ].reset_index(drop=True)

    if debug:
        print("- Total")

    print(f"{len(monster_df.index)} results\n")

    return monster_df


### Non deck cards


def fetch_token(
    token_query: str = None, cg=CG.ALL, step: int = 500, limit: int = 5000, **kwargs
):
    """
    Fetch token cards based on query and properties of the cards.

    Args:
        token_query (str, optional): A string representing a SWM query to search for. Defaults to None.
        step (int, optional): An integer that represents the number of results to fetch at a time. Defaults to 500.
        limit (int, optional): An integer that represents the maximum number of results to fetch. Defaults to 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pandas.DataFrame: A pandas DataFrame object containing the properties of the fetched token cards.

    """
    valid_cg = cg.value
    print("Downloading tokens")

    concept = f"[[Category:Tokens]]"
    if valid_cg != "CG":
        concept += f"[[Category:{valid_cg}%20cards]]"
    else:
        concept += "[[Category:TCG%20cards||OCG%20cards]]"

    if token_query is None:
        token_query = card_query(default="monster")

    token_df = fetch_properties(concept, token_query, step=step, limit=limit, **kwargs)

    print(f"{len(token_df.index)} results\n")

    return token_df


def fetch_counter(
    counter_query: str = None, cg=CG.ALL, step: int = 500, limit: int = 5000, **kwargs
):
    """
    Fetch counter cards based on query and properties of the cards.

    Args:
        counter_query (str, optional): A string representing a SMW query to search for. Defaults to None.
        step (int, optional): An integer that represents the number of results to fetch at a time. Defaults to 500.
        limit (int, optional): An integer that represents the maximum number of results to fetch. Defaults to 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pandas.DataFrame: A pandas DataFrame object containing the properties of the fetched counter cards.
    """
    valid_cg = cg.value
    print("Downloading counters")

    concept = f"[[Category:Counters]][[Page%20type::Card%20page]]"
    if valid_cg != "CG":
        concept += f"[[Medium::{valid_cg}]]"

    if counter_query is None:
        counter_query = card_query(default="counter")

    counter_df = fetch_properties(
        concept, counter_query, step=step, limit=limit, **kwargs
    )

    print(f"{len(counter_df.index)} results\n")

    return counter_df


### Alternative formats


def fetch_speed(speed_query: str = None, step: int = 500, limit: int = 5000, **kwargs):
    """
    Fetches TCG Speed Duel cards from the yugipedia Wiki API.

    Args:
        speed_query (str):  A string representing a SMW query to search for. Defaults to None.
        step (int): The number of results to fetch in each API call. Defaults to 500.
        limit (int): The maximum number of results to fetch. Defaults to 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        A pandas DataFrame containing the fetched TCG Speed Duel cards.
    """
    debug = kwargs.get("debug", False)

    print(f"Downloading Speed duel cards")
    concept = "[[Category:TCG Speed Duel cards]]"
    if speed_query is None:
        speed_query = card_query(default="speed")

    speed_df = fetch_properties(
        concept,
        speed_query,
        step=step,
        limit=limit,
        **kwargs,
    )

    if debug:
        print("- Total")

    print(f"{len(speed_df.index)} results\n")

    return speed_df


def fetch_skill(skill_query: str = None, step: int = 500, limit: int = 5000, **kwargs):
    """
    Fetches skill cards from the yugipedia Wiki API.

    Args:
        skill_query (str): A string representing a SMW query to search for. Defaults to None.
        step (int): The number of results to fetch in each API call. Defaults to 500.
        limit (int): The maximum number of results to fetch. Defaults to 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        A pandas DataFrame containing the fetched skill cards.
    """
    print("Downloading skill cards")

    concept = "[[Category:Skill%20Cards]][[Card type::Skill Card]]"
    if skill_query is None:
        skill_query = card_query(default="skill")

    skill_df = fetch_properties(concept, skill_query, step=step, limit=limit, **kwargs)

    print(f"{len(skill_df.index)} results\n")

    return skill_df


def fetch_rush(rush_query: str = None, step: int = 500, limit: int = 5000, **kwargs):
    """
    Fetches Rush Duel cards from the Yu-Gi-Oh! Wikia API.

    Args:
        rush_query (str): A search query to filter the results. If not provided, it defaults to "rush".
        step (int): The number of results to fetch in each API call. Defaults to 500.
        limit (int): The maximum number of results to fetch. Defaults to 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        A pandas DataFrame containing the fetched Rush Duel cards.
    """
    debug = kwargs.get("debug", False)
    print("Downloading Rush Duel cards")

    concept = f"[[Category:Rush%20Duel%20cards]][[Medium::Rush%20Duel]]"
    if rush_query is None:
        rush_query = card_query(default="rush")

    rush_df = fetch_properties(concept, rush_query, step=step, limit=limit, **kwargs)

    print(f"{len(rush_df.index)} results\n")

    return rush_df


### Unusable cards


def fetch_unusable(
    query: str = None,
    cg: CG = CG.ALL,
    filter=True,
    step: int = 500,
    limit: int = 5000,
    **kwargs,
):
    """
    Fetch unusable cards based on query and properties of the cards. Unusable cards include "Strategy cards", "Tip cards",
    "Card Checklists", etc, which are not actual cards. The filter option enables filtering those out and keeping only cards
    such as Duelist Kingdom "Ticket cards", old video-game promo "Character cards" and "Non-game cards" which have the layout
    of a real card, such as "Everyone's King". This criteria is not free of ambiguity.

    Args:
        query (str, optional): A string representing a SMW query to search for. Defaults to None.
        cg (CG, optional): An Enum that represents the card game to fetch cards from. Defaults to CG.ALL.
        filter (bool, optional): Keep only "Character Cards", "Non-game cards" and "Ticket Cards".
        step (int, optional): An integer that represents the number of results to fetch at a time. Defaults to 500.
        limit (int, optional): An integer that represents the maximum number of results to fetch. Defaults to 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pandas.DataFrame: A pandas DataFrame object containing the properties of the fetched spell/trap cards.

    """
    debug = kwargs.get("debug", False)
    concept = "[[Category:Unusable cards]]"

    valid_cg = cg.value
    if valid_cg == "CG":
        concept = "OR".join([concept + f"[[{s} status::+]]" for s in ["TCG", "OCG"]])
    else:
        concept += f"[[{valid_cg} status::+]]"

    concept = up.quote(concept)

    print(f"Downloading unusable cards")
    if query is None:
        query = card_query()

    unusable_df = fetch_properties(concept, query, step=step, limit=limit, **kwargs)

    if filter and "Card type" in unusable_df:
        unusable_df = unusable_df[
            unusable_df["Card type"].isin(
                ["Character Card", "Non-game card", "Ticket Card"]
            )
        ].reset_index(drop=True)

    unusable_df.dropna(how="all", axis=1, inplace=True)

    if debug:
        print("- Total")

    print(f"{len(unusable_df.index)} results\n")

    return unusable_df


### Extra properties


def fetch_errata(errata: str = "all", step: int = 500, **kwargs):
    """
    Fetches errata information from the yuipedia Wiki API.

    Args:
        errata (str): The type of errata information to fetch. Valid values are 'name', 'type', and 'all'. Defaults to 'all'.
        step (int): The number of results to fetch in each API call. Defaults to 500.
        **kwargs: Additional keyword arguments to pass to fetch_categorymembers.

    Returns:
        A pandas DataFrame containing a boolean table indicating whether each card has errata information for the specified type.
    """
    debug = kwargs.get("debug", False)
    errata = errata.lower()
    valid = {"name", "type", "all"}
    categories = {
        "all": "Card Errata",
        "type": "Cards with card type errata",
        "name": "Cards with name errata",
    }
    if errata not in valid:
        raise ValueError("results: errata must be one of %r." % valid)
    elif errata == "all":
        errata = "all"
        categories = list(categories.values())
    else:
        categories = list(categories["errata"])

    print(f"Downloading {errata} errata")
    errata_df = pd.DataFrame(dtype=bool)
    iterator = tqdm(
        categories,
        leave=False,
        unit="initial",
        disable=("PM_IN_EXECUTION" in os.environ),
    )
    for cat in iterator:
        desc = cat.split("Category:")[-1]
        iterator.set_description(desc)
        if debug:
            tqdm.write(f"- {cat}")

        temp = fetch_categorymembers(
            cat, namespace=3010, step=step, iterator=iterator, debug=debug
        )
        errata_data = temp["title"].apply(lambda x: x.split("Card Errata:")[-1])
        errata_series = pd.Series(data=True, index=errata_data, name=desc)
        errata_df = (
            pd.concat([errata_df, errata_series], axis=1)
            .astype("boolean")
            .fillna(False)
            .sort_index()
        )

    if debug:
        print("- Total")

    print(f"{len(errata_df.index)} results\n")
    return errata_df


## Sets


def fetch_set_list_pages(cg: CG = CG.ALL, step: int = 500, limit=5000, **kwargs):
    """
    Fetches a list of 'Set Card Lists' pages from the yugipedia Wiki API.

    Args:
        cg (CG): A member of the CG enum representing the card game for which set lists are being fetched.
        step (int): The number of pages to fetch in each API request.
        limit (int): The maximum number of pages to fetch.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pd.DataFrame: A DataFrame containing the titles of the set list pages.

    """
    debug = kwargs.get("debug", False)
    valid_cg = cg.value
    if valid_cg == "CG":
        category = ["TCG Set Card Lists", "OCG Set Card Lists"]
    else:
        category = f"{valid_cg}%20Set%20Card%20Lists"

    print("Downloading list of 'Set Card Lists' pages")
    set_list_pages = pd.DataFrame()
    result = pd.DataFrame()
    iterator = tqdm(
        category,
        leave=False,
        unit="category",
        disable=("PM_IN_EXECUTION" in os.environ),
    )
    for cat in iterator:
        iterator.set_description(cat.split("Category:")[-1])
        temp = fetch_categorymembers(
            cat, namespace=None, step=step, iterator=iterator, debug=debug
        )
        sub_categories = pd.DataFrame(temp)["title"]
        sub_iterator = tqdm(
            sub_categories,
            leave=False,
            unit="subcategory",
            disable=("PM_IN_EXECUTION" in os.environ),
        )
        for sub_cat in sub_iterator:
            sub_iterator.set_description(sub_cat.split("Category:")[-1])
            temp = fetch_properties(
                f"[[{sub_cat}]]",
                query="|?Modification date",
                step=limit,
                limit=limit,
                iterator=sub_iterator,
                **kwargs,
            )
            set_list_pages = pd.concat([set_list_pages, pd.DataFrame(temp)])

    return set_list_pages


def fetch_set_lists(titles: List[str], **kwargs):  # Separate formating function
    """
    Fetches card set lists from a list of page titles.

    Args:
        titles (List[str]): A list of page titles from which to fetch set lists.
        **kwargs: Additional keyword arguments.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed card set lists.
    """
    debug = kwargs.get("debug", False)
    if debug:
        print(f"{len(titles)} sets requested")

    titles = up.quote("|".join(titles))
    rarity_dict = load_json(os.path.join(PARENT_DIR, "assets/json/rarities.json"))
    set_lists_df = pd.DataFrame(
        columns=[
            "Set",
            "Card number",
            "Name",
            "Rarity",
            "Print",
            "Quantity",
            "Region",
            "Page name",
        ]
    )
    success = 0
    error = 0

    response = requests.get(
        f"{base_url}{revisions_query_action}{titles}", headers=http_headers
    )
    if debug:
        print(response.url)
    try:
        json = response.json()
    except:
        print(response.url)
        return

    contents = json["query"]["pages"].values()

    for content in contents:
        if "revisions" in content.keys():
            title = None
            raw = content["revisions"][0]["*"]
            parsed = wtp.parse(raw)
            for template in parsed.templates:
                if template.name == "Set page header":
                    for argument in template.arguments:
                        if "set=" in argument:
                            title = argument.value
                if template.name == "Set list":
                    set_df = pd.DataFrame(columns=set_lists_df.columns)
                    page_name = content["title"]

                    region = None
                    rarity = None
                    card_print = None
                    qty = None
                    desc = None
                    opt = None
                    list_df = None
                    extra_df = None

                    for argument in template.arguments:
                        if "region=" in argument:
                            region = argument.value
                            # if region = 'ES': # Remove second identifier for spanish
                            #     region = 'SP'

                        elif "rarities=" in argument:
                            rarity = tuple(
                                rarity_dict.get(
                                    (
                                        i[0].upper() + i[1:] if i[0].islower() else i
                                    ).strip(),  # Correct lower case accronymns (Example: c->C for common)
                                    i.strip(),
                                )
                                for i in (argument.value).split(",")
                            )

                        elif "print=" in argument:
                            card_print = argument.value

                        elif "qty=" in argument:
                            qty = argument.value

                        elif "description=" in argument:
                            desc = argument.value

                        elif "options=" in argument:
                            opt = argument.value

                        else:
                            set_list = argument.value[1:-1]
                            lines = set_list.split("\n")

                            list_df = pd.DataFrame([x.split(";") for x in lines])
                            list_df = list_df[~list_df[0].str.contains("!:")]

                            # Handle extra parameters passed as "// descriptions"
                            extra = list_df.map(
                                lambda x: (
                                    x.split("//")[1]
                                    if isinstance(x, str) and "//" in x
                                    else None
                                )
                            ).dropna(how="all")
                            if not extra.empty:
                                extra = extra.stack().droplevel(1, axis=0)
                                extra_lines = pd.DataFrame()
                                for extra_idx, extra_value in extra.items():
                                    if "::" in extra_value:
                                        col, val = extra_value.split("::")
                                        # Strip and process col and val to extract desired values
                                        col = col.strip().strip("@").lower()
                                        val = (
                                            val.strip()
                                            .strip("(")
                                            .strip(")")
                                            .split("]]")[0]
                                            .split("[[")[-1]
                                        )
                                        extra_lines.loc[extra_idx, col] = val

                                extra_lines = extra_lines.dropna(how="all")
                                if not extra_lines.empty:
                                    extra_df = extra_lines
                            ###

                            list_df = list_df.map(
                                lambda x: x.split("//")[0] if x is not None else x
                            )

                            list_df = list_df.map(
                                lambda x: x.strip() if x is not None else x
                            )
                            list_df.replace(
                                r"^\s*$|^@.*$", None, regex=True, inplace=True
                            )

                    if list_df is None:
                        error += 1
                        if debug:
                            print(f'Error! Unable to parse template for "{page_name}"')
                        continue

                    noabbr = opt == "noabbr"
                    set_df["Name"] = list_df[1 - noabbr].apply(
                        lambda x: (
                            x.strip("\u200e").split(" (")[0] if x is not None else x
                        )
                    )

                    if not noabbr and len(list_df.columns > 1):
                        set_df["Card number"] = list_df[0]

                    if len(list_df.columns) > (2 - noabbr):  # and rare in str
                        set_df["Rarity"] = list_df[2 - noabbr].apply(
                            lambda x: (
                                tuple(
                                    [
                                        rarity_dict.get(y.strip(), y.strip())
                                        for y in x.split(",")
                                    ]
                                )
                                if x is not None and "description::" not in x
                                else rarity
                            )
                        )

                    else:
                        set_df["Rarity"] = [rarity for _ in set_df.index]

                    if len(list_df.columns) > (3 - noabbr):
                        if card_print is not None:  # and new/reprint in str
                            set_df["Print"] = list_df[3 - noabbr].apply(
                                lambda x: (
                                    card_print if (card_print and x is None) else x
                                )
                            )

                            if len(list_df.columns) > (4 - noabbr) and qty:
                                set_df["Quantity"] = list_df[4 - noabbr].apply(
                                    lambda x: x if x is not None else qty
                                )

                        elif qty:
                            set_df["Quantity"] = list_df[3 - noabbr].apply(
                                lambda x: x if x is not None else qty
                            )

                    if not title:
                        title = page_name.split("Lists:")[1]

                    # Handle token name and print in description
                    if extra_df is not None:
                        for row in set_df.index:
                            # Handle token name in description
                            if (
                                "description" in extra_df
                                and row in extra_df["description"].dropna().index
                            ):
                                if (
                                    set_df.at[row, "Name"] is not None
                                    and "Token" in set_df.at[row, "Name"]
                                    and "Token" in extra_df.at[row, "description"]
                                ):
                                    set_df.at[row, "Name"] = extra_df.at[
                                        row, "description"
                                    ]

                            # Handle print in description
                            if (
                                "print" in extra_df
                                and row in extra_df["print"].dropna().index
                            ):
                                set_df.at[row, "Print"] = extra_df.at[row, "print"]
                    ###

                    set_df["Set"] = re.sub(r"\(\w{3}-\w{2}\)\s*$", "", title).strip()
                    set_df["Region"] = region.upper()
                    set_df["Page name"] = page_name
                    set_lists_df = (
                        pd.concat([set_lists_df, set_df], ignore_index=True)
                        .infer_objects(copy=False)
                        .fillna(np.nan)
                    )
                    success += 1

        else:
            error += 1
            if debug:
                print(f"Error! No content for \"{content['title']}\"")

    if debug:
        print(f"{success} set lists received - {error} missing")
        print("-------------------------------------------------")

    return set_lists_df, success, error


def fetch_all_set_lists(cg: CG = CG.ALL, step: int = 40, **kwargs):
    """
    Fetches all set lists for a given card game.

    Args:
        cg (CG, optional): The card game to fetch set lists for. Defaults to CG.ALL.
        step (int, optional): The number of sets to fetch at once. Defaults to 50.
        **kwargs: Additional keyword arguments to pass to fetch_set_list_pages and fetch_set_lists.

    Returns:
        pd.DataFrame: A DataFrame containing all set lists for the specified card game.

    Raises:
        Any exceptions raised by fetch_set_list_pages() or fetch_set_lists().
    """
    debug = kwargs.get("debug", False)
    sets = fetch_set_list_pages(cg, **kwargs)  # Get list of sets
    keys = sets["Page name"]

    all_set_lists_df = pd.DataFrame(
        columns=["Set", "Card number", "Name", "Rarity", "Print", "Quantity", "Region"]
    )
    total_success = 0
    total_error = 0

    for i in trange(np.ceil(len(keys) / step).astype(int), leave=False):
        success = 0
        error = 0
        if debug:
            tqdm.write(f"Iteration {i}:")

        first = i * step
        last = (i + 1) * step

        set_lists_df, success, error = fetch_set_lists(keys[first:last], **kwargs)
        set_lists_df = set_lists_df.merge(sets, on="Page name", how="left").drop(
            "Page name", axis=1
        )
        all_set_lists_df = pd.concat(
            [all_set_lists_df, set_lists_df], ignore_index=True
        )
        total_success += success
        total_error += error

    all_set_lists_df = all_set_lists_df.convert_dtypes()
    all_set_lists_df.sort_values(by=["Set", "Region", "Card number"]).reset_index(
        inplace=True
    )
    print(
        f'{"Total: " if debug else ""}{total_success} set lists received - {total_error} missing'
    )

    return all_set_lists_df


def fetch_set_info(
    sets: List[str], extra_info: List[str] = [], step: int = 15, **kwargs
):
    """
    Fetches information for a list of sets.

    Args:
        sets (List[str]): A list of set names to fetch information for.
        extra_info (List[str], optional): A list of additional information to fetch for each set. Defaults to an empty list.
        step (int, optional): The number of sets to fetch information for at once. Defaults to 15.
        **kwargs: Additional keyword arguments.

    Returns:
        pd.DataFrame: A DataFrame containing information for all sets in the list.

    Raises:
        Any exceptions raised by requests.get().
    """
    debug = kwargs.get("debug", False)
    if debug:
        print(f"{len(titles)} sets requested")

    regions_dict = load_json(os.path.join(PARENT_DIR, "assets/json/regions.json"))
    # Info to ask
    info = extra_info + ["Series", "Set type", "Cover card"]
    # Release to ask
    release = [i + " release date" for i in set(regions_dict.values())]
    # Ask list
    ask = up.quote("|".join(np.append(info, release)))

    # Get set info
    set_info_df = pd.DataFrame()
    for i in trange(np.ceil(len(sets) / step).astype(int), leave=False):
        first = i * step
        last = (i + 1) * step
        titles = up.quote("]]OR[[".join(sets[first:last]))
        response = requests.get(
            f"{base_url}{askargs_query_action}{titles}&printouts={ask}",
            headers=http_headers,
        )
        formatted_response = extract_results(response)
        formatted_response.drop(
            "Page name", axis=1, inplace=True
        )  # Page name not needed - no set errata, set name same as page name
        formatted_df = format_df(
            formatted_response, include_all=(True if extra_info else True)
        )
        if debug:
            tqdm.write(
                f"Iteration {i}\n{len(formatted_df)} set properties downloaded - {step-len(formatted_df)} errors"
            )
            tqdm.write("-------------------------------------------------")

        set_info_df = pd.concat([set_info_df, formatted_df.dropna(axis=1, how="all")])

    set_info_df = set_info_df.convert_dtypes()
    set_info_df.sort_index(inplace=True)

    print(
        f'{"Total:" if debug else ""}{len(set_info_df)} set properties received - {len(sets)-len(set_info_df)} errors'
    )

    return set_info_df


# ================== #
# Plotting functions #
# ================== #

# Variables
colors_dict = load_json(
    os.path.join(PARENT_DIR, "assets/json/colors.json")
)  # Colors dictionary to associate to series and cards

# Functions


def adjust_lightness(color: str, amount: float = 0.5):
    """Adjust the lightness of a given color by a specified amount.

    Args:
        color (str): The color to be adjusted, in string format.
        amount (float): The amount by which to adjust the lightness of the color. Default value is 0.5.

    Returns:
        tuple: The adjusted color in RGB format.

    Raises:
        KeyError: If the specified color is not a valid Matplotlib color name.
    """

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def align_yaxis(ax1: plt.Axes, v1: float, ax2: plt.Axes, v2: float):
    """
    Adjust the y-axis of two subplots so that the specified values in each subplot are aligned.

    Args:
        ax1 (AxesSubplot): The first subplot.
        v1 (float): The value in ax1 that should be aligned with v2 in ax2.
        ax2 (AxesSubplot): The second subplot.
        v2 (float): The value in ax2 that should be aligned with v1 in ax1.

    Returns:
        None
    """
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    adjust_yaxis(ax2, (y1 - y2) / 2, v2)
    adjust_yaxis(ax1, (y2 - y1) / 2, v1)


def adjust_yaxis(ax: plt.Axes, ydif: float, v: float):
    """
    Shift the y-axis of a subplot by a specified amount, while maintaining the location of a specified point.

    Args:
        ax (AxesSubplot): The subplot whose y-axis is to be adjusted.
        ydif (float): The amount by which to adjust the y-axis.
        v (float): The location of the point whose position should remain unchanged.

    Returns:
        None
    """
    inv = ax.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, ydif))
    miny, maxy = ax.get_ylim()
    miny, maxy = miny - v, maxy - v
    if -miny > maxy or (-miny == maxy and dy > 0):
        nminy = miny
        nmaxy = miny * (maxy + dy) / (miny + dy)
    else:
        nmaxy = maxy
        nminy = maxy * (miny + dy) / (maxy + dy)
    ax.set_ylim(nminy + v, nmaxy + v)


def generate_rate_grid(
    dy: pd.DataFrame,
    ax: plt.Axes,
    xlabel: str = "Date",
    size: str = "150%",
    pad: int = 0,
    colors: List[str] = None,
    cumsum: bool = True,
):
    """
    Generate a grid of subplots displaying yearly and monthly rates from a Pandas DataFrame.

    Args:
        dy (pd.DataFrame): A Pandas DataFrame containing the data to be plotted.
        ax (AxesSubplot): The subplot onto which to plot the grid.
        xlabel (str): The label to be used for the x-axis. Default value is 'Date'.
        size (str): The size of the bottom subplot as a percentage of the top subplot. Default value is '150%'.
        pad (int): The amount of padding between the two subplots in pixels. Default value is 0.
        colors (List[str]): A list of colors to be used in the plot. If not provided, the default Matplotlib color cycle is used. Default value is None.
        cumsum (bool): If True, plot the cumulative sum of the data. If False, plot only the yearly and monthly rates. Default value is True.

    Returns:
        None
    """
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if cumsum:
        cumsum_ax = ax
        divider = make_axes_locatable(cumsum_ax)
        yearly_ax = divider.append_axes("bottom", size=size, pad=pad)
        cumsum_ax.figure.add_axes(yearly_ax)
        cumsum_ax.set_xticklabels([])
        axes = [cumsum_ax, yearly_ax]

        y = dy.fillna(0).cumsum()

        if len(dy.columns) == 1:
            cumsum_ax.plot(y, label="Cummulative", c=colors[0], antialiased=True)
            cumsum_ax.fill_between(
                y.index, y.values.T[0], color=colors[0], alpha=0.1, hatch="x"
            )
            cumsum_ax.set_ylabel(f"{y.columns[0]}")  # Wrap text
        else:
            cumsum_ax.stackplot(
                y.index, y.values.T, labels=y.columns, colors=colors, antialiased=True
            )
            cumsum_ax.set_ylabel(f"Cumulative {y.index.name.lower()}")

        yearly_ax.set_ylabel(f"Yearly {dy.index.name.lower()} rate")
        cumsum_ax.legend(loc="upper left", ncols=int(len(dy.columns) / 5 + 1))  # Test

    else:
        yearly_ax = ax
        axes = [yearly_ax]

        if len(dy.columns) == 1:
            yearly_ax.set_ylabel(
                f"{dy.columns[0]}\nYearly {dy.index.name.lower()} rate"
            )
        else:
            yearly_ax.set_ylabel(f"Yearly {dy.index.name.lower()} rate")

    if len(dy.columns) == 1:
        monthly_ax = yearly_ax.twinx()

        yearly_ax.plot(
            dy.resample("Y").sum(),
            label="Yearly rate",
            ls="--",
            c=colors[1],
            antialiased=True,
        )
        yearly_ax.legend(loc="upper left", ncols=int(len(dy.columns) / 8 + 1))
        monthly_ax.plot(
            dy.resample("M").sum(), label="Monthly rate", c=colors[2], antialiased=True
        )
        monthly_ax.set_ylabel(f"Monthly {dy.index.name.lower()} rate")
        monthly_ax.legend(loc="upper right")

    else:
        dy2 = dy.resample("Y").sum()
        yearly_ax.stackplot(
            dy2.index, dy2.values.T, labels=dy2.columns, colors=colors, antialiased=True
        )
        if not cumsum:
            yearly_ax.legend(loc="upper left", ncols=int(len(dy.columns) / 8 + 1))

    if xlabel is not None:
        yearly_ax.set_xlabel(xlabel)
    else:
        yearly_ax.set_xticklabels([])

    for temp_ax in axes:
        temp_ax.set_xlim(
            [
                dy.index.min() - pd.Timedelta(weeks=13),
                dy.index.max() + pd.Timedelta(weeks=52),
            ]
        )
        temp_ax.xaxis.set_minor_locator(AutoMinorLocator())
        temp_ax.yaxis.set_minor_locator(AutoMinorLocator())
        temp_ax.xaxis.set_major_locator(mdates.YearLocator())
        temp_ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        temp_ax.set_axisbelow(True)
        temp_ax.grid()

    if len(dy.columns) == 1:
        align_yaxis(yearly_ax, 0, monthly_ax, 0)
        l = yearly_ax.get_ylim()
        l2 = monthly_ax.get_ylim()
        f = lambda x: l2[0] + (x - l[0]) / (l[1] - l[0]) * (l2[1] - l2[0])
        ticks = f(yearly_ax.get_yticks())
        monthly_ax.yaxis.set_major_locator(FixedLocator(ticks))
        monthly_ax.yaxis.set_minor_locator(AutoMinorLocator())
        axes.append(monthly_ax)

    return axes


def rate_subplots(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = None,
    title: str = "",
    xlabel: str = "Date",
    colors: List[str] = None,
    cumsum: bool = True,
    bg: pd.DataFrame = None,
    vlines: pd.DataFrame = None,
):
    """
    Creates a grid of subplots to visualize rates of change over time of multiple variables in a pandas DataFrame.

    Args:
        df (pd.DataFrame): The pandas DataFrame containing the data to plot.
        figsize (Tuple[int, int] or None): The size of the figure to create. If None, default size is (16, len(df.columns)*2*(1+cumsum)).
        title (str): The title of the figure. Default is an empty string.
        xlabel (str): The label of the x-axis. Default is 'Date'.
        colors (List[str]): The list of colors to use for the lines. If None, default colors are used.
        cumsum (bool): Whether to plot the cumulative sum of the data. Default is True.
        bg (pd.DataFrame): A DataFrame containing the background shading data. Default is None.
        vlines (pd.DataFrame): A DataFrame containing the vertical line data. Default is None.

    Returns:
        None: Displays the generated plot.
    """
    if figsize is None:
        figsize = (16, len(df.columns) * 2 * (1 + cumsum))

    fig, axes = plt.subplots(
        nrows=len(df.columns), ncols=1, figsize=figsize, sharex=True
    )
    fig.suptitle(
        f'{title if title is not None else df.index.name.capitalize()}{f" by {df.columns.name.lower()}" if df.columns.name is not None else ""}',
        y=1,
    )

    if colors is None:
        cmap = plt.cm.tab20
    else:
        if len(colors) == len(df.columns):
            cmap = mc.ListedColormap(
                [adjust_lightness(c, i * 0.5 + 0.75) for c in colors for i in (0, 1)]
            )
        else:
            cmap = mc.ListedColormap(colors)

    c = 0
    for i, col in enumerate(df.columns):
        sub_axes = generate_rate_grid(
            df[col].to_frame(),
            ax=axes[i],
            colors=[cmap(2 * c), cmap(2 * c), cmap(2 * c + 1)],
            size="100%",
            xlabel="Date" if (i + 1) == len(df.columns) else None,
            cumsum=cumsum,
        )

        for ix, ax in enumerate(sub_axes[:2]):
            if bg is not None and all(col in bg.columns for col in ["begin", "end"]):
                bg = bg.copy()
                bg["end"].fillna(df.index.max(), inplace=True)
                for idx, row in bg.iterrows():
                    if row["end"] > pd.to_datetime(ax.get_xlim()[0], unit="d"):
                        filled_poly = ax.axvspan(
                            row["begin"],
                            row["end"],
                            alpha=0.1,
                            color=colors_dict[idx],
                            zorder=-1,
                        )
                        if i == 0 and ix == 0:
                            (x0, y0), (x1, y1) = (
                                filled_poly.get_path().get_extents().get_points()
                            )
                            ax.text(
                                (x0 + x1) / 2,
                                y1,
                                idx,
                                ha="center",
                                va="bottom",
                                transform=ax.get_xaxis_transform(),
                            )

            if vlines is not None:
                for idx, row in vlines.items():
                    if row > pd.to_datetime(ax.get_xlim()[0], unit="d"):
                        line = ax.axvline(row, ls="-.", c="maroon", lw=1)
                        if i == 0 and ix == 0:
                            (x0, y0), (x1, y1) = (
                                line.get_path().get_extents().get_points()
                            )
                            ax.text(
                                (x0 + x1) / 2 + 25,
                                (0.02 if cumsum else 0.98),
                                idx,
                                c="maroon",
                                ha="left",
                                va=("bottom" if cumsum else "top"),
                                rotation=90,
                                transform=ax.get_xaxis_transform(),
                            )

        c += 1
        if 2 * c + 1 >= cmap.N:
            c = 0

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.",
    )

    fig.tight_layout()
    fig.show()

    warnings.filterwarnings(
        "default",
        category=UserWarning,
        message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.",
    )


def rate_plot(
    dy: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 6),
    title: str = None,
    xlabel: str = "Date",
    colors: List[str] = None,
    cumsum: bool = True,
    bg: pd.DataFrame = None,
    vlines: pd.DataFrame = None,
):
    """
    Creates a single plot to visualize the rate of change over time of a single variable in a pandas DataFrame.

    Args:
        dy (pd.DataFrame): The pandas DataFrame containing the data to plot.
        figsize (Tuple[int, int]): The size of the figure to create. Default is (16, 6).
        title (str): The title of the figure. Default is None.
        xlabel (str): The label of the x-axis. Default is 'Date'.
        colors (List[str]): The list of colors to use for the lines. If None, default colors are used.
        cumsum (bool): Whether to plot the cumulative sum of the data. Default is True.
        bg (pd.DataFrame): A DataFrame containing the background shading data. Default is None.
        vlines (pd.DataFrame): A DataFrame containing the vertical line data. Default is None.

    Returns:
        None: Displays the generated plot.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    fig.suptitle(
        f'{title if title is not None else dy.index.name.capitalize()}{f" by {dy.columns.name.lower()}" if dy.columns.name is not None else ""}'
    )

    axes = generate_rate_grid(dy, ax, size="100%", colors=colors, cumsum=cumsum)
    for i, ax in enumerate(axes[:2]):
        if bg is not None and all(col in bg.columns for col in ["begin", "end"]):
            bg = bg.copy()
            bg["end"].fillna(dy.index.max(), inplace=True)
            for idx, row in bg.iterrows():
                if row["end"] > pd.to_datetime(ax.get_xlim()[0], unit="d"):
                    filled_poly = ax.axvspan(
                        row["begin"],
                        row["end"],
                        alpha=0.1,
                        color=colors_dict[idx],
                        zorder=-1,
                    )
                    if i == 0:
                        (x0, y0), (x1, y1) = (
                            filled_poly.get_path().get_extents().get_points()
                        )
                        ax.text(
                            (x0 + x1) / 2,
                            y1,
                            idx,
                            ha="center",
                            va="bottom",
                            transform=ax.get_xaxis_transform(),
                        )

        if vlines is not None:
            for idx, row in vlines.items():
                if row > pd.to_datetime(ax.get_xlim()[0], unit="d"):
                    line = ax.axvline(row, ls="-.", c="maroon", lw=1)
                    if i == 0:
                        (x0, y0), (x1, y1) = line.get_path().get_extents().get_points()
                        ax.text(
                            (x0 + x1) / 2 + 25,
                            (0.02 if cumsum or len(dy.columns) > 1 else 0.98),
                            idx,
                            c="maroon",
                            ha="left",
                            va=("bottom" if cumsum or len(dy.columns) > 1 else "top"),
                            rotation=90,
                            transform=ax.get_xaxis_transform(),
                        )

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.",
    )

    fig.tight_layout()
    fig.show()

    warnings.filterwarnings(
        "default",
        category=UserWarning,
        message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.",
    )


def arrow_plot(arrows: pd.Series, figsize: Tuple[int, int] = (6, 6), **kwargs):
    """
    Create a polar plot to visualize the frequency of each arrow direction in a pandas Series.

    Args:
        arrows (pandas.Series): A pandas Series containing arrow symbols as string data type.
        figsize (Tuple[int, int], optional): The width and height of the figure. Defaults to (6, 6).
        **kwargs: Additional keyword arguments to be passed to the bar() method.

    Returns:
        None: Displays the generated plot.
    """
    # Count the frequency of each arrow direction
    counts = arrows.value_counts().sort_index()

    # Map the arrows to angles
    angle_map = {
        "→": 0,
        "↗": np.pi / 4,
        "↑": np.pi / 2,
        "↖": 3 * np.pi / 4,
        "←": np.pi,
        "↙": 5 * np.pi / 4,
        "↓": 3 * np.pi / 2,
        "↘": 7 * np.pi / 4,
    }
    angles = counts.index.map(angle_map)

    # Create a polar plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(polar=True)
    ax.bar(angles, counts, width=0.5, color=colors_dict["Link Monster"], **kwargs)

    # Set the label for each arrow
    ax.set_xticks(list(angle_map.values()))
    ax.set_xticklabels(["▶", "◥", "▲", "◤", "◀", "◣", "▼", "◢"], fontsize=18)

    # Set radius grid location
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ticks = ax.get_yticks()
    ax.yaxis.set_major_locator(FixedLocator(ticks[1:]))
    ax.set_rorigin(-5)

    # Set the title of the plot
    ax.set_title("Link Arrows")

    # Display the plot
    fig.tight_layout()
    fig.show()


def boxplot(df, mean=True, **kwargs):
    """
    Plots a box plot of a given DataFrame using seaborn, with the year of the Release column on the x-axis and the remaining column on the y-axis.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the Release dates and another numeric column.
        mean (bool, optional): If True, plots a line representing the mean of each box. Defaults to True.
        **kwargs: Additional keyword arguments to pass to seaborn.boxplot().

    Returns:
        None

    Raises:
        ValueError: If the DataFrame has no Release column.
    """
    df = df.dropna().copy()
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot()
    col = df.columns.difference(["Release"])[0]
    df["year"] = df["Release"].dt.strftime("%Y")
    df[col] = df[col].apply(pd.to_numeric, errors="coerce")

    sns.boxplot(ax=ax, data=df, y=col, x="year", width=0.5, **kwargs)
    if mean:
        df.groupby("year").mean(numeric_only=True).plot(
            ax=ax, c="r", ls="--", alpha=0.75, grid=True, legend=False
        )

    if df[col].max() < 5000:
        ax.set_yticks(np.arange(0, df[col].max() + 1, 1))
    elif df[col].max() == 5000:
        ax.set_yticks(np.arange(0, 5500, 500))
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.set_axisbelow(True)
    plt.xticks(rotation=30)
    fig.tight_layout()
    fig.show()


# ======================= #
# Complete execution flow #
# ======================= #


def run(
    reports: Union[str, List[str]] = "all",
    progress_handler=None,
    telegram_first: bool = False,
    suppress_contribs: bool = False,
    cleanup: Union[bool, str] = False,
    dry_run: bool = False,
    **kwargs,
):
    """
    Executes all notebooks in the source directory that match the specified report, updates the page index
    to reflect the last execution timestamp, and clean up redundant data files.

    Args:
        reports (str, optional): The report to generate. Defaults to 'all'.
        progress_handler (function, optional): A progress handler function to report execution progress. Defaults to None.
        telegram_first (bool, optional): Defaults to False.
        suppress_contribs (bool, optional): Defaults to False.
        cleanup (Union[bool,str], optional): whether to cleanup data files after execution. If True, perform cleanup, if False, doesn't perform cleanup. If 'auto', performs cleanup if there are more than 4 data files for each report (assuming one per week). Defaults to 'auto'.
        dry_run (bool, optional): dry_run flag to pass to cleanup_data method call. Defaults to False.
        **kwargs: Additional keyword arguments to pass to run_notebook.

    Returns:
        None: This function does not return a value.
    """
    # Check API status
    if not check_API_status():
        if progress_handler:
            progress_handler.exit(API_status=False)
        return

    # Execute all notebooks in the source directory
    try:
        run_notebooks(
            reports=reports,
            progress_handler=progress_handler,
            telegram_first=telegram_first,
            suppress_contribs=suppress_contribs,
            **kwargs,
        )
    except Exception as e:
        raise e
    finally:
        # Update page index to reflect last execution timestamp
        update_index()

    # Cleanup redundant data files
    if cleanup == "auto":
        data_files_count = len(glob.glob(os.path.join(PARENT_DIR, "data/*.bz2")))
        reports_count = len(glob.glob(os.path.join(SCRIPT_DIR, "*.ipynb")))
        if data_files_count / reports_count > 10:
            cleanup_data(dry_run=dry_run)
    elif cleanup:
        cleanup_data(dry_run=dry_run)


# ========= #
# CLI usage #
# ========= #


def auto_or_bool(value):
    if value is None:
        return True
    elif value.lower() == "auto":
        return "auto"
    else:
        return bool(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--reports",
        nargs="+",
        dest="reports",
        default="all",
        type=str,
        required=False,
        help="The report(s) to be generated.",
    )
    parser.add_argument(
        "-t",
        "--telegram-token",
        dest="telegram_token",
        type=str,
        required=False,
        help="Telegram API token.",
    )
    parser.add_argument(
        "-d",
        "--discord-token",
        dest="discord_token",
        type=str,
        required=False,
        help="Discord API token.",
    )
    parser.add_argument(
        "-c",
        "--channel",
        dest="channel_id",
        type=int,
        required=False,
        help="Discord or Telegram Channel/chat ID.",
    )
    parser.add_argument(
        "-s",
        "--suppress-contribs",
        action="store_true",
        required=False,
        help="Disables using TQDM contribs entirely.",
    )
    parser.add_argument(
        "-f",
        "--telegram-first",
        action="store_true",
        required=False,
        help="Force TQDM to try using Telegram as progress bar before Discord.",
    )
    parser.add_argument(
        "--cleanup",
        default="auto",
        type=auto_or_bool,
        nargs="?",
        const=True,
        action="store",
        help="Wether to run the cleanup routine. Options are True, False and 'auto'. Defaults to auto.",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        required=False,
        help="Whether to dry run the cleanup routine. No effect if cleanup is False.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        required=False,
        help="Enables debug flag.",
    )
    args = vars(parser.parse_args())
    # Assures the script is within a git repository before proceesing
    assure_repo()
    # Change working directory to script location
    os.chdir(SCRIPT_DIR)
    # Execute the complete workflow
    run(**args)
    # Exit python
    quit()
