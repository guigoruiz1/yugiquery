# yugiquery/utils/helpers.py

# -*- coding: utf-8 -*-

# ============== #
# Helpers module #
# ============== #

import os
import sys
import json
import hashlib
import arrow
import re
import calendar
import subprocess
import sysconfig
from dotenv import dotenv_values
from tqdm.auto import tqdm, trange
from typing import Any, Callable, Dict, List, Tuple, Union

# Overwrite packages with versions specific for jupyter notebook
try:
    if get_ipython() is not None:
        from itables import init_notebook_mode
        from halo import HaloNotebook as Halo
except:
    from halo import Halo

# ========= #
# Variables #
# ========= #

UTILS_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_DIR = os.path.dirname(UTILS_DIR)
WORK_DIR = os.getcwd()  # Placeholder
DATA_DIR = os.path.join(WORK_DIR, "data")

# Handle cases where the package is installed
REPORTS_DIR = os.path.join(WORK_DIR, "reports")
if not os.path.isdir(REPORTS_DIR):
    REPORTS_DIR = os.path.join(
        sysconfig.get_path("data"), "share", "yugiquery", "reports"
    )

SECRETS_FILE = os.path.join(SCRIPT_DIR, "assets", "secrets.env")
if not os.path.isfile(SECRETS_FILE):
    SECRETS_FILE = os.path.join(
        sysconfig.get_path("data"), "share", "yugiquery", "secrets.env"
    )

# ================== #
# TQDM temporary fix #
# ================== #


def ensure_tqdm():
    loop = 0
    while True:
        try:
            from tqdm.contrib.discord import tqdm as discord_pbar

            return discord_pbar

        except ImportError:
            if loop > 1:
                print("Failed to import required tqdm fork twice. Aborting...")
                quit()

            loop += 1
            print(
                "Missing required tqdm fork for Discord progress bar. Trying to install now..."
            )
            subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "git+https://github.com/guigoruiz1/tqdm.git",
                ]
            )


# ============ #
# Data loaders #
# ============ #


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


# ========== #
# Validators #
# ========== #


def auto_or_bool(value):
    if value is None:
        return True
    elif value.lower() == "auto":
        return "auto"
    else:
        return bool(value)


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
