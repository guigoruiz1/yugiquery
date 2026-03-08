# yugiquery/utils/helpers.py

# -*- coding: utf-8 -*-

# ============== #
# Helpers module #
# ============== #

# ======= #
# Imports #
# ======= #

# Standard library imports
import calendar  # Used in notebooks
import hashlib
import json
import os
import platform
from ast import literal_eval
from pathlib import Path
from typing import Literal, List, Dict

# Third-party imports
import arrow
from dotenv import dotenv_values
from termcolor import cprint

# Local application imports
from .dirs import dirs

# ============ #
# Global Debug #
# ============ #


# TODO: find more elegant way to handle debuging
def check_debug(local_debug: bool = False) -> bool:
    """
    Check if the debug mode is enabled.

    Args:
        local_debug (bool, optional): A boolean indicating whether the debug mode is enabled locally. Defaults to False.

    Returns:
        bool: A boolean indicating whether the debug mode is enabled.
    """
    return literal_eval(os.environ.get("YQ_DEBUG", "False")) or local_debug


# ============ #
# Data loaders #
# ============ #


def load_secrets(
    requested_secrets: List[str] = [], secrets_file: str | Path | None = None, required: bool = False
) -> Dict[str, str | None]:
    """
    Load secrets from environment variables and/or a .env file.

    The secrets can be specified by name using the `requested_secrets` argument, which should be a list of strings. If `requested_secrets` is not specified, all available secrets will be returned.

    The `secrets_file` argument is the path to a .env file containing additional secrets to load. If `secrets_file` is specified and the file exists, the function will load the secrets from the file and merge them with the secrets loaded from the environment variables giving priority to secrets obtained from the environment.

    The `required` argument is a boolean or list of booleans indicating whether each requested secret is required to be present. If `required` is True, a KeyError will be raised if the secret is not found. If `required` is False or not specified, missing secrets will be skipped.

    Args:
        requested_secrets (List[str], optional): A list of names of the secrets to retrieve. If empty or not specified, all available secrets will be returned. Defaults to [].
        secrets_file (str | Path | None, optional): The path to a .env file containing additional secrets to load. Defaults to None.
        required (bool or List[bool], optional): A boolean or list of booleans indicating whether each requested secret is required to be present. If True, a KeyError will be raised if the secret is not found. If False or not specified, missing secrets will be skipped. Defaults to False.

    Returns:
        Dict[str, str | None]: A dictionary containing the requested secrets as key-value pairs.

    Raises:
        KeyError: If a required secret is not found in the environment variables or .env file.

    """
    secrets: Dict[str, str | None] = {
        key: value
        for key in requested_secrets
        if (value := os.environ.get(key, os.environ.get(f"TQDM_{key}")))  # Using walrus operator to assign and check value
    }
    if secrets_file is not None and Path(secrets_file).is_file():
        secrets = dotenv_values(secrets_file) | secrets

        if not requested_secrets:
            return secrets
        else:
            secrets = {key: secrets[key] for key in requested_secrets if key in secrets.keys() and secrets[key]}
    if required:
        for i, key in enumerate(requested_secrets):
            check = required if isinstance(required, bool) else required[i]
            if check and key not in secrets.keys():
                raise KeyError(f'Secret "{requested_secrets[i]}" not found')

    return secrets


def load_json(json_file: str | Path) -> dict:
    """
    Load data from a JSON file.

    Args:
        json_file (str | Path): The file path to the JSON file.

    Returns:
        dict: A dictionary containing the data from the JSON file. If the file does not exist, an empty dictionary is returned.
    """
    try:
        with open(json_file, "r") as file:
            data = json.load(file)
            return data
    except:
        cprint(text=f"Error loading {json_file}! Returning empty dictionary. This may break some features.", color="yellow")
        return {}


def auto_or_bool(value: str) -> bool | Literal["auto"]:
    """
    Convert a string to a boolean (True or False) or "auto".
    """

    if value is None:
        return True
    elif value.lower() == "auto":
        return "auto"
    else:
        return bool(value)


# ========== #
# Validators #
# ========== #


def md5(name: str) -> str:
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


# =================== #
# String Manipulators #
# =================== #


def escape_chars(string: str, chars: List[str] = ["_", ".", "-", "+", "#", "@", "="]) -> str:
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


# ====================== #
# Timestamp Manipulators #
# ====================== #


def get_ts_granularity(seconds: int) -> List[arrow.arrow._GRANULARITY]:
    """
    Humanizes a time interval given in seconds.

    Args:
        seconds (int): The time interval in seconds.

    Returns:
        List[arrow.arrow._GRANULARITY]: A list of human-readable granularities for the time interval.
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

    # Ensure at least "second" is returned
    if not selected_granularity:
        selected_granularity.append("second")

    return selected_granularity


def make_filename(report: str, timestamp: arrow.Arrow, previous_timestamp: arrow.Arrow | None = None) -> str:
    """
    Generates a standardized filename based on the provided parameters.

    Args:
        report (str): The name or identifier of the report.
        timestamp (arrow.Arrow): The timestamp to be included in the filename.
        previous_timestamp (arrow.Arrow | None): The previous timestamp, if applicable. Defaults to None.

    Returns:
        str: The generated filename.
    """
    report = report.lower()
    formated_ts = timestamp.isoformat(timespec="minutes").replace("+00:00", "Z").replace(":", "").replace("-", "")
    if previous_timestamp is None:
        return f"{report}_data_{formated_ts}.bz2"
    else:
        formated_previous_ts = (
            previous_timestamp.isoformat(timespec="minutes").replace("+00:00", "Z").replace(":", "-").replace("-", "")
        )
        return f"{report}_changelog_{formated_previous_ts}_{formated_ts}.bz2"


# ============== #
# Lock Mechanism #
# ============== #


def lock(file_name: str) -> None:
    """
    Acquire a file lock and handle stale locks using the same lock file.

    Args:
        file_name (str): The name of the lock file to create

    Raises:
        RuntimeError: If another instance is already running.
    """
    lock_file_path = dirs.temp.joinpath(file_name).with_suffix(".lock")

    # Open (or create if doesn't exist) the lock file in a+ mode
    with open(lock_file_path, "a+") as lock_file:
        try:
            # Try to acquire an exclusive lock on the file
            if platform.system() == "Windows":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
            else:
                import fcntl

                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Read the existing PID in the file (if any)
            existing_pid = lock_file.read().strip()

            if existing_pid:
                print(f"Stale lock file held by process {existing_pid}. Replacing with current PID.")

            # Write the current process PID into the file
            lock_file.seek(0)  # Go back to the beginning of the file
            lock_file.write(str(os.getpid()))
            lock_file.truncate()  # Ensure to remove any leftover content
        except (OSError, IOError):
            raise RuntimeError("Another instance is running")


def unlock(file_name: str) -> None:
    """
    Release a file lock and remove the lock file.

    Args:
        file_name (str): The name of the lock file to remove

    """
    lock_file_path = dirs.temp.joinpath(file_name).with_suffix(".lock")

    if not lock_file_path.exists():
        print("Lock file does not exist. Ignoring unlock request.")

    lock_file = open(lock_file_path, "w")
    try:
        if platform.system() == "Windows":
            import msvcrt

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl

            fcntl.flock(lock_file, fcntl.LOCK_UN)
    finally:
        lock_file.close()
        os.remove(lock_file_path)
