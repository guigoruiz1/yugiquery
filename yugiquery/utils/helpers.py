# yugiquery/utils/helpers.py

# -*- coding: utf-8 -*-

# ============== #
# Helpers module #
# ============== #

# ======= #
# Imports #
# ======= #

# Standard library imports
import argparse
import calendar  # Used in notebooks
import hashlib
import importlib.util
import json
import os
import re
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
    requested_secrets: List[str] = [], secrets_file: str | None = None, required: bool = False
) -> Dict[str, str]:
    """
    Load secrets from environment variables and/or a .env file.

    The secrets can be specified by name using the `requested_secrets` argument, which should be a list of strings. If `requested_secrets` is not specified, all available secrets will be returned.

    The `secrets_file` argument is the path to a .env file containing additional secrets to load. If `secrets_file` is specified and the file exists, the function will load the secrets from the file and merge them with the secrets loaded from the environment variables giving priority to secrets obtained from the environment.

    The `required` argument is a boolean or list of booleans indicating whether each requested secret is required to be present. If `required` is True, a KeyError will be raised if the secret is not found. If `required` is False or not specified, missing secrets will be skipped.

    Args:
        requested_secrets (List[str], optional): A list of names of the secrets to retrieve. If empty or not specified, all available secrets will be returned. Defaults to [].
        secrets_file (str | None, optional): The path to a .env file containing additional secrets to load. Defaults to None.
        required (bool or List[bool], optional): A boolean or list of booleans indicating whether each requested secret is required to be present. If True, a KeyError will be raised if the secret is not found. If False or not specified, missing secrets will be skipped. Defaults to False.

    Returns:
        Dict[str, str]: A dictionary containing the requested secrets as key-value pairs.

    Raises:
        KeyError: If a required secret is not found in the environment variables or .env file.

    """
    secrets = {
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


def load_json(json_file: str) -> dict:
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
        cprint(text=f"Error loading {json_file}! Returning empty dictionary. This may break some features.", color="yellow")
        return {}


# ======== #
# Argparse #
# ======== #
class CustomHelpFormatter(argparse.HelpFormatter):
    def __init__(self, prog):
        super().__init__(prog, max_help_position=60)

    def _format_action_invocation(self, action):
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        elif isinstance(action, CredAction):
            # Override to show [TOKEN] [CHANNEL] format
            return ", ".join(action.option_strings) + " " + " ".join(f"[{metavar}]" for metavar in action.metavar)
        else:
            # Override to show -a, --arg [ARG] format
            default = self._get_default_metavar_for_optional(action)
            args_string = self._format_args(action, default)
            return ", ".join(action.option_strings) + " " + args_string

    def _format_actions_usage(self, actions, groups):
        # Find group indices and identify actions in groups
        group_actions = set()
        inserts = {}
        for group in groups:
            if not group._group_actions:
                raise ValueError(f"empty group {group}")

            try:
                start = actions.index(group._group_actions[0])
            except ValueError:
                continue
            else:
                group_action_count = len(group._group_actions)
                end = start + group_action_count
                if actions[start:end] == group._group_actions:

                    suppressed_actions_count = 0
                    for action in group._group_actions:
                        group_actions.add(action)
                        if action.help is argparse.SUPPRESS:
                            suppressed_actions_count += 1

                    exposed_actions_count = group_action_count - suppressed_actions_count
                    if not exposed_actions_count:
                        continue

                    if not group.required:
                        if start in inserts:
                            inserts[start] += " ["
                        else:
                            inserts[start] = "["
                        if end in inserts:
                            inserts[end] += "]"
                        else:
                            inserts[end] = "]"
                    elif exposed_actions_count > 1:
                        if start in inserts:
                            inserts[start] += " ("
                        else:
                            inserts[start] = "("
                        if end in inserts:
                            inserts[end] += ")"
                        else:
                            inserts[end] = ")"
                    for i in range(start + 1, end):
                        inserts[i] = "|"

        # Collect all actions format strings
        parts = []
        for i, action in enumerate(actions):

            # Suppressed arguments are marked with None
            # Remove | separators for suppressed arguments
            if action.help is argparse.SUPPRESS:
                parts.append(None)
                if inserts.get(i) == "|":
                    inserts.pop(i)
                elif inserts.get(i + 1) == "|":
                    inserts.pop(i + 1)

            # Produce all arg strings
            elif not action.option_strings:
                default = self._get_default_metavar_for_positional(action)
                part = self._format_args(action, default)

                # If it's in a group, strip the outer []
                if action in group_actions:
                    if part[0] == "[" and part[-1] == "]":
                        part = part[1:-1]

                # Add the action string to the list
                parts.append(part)

            # Produce the first way to invoke the option in brackets
            else:
                option_string = action.option_strings[0]

                # Handle CredAction separately
                if isinstance(action, CredAction):
                    # Format for CredAction
                    args_string = " ".join(f"[{metavar}]" for metavar in action.metavar)
                    part = "%s %s" % (option_string, args_string)
                    part = f"[{part}]"
                else:
                    # Default format for other actions
                    if action.nargs == 0:
                        part = action.format_usage()
                    else:
                        default = self._get_default_metavar_for_optional(action)
                        args_string = self._format_args(action, default)
                        part = "%s %s" % (option_string, args_string)

                    # Make it look optional if it's not required or in a group
                    if not action.required and action not in group_actions:
                        part = "[%s]" % part

                # Add the action string to the list
                parts.append(part)

        # Insert things at the necessary indices
        for i in sorted(inserts, reverse=True):
            parts[i:i] = [inserts[i]]

        # Join all the action items with spaces
        text = " ".join([item for item in parts if item is not None])

        # Clean up separators for mutually exclusive groups
        open = r"[\[(]"
        close = r"[\])]"
        text = re.sub(r"(%s) " % open, r"\1", text)
        text = re.sub(r" (%s)" % close, r"\1", text)
        text = re.sub(r"%s *%s" % (open, close), r"", text)
        text = text.strip()

        # Return the text
        return text


class CredAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) == 0:
            setattr(namespace, self.dest, True)
        elif len(values) == 2:
            setattr(namespace, self.dest, argparse.Namespace(tkn=values[0], ch=values[1]))
        else:
            raise argparse.ArgumentError(self, "must provide either zero or exactly two arguments")


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


def get_ts_granularity(seconds: int) -> list[str]:
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

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
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

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file, fcntl.LOCK_UN)
    finally:
        lock_file.close()
        os.remove(lock_file_path)
