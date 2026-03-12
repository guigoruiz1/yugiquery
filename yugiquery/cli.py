# yugiquery/cli.py

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import argparse
import re
from typing import Literal

# --- Imports: Local Application --- #
from .core import run
from .utils import auto_or_bool, dirs, git, setup_logging


# --- Argparse Utilities --- #


class CustomHelpFormatter(argparse.HelpFormatter):
    def __init__(self, prog):
        super().__init__(prog, max_help_position=60)

    def _format_action_invocation(self, action):
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        elif isinstance(action, CredAction):
            # Override to show [TOKEN] [CHANNEL] format
            metavars = action.metavar or (self._get_default_metavar_for_optional(action),)
            return ", ".join(action.option_strings) + " " + " ".join(f"[{metavar}]" for metavar in metavars)
        else:
            # Override to show -a, --arg [ARG] format
            default = self._get_default_metavar_for_optional(action)
            args_string = self._format_args(action, default)
            return ", ".join(action.option_strings) + " " + args_string

    def _format_actions_usage(self, actions, groups):
        # Find group indices and identify actions in groups
        actions = list(actions)
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
                    metavars = action.metavar or (self._get_default_metavar_for_optional(action),)
                    args_string = " ".join(f"[{metavar}]" for metavar in metavars)
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
    """Custom argparse action for handling credential arguments."""

    def __call__(self, parser, namespace, values, option_string=None):
        values_list = list(values) if values is not None else []
        if len(values_list) == 0:
            setattr(namespace, self.dest, True)
        elif len(values_list) == 2:
            setattr(namespace, self.dest, argparse.Namespace(tkn=values_list[0], ch=values_list[1]))
        else:
            raise argparse.ArgumentError(self, "must provide either zero or exactly two arguments")


# --- CLI Main --- #


def main(args):
    setup_logging(level=args.log_level, log_file=args.log_file)
    # Assures the script is within a git repository before processing
    _ = git.ensure_repo()
    # Execute the complete workflow
    run(
        reports=args.reports,
        cleanup=args.cleanup,
        dry_run=args.dryrun,
        jekyll=args.jekyll,
        discord=args.discord,
        telegram=args.telegram,
    )


def set_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "reports",
        nargs="*",
        metavar="REPORT",
        default="all",
        type=str,
        help="the report(s) to be generated. Defaults to 'all'",
    )
    parser.add_argument(
        "-j",
        "--jekyll",
        action="store_true",
        help="generate Jekyll markdown pages for HTML reports. Defaults to False",
    )
    parser.add_argument(
        "-c",
        "--cleanup",
        default="auto",
        type=auto_or_bool,
        nargs="?",
        const=True,
        action="store",
        help="wether to run the cleanup routine. Options: {True,False,'auto'}. Defaults to auto",
    )
    pbar_group = parser.add_argument_group("Progress bars")
    pbar_group.add_argument(
        "-d",
        "--discord",
        nargs="*",
        metavar=("DISCORD_TOKEN", "DISCORD_CHANNEL_ID"),
        dest="discord",
        default=False,
        action=CredAction,
        help="Discord TOKEN and CHANNEL_ID, respectively, or no arguments to search for values in secrets",
    )

    pbar_group.add_argument(
        "-t",
        "--telegram",
        nargs="*",
        metavar=("TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID"),
        dest="telegram",
        default=False,
        action=CredAction,
        help="Telegram TOKEN and CHAT_ID, respectively, or no arguments to search for values in secrets",
    )
    debug_group = parser.add_argument_group("Debugging")
    debug_group.add_argument(
        "--dryrun",
        action="store_true",
        required=False,
        help="run in dry run mode",
    )
    debug_group.add_argument(
        "--log-level",
        type=str,
        required=False,
        default=None,
        help="set log verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    debug_group.add_argument(
        "--log-file",
        type=str,
        required=False,
        default=None,
        metavar="PATH",
        help="write log output to a file in addition to stderr",
    )
    debug_group.add_argument("-p", "--paths", action="store_true", help="print YugiQuery paths and exit")
