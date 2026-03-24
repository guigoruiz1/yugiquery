# yugiquery/cli.py

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import argparse
import re

# --- Imports: Local Application --- #
from .core import run, cleanup_data
from .utils import auto_or_bool, git, LoggerConfig

# --- Argparse Utilities --- #


class CustomHelpFormatter(argparse.HelpFormatter):
    """Custom help formatter to improve the formatting of command-line arguments in the help message."""

    def __init__(self, prog):
        super().__init__(prog, max_help_position=60)

    def _format_args(self, action, default_metavar):
        """Format the argument list for the help message."""
        if action.nargs == argparse.ZERO_OR_MORE:
            metavar = "[" + " ".join(self._metavar_formatter(action, default_metavar)(1)) + "]"
            return metavar
        else:
            return super()._format_args(action, default_metavar)


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


def handle_run(args):
    """Handle the 'run' subcommand: set up logging, ensure git repo, and execute the main workflow."""
    LoggerConfig.setup(level=args.log_level, log_file=args.log_file)
    _ = git.ensure_repo()

    run(
        reports=args.reports,
        cleanup=args.cleanup,
        dry_run=args.dryrun,
        jekyll=args.jekyll,
        changelog=args.changelog,
        benchmark=args.benchmark,
        discord=args.discord,
        telegram=args.telegram,
        operation="all",
    )


def handle_fetch(args):
    """Handle the 'fetch' subcommand: set up logging, ensure git repo, and execute the data update workflow."""
    LoggerConfig.setup(level=args.log_level, log_file=args.log_file)
    _ = git.ensure_repo()

    run(
        reports=args.data,
        cleanup=args.cleanup,
        dry_run=args.dryrun,
        discord=args.discord,
        telegram=args.telegram,
        changelog=args.changelog,
        benchmark=args.benchmark,
        operation="data",
    )


def handle_report(args):
    """Handle the 'report' subcommand: set up logging, ensure git repo, and execute the report generation workflow."""
    LoggerConfig.setup(level=args.log_level, log_file=args.log_file)
    _ = git.ensure_repo()
    # Only run notebooks and git ops, no API/data update
    run(
        reports=args.reports,
        dry_run=args.dryrun,
        jekyll=args.jekyll,
        discord=args.discord,
        telegram=args.telegram,
        operation="reports",
    )


def handle_cleanup(args):
    """Handle the 'cleanup' subcommand: set up logging, ensure git repo, and execute the cleanup routine."""
    LoggerConfig.setup(level=args.log_level, log_file=args.log_file)
    _ = git.ensure_repo()

    cleanup_data(dry_run=args.dryrun)


def set_run_parser(target: argparse._SubParsersAction | argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Configure the run subparser with arguments for reports, cleanup, progress bars, and debugging. Can be used as a subparser or standalone parser."""

    desc = "Run the full Yugiquery flow"
    if target is None:
        parser = argparse.ArgumentParser(description=desc, formatter_class=CustomHelpFormatter)
    elif isinstance(target, argparse._SubParsersAction):
        parser = target.add_parser("run", description=desc, help=desc, formatter_class=CustomHelpFormatter)
    else:
        parser = target

    parser.add_argument(
        "reports",
        nargs="*",
        metavar="REPORT",
        default="all",
        type=str,
        help="the report(s) to be generated. Defaults to 'all'",
    )
    report_group = parser.add_argument_group("report options")
    data_group = parser.add_argument_group("data options")
    _set_report_args(report_group)
    _set_data_args(data_group)
    _set_progress_args(parser)
    set_debug_args(parser)
    return parser


def set_fetch_parser(target: argparse._SubParsersAction | argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Configure the fetch subparser with arguments for data update options, progress bars, and debugging. Can be used as a subparser or standalone parser."""

    desc = "Fetch/update data only (no reports)"
    if target is None:
        parser = argparse.ArgumentParser(description=desc, formatter_class=CustomHelpFormatter)
    elif isinstance(target, argparse._SubParsersAction):
        parser = target.add_parser("fetch", description=desc, help=desc, formatter_class=CustomHelpFormatter)
    else:
        parser = target

    parser.add_argument(
        "data",
        nargs="*",
        metavar="DATA",
        default="all",
        type=str,
        help="The data to update (cards, rush, speed, bandai, sets). Defaults to 'all' if omitted.",
    )
    _set_data_args(parser)
    _set_progress_args(parser)
    set_debug_args(parser)
    return parser


def set_report_parser(target: argparse._SubParsersAction | argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Configure the report subparser with arguments for report generation options, progress bars, and debugging. Can be used as a subparser or standalone parser."""

    desc = "Generate reports only (no data update)"
    if target is None:
        parser = argparse.ArgumentParser(description=desc, formatter_class=CustomHelpFormatter)
    elif isinstance(target, argparse._SubParsersAction):
        parser = target.add_parser("report", description=desc, help=desc, formatter_class=CustomHelpFormatter)
    else:
        parser = target

    parser.add_argument(
        "reports",
        nargs="*",
        metavar="REPORT",
        default="all",
        type=str,
        help="the report(s) to be generated. Defaults to 'all'",
    )
    _set_report_args(parser)
    _set_progress_args(parser)
    set_debug_args(parser)
    return parser


def set_cleanup_parser(
    target: argparse._SubParsersAction | argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """
    Configure the cleanup subparser with debug flags (e.g., --dryrun, --log-level, --log-file). Can be used as a subparser or standalone parser.
    """
    desc = "Clean up redundant data files and compact benchmark/changelog history."
    if target is None:
        parser = argparse.ArgumentParser(description=desc, formatter_class=CustomHelpFormatter)
    elif isinstance(target, argparse._SubParsersAction):
        parser = target.add_parser("cleanup", description=desc, help=desc, formatter_class=CustomHelpFormatter)
    else:
        parser = target
    set_debug_args(parser)
    return parser


def _set_report_args(parser: argparse.ArgumentParser | argparse._ArgumentGroup) -> None:
    """Add report generation related arguments to the given parser or argument group."""
    parser.add_argument(
        "-j",
        "--jekyll",
        action="store_true",
        help="generate Jekyll markdown pages for HTML reports. Defaults to False",
    )


def _set_data_args(parser: argparse.ArgumentParser | argparse._ArgumentGroup) -> None:
    """Add data update related arguments to the given parser or argument group."""
    parser.add_argument(
        "--no-benchmark", dest="benchmark", action="store_false", help="disable benchmark saving for data update"
    )
    parser.add_argument(
        "--no-changelog", dest="changelog", action="store_false", help="disable changelog saving for data update"
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


def _set_progress_args(parser: argparse.ArgumentParser | argparse._ArgumentGroup) -> None:
    """Add progress bar related arguments (Discord and Telegram) to the given parser or argument group."""
    pbar_group = parser.add_argument_group("progress bars")
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


def set_debug_args(parser: argparse.ArgumentParser | argparse._ArgumentGroup, log_only: bool = False) -> None:
    """
    Add debugging related arguments (e.g., --dryrun, --log-level, --log-file) to the given parser or argument group.

    Args:
        parser (argparse.ArgumentParser | argparse._ArgumentGroup): The parser or argument group to which the debug arguments should be added.
        log_only (bool): If True, only add logging related arguments and skip the --dryrun argument. Defaults to False.
    """
    debug_group = parser.add_argument_group("debugging")
    if not log_only:
        debug_group.add_argument(
            "--dryrun",
            action="store_true",
            required=False,
            help="Perform a dry run: skip notebook execution and do not commit any changes. Useful for testing the workflow without making modifications",
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
        help="write log output to a file in addition to stderr",
    )
