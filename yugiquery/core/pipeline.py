# yugiquery/core/pipeline.py

# -*- coding: utf-8 -*-

import argparse
import io
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Callable, List, Literal

import nbformat
import papermill as pm
from jupyter_client import kernelspec
from termcolor import cprint
from tqdm.auto import tqdm

from .. import api
from . import cleanup_data, update_index
from ..utils import ProgressHandler, check_debug, dirs, git, load_secrets, lock, make_jekyll_page, unlock


def _setup_progress_bars(
    reports: str | list[str] | List[Path],
    external_pbar: Callable[..., tqdm | None] | None,
    discord: bool | argparse.Namespace,
    telegram: bool | argparse.Namespace,
) -> List[tqdm]:
    """
    Setup progress bars for notebook execution including local, Discord, and Telegram.

    Args:
        reports (str | list[str] | List[Path]): List of notebook paths to execute.
        external_pbar (Callable[..., tqdm | None] | None): External progress bar callable.
        discord (bool | argparse.Namespace): Discord configuration.
        telegram (bool | argparse.Namespace): Telegram configuration.

    Returns:
        List[tqdm]: List of configured progress bars.
    """
    contribs = {"discord": discord, "telegram": telegram}
    pbars = []

    pbar_kwargs: dict[str, Any] = dict(
        iterable=reports,
        unit="report",
        unit_scale=True,
        dynamic_ncols=(not dirs.is_notebook),
        delay=2,
        desc="Completion",
    )

    # Setup primary progress bar
    if external_pbar is not None:
        pbars.append(external_pbar(position=0, **pbar_kwargs))
    else:
        pbars.append(
            tqdm(
                position=0,
                **pbar_kwargs,
            )
        )

    # Helper to setup contrib progress bars (Discord/Telegram)
    def setup_contrib(contrib: str) -> tqdm | None:
        """Setup a single contrib progress bar (Discord or Telegram)."""
        contrib_upper = contrib.upper()
        contrib_value = contribs.get(contrib)

        # Determine channel key based on contrib type
        if contrib_upper == "DISCORD":
            ch_key = "channel_id"
        elif contrib_upper == "TELEGRAM":
            ch_key = "chat_id"
        else:
            cprint(text=f"Unsupported contrib: {contrib}. Ignoring...", color="yellow")
            return None

        # Get credentials from contrib_value or secrets
        if contrib_value is True:
            required_secrets = [f"{contrib_upper}_TOKEN", f"{contrib_upper}_{ch_key.upper()}"]
            try:
                secrets = load_secrets(
                    required_secrets,
                    secrets_file=dirs.secrets_file,
                    required=True,
                )
                tkn = secrets.get(required_secrets[0])
                ch = secrets.get(required_secrets[1])
            except Exception:
                cprint(text=f"Missing {contrib} secrets. Ignoring...", color="yellow")
                return None
        elif isinstance(contrib_value, argparse.Namespace):
            tkn = contrib_value.tkn
            ch = contrib_value.ch
        else:
            return None

        # Validate credentials
        if not tkn or not ch:
            cprint(text=f"Missing {contrib} credentials. Ignoring...", color="yellow")
            return None

        # Import and initialize the appropriate tqdm contrib
        try:
            if contrib_upper == "DISCORD":
                from tqdm.contrib.discord import tqdm as contrib_tqdm
            elif contrib_upper == "TELEGRAM":
                from tqdm.contrib.telegram import tqdm as contrib_tqdm
            else:
                cprint(text=f"Unsupported contrib: {contrib}. Ignoring...", color="yellow")
                return None
            return contrib_tqdm(
                token=tkn,
                file=open(os.devnull, "w"),
                **{ch_key.lower(): ch},
                **pbar_kwargs,
            )
        except Exception as e:
            print(e)
            cprint(text=f"Error setting up {contrib} progress bar. Ignoring...", color="yellow")
            return None

    # Setup contrib progress bars
    for contrib in contribs:
        pbar = setup_contrib(contrib)
        if pbar:
            pbars.append(pbar)

    return pbars


def run_notebooks(
    reports: str | list[str] | List[Path],
    external_pbar: Callable[..., tqdm | None] | None = None,
    discord: bool | argparse.Namespace = False,
    telegram: bool | argparse.Namespace = False,
    dry_run: bool = False,
    debug: bool = False,
) -> None:
    """
    Execute specified Jupyter notebooks using Papermill.

    Args:
        reports (str | List[str] | List[Path]): List of notebooks to execute.
        external_pbar (Callable[..., tqdm | None] | None, optional): A callable that returns a tqdm progress bar instance. Defaults to None.
        discord (bool | argparse.Namespace, optional): Discord configuration, either as a boolean or argparse.Namespace. Default is False.
        telegram (bool | argparse.Namespace, optional): Telegram configuration, either as a boolean or argparse.Namespace. Default is False.
        dry_run (bool, optional): Whether to run in dry run mode. Default is False.
        debug (bool, optional): Whether to enable debug mode. Default is False.

    Returns:
        None

    Raises:
        Exception: Raised if any exceptions occur during notebook execution.
    """
    # Setup progress bars
    warnings.filterwarnings("ignore", message=".*clamping frac to range.*")
    pbars = _setup_progress_bars(reports, external_pbar, discord, telegram)

    # Create the main logger
    logger = logging.getLogger("papermill")
    logger.setLevel(logging.INFO)

    # Create a StreamHandler and attach it to the logger
    stream_handler = logging.StreamHandler(io.StringIO())
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    stream_handler.addFilter(lambda record: record.getMessage().startswith("Ending Cell"))
    logger.addHandler(stream_handler)

    exceptions = []
    os.environ["YQ_DEBUG"] = str(check_debug(debug))
    tqdm.write("\nExecution started")
    for i, report in enumerate(reports):
        report_name = Path(report).stem
        dest_report = str(dirs.NOTEBOOKS.user / f"{report_name}.ipynb")

        lock(report_name)

        # Update the postfix
        for pbar in pbars:
            pbar.set_postfix(report=report_name)

        if dry_run:
            tqdm.write(f"Dry run - Generating {report_name} report")
            continue

        with open(report) as f:
            nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)
            cells = len(nb.cells)

        # Define a function to update the output variable
        def update_pbar():
            for pbar in pbars:
                pbar.update(1 / cells)
                pbar.refresh()

        # Attach the update_pbar function to the stream_handler
        stream_handler.flush = update_pbar

        tqdm.write(f"\nGenerating {report_name} report")

        # execute the notebook with papermill
        os.environ["PM_IN_EXECUTION"] = dest_report
        if "yugiquery" in kernelspec.find_kernel_specs():
            kernel_name = "yugiquery"
        else:
            kernel_name = "python3"

        try:
            pm.execute_notebook(
                input_path=report,
                output_path=dest_report,
                log_output=True,
                progress_bar=True,
                kernel_name=kernel_name,
            )
        except pm.PapermillExecutionError as e:
            tqdm.write(str(e))
            exceptions.append(e)
        finally:
            os.environ.pop("PM_IN_EXECUTION", default=None)
            for pbar in pbars:
                pbar.update(1 + i - pbar.n)
                pbar.refresh()

            unlock(report_name)

    # Empty character for better readability
    tqdm.write("\nExecution completed")
    os.environ.pop("YQ_DEBUG", default=None)

    # Close the iterator
    for pbar in pbars:
        pbar.close()
        if pbar.fp:
            pbar.fp.close()

    # Close the stream_handler
    stream_handler.close()
    # Clear custom handler
    logger.handlers.clear()

    warnings.filterwarnings("default")

    if exceptions:
        combined_message = "\n".join(str(e) for e in exceptions)
        raise Exception(combined_message)


def run(
    reports: str | List[str] | List[Path] = "all",
    progress_handler: ProgressHandler | None = None,
    cleanup: bool | Literal["auto"] = "auto",
    dry_run: bool = False,
    squash: bool = True,
    jekyll: bool = False,
    discord: bool | argparse.Namespace = False,
    telegram: bool | argparse.Namespace = False,
    debug: bool = False,
) -> None:
    """
    Executes all notebooks in the user and package `NOTEBOOKS` directories that match the specified report, updates the page index
    to reflect the last execution timestamp, and clean up redundant data files.

    Args:
        reports (str | List[str] | List[Path], optional): The report to generate. Defaults to 'all'.
        progress_handler (ProgressHandler | None, optional): An optional ProgressHandler instance to report execution progress. Defaults to None.
        cleanup (bool | Literal["auto"], optional): whether to cleanup data files after execution. If True, perform cleanup, if False, doesn't perform cleanup. If 'auto', performs cleanup if there are more than 4 data files for each report (assuming one per week). Defaults to 'auto'.
        dry_run (bool, optional): dry_run flag to pass to notebook execution and other operations. If True, notebooks will be executed but no changes will be committed to Git, Jekyll pages will not be created, and the index will not be updated. Defaults to False.
        squash (bool, optional): squash commits after execution. Defaults to True.
        jekyll (bool, optional): whether to generate Jekyll markdown pages for HTML reports. Defaults to False.
        discord (bool | argparse.Namespace, optional): Discord configuration, either as a boolean or argparse.Namespace. Default is False.
        telegram (bool | argparse.Namespace, optional): Telegram configuration, either as a boolean or argparse.Namespace. Default is False.
        debug (bool, optional): Whether to enable debug mode. Default is False.

    Raises:
        Exception: Raised if any exceptions occur during notebook execution.

    Returns:
        None: This function does not return a value.
    """
    report_paths = dirs.find_notebooks(reports)

    # Check API status
    api_status = api.check_status()
    if progress_handler:
        progress_handler.send(API_status=api_status)
    if not api_status:
        return

    # Get the current commit hash
    start_commit = git.get_repo().head.commit

    lock("run")
    try:
        # Execute notebooks
        try:
            if len(report_paths) > 0:
                run_notebooks(
                    reports=report_paths,
                    external_pbar=progress_handler.pbar if progress_handler else None,
                    discord=discord,
                    telegram=telegram,
                    debug=debug,
                    dry_run=dry_run,
                )
            else:
                cprint(text="No reports found. Ignoring... \n", color="yellow")
        except Exception as e:
            if progress_handler:
                progress_handler.send(error=str(e))
            raise
        finally:
            # Update page index to reflect last execution timestamp
            # Error is not critical but should be noted
            try:
                index_result = update_index(dry_run=dry_run)
                print(index_result)
            except Exception as e:
                if progress_handler:
                    progress_handler.send(error=str(e))
                cprint(text=f"Error updating index. Ignoring... \n", color="yellow")
                print(e)

        # Cleanup redundant data files
        if cleanup == "auto":
            data_files_count = len(list(dirs.DATA.glob("*.bz2")))
            reports_count = len(list(dirs.REPORTS.glob("*.html")))
            cleanup = data_files_count / max(reports_count, 1) > 10
        if cleanup:
            try:
                cleanup_data(dry_run=dry_run)
            except Exception as e:
                if progress_handler:
                    progress_handler.send(error=str(e))
                cprint(text=f"Error cleaning up data. Ignoring... \n", color="yellow")
                print(e)

        # Generate Jekyll pages for reports
        if jekyll:
            print("\nGenerating Jekyll pages")
            for report_path in report_paths:
                title = Path(report_path).stem
                if dry_run:
                    print(f"Dry run - Would create Jekyll page for: {title}")
                else:
                    try:
                        make_jekyll_page(title=title)
                    except Exception as e:
                        if progress_handler:
                            progress_handler.send(error=str(e))
                        cprint(text=f"Error creating Jekyll page for {title}. Ignoring... \n", color="yellow")
                        print(e)

        # Squash commits if any
        if squash:
            if dry_run:
                print("\nDry run - Squashing commits")
            else:
                # Error is not critical but should be noted
                print("\nSquashing commits")
                try:
                    squash_results = git.squash_commits(start_commit)
                    print(squash_results)
                except Exception as e:
                    cprint(text=f"Error squashing commits. Ignoring... \n", color="yellow")
                    print(e)
    finally:
        try:
            unlock("run")
        except Exception as e:
            cprint(text=f"Error unlocking run lock.\n", color="red")
            print(e)
