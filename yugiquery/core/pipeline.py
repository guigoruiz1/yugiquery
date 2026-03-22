# yugiquery/core/pipeline.py

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import argparse
import io
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, List, Literal

# --- Imports: Third-Party --- #
import arrow
import nbformat
import papermill as pm
import pandas as pd
from jupyter_client import kernelspec
from tqdm.auto import tqdm

# --- Imports: Local Application --- #
from .. import api
from . import cleanup_data, update_index
from ..metadata import __title__
from ..utils import ProgressHandler, dirs, git, load_secrets, lock, make_jekyll_page, unlock, LoggerConfig, make_filename
from .data import load_latest, merge_errata, merge_set_info
from .maintenance import generate_changelog, benchmark

# --- Logger Setup --- #
logger = LoggerConfig.get_logger()


# --- Main Pipeline Function --- #


def run(
    reports: str | List[str] | List[Path] = "all",
    progress_handler: ProgressHandler | None = None,
    cleanup: bool | Literal["auto"] = "auto",
    dry_run: bool = False,
    changelog: bool = True,
    benchmark: bool = True,
    squash: bool = True,
    jekyll: bool = False,
    operation: Literal["data", "reports", "both", "all"] = "all",
    discord: bool | argparse.Namespace = False,
    telegram: bool | argparse.Namespace = False,
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

    Raises:
        Exception: Raised if any exceptions occur during notebook execution.

    Returns:
        None: This function does not return a value.
    """
    if operation not in ("data", "reports", "all"):
        raise ValueError("Invalid operation. Must be 'data', 'reports', 'both', or 'all'.")

    print("\nExecution started")
    logger.info("Execution started")
    print()  # Empty space for better readability

    # Get the current commit hash
    start_commit = git.get_repo().head.commit

    # Setup progress bars
    report_paths = dirs.find_notebooks(reports) if operation in ("reports", "both", "all") else []
    all_data_flows = ["bandai", "cards", "rush", "speed", "sets"]
    data_flows = (
        [flow for flow in all_data_flows if flow in reports or reports == "all"]
        if operation in ("data", "both", "all")
        else []
    )

    if operation in ("data", "both", "all") and len(data_flows) > 0:
        total = len(data_flows)
        # Need to account for errata
        if any(flow in ["cards", "rush", "speed"] for flow in data_flows):
            total += 1

        data_pbars, data_pbar_kwargs = _setup_pbars(
            total=total,
            external_pbar=progress_handler.pbar if progress_handler else None,
            discord=discord,
            telegram=telegram,
        )
        _run_data(
            data_flows=data_flows,
            benchmark=benchmark,
            changelog=changelog,
            dry_run=dry_run,
            cleanup=cleanup,
            progress_handler=progress_handler,
            pbars=data_pbars,
        )

    if operation in ("reports", "both", "all") and len(report_paths) > 0:
        reports_pbars, reports_pbar_kwargs = _setup_pbars(
            total=len(report_paths),
            external_pbar=progress_handler.pbar if progress_handler else None,
            discord=discord,
            telegram=telegram,
        )
        # Only show local progress bar if no external progress handler is provided. Not needed for data flows
        if progress_handler is None:
            reports_pbars.append(
                tqdm(
                    position=0,
                    **reports_pbar_kwargs,
                )
            )

        _run_reports(
            report_paths=report_paths,
            dry_run=dry_run,
            jekyll=jekyll,
            progress_handler=progress_handler,
            pbars=reports_pbars,
        )

    # 7. Squash commits into one with a message referencing the new index timestamp
    if squash and not dry_run:
        _squash_commits(start_commit=start_commit, progress_handler=progress_handler)

    print("\nExecution completed")
    logger.info("Execution completed")


def _squash_commits(start_commit, progress_handler: ProgressHandler | None = None):
    """
    Helper function to squash commits into one after completing the data update and report generation processes. This is done at the end to keep the commit history clean, with one commit per execution that references the new index timestamp.
    Args:
        start_commit: The commit hash before the execution started, used as the starting point for squashing commits.
        progress_handler (ProgressHandler | None, optional): An optional ProgressHandler instance to report progress. Defaults to None.

    Returns:
        None
    """
    lock("squash_commits")
    try:
        logger.info("Squashing commits")
        print("\nSquashing commits")
        squash_results = git.squash_commits(start_commit)
        print(squash_results)
        logger.info("%s", squash_results)
    except Exception as e:
        if progress_handler:
            progress_handler.send(error=str(e))
        logger.warning("Error squashing commits. Ignoring... %s", e)
    finally:
        try:
            unlock("squash_commits")
        except Exception as e:
            logger.error("Error unlocking squash_commits lock. %s", e)


# --- Data Update --- #


def _run_data(
    data_flows: list[str],
    dry_run: bool = False,
    benchmark: bool = True,
    changelog: bool = True,
    cleanup: bool | Literal["auto"] = "auto",
    progress_handler: ProgressHandler | None = None,
    pbars: list[tqdm] = [],
):
    """
    Helper function to run data operations: API status check, data update, and data cleanup.

    Args:
        data_flows (list[str]): List of data flows to update (e.g., ['cards', 'rush']).
        dry_run (bool, optional): If True, runs the data update without committing changes. Defaults to False.
        benchmark (bool, optional): If True, benchmarks the data update process. Defaults to True.
        changelog (bool, optional): If True, generates changelogs for the updated data. Defaults to True.
        cleanup (bool | Literal["auto"], optional): Whether to clean up redundant data files after updating. If 'auto', performs cleanup if there are more than 4 data files for each report. Defaults to 'auto'.
        progress_handler (ProgressHandler | None, optional): An optional ProgressHandler instance to report progress. Defaults to None.
        pbars (list[tqdm], optional): A list of tqdm progress bars to update during the process. Defaults to [].

    Returns:
        None

    Raises:
        Exception: If any errors occur during the API status check, data update, or cleanup processes, the exception will be raised after attempting to clean up progress bars and release locks.
    """
    lock("run_data")
    # 1. Check API status
    api_status = api.check_status()
    if progress_handler:
        progress_handler.send(API_status=api_status)
    if not api_status:
        return

    # 2. Update data for relevant flows
    try:
        update_data(
            flows=data_flows,
            pbars=pbars,
            changelog=changelog,
            benchmark=benchmark,
            commit=not dry_run,
        )
    except Exception as e:
        if progress_handler:
            progress_handler.send(error=str(e))
        raise
    finally:
        _close_pbars(pbars)

    # 3. Cleanup redundant data files
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
        raise


def update_data(
    flows: List[str] | str = "all", pbars: List[tqdm] = [], changelog=True, benchmark=True, commit=True
) -> dict[str, tuple[pd.DataFrame, Path, Path | None]]:
    """
    Update data for the specified flow(s). Accepts a string or a list of strings.
    If 'all', updates all flows. Ignores unknown flows.
    Returns a dictionary with results for each flow.

    Args:
        flows (List[str] | str, optional): The flow(s) to update. Defaults to 'all'.
        pbars (List[tqdm], optional): List of progress bars to update during the process. Defaults to [].
        changelog (bool, optional): Whether to generate and save changelogs for the updated data. Defaults to True.
        benchmark (bool, optional): Whether to benchmark the data update process and save the results. Defaults to True.
        commit (bool, optional): Whether to commit the updated data and benchmark results to Git. Defaults to True.

    Returns:
        dict[str, tuple[pd.DataFrame, Path, Path | None]]: A dictionary where keys are flow names and values are tuples containing the updated DataFrame, the path to the saved data file, and the path to the generated changelog file (or None if no changelog was generated).
    """
    valid_flows = {
        "cards": _update_cards_data,
        "rush": _update_rush_data,
        "speed": _update_speed_data,
        "bandai": _update_bandai_data,
        "sets": _update_sets_data,
    }
    has_errata = {"cards", "rush", "speed"}
    results = {}
    exceptions = []

    # Always work with lowercase flow names for case-insensitivity
    if isinstance(flows, str):
        flows = [flows]
    else:
        flows = list(flows)
    flows = [f.lower() if isinstance(f, str) else f for f in flows]
    if len(flows) == 1 and flows[0] == "all":
        flows_to_run = list(valid_flows.keys())
    else:
        flows_to_run = flows

    if any(flow in has_errata for flow in flows_to_run):
        for pbar in pbars:
            pbar.set_postfix(report=f"Errata data update")
        errata_df = api.fetch_errata()
        for pbar in pbars:
            pbar.update(1)

    for flow in flows_to_run:
        func = valid_flows.get(flow)
        if func:
            lock(f"update_{flow}")
            try:
                for pbar in pbars:
                    pbar.set_postfix(data=flow)
                if flow in has_errata:
                    results[flow] = func(
                        save_changelog=changelog, save_benchmark=benchmark, commit=commit, errata_df=errata_df
                    )
                else:
                    results[flow] = func(save_changelog=changelog, save_benchmark=benchmark, commit=commit)
                for pbar in pbars:
                    pbar.update(1)
                logger.info(f"Data update for {flow} succeeded.")
            except Exception as e:
                logger.error("Data update failed for flow '%s'\n: %s", flow, e)
                exceptions.append(e)
            finally:
                unlock(f"update_{flow}")
        else:
            logger.warning(f"Unknown data update flow: {flow} (skipped)")

    if exceptions:
        combined_message = "\n".join(str(e) for e in exceptions)
        raise Exception(combined_message)

    return results


# --- Data Update Flows --- #


def _update_cards_data(
    errata_df=None, save_changelog=True, save_benchmark=True, commit=True
) -> tuple[pd.DataFrame | None, Path | None, Path | None]:
    """
    Retrieve, process, and save cards data. Optionally generate a changelog.

    Args:
        errata_df (pd.DataFrame, optional): DataFrame containing errata information. If not provided, it will be fetched from the API. Defaults to None.
        save_changelog (bool, optional): Whether to generate and save a changelog comparing the new data with the previous version. Defaults to True.
        save_benchmark (bool, optional): Whether to benchmark the data update process and save the results. Defaults to True.
        commit (bool, optional): Whether to commit the updated data and benchmark results to Git. Defaults to True.

    returns:
        tuple[pd.DataFrame, Path, Path | None]: A tuple containing the updated Cards DataFrame, the path to the saved data file, and the path to the generated changelog file (or None if no changelog was generated).
    """
    lock("cards_data")
    try:
        now = arrow.utcnow()
        logger.info("Updating cards data...")

        if errata_df is None:
            errata_df = api.fetch_errata()

        monster_df = api.fetch_monster()
        st_df = api.fetch_st()
        token_df = api.fetch_token()
        counter_df = api.fetch_counter()
        unusable_df = api.fetch_unusable()

        full_cards_df = pd.concat(
            [monster_df, st_df, token_df, counter_df, unusable_df], ignore_index=True, axis=0
        ).sort_values("Name", ignore_index=True)

        if errata_df is not None:
            full_cards_df = merge_errata(full_cards_df, errata_df)

        changelog_path = None
        if save_changelog:
            prev_df, prev_ts = load_latest("cards", return_ts=True)
            if prev_df is not None:
                changelog_path = _write_changelog(prev_df, prev_ts, full_cards_df, now, "cards", col="Name")

        cards_path = dirs.DATA / make_filename(report="cards", timestamp=now)
        full_cards_df.to_csv(cards_path, index=False)

        if save_benchmark:
            benchmark(now, "fetch", "cards")

        if commit:
            git.commit(["*[Cc]ards*", "data/benchmark.json"], message=f"Cards data updated - {now.isoformat()}")

        logger.info(f"Cards data saved to {cards_path}")

        return full_cards_df, cards_path, changelog_path
    finally:
        try:
            unlock("cards_data")
        except Exception as e:
            logger.error("Error unlocking cards_data lock. %s", e)


def _update_rush_data(
    errata_df=None, save_changelog=True, save_benchmark=True, commit=True
) -> tuple[pd.DataFrame | None, Path | None, Path | None]:
    """
    Retrieve, process, and save Rush Duel cards data. Optionally generate a changelog.

    Args:
        errata_df (pd.DataFrame, optional): DataFrame containing errata information. If not provided, it will be fetched from the API. Defaults to None.
        save_changelog (bool, optional): Whether to generate and save a changelog comparing the new data with the previous version. Defaults to True.
        save_benchmark (bool, optional): Whether to benchmark the data update process and save the results. Defaults to True.
        commit (bool, optional): Whether to commit the updated data and benchmark results to Git. Defaults to True.

    returns:
        tuple[pd.DataFrame, Path, Path | None]: A tuple containing the updated Rush Duel cards DataFrame, the path to the saved data file, and the path to the generated changelog file (or None if no changelog was generated).
    """
    lock("rush_data")
    try:
        now = arrow.utcnow()
        logger.info("Updating Rush Duel cards data...")
        if errata_df is None:
            errata_df = api.fetch_errata()

        rush_df = api.fetch_rush()
        if errata_df is not None:
            rush_df = merge_errata(rush_df, errata_df)

        changelog_path = None
        if save_changelog:
            prev_df, prev_ts = load_latest("rush", return_ts=True)
            if prev_df is not None:
                changelog_path = _write_changelog(prev_df, prev_ts, rush_df, now, "rush", col="Name")

        rush_path = dirs.DATA / make_filename(report="rush", timestamp=now)
        rush_df.to_csv(rush_path, index=False)

        if save_benchmark:
            benchmark(now, "fetch", "rush")

        if commit:
            git.commit(["*[Rr]ush*", "data/benchmark.json"], message=f"Rush Duel data updated - {now.isoformat()}")

        logger.info(f"Rush Duel cards data saved to {rush_path}")

        return rush_df, rush_path, changelog_path
    finally:
        try:
            unlock("rush_data")
        except Exception as e:
            logger.error("Error unlocking rush_data lock. %s", e)


def _update_speed_data(
    errata_df=None, save_changelog=True, save_benchmark=True, commit=True
) -> tuple[pd.DataFrame, Path, Path | None]:
    """
    Retrieve, process, and save Speed Duel cards data. Optionally generate a changelog.

    Args:
        errata_df (pd.DataFrame, optional): DataFrame containing errata information. If not provided, it will be fetched from the API. Defaults to None.
        save_changelog (bool, optional): Whether to generate and save a changelog comparing the new data with the previous version. Defaults to True.
        save_benchmark (bool, optional): Whether to benchmark the data update process and save the results. Defaults to True.
        commit (bool, optional): Whether to commit the updated data and benchmark results to Git. Defaults to True.

    returns:
        tuple[pd.DataFrame, Path, Path | None]: A tuple containing the updated Speed Duel cards DataFrame, the path to the saved data file, and the path to the generated changelog file (or None if no changelog was generated).
    """
    lock("speed_data")
    try:
        now = arrow.utcnow()
        logger.info("Updating Speed Duel cards data...")
        if errata_df is None:
            errata_df = api.fetch_errata()

        speed_df = api.fetch_speed()
        skill_df = api.fetch_skill()
        full_speed_df = pd.concat([speed_df, skill_df], ignore_index=True, axis=0).sort_values("Name", ignore_index=True)

        if errata_df is not None:
            full_speed_df = merge_errata(full_speed_df, errata_df)

        changelog_path = None
        if save_changelog:
            prev_df, prev_ts = load_latest("speed", return_ts=True)
            if prev_df is not None:
                changelog_path = _write_changelog(prev_df, prev_ts, full_speed_df, now, "speed", col="Name")

        speed_path = dirs.DATA / make_filename(report="speed", timestamp=now)
        full_speed_df.to_csv(speed_path, index=False)

        if save_benchmark:
            benchmark(now, "fetch", "speed")

        if commit:
            git.commit(["*[Ss]peed*", "data/benchmark.json"], message=f"Speed Duel data updated - {now.isoformat()}")

        logger.info(f"Speed Duel cards data saved to {speed_path}")

        return full_speed_df, speed_path, changelog_path
    finally:
        try:
            unlock("speed_data")
        except Exception as e:
            logger.error("Error unlocking speed_data lock. %s", e)


def _update_bandai_data(
    save_changelog=True, save_benchmark=True, commit=True
) -> tuple[pd.DataFrame | None, Path | None, Path | None]:
    """
    Retrieve, process, and save Bandai cards data. Optionally generate a changelog.

    Args:
        save_changelog (bool, optional): Whether to generate and save a changelog comparing the new data with the previous version. Defaults to True.
        save_benchmark (bool, optional): Whether to benchmark the data update process and save the results. Defaults to True.
        commit (bool, optional): Whether to commit the updated data and benchmark results to Git. Defaults to True.

    returns:
        tuple[pd.DataFrame, Path, Path | None]: A tuple containing the updated Bandai cards DataFrame, the path to the saved data file, and the path to the generated changelog file (or None if no changelog was generated).
    """
    lock("bandai_data")
    try:
        now = arrow.utcnow()
        logger.info("Updating Bandai cards data...")
        bandai_df = api.fetch_bandai()

        changelog_path = None
        if save_changelog:
            prev_df, prev_ts = load_latest("bandai", return_ts=True)
            if prev_df is not None:
                changelog_path = _write_changelog(prev_df, prev_ts, bandai_df, now, "bandai", col="Name")

        bandai_path = dirs.DATA / make_filename(report="bandai", timestamp=now)
        bandai_df.to_csv(bandai_path, index=False)

        if save_benchmark:
            benchmark(now, "fetch", "bandai")

        if commit:
            git.commit(["*[Bb]andai*", "data/benchmark.json"], message=f"Bandai data updated - {now.isoformat()}")

        logger.info(f"Bandai cards data saved to {bandai_path}")

        return bandai_df, bandai_path, changelog_path
    finally:
        try:
            unlock("bandai_data")
        except Exception as e:
            logger.error("Error unlocking bandai_data lock. %s", e)


def _update_sets_data(
    save_changelog=True, save_benchmark=True, commit=True
) -> tuple[pd.DataFrame | None, Path | None, Path | None]:
    """
    Retrieve, process, and save sets data. Optionally generate a changelog.

    Args:
        save_changelog (bool, optional): Whether to generate and save a changelog comparing the new data with the previous version. Defaults to True.
        save_benchmark (bool, optional): Whether to benchmark the data update process and save the results. Defaults to True.
        commit (bool, optional): Whether to commit the updated data and benchmark results to Git. Defaults to True.

    returns:
        tuple[pd.DataFrame, Path, Path | None]: A tuple containing the updated Sets DataFrame, the path to the saved data file, and the path to the generated changelog file (or None if no changelog was generated).
    """
    lock("sets_data")
    try:
        now = arrow.utcnow()
        logger.info("Updating sets data...")
        all_set_lists_df = api.fetch_all_set_lists()
        sets = all_set_lists_df["Set"].unique()
        set_info_df = api.fetch_set_info(*sets)
        all_set_lists_df = merge_set_info(all_set_lists_df, set_info_df)

        changelog_path = None
        if save_changelog:
            prev_df, prev_ts = load_latest("sets", return_ts=True)
            if prev_df is not None:
                changelog_path = _write_changelog(prev_df, prev_ts, all_set_lists_df, now, "sets", col="Card number")

        sets_path = dirs.DATA / make_filename(report="sets", timestamp=now)
        all_set_lists_df.to_csv(sets_path, index=False)

        if save_benchmark:
            benchmark(now, "fetch", "sets")

        if commit:
            git.commit(["*[Ss]ets*", "data/benchmark.json"], message=f"Sets data updated - {now.isoformat()}")

        logger.info(f"Sets data saved to {sets_path}")

        return all_set_lists_df, sets_path, changelog_path
    finally:
        try:
            unlock("sets_data")
        except Exception as e:
            logger.error("Error unlocking sets_data lock. %s", e)


def _write_changelog(prev_df, prev_ts, new_df, new_ts, file_name: str, col: str = "Name") -> Path | None:
    """
    Helper function to generate and save a changelog comparing prev_df and new_df.
    Returns the changelog path if changes are found, else None.

    Args:
        prev_df (pd.DataFrame): The previous version of the data.
        prev_ts (arrow.Arrow): The timestamp of the previous data version.
        new_df (pd.DataFrame): The new version of the data.
        new_ts (arrow.Arrow): The timestamp of the new data version.
        file_name (str): The base name to use for the changelog file.
        col (str, optional): The column to use as the key for generating the changelog. Defaults to "Name".

    Returns:
        Path | None: The path to the saved changelog file if changes are detected, otherwise None.
    """
    try:
        changelog_df = generate_changelog(prev_df, new_df, col=col)
        if not changelog_df.empty:
            changelog_path = dirs.DATA / make_filename(report=file_name, timestamp=new_ts, previous_timestamp=prev_ts)
            changelog_df.to_csv(changelog_path, index=True)
            logger.info(f"{file_name.capitalize()} changelog saved to {changelog_path}")
            return changelog_path
        else:
            logger.info(f"No changes detected for {file_name}, no changelog generated.")
    except Exception as e:
        logger.error(f"Changelog for {file_name} failed: {e}")
    return None


# --- Notebook Execution --- #


def _run_reports(
    report_paths: List[Path],
    dry_run: bool = False,
    jekyll: bool = False,
    progress_handler: ProgressHandler | None = None,
    pbars: list[tqdm] = [],
):
    """
    Helper function to run report generation: notebook execution, Jekyll page generation and index update.

    Args:
        report_paths (List[Path]): List of paths to the report notebooks to execute.
        dry_run (bool, optional): If True, runs the report generation without committing changes or creating Jekyll pages. Defaults to False.
        jekyll (bool, optional): Whether to generate Jekyll pages for the reports after executing the notebooks. Defaults to False.
        progress_handler (ProgressHandler | None, optional): An optional ProgressHandler instance to report progress. Defaults to None.
        pbars (list[tqdm], optional): A list of tqdm progress bars to update during the process. Defaults to [].

    Returns:
        None

    Raises:
        Exception: If any errors occur during notebook execution, Jekyll page generation, or index update, the exception will be raised after attempting to clean up progress bars and release locks.
    """
    # 4. Execute notebooks
    try:
        run_notebooks(
            reports=report_paths,
            pbars=pbars,
            dry_run=dry_run,
        )
    except Exception as e:
        if progress_handler:
            progress_handler.send(error=str(e))
        raise
    finally:
        _close_pbars(pbars)

    # 6. Generate Jekyll pages for reports
    if jekyll:
        logger.info("Generating Jekyll pages")
        print("\nGenerating Jekyll pages")
        for report_path in report_paths:
            title = Path(report_path).stem
            if dry_run:
                logger.info("Dry run - Would create Jekyll page for: %s", title)
                print(f"Dry run - Would create Jekyll page for: {title}")
            else:
                try:
                    make_jekyll_page(title=title)
                except Exception as e:
                    if progress_handler:
                        progress_handler.send(error=str(e))
                    logger.warning("Error creating Jekyll page for %s. Ignoring... %s", title, e)

    # 5. Update page index
    try:
        index_result = update_index(dry_run=dry_run)
        print(index_result)
        logger.info(index_result)
    except Exception as e:
        if progress_handler:
            progress_handler.send(error=str(e))
        logger.warning("Error updating index. Ignoring... %s", e)


def run_notebooks(
    reports: str | list[str] | List[Path],
    pbars: List[tqdm] = [],
    dry_run: bool = False,
) -> None:
    """
    Execute specified Jupyter notebooks using Papermill.

    Args:
        reports (str | List[str] | List[Path]): List of notebooks to execute.
        pbars (List[tqdm], optional): List of progress bars to update during execution. Default is an empty list.
        dry_run (bool, optional): Whether to run in dry run mode. Default is False.

    Returns:
        None

    Raises:
        Exception: Raised if any exceptions occur during notebook execution.
    """

    # Create the main logger
    papermill_logger = logging.getLogger("papermill")
    papermill_logger.setLevel(logging.INFO)

    # Create a StreamHandler and attach it to the logger
    stream_handler = logging.StreamHandler(io.StringIO())
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    stream_handler.addFilter(lambda record: record.getMessage().startswith("Ending Cell"))
    papermill_logger.addHandler(stream_handler)

    exceptions = []

    # Setup progress bars
    warnings.filterwarnings("ignore", message=".*clamping frac to range.*")

    for i, report in enumerate(reports):
        report_name = Path(report).stem
        dest_report = str(dirs.NOTEBOOKS.user / f"{report_name}.ipynb")

        lock(report_name)
        try:
            # Update the postfix
            for pbar in pbars:
                pbar.set_postfix(report=report_name)

            if dry_run:
                logger.info("Dry run - Generating %s report", report_name)
                print(f"\nDry run - Generating {report_name} report")
                continue

            with open(report) as f:
                nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)
                cells = len(nb.cells)

            # Define a function to update the output variable
            def update_pbar():
                for pbar in pbars:
                    pbar.update(1 / cells)

            # Attach the update_pbar function to the stream_handler
            stream_handler.flush = update_pbar

            tqdm.write(f"\nGenerating {report_name} report")
            logger.info("Generating %s report", report_name)

            # Set logger environment variables for notebook execution
            os.environ["PM_IN_EXECUTION"] = dest_report
            LoggerConfig.propagate_env()

            if __title__.lower() in kernelspec.find_kernel_specs():
                kernel_name = __title__.lower()
            else:
                kernel_name = "python3"

            try:
                pm.execute_notebook(
                    input_path=report,
                    output_path=dest_report,
                    log_output=True,
                    progress_bar={"position": 1, "desc": report_name},  # pyright: ignore[reportArgumentType]
                    kernel_name=kernel_name,
                )
                logger.info("Report '%s' generated successfully at %s", report_name, dest_report)
            except pm.PapermillExecutionError as e:
                logger.error("Report execution failed for '%s'\n: %s", report_name, e)
                exceptions.append(e)
            finally:
                os.environ.pop("PM_IN_EXECUTION", None)
                LoggerConfig.clean_env()
                for pbar in pbars:
                    pbar.update(1 + i - pbar.n)
                    # pbar.refresh()
        finally:
            unlock(report_name)

    # Close and clear the stream_handler
    stream_handler.close()
    papermill_logger.handlers.clear()

    warnings.filterwarnings("default")

    if exceptions:
        combined_message = "\n".join(str(e) for e in exceptions)
        raise Exception(combined_message)


# --- Progress Bar Setup --- #


def _setup_pbars(
    total: int,
    external_pbar: Callable[..., tqdm | None] | None,
    discord: bool | argparse.Namespace,
    telegram: bool | argparse.Namespace,
) -> tuple[List[tqdm], dict[str, Any]]:
    """
    Setup progress bars for notebook execution including local, Discord, and Telegram.

    Args:
        total (int): Total number of items to process.
        external_pbar (Callable[..., tqdm | None] | None): External progress bar callable.
        discord (bool | argparse.Namespace): Discord configuration.
        telegram (bool | argparse.Namespace): Telegram configuration.

    Returns:
        tuple[List[tqdm], dict[str, Any]]: List of configured progress bars and their keyword arguments.
    """
    contribs = {"discord": discord, "telegram": telegram}
    pbars = []

    pbar_kwargs: dict[str, Any] = dict(
        total=total,
        unit="report",
        unit_scale=True,
        dynamic_ncols=(not dirs.is_notebook),
        delay=2,
        desc="Completion",
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
            logger.warning("Unsupported contrib: %s. Ignoring...", contrib)
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
                logger.warning("Missing %s secrets. Ignoring...", contrib)
                return None
        elif isinstance(contrib_value, argparse.Namespace):
            tkn = contrib_value.tkn
            ch = contrib_value.ch
        else:
            return None

        # Validate credentials
        if not tkn or not ch:
            logger.warning("Missing %s credentials. Ignoring...", contrib)
            return None

        # Import and initialize the appropriate tqdm contrib
        try:
            if contrib_upper == "DISCORD":
                from tqdm.contrib.discord import tqdm as contrib_tqdm
            elif contrib_upper == "TELEGRAM":
                from tqdm.contrib.telegram import tqdm as contrib_tqdm
            else:
                logger.warning("Unsupported contrib: %s. Ignoring...", contrib)
                return None
            contrib_pbar = contrib_tqdm(
                token=tkn,
                position=len(pbars) + 1,
                file=open(os.devnull, "w"),
                **{ch_key.lower(): ch},
                **pbar_kwargs,
            )
            return contrib_pbar
        except Exception as e:
            logger.warning("Error setting up %s progress bar. Ignoring... %s", contrib, e)
            return None

    # Setup primary progress bar
    if external_pbar is not None:
        pbars.append(external_pbar(position=0, **pbar_kwargs))

    # Setup contrib progress bars
    for contrib in contribs:
        pbar = setup_contrib(contrib)
        if pbar:
            pbars.append(pbar)

    return pbars, pbar_kwargs


def _close_pbars(pbars: List[tqdm]) -> None:
    """
    Close all progress bars and their file handlers if applicable.

    Args:
        pbars (List[tqdm]): List of progress bars to close.
    """
    for pbar in pbars:
        pbar.close()
        if pbar.fp is not None and pbar.fp not in (sys.stdout, sys.stderr):
            pbar.fp.close()
