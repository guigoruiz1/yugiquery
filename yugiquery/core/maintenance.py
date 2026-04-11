# yugiquery/core/maintenance.py

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import json
import os
import re
from pathlib import Path
from typing import Dict, List, TypedDict

# --- Imports: Third-Party --- #
import arrow
import pandas as pd

# --- Imports: Local Application --- #
from ..utils import (
    dirs,
    get_notebook_path,
    git,
    load_json,
    make_filename,
    LoggerConfig,
    lock,
    unlock,
    filename_ts_fmt,
)

# --- Halo Spinner Import --- #
if dirs.is_notebook:
    from halo import HaloNotebook as Halo
else:
    from halo import Halo

# --- Logger Setup --- #
logger = LoggerConfig.get_logger()


# --- Benchmark Entry Type --- #


class BenchmarkEntry(TypedDict):
    ts: str
    average: float
    weight: float


# --- Changelog Generation --- #


def generate_changelog(previous_df: pd.DataFrame, current_df: pd.DataFrame, col: str | List[str]) -> pd.DataFrame:
    """
    Generate a changelog DataFrame by comparing two DataFrames on key columns.

    Args:
        previous_df (pd.DataFrame): Previous version of the data.
        current_df (pd.DataFrame): Current version of the data.
        col (str | List[str]): Column(s) used as comparison keys.

    Returns:
        pd.DataFrame: DataFrame containing old/new rows for changed records.
    """
    if isinstance(col, str):
        col = [col]

    changelog = (  # Pylance thinks this is a Series but it's a DataFrame
        previous_df.merge(current_df, indicator=True, how="outer")
        .loc[lambda x: x["_merge"] != "both"]
        .sort_values(col, ignore_index=True)
    )  # pyright: ignore[reportCallIssue]
    changelog["_merge"] = changelog["_merge"].cat.rename_categories({"left_only": "Old", "right_only": "New"})
    changelog.rename(columns={"_merge": "Version"}, inplace=True)
    nunique = changelog.groupby(col).nunique(dropna=False)
    cols_to_drop = nunique[nunique < 2].dropna(axis=1).columns.difference(["Modification date", "Version"])
    changelog.drop(cols_to_drop, axis=1, inplace=True)
    changelog = changelog.set_index(col)

    if all(col in changelog.columns for col in ["Modification date", "Version"]):
        true_changes = changelog.drop(["Modification date", "Version"], axis=1)[nunique > 1].dropna(axis=0, how="all").index
        new_entries = nunique[nunique["Version"] == 1].dropna(axis=0, how="all").index
        rows_to_keep = true_changes.union(new_entries).unique()
        changelog = changelog.loc[rows_to_keep].sort_values(by=[*col, "Version"])

    return changelog


# --- Benchmarking --- #


def benchmark(timestamp: arrow.Arrow, group: str = "report", entry: str | None = None) -> None:
    """
    Record report execution time and persist benchmark history.

    Args:
        timestamp (arrow.Arrow): Start timestamp for execution.
        entry (str | None, optional): Entry name. If None, infer notebook stem.
        group (str | None, optional): Group name.
    """
    if entry is None:
        path = get_notebook_path()
        entry = path.stem if path else "Unnamed"

    now = arrow.utcnow()
    timedelta = now - timestamp
    benchmark_file = dirs.DATA / "benchmark.json"
    data = load_json(benchmark_file)

    if group not in data:
        data[group] = {}
    if entry not in data[group]:
        data[group][entry] = []
    data[group][entry].append({"ts": now.isoformat(), "average": timedelta.total_seconds(), "weight": 1})

    with open(benchmark_file, "w+") as file:
        json.dump(data, file, indent=4)

    logger.info("%s", f"{entry.capitalize()} {group.capitalize()} benchmarked")


# --- Index Updating --- #


def update_index(commit: bool = False, page_paths: List[Path | str] | None = None) -> str:
    """
    Update `index.md` and `README.md` report table and last execution timestamp.

    Args:
        commit (bool, optional): If True, commit changes after updating the index.
        page_paths (List[Path | str] | None, optional): Additional .md pages to consider.

    Returns:
        str: Git commit output or dry-run advisory message.
    """
    lock("update_index")
    try:
        print("\nUpdating index")
        logger.info("Updating index")

        def extract_permalink(md_file: Path) -> str | None:
            """Extract permalink from Jekyll frontmatter, returning None if not found."""
            with open(md_file, "r", encoding="utf-8") as f:
                in_frontmatter = False
                for line in f:
                    if line.strip() == "---":
                        if in_frontmatter:
                            return None
                        in_frontmatter = True
                        continue
                    if in_frontmatter and line.strip().startswith("permalink:"):
                        permalink = line.split(":", 1)[1].strip()
                        return permalink.strip('"').strip("'")
            return None

        index_path = dirs.WORK / "index.md"
        readme_path = dirs.WORK / "README.md"

        timestamp = arrow.utcnow()

        with open(index_path, encoding="utf-8") as f:
            index = f.read()
        with open(readme_path, encoding="utf-8") as f:
            readme = f.read()

        all_reports = {f.stem: ("html", f) for f in dirs.REPORTS.glob("*.html")}

        for f in dirs.REPORTS.glob("*.md"):
            all_reports[f.stem] = ("md", f)

        if page_paths:
            for path in page_paths:
                path = Path(path)
                if path.is_dir():
                    for f in path.glob("*.md"):
                        all_reports[f.stem] = ("md", f)
                elif path.is_file() and path.suffix == ".md":
                    all_reports[path.stem] = ("md", path)

        rows = []
        for stem in sorted(all_reports.keys()):
            file_type, report_file = all_reports[stem]

            if file_type == "md":
                permalink = extract_permalink(report_file)
                if permalink:
                    link_path = permalink.strip("/")
                else:
                    link_path = f"reports/{stem}"
            else:
                link_path = str(report_file.relative_to(dirs.WORK))

            timestamp_str = pd.to_datetime(report_file.stat().st_mtime, unit="s", utc=True).strftime("%d/%m/%Y %H:%M %Z")
            rows.append(f"| [{stem}]({link_path}) | {timestamp_str} |")

        table = "\n".join(
            [
                "|                    Report | Last execution       |",
                "| -------------------------:|:-------------------- |",
                *rows,
            ]
        )

        def replace_table(content: str) -> str:
            start_marker = "<!-- REPORT_TABLE_START -->"
            end_marker = "<!-- REPORT_TABLE_END -->"
            start = content.find(start_marker)
            end = content.find(end_marker)
            if start == -1 or end == -1 or end < start:
                raise ValueError("Table markers not found in file.")
            before = content[: start + len(start_marker)]
            after = content[end:]
            updated = before + "\n" + table + "\n" + after

            ts_pattern = r"(last executed at `)([^`]+)(`)"
            new_ts = timestamp.strftime("%d/%m/%Y %H:%M %Z")
            updated_new, n_subs = re.subn(ts_pattern, r"\g<1>" + new_ts + r"\g<3>", updated)
            if n_subs == 0:
                raise ValueError("Last executed timestamp pattern not found in file.")
            return updated_new

        index = replace_table(index)
        readme = replace_table(readme)

        with open(index_path, "w", encoding="utf-8") as o:
            o.write(index)
        with open(readme_path, "w", encoding="utf-8") as o:
            o.write(readme)

        if commit:
            return git.commit(
                files=[index_path, readme_path],
                message=f"Index and README timestamp update - {timestamp.isoformat()}",
            )
        else:
            return "README and index updated - Dry run, not commited."
    finally:
        try:
            unlock("update_index")
        except Exception as e:
            logger.error("Error unlocking update_index lock. %s", e)


# --- Data Cleanup --- #


def cleanup_data(dryrun: bool = False) -> None:
    """
    Clean up redundant data files and compact benchmark/changelog history.

    Performs two main operations:
    1. Condenses benchmark history using weighted averages.
    2. Removes redundant changelog and data files, keeping only the most recent from each month
       and condensing multiple changelogs within the same month into a single consolidated file.

    Args:
        dryrun (bool, optional): If True, log intended actions without modifying files. Defaults to False.
    """
    lock("cleanup_data")
    try:
        dryrun_str = " (dry run)" if dryrun else ""
        print(f"\nStarting data cleanup{dryrun_str}")
        logger.info("Starting data cleanup%s", dryrun_str)

        with Halo(text="Cleaning up data...", spinner="line", enabled=("PM_IN_EXECUTION" not in os.environ)) as spinner:
            benchmark_file = dirs.DATA / "benchmark.json"
            if benchmark_file.is_file():
                spinner.text = "Condensing benchmark history..."
                benchmark = load_json(benchmark_file)
                new_benchmark = condense_benchmark(benchmark)
                if dryrun:
                    logger.info("Condensed Benchmark: %s", new_benchmark)
                else:
                    logger.info("Condensed benchmark history saved to %s", benchmark_file)
                    with open(benchmark_file, "w+") as f:
                        json.dump(new_benchmark, f, indent=4)

            file_list = list(dirs.DATA.glob("*.bz2"))
            if not file_list:
                return

            df = _build_file_index(file_list)

            for is_changelog, label in [(True, "changelog"), (False, "data")]:
                spinner.text = f"Cleaning {label} files..."
                logger.info("Processing %s files...", label)
                if dryrun:
                    print(
                        "Processing {label} files...",
                    )

                subdf = df[df["IsChangelog"] == is_changelog].copy()
                if subdf.empty:
                    continue

                cutoff = subdf["MaxTS_pd"].max() - pd.Timedelta(days=30)

                # Recent → weekly buckets
                recent = subdf[subdf["MaxTS_pd"] >= cutoff].copy()
                recent["Bucket"] = recent["MaxTS_pd"].dt.tz_localize(None).dt.to_period("W").dt.to_timestamp()

                # Older → monthly buckets
                older = subdf[subdf["MaxTS_pd"] < cutoff].copy()
                older["Bucket"] = older["MaxTS_pd"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()

                combined = pd.concat([recent, older])

                for (group, bucket), group_df in combined.groupby(["Group", "Bucket"]):
                    logger.debug(
                        "Processing %s files for group %s bucket %s",
                        label,
                        group,
                        bucket.strftime("%Y-%m-%d"),
                    )

                    _process_group(
                        group_df=group_df,
                        is_changelog=is_changelog,
                        dryrun=dryrun,
                    )

            spinner.text = "Updating index..."
            if not dryrun:
                result = git.commit(
                    files=[
                        dirs.DATA / "benchmark.json",
                        dirs.DATA / "*bz2",
                    ],
                    message=f"Data cleanup {arrow.utcnow().isoformat()}",
                )
                logger.info("%s", result)

            spinner.succeed("Data cleanup completed")

    finally:
        try:
            unlock("cleanup_data")
        except Exception as e:
            logger.error("Error unlocking cleanup_data lock. %s", e)


# Benchmark Condensing
def condense_benchmark(benchmark: Dict[str, Dict[str, List[BenchmarkEntry]]]) -> Dict[str, Dict[str, List[BenchmarkEntry]]]:
    """
    Condense benchmark history by weighted average and total weight for each key.

    Args:
        benchmark (Dict[str, List[BenchmarkEntry]]): Benchmark data dictionary.

    Returns:
        Dict[str, List[BenchmarkEntry]]: Condensed benchmark dictionary.
    """
    now = arrow.utcnow()
    for group_key, group_val in benchmark.items():
        for entry_key, entry_val in group_val.items():
            weighted_sum = 0.0
            total_weight = 0.0
            for entry in entry_val:
                weighted_sum += entry["average"] * entry["weight"]
                total_weight += entry["weight"]
            weighted_average = weighted_sum / total_weight if total_weight else 0.0
            benchmark[group_key][entry_key] = [
                {
                    "ts": now.isoformat(),
                    "average": weighted_average,
                    "weight": total_weight,
                }
            ]

    return benchmark


# Changelog Condensing
def condense_changelogs(files: List[Path | str]) -> pd.DataFrame:
    """
    Condense multiple changelog files into a consolidated dataframe.

    Args:
        files (List[Path | str]): A list of changelog file paths.

    Returns:
        pd.DataFrame: The consolidated changelog dataframe.
    """
    dfs = []

    for file in files:
        file_path = Path(file)

        from_ts, to_ts = _parse_changelog_ts(file_path)
        if from_ts is None or to_ts is None:
            continue

        df = pd.read_csv(file_path, dtype=object)

        # Map version to actual timestamps
        df["Version"] = df["Version"].map(
            {
                "Old": filename_ts_fmt(from_ts),
                "New": filename_ts_fmt(to_ts),
            }
        )

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    new_changelog = pd.concat(dfs, ignore_index=True)

    # Sort deterministically
    new_changelog = new_changelog.sort_values(
        by=[new_changelog.columns[0], "Version"],
        ascending=[True, True],
    )

    # Remove duplicates (keep latest state)
    new_changelog = new_changelog.drop_duplicates(keep="last")

    # Remove empty rows
    new_changelog = new_changelog.dropna(how="all")

    # Keep only latest version per row identity
    index = new_changelog.drop(["Modification date", "Version"], axis=1, errors="ignore").drop_duplicates(keep="last").index

    return new_changelog.loc[index]


# Build file index (single source of truth)
def _build_file_index(file_list: List[Path]) -> pd.DataFrame:
    """
    Build a DataFrame index of data and changelog files organising them by group and timestamp.

    Args:
        file_list (List[Path]): A list of file paths to index.

    Returns:
        pd.DataFrame: A DataFrame containing the indexed file information.
    """
    rows = []

    for path in file_list:
        name = path.name
        is_changelog = "changelog" in name

        if is_changelog:
            min_ts, max_ts = _parse_changelog_ts(path)
        else:
            max_ts = _parse_data_ts(path)
            min_ts = max_ts

        # skip files where timestamp parsing failed
        if min_ts is None or max_ts is None:
            logger.warning(f"Skipping file with invalid timestamp: {path}")
            continue

        group = _extract_group(path)
        if group is None:
            logger.warning(f"Skipping file with unrecognized group: {path}")
            continue

        rows.append({"Name": path, "IsChangelog": is_changelog, "MinTS": min_ts, "MaxTS": max_ts, "Group": group})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # convert to pandas datetime for grouping
    df["MaxTS_pd"] = pd.to_datetime(df["MaxTS"].apply(lambda x: x.datetime))
    df["MinTS_pd"] = pd.to_datetime(df["MinTS"].apply(lambda x: x.datetime))

    return df


def _parse_data_ts(path: Path | str) -> arrow.Arrow | None:
    """
    Parse timestamp from data filename, returning None if parsing fails.

    Args:
        path (str | Path): The file path to parse.

    Returns:
        arrow.Arrow | None: The parsed timestamp as an Arrow object, or None if parsing fails.
    """
    try:
        return arrow.get(Path(path).stem.split("_")[-1])
    except Exception:
        return None


def _parse_changelog_ts(path: Path | str) -> tuple[arrow.Arrow | None, arrow.Arrow | None]:
    """
    Parse from/to timestamps from changelog filename, returning (None, None) if parsing fails.
    Args:
        path (str | Path): The file path to parse.

    Returns:
        tuple[arrow.Arrow | None, arrow.Arrow | None]: A tuple containing the parsed from and to timestamps as Arrow objects, or (None, None) if parsing fails.
    """
    parts = Path(path).stem.split("_")
    try:
        return arrow.get(parts[-2]), arrow.get(parts[-1])
    except Exception:
        return None, None


# Process a single group bucket
def _process_group(group_df: pd.DataFrame, is_changelog: bool, dryrun: bool) -> None:
    """
    Process a single group bucket of files, deleting redundant files and condensing changelogs if necessary.

    Args:
        group_df (pd.DataFrame): DataFrame containing files for a specific group and bucket.
        is_changelog (bool): Whether the files are changelogs or data files.
        dryrun (bool): If True, log intended actions without modifying files.
    Returns:
        None
    """
    deleted_count = 0
    kept_count = 0

    # Delete files older than 12 months using arrow
    cutoff = arrow.utcnow().shift(months=-12)
    old_files = group_df[group_df["MaxTS_pd"].apply(lambda x: arrow.get(x) < cutoff)]["Name"].tolist()
    for file in old_files:
        logger.info("Delete old file %s", file)
        deleted_count += 1
        if not dryrun:
            os.remove(file)
        else:
            print(f"Dry run: would delete old file {file}")

    files = group_df.sort_values("MaxTS_pd", ascending=False)["Name"].tolist()
    if not files:
        logger.info("No files to process for group %s", group_df["Group"].iloc[0])
        return

    group = group_df["Group"].iloc[0]

    if is_changelog and len(files) > 1:
        new_changelog = condense_changelogs(files)
        if new_changelog.empty:
            logger.warning("Skipping empty changelog merge for group %s", group)
            return

        min_ts = arrow.get(new_changelog["timestamp"].min())
        max_ts = arrow.get(new_changelog["timestamp"].max())
        new_filename = make_filename(report=group, timestamp=max_ts, previous_timestamp=min_ts)
        new_filepath = dirs.DATA / new_filename
        logger.info("New changelog file: %s", new_filepath)
        if not dryrun:
            new_changelog.to_csv(new_filepath, index=False)
        else:
            print(f"Dry run: would save new changelog to {new_filepath}")

        for file in files:
            if Path(file) != new_filepath:
                logger.info("Delete %s", file)
                deleted_count += 1
                if not dryrun:
                    os.remove(file)
                else:
                    print(f"Dry run: would delete file {file}")
            else:
                logger.info("Keep %s", file)
                if dryrun:
                    print(f"Dry run: would keep file {file}")
                kept_count += 1
    else:
        most_recent = group_df.loc[group_df["MaxTS_pd"].idxmax(), "Name"]
        for file in files:
            if file != most_recent:
                logger.info("Delete %s", file)
                deleted_count += 1
                if not dryrun:
                    os.remove(file)
                else:
                    print(f"Dry run: would delete file {file}")
            else:
                logger.info("Keep %s", file)
                if dryrun:
                    print(f"Dry run: would keep file {file}")
                kept_count += 1

    message = f"Summary for group {group} ({'changelog' if is_changelog else 'data'}): {deleted_count} would be deleted, {kept_count} would be kept"
    logger.info(message)
    if dryrun:
        print(f"Dry run: {message}")


def _extract_group(path: Path | str) -> str | None:
    """
    Extract the group name from a filename, returning None if it cannot be determined.

    Args:
        path (str | Path): The file path to extract the group from.

    Returns:
        str | None: The extracted group name or None if it cannot be determined.
    """
    name = Path(path).stem

    if "_data_" in name:
        return name.split("_data_")[0]
    if "_changelog_" in name:
        return name.split("_changelog_")[0]

    return None
