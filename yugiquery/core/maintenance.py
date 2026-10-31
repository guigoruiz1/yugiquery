# yugiquery/core/maintenance.py

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import json
import os
import re
import tempfile
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

        index_path = dirs.WORK / "index.md"
        readme_path = dirs.WORK / "README.md"

        timestamp = arrow.utcnow()

        with open(index_path, encoding="utf-8") as f:
            index = f.read()
        with open(readme_path, encoding="utf-8") as f:
            readme = f.read()

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

        all_reports: Dict[str, Dict[str, Path]] = {}

        for f in dirs.REPORTS.glob("*.html"):
            all_reports.setdefault(f.stem, {})["html"] = f

        for f in dirs.REPORTS.glob("*.md"):
            all_reports.setdefault(f.stem, {})["md"] = f

        if page_paths:
            for path in page_paths:
                path = Path(path)
                if path.is_dir():
                    for f in path.glob("*.md"):
                        all_reports.setdefault(f.stem, {})["md"] = f
                elif path.is_file() and path.suffix == ".md":
                    all_reports.setdefault(path.stem, {})["md"] = path

        rows = []
        for stem in sorted(all_reports.keys()):
            report_entry = all_reports[stem]
            report_file = report_entry.get("md") or report_entry.get("html")
            if report_file is None:
                continue

            # Prefer HTML file modification time since reports are generated as HTML.
            timestamp_file = report_entry.get("html") or report_file
            try:
                relative_path = report_file.relative_to(dirs.WORK)
            except ValueError:
                relative_path = report_file

            if relative_path.suffix == ".md":
                permalink = extract_permalink(report_file)
                if permalink:
                    link_path = permalink.strip("/") + "/"
                else:
                    link_path = str(relative_path.with_suffix("")) + "/"
            elif relative_path.suffix == ".html":
                link_path = str(relative_path.with_suffix(""))
            else:
                link_path = str(relative_path)

            timestamp_str = pd.to_datetime(timestamp_file.stat().st_mtime, unit="s", utc=True).strftime("%d/%m/%Y %H:%M %Z")
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
            # Keep blank lines around the markdown table so Jekyll/kramdown parses it reliably.
            updated = before + "\n\n" + table + "\n\n" + after

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
            return "README and index updated - Dry run, not committed."
    finally:
        try:
            unlock("update_index")
        except Exception as e:
            logger.error("Error unlocking update_index lock. %s", e)


# --- Data Cleanup --- #


def cleanup_data(dryrun: bool = False) -> None:
    """
    Clean up redundant data files and compact benchmark/changelog history.

    Changelogs are consolidated when they belong to the same retention
    bucket or have overlapping/touching timestamp intervals.

    Data snapshots retain the existing weekly/monthly bucket policy.
    """
    lock("cleanup_data")

    try:
        dryrun_str = " (dry run)" if dryrun else ""
        print(f"\nStarting data cleanup{dryrun_str}")
        logger.info("Starting data cleanup%s", dryrun_str)

        with Halo(
            text="Cleaning up data...",
            spinner="line",
            enabled=("PM_IN_EXECUTION" not in os.environ),
        ) as spinner:

            # Condense benchmark history
            benchmark_file = dirs.DATA / "benchmark.json"

            if benchmark_file.is_file():
                spinner.text = "Condensing benchmark history..."

                benchmark = load_json(benchmark_file)
                new_benchmark = condense_benchmark(benchmark)

                if dryrun:
                    logger.info("Condensed Benchmark: %s", new_benchmark)
                else:
                    logger.info(
                        "Condensed benchmark history saved to %s",
                        benchmark_file,
                    )

                    with open(benchmark_file, "w+") as f:
                        json.dump(new_benchmark, f, indent=4)

            file_list = list(dirs.DATA.glob("*.bz2"))

            if not file_list:
                spinner.succeed("Data cleanup completed")
                return

            df = _build_file_index(file_list)
            if df.empty:
                logger.warning("No valid data or changelog files found.")
                spinner.succeed("Data cleanup completed")
                return

            # Process all changelogs together to detect connections
            # across weekly and monthly bucket boundaries
            spinner.text = "Cleaning changelog files..."
            logger.info("Processing changelog files...")

            if dryrun:
                print("Processing changelog files...")

            changelogs = df[df["IsChangelog"]].copy()

            for group, group_df in changelogs.groupby("Group"):
                _process_group(
                    group_df=group_df,
                    is_changelog=True,
                    dryrun=dryrun,
                )

            # Process data snapshots using the existing bucket policy
            spinner.text = "Cleaning data files..."
            logger.info("Processing data files...")

            if dryrun:
                print("Processing data files...")

            data_files = df[~df["IsChangelog"]].copy()

            if not data_files.empty:
                cutoff = (
                    data_files["MaxTS_pd"].max()
                    - pd.Timedelta(days=30)
                )

                # Recent → weekly buckets
                recent = data_files[
                    data_files["MaxTS_pd"] >= cutoff
                ].copy()

                recent["Bucket"] = (
                    recent["MaxTS_pd"]
                    .dt.tz_localize(None)
                    .dt.to_period("W")
                    .dt.to_timestamp()
                )

                # Older → monthly buckets
                older = data_files[
                    data_files["MaxTS_pd"] < cutoff
                ].copy()

                older["Bucket"] = (
                    older["MaxTS_pd"]
                    .dt.tz_localize(None)
                    .dt.to_period("M")
                    .dt.to_timestamp()
                )

                combined = pd.concat([recent, older])

                for (group, bucket), group_df in combined.groupby(
                    ["Group", "Bucket"]
                ):
                    logger.debug(
                        "Processing data files for group %s bucket %s",
                        group,
                        bucket.strftime("%Y-%m-%d"),
                    )

                    _process_group(
                        group_df=group_df,
                        is_changelog=False,
                        dryrun=dryrun,
                    )

            # Commit cleanup changes
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
            logger.error(
                "Error unlocking cleanup_data lock. %s",
                e,
            )


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
    Consolidate changelogs while preserving timestamp-based history.

    Fresh Old/New versions are converted using their respective filename
    boundaries. Existing timestamps are preserved.

    Rows with missing Version values are discarded because their historical
    timestamps cannot be reliably recovered.

    Invalid or out-of-interval timestamps raise an error.

    Identical states are deduplicated, ignoring Modification date and
    Version, keeping the latest occurrence.
    """
    dfs = []
    timestamp_pattern = re.compile(r"^\d{8}T\d{4}Z$")

    # Process source files in chronological order for deterministic results
    file_intervals = []

    for file in files:
        file_path = Path(file)
        from_ts, to_ts = _parse_changelog_ts(file_path)

        if from_ts is None or to_ts is None:
            raise ValueError(f"Invalid changelog filename: {file_path}")

        if from_ts >= to_ts:
            raise ValueError(f"Invalid changelog interval: {file_path}")

        file_intervals.append((from_ts, to_ts, file_path))

    file_intervals.sort(
        key=lambda item: (item[0], item[1], item[2].name)
    )

    for from_ts, to_ts, file_path in file_intervals:
        start_ts = filename_ts_fmt(from_ts)
        end_ts = filename_ts_fmt(to_ts)

        df = pd.read_csv(
            file_path,
            dtype=object,
            keep_default_na=False,
            na_values="",
        )

        if "Version" not in df.columns:
            raise ValueError(f"Missing Version column in {file_path}")

        # Convert fresh versions while preserving existing timestamps
        df["Version"] = df["Version"].replace({
            "Old": start_ts,
            "New": end_ts,
        })

        # Discard rows whose historical version cannot be established
        missing_versions = (
            df["Version"].isna()
            | df["Version"].astype(str).str.strip().eq("")
        )

        missing_count = int(missing_versions.sum())

        if missing_count:
            logger.warning(
                "Discarding %d rows with missing Version in %s",
                missing_count,
                file_path,
            )

        df = df.loc[~missing_versions].copy()

        # Validate every remaining version against this source file's interval
        valid_versions = df["Version"].map(
            lambda value: (
                bool(timestamp_pattern.fullmatch(str(value)))
                and start_ts <= str(value) <= end_ts
            )
        )

        if not valid_versions.all():
            invalid = df.loc[~valid_versions, "Version"].unique().tolist()
            raise ValueError(
                f"Invalid or out-of-interval Version values in "
                f"{file_path}: {invalid}"
            )

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    new_changelog = pd.concat(dfs, ignore_index=True)

    # Remove completely empty rows
    new_changelog = new_changelog.dropna(how="all")

    # Deduplicate identical states, ignoring Modification date and Version
    identity_columns = [
        column for column in new_changelog.columns
        if column not in ["Modification date", "Version"]
    ]

    if identity_columns:
        # Keep the latest occurrence of each identical state
        new_changelog = new_changelog.sort_values(
            "Version",
            kind="stable",
        )

        new_changelog = new_changelog.drop_duplicates(
            subset=identity_columns,
            keep="last",
        )

    # Sort the result deterministically
    sort_columns = [
        column for column in ["Name", "Version"]
        if column in new_changelog.columns
    ]

    if sort_columns:
        new_changelog = new_changelog.sort_values(
            sort_columns,
            kind="stable",
        )

    return new_changelog.reset_index(drop=True)


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
def _process_data(group_df: pd.DataFrame, dryrun: bool) -> None:
    """Keep the newest data snapshot in a retention bucket."""
    group = group_df["Group"].iloc[0]

    most_recent = group_df.loc[
        group_df["MaxTS_pd"].idxmax(), "Name"
    ]

    deleted_count = 0

    for file in group_df["Name"]:
        if file == most_recent:
            logger.info("Keep %s", file)

            if dryrun:
                print(f"Dry run: would keep file {file}")

            continue

        logger.info("Delete %s", file)
        deleted_count += 1

        if dryrun:
            print(f"Dry run: would delete file {file}")
        else:
            os.remove(file)

    message = (
        f"Summary for group {group} (data): "
        f"{deleted_count} would be deleted, 1 would be kept"
    )

    logger.info(message)

    if dryrun:
        print(f"Dry run: {message}")


def _process_changelogs(subdf: pd.DataFrame, dryrun: bool) -> None:
    """
    Consolidate changelogs belonging to the same retention bucket or
    having overlapping/touching timestamp intervals.
    """
    if subdf.empty:
        return

    # Preserve existing weekly/monthly retention buckets
    cutoff = subdf["MaxTS_pd"].max() - pd.Timedelta(days=30)

    recent = subdf[subdf["MaxTS_pd"] >= cutoff].copy()
    recent["Bucket"] = (
        recent["MaxTS_pd"]
        .dt.tz_localize(None)
        .dt.to_period("W")
        .dt.to_timestamp()
    )

    older = subdf[subdf["MaxTS_pd"] < cutoff].copy()
    older["Bucket"] = (
        older["MaxTS_pd"]
        .dt.tz_localize(None)
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    combined = pd.concat([recent, older]).reset_index(drop=True)

    if combined.empty:
        return

    # Build connected components
    parent = list(range(len(combined)))

    def find(index):
        if parent[index] != index:
            parent[index] = find(parent[index])
        return parent[index]

    def union(first, second):
        first_root = find(first)
        second_root = find(second)

        if first_root != second_root:
            parent[second_root] = first_root

    # Files in the same group and retention bucket are connected
    for _, indices in combined.groupby(["Group", "Bucket"]).groups.items():
        indices = list(indices)

        for index in indices[1:]:
            union(indices[0], index)

    # Overlapping or touching intervals are connected across buckets
    for group, group_df in combined.groupby("Group"):
        intervals = []

        for index, row in group_df.iterrows():
            start, end = _parse_changelog_ts(row["Name"])

            if start is None or end is None or start >= end:
                raise ValueError(
                    f"Invalid changelog interval for group {group}: "
                    f"{row['Name']}"
                )

            intervals.append((start, end, index))

        intervals.sort(key=lambda item: (item[0], item[1]))

        component_root = None
        component_end = None

        for start, end, index in intervals:
            # A shared endpoint counts as connected
            if component_root is None or start > component_end:
                component_root = index
                component_end = end
                continue

            union(component_root, index)
            component_end = max(component_end, end)

    combined["_Component"] = [
        find(index) for index in range(len(combined))
    ]

    components = combined.groupby("_Component", sort=False)

    logger.info(
        "Processing %d connected changelog groups",
        len(components),
    )

    for _, component_df in components:
        group = component_df["Group"].iloc[0]
        files = component_df.sort_values(
            "MaxTS_pd",
            ascending=False,
        )["Name"].tolist()

        # A single file needs no consolidation
        if len(files) == 1:
            logger.info("Keep %s", files[0])

            if dryrun:
                print(f"Dry run: would keep file {files[0]}")

            continue

        # Consolidate and validate source changelogs
        new_changelog = condense_changelogs(files)

        if new_changelog.empty:
            logger.warning(
                "Skipping empty changelog merge for group %s",
                group,
            )
            continue

        intervals = [_parse_changelog_ts(file) for file in files]

        min_ts = min(start for start, _ in intervals)
        max_ts = max(end for _, end in intervals)

        new_filename = make_filename(
            report=group,
            timestamp=max_ts,
            previous_timestamp=min_ts,
        )

        new_filepath = dirs.DATA / new_filename
        source_paths = {Path(file).resolve() for file in files}

        if (
            new_filepath.exists()
            and new_filepath.resolve() not in source_paths
        ):
            raise FileExistsError(
                f"Consolidated output already exists and is not a source "
                f"file: {new_filepath}"
            )

        if dryrun:
            print(
                f"Dry run: would consolidate {len(files)} files into "
                f"{new_filename} ({len(new_changelog):,} rows)"
            )
            continue

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{group}_changelog_",
                suffix=".bz2",
                dir=dirs.DATA,
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)

            new_changelog.to_csv(
                temp_path,
                index=False,
                compression="bz2",
            )

            # Verify the written file before replacing any sources
            verified = pd.read_csv(
                temp_path,
                compression="bz2",
                dtype=object,
                keep_default_na=False,
                na_values="",
            )

            pd.testing.assert_frame_equal(
                verified,
                new_changelog,
                check_dtype=False,
                check_exact=True,
            )

            timestamp_pattern = re.compile(r"^\d{8}T\d{4}Z$")

            valid_versions = verified["Version"].map(
                lambda value: (
                    bool(timestamp_pattern.fullmatch(str(value)))
                    and filename_ts_fmt(min_ts)
                    <= str(value)
                    <= filename_ts_fmt(max_ts)
                ) if pd.notna(value) else False
            )

            if not valid_versions.all():
                raise ValueError(
                    f"Invalid Version values in consolidated "
                    f"changelog {new_filename}"
                )

            temp_path.replace(new_filepath)
            temp_path = None

        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

        # Delete sources only after successful output verification
        for file in files:
            if Path(file).resolve() != new_filepath.resolve():
                os.remove(file)
                logger.info("Delete %s", file)

        logger.info("Keep consolidated changelog %s", new_filepath)

def _process_group(group_df: pd.DataFrame, is_changelog: bool, dryrun: bool) -> None:
    """Apply the 12-month retention policy and delegate file processing."""
    group = group_df["Group"].iloc[0]
    label = "changelog" if is_changelog else "data"

    cutoff = arrow.utcnow().shift(months=-12)

    expired = group_df[
        group_df["MaxTS_pd"].apply(
            lambda value: arrow.get(value) < cutoff
        )
    ]

    deleted_count = 0

    for file in expired["Name"]:
        logger.info("Delete old file %s", file)
        deleted_count += 1

        if dryrun:
            print(f"Dry run: would delete old file {file}")
        else:
            os.remove(file)

    active_df = group_df[
        ~group_df["Name"].isin(expired["Name"])
    ].copy()

    if not active_df.empty:
        if is_changelog:
            _process_changelogs(active_df, dryrun)
        else:
            _process_data(active_df, dryrun)

    logger.info(
        "Retention summary for %s (%s): %d expired files",
        group,
        label,
        deleted_count,
    )


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
