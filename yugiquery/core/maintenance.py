# yugiquery/core/maintenance.py

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict

# --- Imports: Third-Party --- #
import arrow
from IPython.display import display
import pandas as pd

# --- Imports: Local Application --- #
from ..utils import dirs, get_notebook_path, git, load_json, make_filename

logger = logging.getLogger(__name__)


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

    if changelog.empty:
        logger.info("No changes")

    return changelog


# --- Benchmarking --- #


def benchmark(timestamp: arrow.Arrow, report: str | None = None) -> None:
    """
    Record report execution time and persist benchmark history.

    Args:
        timestamp (arrow.Arrow): Start timestamp for execution.
        report (str | None, optional): Report name. If None, infer notebook stem.
    """
    if report is None:
        path = get_notebook_path()
        report = path.stem if path else "Unnamed"

    now = arrow.utcnow()
    timedelta = now - timestamp
    benchmark_file = dirs.DATA / "benchmark.json"
    data = load_json(benchmark_file)

    if report not in data:
        data[report] = []
    data[report].append({"ts": now.isoformat(), "average": timedelta.total_seconds(), "weight": 1})

    with open(benchmark_file, "w+") as file:
        json.dump(data, file, indent=4)

    result = git.commit(
        files=[benchmark_file],
        message=f"{report.capitalize()} report benchmarked - {now.isoformat()}",
    )
    logger.info("%s", result)


# --- Changelog Condensing --- #


def condense_changelogs(files: List[Path | str]) -> Tuple[pd.DataFrame, Path]:
    """
    Condense multiple changelog files into a consolidated dataframe and generate a new filename.

    Args:
        files (List[Path | str]): A list of changelog file paths.

    Returns:
        Tuple[pd.DataFrame, Path]: The consolidated changelog dataframe and new file path.
    """
    new_changelog = pd.DataFrame()
    changelog_name = None
    first_date = None
    last_date = None
    last_file_path = None

    for file in files:
        file_path = Path(str(file))
        match = re.search(
            r"(\w+)_\w+_(\d{8}T\d{4})Z_(\d{8}T\d{4})Z.bz2",
            file_path.name,
        )
        if match is None:
            continue
        last_file_path = file_path
        name = match.group(1)
        from_date = match.group(2)
        to_date = match.group(3)
        if changelog_name is not None and changelog_name != name:
            logger.warning("Names mismatch!")
        changelog_name = name
        if first_date is None or first_date > from_date:
            first_date = from_date
        if last_date is None or last_date < to_date:
            last_date = to_date
        df = pd.read_csv(file_path, dtype=object)
        df["Version"] = df["Version"].map({"Old": from_date, "New": to_date})
        new_changelog = pd.concat([new_changelog, df], axis=0, ignore_index=True)

    new_changelog.sort_values(
        by=[new_changelog.columns[0], "Version"],
        ascending=[True, True],
        axis=0,
        inplace=True,
    )
    new_changelog = new_changelog.drop_duplicates(keep="last").dropna(how="all", axis=0)
    index = new_changelog.drop(["Modification date", "Version"], axis=1).drop_duplicates(keep="last").index

    assert last_file_path is not None, "No valid changelog files found"
    assert changelog_name is not None, "Unable to determine changelog name"
    assert last_date is not None, "Unable to determine last date"
    assert first_date is not None, "Unable to determine first date"

    new_filename = last_file_path.parent.joinpath(
        make_filename(
            report=changelog_name,
            timestamp=arrow.get(last_date),
            previous_timestamp=arrow.get(first_date),
        ),
    )
    return new_changelog.loc[index], new_filename


# --- Benchmark Condensing --- #


def condense_benchmark(benchmark: Dict[str, List[BenchmarkEntry]]) -> Dict[str, List[BenchmarkEntry]]:
    """
    Condense benchmark history by weighted average and total weight for each key.

    Args:
        benchmark (Dict[str, List[BenchmarkEntry]]): Benchmark data dictionary.

    Returns:
        Dict[str, List[BenchmarkEntry]]: Condensed benchmark dictionary.
    """
    now = arrow.utcnow()
    for key, _ in benchmark.items():
        for key, values in benchmark.items():
            weighted_sum = 0.0
            total_weight = 0.0
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


# --- Index Updating --- #


def update_index(dry_run: bool = False, page_paths: List[Path | str] | None = None) -> str:
    """
    Update `index.md` and `README.md` report table and last execution timestamp.

    Args:
        dry_run (bool, optional): If True, skip writing and committing changes.
        page_paths (List[Path | str] | None, optional): Additional .md pages to consider.

    Returns:
        str: Git commit output or dry-run advisory message.
    """

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
        rows.append(f"[{stem}]({link_path}) | {timestamp_str}")

    table = " |\n| ".join(rows)

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

    if dry_run:
        return "Dry run - README and index updated"

    with open(index_path, "w", encoding="utf-8") as o:
        o.write(index)
    with open(readme_path, "w", encoding="utf-8") as o:
        o.write(readme)
    return git.commit(
        files=[index_path, readme_path],
        message=f"Index and README timestamp update - {timestamp.isoformat()}",
    )


# --- Data Cleanup --- #


def cleanup_data(dry_run: bool = False) -> None:
    """
    Clean up redundant data files and compact benchmark/changelog history.

    Performs two main operations:
    1. Condenses benchmark history using weighted averages.
    2. Removes redundant changelog and data files, keeping only the most recent from each month
       and condensing multiple changelogs within the same month into a single consolidated file.

    Args:
        dry_run (bool, optional): If True, log intended actions without modifying files. Defaults to False.
    """
    dry_run_str = " (dry run)" if dry_run else ""
    logger.info("Starting data cleanup%s", dry_run_str)

    benchmark_file = dirs.DATA / "benchmark.json"
    if benchmark_file.is_file():
        benchmark = load_json(benchmark_file)
        new_benchmark = condense_benchmark(benchmark)
        if dry_run:
            logger.info("Benchmark: %s", new_benchmark)
        else:
            with open(benchmark_file, "w+") as f:
                json.dump(new_benchmark, f, indent=4)

    file_list = list(dirs.DATA.glob("*.bz2"))
    if not file_list:
        return

    df = pd.DataFrame(file_list, columns=["Name"])
    df["Date"] = pd.to_datetime(df["Name"].apply(os.path.getctime), unit="s")
    df["Group"] = df["Name"].apply(lambda x: "_".join(Path(x).name.split("_", 2)[:2]))
    grouped = df.groupby(["Group", pd.Grouper(key="Date", freq="MS")])

    same_month_files = {
        "changelog": [group[1]["Name"].tolist() for group in grouped if "changelog" in str(group[0][0])],
        "data": [group[1]["Name"].tolist() for group in grouped if "changelog" not in str(group[0][0])],
    }

    last_month_files = df[df["Date"] >= df["Date"].max() - pd.DateOffset(months=1)].resample("W", on="Date").first()
    last_month_files = {
        "changelog": last_month_files[last_month_files["Group"].str.contains("changelog", na=False)]["Name"]
        .dropna()
        .tolist(),
        "data": last_month_files[~last_month_files["Group"].str.contains("changelog", na=False)]["Name"].dropna().tolist(),
    }

    last_month_changelog = set(last_month_files["changelog"])
    last_month_data = set(last_month_files["data"])

    same_month_files["changelog"] = [
        files for files in same_month_files["changelog"] if not any(file in last_month_changelog for file in files)
    ]
    same_month_files["data"] = [
        files for files in same_month_files["data"] if not any(file in last_month_data for file in files)
    ]

    logger.info("same month (with changelog)")
    for files in same_month_files["changelog"]:
        if len(files) > 1:
            new_changelog, new_filepath = condense_changelogs(files)
            logger.info("New changelog file: %s", new_filepath)
            if dry_run:
                display(new_changelog)
            else:
                new_changelog.to_csv(new_filepath)
            for file in files:
                if dry_run:
                    logger.info("Delete %s", file)
                else:
                    os.remove(file)

    logger.info("same month (without changelog)")
    for files in same_month_files["data"]:
        for file in files[:-1]:
            if dry_run:
                logger.info("Delete %s", file)
            else:
                os.remove(file)
        if dry_run:
            logger.info("Keep %s", files[-1])

    if (files := last_month_files["changelog"]) and (len(files) > 1):
        logger.info("Last month (with changelog)")
        new_changelog, new_filepath = condense_changelogs(files)
        logger.info("New changelog file: %s", new_filepath)
        if dry_run:
            display(new_changelog)
        else:
            new_changelog.to_csv(new_filepath)
        for file in last_month_files["changelog"]:
            if dry_run:
                logger.info("Delete %s", file)
            else:
                os.remove(file)

    if files := last_month_files["data"]:
        logger.info("Last month (without changelog)")
        for file in files[:-1]:
            if dry_run:
                logger.info("Delete %s", file)
            else:
                os.remove(file)
        if dry_run:
            logger.info("Keep %s", files[-1])

    if not dry_run:
        result = git.commit(
            files=[
                dirs.DATA / "benchmark.json",
                dirs.DATA / "*bz2",
            ],
            message=f"Data cleanup {arrow.utcnow().isoformat()}",
        )
        logger.info("%s", result)
