#!/usr/bin/env python3

# -*- coding: utf-8 -*-

# ======================================================================== #
#                                                                          #
# ██    ██ ██    ██  ██████  ██  ██████  ██    ██ ███████ ██████  ██    ██ #
#  ██  ██  ██    ██ ██       ██ ██    ██ ██    ██ ██      ██   ██  ██  ██  #
#   ████   ██    ██ ██   ███ ██ ██    ██ ██    ██ █████   ██████    ████   #
#    ██    ██    ██ ██    ██ ██ ██ ▄▄ ██ ██    ██ ██      ██   ██    ██    #
#    ██     ██████   ██████  ██  ██████   ██████  ███████ ██   ██    ██    #
#                                   ▀▀                                     #
# ======================================================================== #

__author__ = "Guilherme Ruiz"
__copyright__ = "2023, Guilherme Ruiz"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Guilherme Ruiz"
__email__ = "57478888+guigoruiz1@users.noreply.github.com"
__status__ = "Development"

# ======= #
# Imports #
# ======= #

# Native python packages
import argparse
import glob

import io
import json
import logging
import os
import subprocess
import time
from enum import Enum


# PIP packages
loop = 0
while True:
    try:
        import urllib.parse as up
        from ast import literal_eval

        import jupyter_client

        import nbformat
        import numpy as np
        import pandas as pd
        import papermill as pm

        from ipylab import JupyterFrontEnd
        from IPython.display import Markdown, display
        from utils import *

        break

    except ImportError:
        if loop > 1:
            print("Failed to install required packages twice. Aborting...")
            quit()

        loop += 1
        print("Missing required packages. Trying to install now...")
        subprocess.call(["sh", os.path.join(SCRIPT_DIR, "./install.sh")])

# Default settings overrides
pd.set_option("display.max_columns", 40)

# ========= #
# Variables #
# ========= #


class CG(Enum):
    """
    Enum representing the card game formats.

    Attributes:
        CG (str): Both TCG and OCG.
        ALL (CG): Alias for CG, representing all card games.
        BOTH (CG): Alias for CG, representing both card games.
        TCG (str): The 'trading card game' type.
        OCG (str): The 'official card game' type.
    """

    CG = "CG"
    ALL = CG
    BOTH = CG
    TCG = "TCG"
    OCG = "OCG"


# =============== #
# Data management #
# =============== #


def generate_changelog(
    previous_df: pd.DataFrame, current_df: pd.DataFrame, col: Union[str, List[str]]
):
    """
    Generates a changelog DataFrame by comparing two DataFrames based on a specified column.

    Args:
        previous_df (pd.DataFrame): A DataFrame containing the previous version of the data.
        current_df (pd.DataFrame): A DataFrame containing the current version of the data.
        col (Union[str, List[str]]): The name of the column to compare the DataFrames on.

    Returns:
        pd.DataFrame: A DataFrame containing the changes made between the previous and current versions of the data. The DataFrame will have the following columns: the specified column name, the modified data, and the indicator for whether the data is new or modified renamed as version (either "Old" or "New"). If there are no changes, the function will return a DataFrame with no rows.
    """
    if isinstance(col, str):
        col = [col]
    changelog = (
        previous_df.merge(current_df, indicator=True, how="outer")
        .loc[lambda x: x["_merge"] != "both"]
        .sort_values(col, ignore_index=True)
    )
    changelog["_merge"] = changelog["_merge"].cat.rename_categories(
        {"left_only": "Old", "right_only": "New"}
    )
    changelog.rename(columns={"_merge": "Version"}, inplace=True)
    nunique = changelog.groupby(col).nunique(dropna=False)
    cols_to_drop = (
        nunique[nunique < 2]
        .dropna(axis=1)
        .columns.difference(["Modification date", "Version"])
    )
    changelog.drop(cols_to_drop, axis=1, inplace=True)
    changelog = changelog.set_index(col)

    if all(col in changelog.columns for col in ["Modification date", "Version"]):
        true_changes = (
            changelog.drop(["Modification date", "Version"], axis=1)[nunique > 1]
            .dropna(axis=0, how="all")
            .index
        )
        new_entries = nunique[nunique["Version"] == 1].dropna(axis=0, how="all").index
        rows_to_keep = true_changes.union(new_entries).unique()
        changelog = changelog.loc[rows_to_keep].sort_values(by=[*col, "Version"])

    if changelog.empty:
        print("No changes")

    return changelog


def benchmark(timestamp: arrow.Arrow, report: str = None):
    """
    Records the execution time of a report and saves the data to a JSON file.

    Args:
        timestamp (arrow.Arrow): The timestamp when the report execution began.
         report (str): The name of the report being benchmarked. If None, tries obtaining report name from JPY_SESSION_NAME environment variable.

    Returns:
        None
    """
    if report is None:
        try:
            report = os.path.basename(os.environ["JPY_SESSION_NAME"]).split(".")[0]
        except:
            report = ""

    now = arrow.utcnow()
    timedelta = now - timestamp
    benchmark_file = os.path.join(PARENT_DIR, "data/benchmark.json")
    data = load_json(benchmark_file)
    # Add the new data to the existing data
    if report not in data:
        data[report] = []
    data[report].append(
        {"ts": now.isoformat(), "average": timedelta.total_seconds(), "weight": 1}
    )
    # Save new data to file
    with open(benchmark_file, "w+") as file:
        json.dump(data, file)

    result = git.commit(
        files=[benchmark_file],
        commit_message=f"{report} report benchmarked - {now.isoformat()}",
    )
    print(result)


def condense_changelogs(files: pd.DataFrame):
    """
    Condenses multiple changelog files into a consolidated dataframe and generates a new filename.

    Args:
        files (pd.DataFrame): A dataframe containing the changelog files.

    Returns:
        Tuple[pd.DataFrame, str]: A tuple containing the consolidated changelog dataframe and the new filename.
    """
    new_changelog = pd.DataFrame()
    changelog_name = None
    first_date = None
    last_date = None
    for file in files:
        match = re.search(
            r"(\w+)_\w+_(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})Z_(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})Z.bz2",
            os.path.basename(file),
        )
        name = match.group(1)
        from_date = match.group(2)
        to_date = match.group(3)
        if changelog_name is not None and not changelog_name == name:
            print("Names mismatch!")
        changelog_name = name
        if first_date is None or first_date > from_date:
            first_date = from_date
        if last_date is None or last_date < to_date:
            last_date = to_date
        df = pd.read_csv(file, dtype=object)
        df["Version"] = df["Version"].map({"Old": from_date, "New": to_date})
        new_changelog = pd.concat([new_changelog, df], axis=0, ignore_index=True)

    new_changelog.sort_values(
        by=[new_changelog.columns[0], "Version"],
        ascending=[True, True],
        axis=0,
        inplace=True,
    )
    new_changelog = new_changelog.drop_duplicates(keep="last").dropna(how="all", axis=0)
    index = (
        new_changelog.drop(["Modification date", "Version"], axis=1)
        .drop_duplicates(keep="last")
        .index
    )
    new_filename = os.path.join(
        os.path.dirname(file),
        make_filename(
            report=changelog_name,
            timestamp=arrow.get(last_date),
            previous_timestamp=arrow.get(first_date),
        ),
    )
    return new_changelog.loc[index], new_filename


def condense_benchmark(benchmark: dict):
    """
    Condenses a benchmark dictionary by calculating the weighted average and total weight for each key.

    Args:
        benchmark (dict): A dictionary containing benchmark data.

    Returns:
        dict: The condensed benchmark dictionary with updated entries.
    """
    now = arrow.utcnow()
    for key, pair in benchmark.items():
        for key, values in benchmark.items():
            weighted_sum = 0
            total_weight = 0
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


def cleanup_data(dry_run=False):
    """
    Cleans up data files, keeping only the most recent file from each month and week.

    Args:
        dry_run (bool): If True, the function will only print the files that would be deleted without actually deleting them. Defaults to False.

    Returns:
        None
    """
    # Benchmark
    now = arrow.utcnow()
    benchmark_path = os.path.join(PARENT_DIR, "data/benchmark.json")
    benchmark = load_json(benchmark_path)
    new_benchmark = condense_benchmark(benchmark)
    if dry_run:
        print("Benchmark:", new_benchmark)
    else:
        with open(benchmark_path, "w+") as f:
            json.dump(new_benchmark, f)

    # Data CSV files
    file_list = glob.glob(os.path.join(PARENT_DIR, "data/*.bz2"))

    # Create a DataFrame
    df = pd.DataFrame(file_list, columns=["Name"])

    # Convert the 'Date' column to a datetime type
    df["Date"] = pd.to_datetime(df["Name"].apply(os.path.getctime), unit="s")

    # Create a new column 'Group' based on the first two elements after splitting the filename
    df["Group"] = df["Name"].apply(
        lambda x: "_".join(os.path.basename(x).split("_", 2)[:2])
    )

    # Group the DataFrame by 'Group' and 'Date' (year and month)
    grouped = df.groupby(["Group", pd.Grouper(key="Date", freq="MS")])

    # Get a list of all the files created on the same month of the same year, separated by whether they contain "changelog"
    same_month_files = {
        "changelog": [
            group[1]["Name"].tolist() for group in grouped if "changelog" in group[0][0]
        ],
        "data": [
            group[1]["Name"].tolist()
            for group in grouped
            if not "changelog" in group[0][0]
        ],
    }

    # Get a list of all the files created in the last month and split them into weeks
    last_month_files = (
        df[df["Date"] >= df["Date"].max() - pd.Timedelta("1MS")]
        .resample("W", on="Date")
        .first()
    )

    # Separate the last_month_files by whether they contain "changelog"
    last_month_files = {
        "changelog": last_month_files[
            last_month_files["Group"].str.contains("changelog")
        ]["Name"].tolist(),
        "data": last_month_files[~last_month_files["Group"].str.contains("changelog")][
            "Name"
        ].tolist(),
    }

    # Remove the last_month_files from the same_month_files
    same_month_files["changelog"] = [
        files
        for files in same_month_files["changelog"]
        if files not in last_month_files["changelog"]
    ]
    same_month_files["data"] = [
        files
        for files in same_month_files["data"]
        if files not in last_month_files["data"]
    ]

    print("\n- same month (with changelog)")
    for files in same_month_files["changelog"]:
        if len(files) > 1:
            new_changelog, new_filepath = condense_changelogs(files)
            print(f"New changelog file: {new_filepath}")
            if dry_run:
                display(new_changelog)
            else:
                new_changelog.to_csv(new_filepath)
            for file in files:
                if dry_run:
                    print("Delete", file)
                else:
                    os.remove(file)

    print("\n- same month (without changelog)")
    for files in same_month_files["data"]:
        for file in files[:-1]:
            if dry_run:
                print("Delete", file)
            else:
                os.remove(file)
        if dry_run:
            print("Keep", files[-1])

    if (files := last_month_files["changelog"]) and (len(files) > 1):
        print("\n- Last month (with changelog)")
        new_changelog, new_filepath = condense_changelogs(files)
        print(f"New changelog file: {new_filepath}")
        if dry_run:
            display(new_changelog)
        else:
            new_changelog.to_csv(new_filepath)
        for file in last_month_files["changelog"]:
            if dry_run:
                print("Delete", file)
            else:
                os.remove(file)

    if files := last_month_files["data"]:
        print("\n- Last month (without changelog)")
        for file in files[:-1]:
            if dry_run:
                print("Delete", file)
            else:
                os.remove(file)
        if dry_run:
            print("Keep", files[-1])

    if not dry_run:
        result = git.commit(
            files=[
                os.path.join(PARENT_DIR, "data/benchmark.json"),
                os.path.join(PARENT_DIR, "data/*bz2"),  # May not work
            ],
            commit_message=f"Data cleanup {arrow.utcnow().isoformat()}",
        )
        print(result)


def load_corrected_latest(name_pattern: str, tuple_cols: List[str] = []):
    """
    Loads the most recent data file matching the specified name pattern and applies corrections.

    Args:
        name_pattern (str): Data file name pattern to load.
        tuple_cols (List[str]): List of columns containing tuple values to apply literal_eval.

    Returns:
        Tuple[pd.DataFrame, arrow.Arrow]: A tuple containing the loaded dataframe and the timestamp of the file.
    """
    files = sorted(
        glob.glob(f"../data/{name_pattern}_data_*.bz2"),
        key=os.path.getctime,
        reverse=True,
    )

    if files:
        df = pd.read_csv(files[0], dtype=object)
        for col in tuple_cols:
            if col in df:
                df[col] = df[col].dropna().apply(literal_eval)

        for col in ["Modification date", "Release"]:
            if col in df:
                df[col] = pd.to_datetime(df[col])

        ts = arrow.get(os.path.basename(files[0]).split("_")[-1].split(".bz2")[0])
        print(f"{name_pattern} file loaded")
        return df, ts
    else:
        print(f"No {name_pattern} files")
        return None, None


# Sets


def merge_set_info(input_df: pd.DataFrame, input_info_df: pd.DataFrame):
    """
    Merges set information from an input set info DataFrame into an input set list DataFrame based on set and region.

    Args:
        input_df (pd.DataFrame): A pandas DataFrame containing set lists.
        input_info_df (pd.DataFrame): A pandas DataFrame containing set information.

    Returns:
        pd.DataFrame: A pandas DataFrame with set information merged into it.
    """
    if all([col in input_df.columns for col in ["Set", "Region"]]):
        regions_dict = load_json(os.path.join(PARENT_DIR, "assets/json/regions.json"))
        input_df["Release"] = input_df[["Set", "Region"]].apply(
            lambda x: (
                input_info_df[regions_dict[x["Region"]] + " release date"][x["Set"]]
                if (
                    x["Region"] in regions_dict.keys()
                    and x["Set"] in input_info_df.index
                )
                else np.nan
            ),
            axis=1,
        )
        input_df["Release"] = pd.to_datetime(
            input_df["Release"].astype(str), errors="coerce"
        )  # Bug fix
        input_df = input_df.merge(
            input_info_df.loc[:, :"Cover card"],
            left_on="Set",
            right_index=True,
            how="outer",
            indicator=False,
        ).reset_index(drop=True)
        print("Set properties merged")
    else:
        print('Error! No "Set" and/or "Region" column(s) to join set info')

    return input_df


# =================== #
# Notebook management #
# =================== #


def save_notebook():
    """
    Save the current notebook opened in JupyterLab to disk.

    Args:
        None

    Returns:
        None
    """
    app = JupyterFrontEnd()
    app.commands.execute("docmanager:save")
    print("Notebook saved to disk")


# ================ #
# Markdown editing #
# ================ #


def update_index():  # Handle index and readme properly
    """
    Update the index.md and README.md files with a table of links to all HTML reports in the parent directory.
    Also update the @REPORT_|_TIMESTAMP@ and @TIMESTAMP@ placeholders in the index.md file with the latest timestamp.
    If the update is successful, commit the changes to Git with a commit message that includes the timestamp.
    If there is no index.md or README.md files in the assets directory, print an error message and abort.

    Raises:
        FileNotFoundError: If the "index.md" or "README.md" files in "assets" are not found.

    Returns:
        None
    """

    index_file_name = "index.md"
    readme_file_name = "README.md"

    index_input_path = os.path.join(PARENT_DIR, "assets/markdown", index_file_name)
    readme_input_path = os.path.join(PARENT_DIR, "assets/markdown", readme_file_name)
    index_output_path = os.path.join(PARENT_DIR, index_file_name)
    readme_output_path = os.path.join(PARENT_DIR, readme_file_name)

    timestamp = arrow.utcnow()
    try:
        with open(index_input_path) as f:
            index = f.read()

        with open(readme_input_path) as f:
            readme = f.read()
    except:
        print('Missing template files in "assets". Aborting...')

    reports = sorted(glob.glob(os.path.join(PARENT_DIR, "*.html")))
    rows = []
    for report in reports:
        rows.append(
            f"[{os.path.basename(report).split('.')[0]}]({os.path.basename(report)}) | {pd.to_datetime(os.path.getmtime(report),unit='s', utc=True).strftime('%d/%m/%Y %H:%M %Z')}"
        )
    table = " |\n| ".join(rows)

    index = index.replace(f"@REPORT_|_TIMESTAMP@", table)
    index = index.replace(f"@TIMESTAMP@", timestamp.strftime("%d/%m/%Y %H:%M %Z"))

    with open(index_output_path, "w+") as o:
        print(index, file=o)

    readme = readme.replace(f"@REPORT_|_TIMESTAMP@", table)
    readme = readme.replace(f"@TIMESTAMP@", timestamp.strftime("%d/%m/%Y %H:%M %Z"))

    with open(readme_output_path, "w+") as o:
        print(readme, file=o)

    result = git.commit(
        files=[index_output_path, readme_output_path],
        commit_message=f"Index and README timestamp update - {timestamp.isoformat()}",
    )
    print(result)


def header(name: str = None):
    """
    Generates a Markdown header with a timestamp and the name of the notebook (if provided).

    Args:
        name (str, optional): The name of the notebook. If None, attempts to extract the name from the environment variable JPY_SESSION_NAME. Defaults to None.

    Returns:
        Markdown: The generated Markdown header.
    """
    if name is None:
        try:
            name = os.path.basename(os.environ["JPY_SESSION_NAME"]).split(".")[0]
        except:
            name = ""

    with open(os.path.join(PARENT_DIR, "assets/markdown/header.md")) as f:
        header = f.read()
        header = header.replace(
            "@TIMESTAMP@",
            arrow.utcnow().strftime("%d/%m/%Y %H:%M %Z"),
        )
        header = header.replace("@NOTEBOOK@", name)
        return Markdown(header)


def footer(timestamp: arrow.Arrow = None):
    """
    Generates a Markdown footer with a timestamp.

    Args:
        timestamp (arrow.Arrow, optional): The timestamp to use. If None, uses the current time. Defaults to None.

    Returns:
        Markdown: The generated Markdown footer.
    """
    with open(os.path.join(PARENT_DIR, "assets/markdown/footer.md")) as f:
        footer = f.read()
        now = arrow.utcnow()
        footer = footer.replace("@TIMESTAMP@", now.strftime("%d/%m/%Y %H:%M %Z"))

        return Markdown(footer)


# ================== #
# API Query Wrappers #
# ================== #


# Query builder
def card_query(default: str = None, *args, **kwargs):
    """
    Builds a string of arguments to be passed to the yugipedia Wiki API for a card search query.

    Args:
        default (str, optional): The default card type to build a query string for. Can be one of {'spell', 'trap', 'st', 'monster', 'skill', 'counter', 'speed', 'rush', None}. Defaults to None.
        *args: Additional positional arguments to be passed to the API.
        **kwargs: Additional keyword arguments to be passed to the API.

    Raises:
        ValueError: If default is not a valid card type.

    Returns:
        str: A string containing the arguments to be passed to the API for the card search query.
    """
    # Default card query
    prop_bool = {
        "password": True,
        "card_type": True,
        "property": True,
        "primary": True,
        "secondary": True,
        "attribute": True,
        "monster_type": True,
        "stars": True,
        "atk": True,
        "def": True,
        "scale": True,
        "link": True,
        "arrows": True,
        "effect_type": True,
        "archseries": True,
        "alternate_artwork": True,
        "edited_artwork": True,
        "tcg": True,
        "ocg": True,
        "date": True,
    }

    if default is not None:
        default = default.lower()
    valid_default = {
        "spell",
        "trap",
        "st",
        "monster",
        "skill",
        "counter",
        "speed",
        "rush",
        None,
    }
    if default not in valid_default:
        raise ValueError("results: default must be one of %r." % valid_default)
    elif default == "monster":
        prop_bool.update({"property": False})
    elif default == "st" or default == "trap" or default == "spell":
        prop_bool.update(
            {
                "primary": False,
                "secondary": False,
                "attribute": False,
                "monster_type": False,
                "stars": False,
                "atk": False,
                "def": False,
                "scale": False,
                "link": False,
                "arrows": False,
            }
        )
    elif default == "counter":
        prop_bool.update(
            {
                "primary": False,
                "secondary": False,
                "attribute": False,
                "monster_type": False,
                "property": False,
                "stars": False,
                "atk": False,
                "def": False,
                "scale": False,
                "link": False,
                "arrows": False,
            }
        )
    elif default == "skill":
        prop_bool.update(
            {
                "password": False,
                "primary": False,
                "secondary": False,
                "attribute": False,
                "monster_type": False,
                "stars": False,
                "atk": False,
                "def": False,
                "scale": False,
                "link": False,
                "arrows": False,
                "effect_type": False,
                "edited_artwork": False,
                "alternate_artwork": False,
                "ocg": False,
                "speed": True,
                "character": True,
            }
        )
    elif default == "speed":
        prop_bool.update(
            {
                "speed": True,
                "scale": False,
                "link": False,
                "arrows": False,
            }
        )
    elif default == "rush":
        prop_bool.update(
            {
                "password": False,
                "secondary": False,
                "scale": False,
                "link": False,
                "arrows": False,
                "tcg": False,
                "ocg": False,
                "maximum_atk": True,
                "edited_artwork": False,
                "alternate_artwork": False,
                "rush_alt_artwork": True,
                "rush_edited_artwork": True,
                "misc": True,
            }
        )

    # Card properties dictionary
    prop_dict = {
        "password": "|?Password",
        "card_type": "|?Card%20type",
        "property": "|?Property",
        "primary": "|?Primary%20type",
        "secondary": "|?Secondary%20type",
        "attribute": "|?Attribute",
        "monster_type": "|?Type=Monster%20type",
        "stars": "|?Stars%20string=Level%2FRank%20",
        "atk": "|?ATK%20string=ATK",
        "def": "|?DEF%20string=DEF",
        "scale": "|?Pendulum%20Scale",
        "link": "|?Link%20Rating=Link",
        "arrows": "|?Link%20Arrows",
        "effect_type": "|?Effect%20type",
        "archseries": "|?Archseries",
        "alternate_artwork": "|?Category:OCG/TCG%20cards%20with%20alternate%20artworks",
        "edited_artwork": "|?Category:OCG/TCG%20cards%20with%20edited%20artworks",
        "tcg": "|?TCG%20status",
        "ocg": "|?OCG%20status",
        "date": "|?Modification%20date",
        "image_URL": "|?Card%20image",
        "misc": "|?Misc",
        "summoning": "|?Summoning",
        # Speed duel specific
        "speed": "|?TCG%20Speed%20Duel%20status",
        "character": "|?Character",
        # Rush duel specific
        "rush_alt_artwork": "|?Category:Rush%20Duel%20cards%20with%20alternate%20artworks",
        "rush_edited_artwork": "|?Category:Rush%20Duel%20cards%20with%20edited%20artworks",
        "maximum_atk": "|?MAXIMUM%20ATK",
        # Deprecated - Use for debuging
        "category": "|?category",
    }
    # Change default values to kwargs values
    prop_bool.update(kwargs)
    # Initialize string
    search_string = "|?English%20name=Name"
    # Iterate default plus kwargs items
    for arg, value in prop_bool.items():
        # If property is true
        if value:
            # If property in the dictionary, get its value
            if arg in prop_dict.keys():
                search_string += f"{prop_dict[arg]}"
            # If property is not in the dictionary, assume generic property
            else:
                print(f"Unrecognized property {arg}. Assuming |?{up.quote(arg)}.")
                search_string += f"|?{up.quote(arg)}"

    for arg in args:
        search_string += f"|?{up.quote(arg)}"

    return search_string


# Rarities dictionary
def fetch_rarities_dict(rarities_list: List[str] = []):
    """
    Fetches backlinks and redirects for a list of rarities, including abbreviations, to generate a map of rarity abbreviations to their corresponding names.

    Args:
        rarities_list (List[str]): A list of rarities.

    Returns:
        Dict[str, str]: A dictionary mapping rarity abbreviations to their corresponding names.

    """
    words, acronyms = separate_words_and_acronyms(rarities_list)
    if len(rarities_list) > 0:
        print(f"Words: {words}")
        print(f"Acronyms: {acronyms}")

    titles = api.fetch_categorymembers(category="Rarities", namespace=0)["title"]
    words = words + titles.tolist()
    rarity_backlinks = api.fetch_backlinks(words)
    rarity_redirects = api.fetch_redirects(acronyms)
    rarity_dict = rarity_backlinks | rarity_redirects

    return rarity_dict


# Bandai
def fetch_bandai(limit: int = 200, *args, **kwargs):
    """
    Fetch Bandai cards.

    Args:
        limit (int, optional): An integer that represents the maximum number of results to fetch. Defaults to 200.
        *args: Additional properties to query.
        **kwargs: keyword arguments to disable specific properties from query. Remaining keword arguments are passed to fetch_properties()

    Returns:
        pandas.DataFrame: A pandas DataFrame object containing the properties of the fetched Bandai cards.
    """
    debug = kwargs.get("debug", False)
    bandai_query = "|?English%20name=Name"
    bandai_prop_dict = {
        "card_type": "|?Card%20type",
        "level": "|?Level",
        "atk": "|?ATK",
        "def": "|?DEF",
        "number": "|?Bandai%20number=Card%20number",
        "type": "|?Type=Monster%20type",
        "rule": "|?Bandai%20rule=Rule",
        "sets": "|?Sets=Set",
        "rarity": "|?Rarity",
        "ability": "|?Ability",
        "date": "|?Modification%20date",
    }
    for key, value in bandai_prop_dict.items():
        if key in kwargs:
            disable = not kwargs.pop(key)
            if disable:
                continue
        bandai_query += value

    for arg in args:
        bandai_query += f"|?{up.quote(arg)}"

    print(f"Downloading bandai cards")
    concept = "[[Medium::Bandai]]"
    bandai_df = api.fetch_properties(
        concept, bandai_query, step=limit, limit=limit, **kwargs
    )
    if "Monster type" in bandai_df:
        bandai_df["Monster type"] = (
            bandai_df["Monster type"].dropna().apply(lambda x: x.split("(")[0])
        )  # Temporary
    if debug:
        print("- Total")

    print(f"{len(bandai_df.index)} results\n")

    time.sleep(0.5)

    return bandai_df


# Cards
def fetch_st(
    st_query: str = None,
    st: str = "both",
    cg: CG = CG.ALL,
    step: int = 500,
    limit: int = 5000,
    **kwargs,
):
    """
    Fetch spell or trap cards based on query and properties of the cards.

    Args:
        st_query (str, optional): A string representing a SMW query to search for. Defaults to None.
        st (str, optional): A string representing the type of cards to fetch, either "spell", "trap", "both", or "all". Defaults to "both".
        cg (CG, optional): An Enum that represents the card game to fetch cards from. Defaults to CG.ALL.
        step (int, optional): An integer that represents the number of results to fetch at a time. Defaults to 500.
        limit (int, optional): An integer that represents the maximum number of results to fetch. Defaults to 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pandas.DataFrame: A pandas DataFrame object containing the properties of the fetched spell/trap cards.

    Raises:
        ValueError: Raised if the "st" argument is not one of "spell", "trap", "both", or "all".
    """
    debug = kwargs.get("debug", False)
    st = st.capitalize()
    valid_st = {"Spell", "Trap", "Both", "All"}
    valid_cg = cg.value
    concept = f"[[Concept:CG%20non-monster%20cards]]"
    if st not in valid_st:
        raise ValueError("results: st must be one of %r." % valid_st)
    elif st == "Both" or st == "All":
        st = "Spells and Trap"
    else:
        concept += f"[[Card type::{st} card]]"
    if valid_cg != "CG":
        concept += f"[[Medium::{valid_cg}]]"
    print(f"Downloading {st}s")
    if st_query is None:
        st_query = card_query(default="st")

    st_df = api.fetch_properties(concept, st_query, step=step, limit=limit, **kwargs)

    if debug:
        print("- Total")

    print(f"{len(st_df.index)} results\n")

    return st_df


def fetch_monster(
    monster_query: str = None,
    cg: CG = CG.ALL,
    step: int = 500,
    limit: int = 5000,
    exclude_token=True,
    **kwargs,
):
    """
    Fetch monster cards based on query and properties of the cards.

    Args:
        monster_query (str, optional): A string representing a SMW query to search for. Defaults to None.
        cg (CG, optional): An Enum that represents the card game to fetch cards from. Defaults to CG.ALL.
        step (int, optional): An integer that represents the number of results to fetch at a time. Defaults to 500.
        limit (int, optional): An integer that represents the maximum number of results to fetch. Defaults to 5000.
        exclude_token (bool, optional): A boolean that determines whether to exclude Monster Tokens or not. Defaults to True.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pandas.DataFrame: A pandas DataFrame object containing the properties of the fetched monster cards.
    """
    debug = kwargs.get("debug", False)
    valid_cg = cg.value
    attributes = ["DIVINE", "LIGHT", "DARK", "WATER", "EARTH", "FIRE", "WIND", "?"]
    print("Downloading monsters")
    if monster_query is None:
        monster_query = card_query(default="monster")

    monster_df = pd.DataFrame()
    iterator = tqdm(
        attributes,
        leave=False,
        unit="attribute",
        disable=("PM_IN_EXECUTION" in os.environ),
    )
    for att in iterator:
        iterator.set_description(att)
        if debug:
            tqdm.write(f"- {att}")

        concept = f"[[Concept:CG%20monsters]][[Attribute::{att}]]"

        if valid_cg != "CG":
            concept += f"[[Medium::{valid_cg}]]"

        temp_df = api.fetch_properties(
            concept, monster_query, step=step, limit=limit, iterator=iterator, **kwargs
        )
        monster_df = pd.concat([monster_df, temp_df], ignore_index=True, axis=0)

    if exclude_token and "Primary type" in monster_df:
        monster_df = monster_df[
            monster_df["Primary type"] != "Monster Token"
        ].reset_index(drop=True)

    if debug:
        print("- Total")

    print(f"{len(monster_df.index)} results\n")

    return monster_df


# Non deck cards


def fetch_token(
    token_query: str = None, cg=CG.ALL, step: int = 500, limit: int = 5000, **kwargs
):
    """
    Fetch token cards based on query and properties of the cards.

    Args:
        token_query (str, optional): A string representing a SWM query to search for. Defaults to None.
        step (int, optional): An integer that represents the number of results to fetch at a time. Defaults to 500.
        limit (int, optional): An integer that represents the maximum number of results to fetch. Defaults to 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pandas.DataFrame: A pandas DataFrame object containing the properties of the fetched token cards.

    """
    valid_cg = cg.value
    print("Downloading tokens")

    concept = f"[[Category:Tokens]]"
    if valid_cg != "CG":
        concept += f"[[Category:{valid_cg}%20cards]]"
    else:
        concept += "[[Category:TCG%20cards||OCG%20cards]]"

    if token_query is None:
        token_query = card_query(default="monster")

    token_df = api.fetch_properties(
        concept, token_query, step=step, limit=limit, **kwargs
    )

    print(f"{len(token_df.index)} results\n")

    return token_df


def fetch_counter(
    counter_query: str = None, cg=CG.ALL, step: int = 500, limit: int = 5000, **kwargs
):
    """
    Fetch counter cards based on query and properties of the cards.

    Args:
        counter_query (str, optional): A string representing a SMW query to search for. Defaults to None.
        step (int, optional): An integer that represents the number of results to fetch at a time. Defaults to 500.
        limit (int, optional): An integer that represents the maximum number of results to fetch. Defaults to 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pandas.DataFrame: A pandas DataFrame object containing the properties of the fetched counter cards.
    """
    valid_cg = cg.value
    print("Downloading counters")

    concept = f"[[Category:Counters]][[Page%20type::Card%20page]]"
    if valid_cg != "CG":
        concept += f"[[Medium::{valid_cg}]]"

    if counter_query is None:
        counter_query = card_query(default="counter")

    counter_df = api.fetch_properties(
        concept, counter_query, step=step, limit=limit, **kwargs
    )

    print(f"{len(counter_df.index)} results\n")

    return counter_df


# Alternative formats


def fetch_speed(speed_query: str = None, step: int = 500, limit: int = 5000, **kwargs):
    """
    Fetches TCG Speed Duel cards from the yugipedia Wiki API.

    Args:
        speed_query (str):  A string representing a SMW query to search for. Defaults to None.
        step (int): The number of results to fetch in each API call. Defaults to 500.
        limit (int): The maximum number of results to fetch. Defaults to 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        A pandas DataFrame containing the fetched TCG Speed Duel cards.
    """
    debug = kwargs.get("debug", False)

    print(f"Downloading Speed duel cards")
    concept = "[[Category:TCG Speed Duel cards]]"
    if speed_query is None:
        speed_query = card_query(default="speed")

    speed_df = api.fetch_properties(
        concept,
        speed_query,
        step=step,
        limit=limit,
        **kwargs,
    )

    if debug:
        print("- Total")

    print(f"{len(speed_df.index)} results\n")

    return speed_df


def fetch_skill(skill_query: str = None, step: int = 500, limit: int = 5000, **kwargs):
    """
    Fetches skill cards from the yugipedia Wiki API.

    Args:
        skill_query (str): A string representing a SMW query to search for. Defaults to None.
        step (int): The number of results to fetch in each API call. Defaults to 500.
        limit (int): The maximum number of results to fetch. Defaults to 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        A pandas DataFrame containing the fetched skill cards.
    """
    print("Downloading skill cards")

    concept = "[[Category:Skill%20Cards]][[Card type::Skill Card]]"
    if skill_query is None:
        skill_query = card_query(default="skill")

    skill_df = api.fetch_properties(
        concept, skill_query, step=step, limit=limit, **kwargs
    )

    print(f"{len(skill_df.index)} results\n")

    return skill_df


def fetch_rush(rush_query: str = None, step: int = 500, limit: int = 5000, **kwargs):
    """
    Fetches Rush Duel cards from the Yu-Gi-Oh! Wikia API.

    Args:
        rush_query (str): A search query to filter the results. If not provided, it defaults to "rush".
        step (int): The number of results to fetch in each API call. Defaults to 500.
        limit (int): The maximum number of results to fetch. Defaults to 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        A pandas DataFrame containing the fetched Rush Duel cards.
    """
    debug = kwargs.get("debug", False)
    print("Downloading Rush Duel cards")

    concept = f"[[Category:Rush%20Duel%20cards]][[Medium::Rush%20Duel]]"
    if rush_query is None:
        rush_query = card_query(default="rush")

    rush_df = api.fetch_properties(
        concept, rush_query, step=step, limit=limit, **kwargs
    )

    print(f"{len(rush_df.index)} results\n")

    return rush_df


# Unusable cards


def fetch_unusable(
    query: str = None,
    cg: CG = CG.ALL,
    filter=True,
    step: int = 500,
    limit: int = 5000,
    **kwargs,
):
    """
    Fetch unusable cards based on query and properties of the cards. Unusable cards include "Strategy cards", "Tip cards",
    "Card Checklists", etc, which are not actual cards. The filter option enables filtering those out and keeping only cards
    such as Duelist Kingdom "Ticket cards", old video-game promo "Character cards" and "Non-game cards" which have the layout
    of a real card, such as "Everyone's King". This criteria is not free of ambiguity.

    Args:
        query (str, optional): A string representing a SMW query to search for. Defaults to None.
        cg (CG, optional): An Enum that represents the card game to fetch cards from. Defaults to CG.ALL.
        filter (bool, optional): Keep only "Character Cards", "Non-game cards" and "Ticket Cards".
        step (int, optional): An integer that represents the number of results to fetch at a time. Defaults to 500.
        limit (int, optional): An integer that represents the maximum number of results to fetch. Defaults to 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pandas.DataFrame: A pandas DataFrame object containing the properties of the fetched spell/trap cards.

    """
    debug = kwargs.get("debug", False)
    concept = "[[Category:Unusable cards]]"

    valid_cg = cg.value
    if valid_cg == "CG":
        concept = "OR".join([concept + f"[[{s} status::+]]" for s in ["TCG", "OCG"]])
    else:
        concept += f"[[{valid_cg} status::+]]"

    concept = up.quote(concept)

    print(f"Downloading unusable cards")
    if query is None:
        query = card_query()

    unusable_df = api.fetch_properties(concept, query, step=step, limit=limit, **kwargs)

    if filter and "Card type" in unusable_df:
        unusable_df = unusable_df[
            unusable_df["Card type"].isin(
                ["Character Card", "Non-game card", "Ticket Card"]
            )
        ].reset_index(drop=True)

    unusable_df.dropna(how="all", axis=1, inplace=True)

    if debug:
        print("- Total")

    print(f"{len(unusable_df.index)} results\n")

    return unusable_df


# Extra properties


def fetch_errata(errata: str = "all", step: int = 500, **kwargs):
    """
    Fetches errata information from the yuipedia Wiki API.

    Args:
        errata (str): The type of errata information to fetch. Valid values are 'name', 'type', and 'all'. Defaults to 'all'.
        step (int): The number of results to fetch in each API call. Defaults to 500.
        **kwargs: Additional keyword arguments to pass to fetch_categorymembers.

    Returns:
        A pandas DataFrame containing a boolean table indicating whether each card has errata information for the specified type.
    """
    debug = kwargs.get("debug", False)
    errata = errata.lower()
    valid = {"name", "type", "all"}
    categories = {
        "all": "Card Errata",
        "type": "Cards with card type errata",
        "name": "Cards with name errata",
    }
    if errata not in valid:
        raise ValueError("results: errata must be one of %r." % valid)
    elif errata == "all":
        errata = "all"
        categories = list(categories.values())
    else:
        categories = list(categories["errata"])

    print(f"Downloading {errata} errata")
    errata_df = pd.DataFrame(dtype=bool)
    iterator = tqdm(
        categories,
        leave=False,
        unit="initial",
        disable=("PM_IN_EXECUTION" in os.environ),
    )
    for cat in iterator:
        desc = cat.split("Category:")[-1]
        iterator.set_description(desc)
        if debug:
            tqdm.write(f"- {cat}")

        temp = api.fetch_categorymembers(
            cat, namespace=3010, step=step, iterator=iterator, debug=debug
        )
        errata_data = temp["title"].apply(lambda x: x.split("Card Errata:")[-1])
        errata_series = pd.Series(data=True, index=errata_data, name=desc)
        errata_df = (
            pd.concat([errata_df, errata_series], axis=1)
            .astype("boolean")
            .fillna(False)
            .sort_index()
        )

    if debug:
        print("- Total")

    print(f"{len(errata_df.index)} results\n")
    return errata_df


# Sets


def fetch_set_list_pages(cg: CG = CG.ALL, step: int = 500, limit=5000, **kwargs):
    """
    Fetches a list of 'Set Card Lists' pages from the yugipedia Wiki API.

    Args:
        cg (CG): A member of the CG enum representing the card game for which set lists are being fetched.
        step (int): The number of pages to fetch in each API request.
        limit (int): The maximum number of pages to fetch.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pd.DataFrame: A DataFrame containing the titles of the set list pages.

    """
    debug = kwargs.get("debug", False)
    valid_cg = cg.value
    if valid_cg == "CG":
        category = ["TCG Set Card Lists", "OCG Set Card Lists"]
    else:
        category = f"{valid_cg}%20Set%20Card%20Lists"

    print("Downloading list of 'Set Card Lists' pages")
    set_list_pages = pd.DataFrame()
    result = pd.DataFrame()
    iterator = tqdm(
        category,
        leave=False,
        unit="category",
        disable=("PM_IN_EXECUTION" in os.environ),
    )
    for cat in iterator:
        iterator.set_description(cat.split("Category:")[-1])
        temp = api.fetch_categorymembers(
            cat, namespace=None, step=step, iterator=iterator, debug=debug
        )
        sub_categories = pd.DataFrame(temp)["title"]
        sub_iterator = tqdm(
            sub_categories,
            leave=False,
            unit="subcategory",
            disable=("PM_IN_EXECUTION" in os.environ),
        )
        for sub_cat in sub_iterator:
            sub_iterator.set_description(sub_cat.split("Category:")[-1])
            temp = api.fetch_properties(
                f"[[{sub_cat}]]",
                query="|?Modification date",
                step=limit,
                limit=limit,
                iterator=sub_iterator,
                **kwargs,
            )
            set_list_pages = pd.concat([set_list_pages, pd.DataFrame(temp)])

    return set_list_pages


def fetch_all_set_lists(cg: CG = CG.ALL, step: int = 40, **kwargs):
    """
    Fetches all set lists for a given card game.

    Args:
        cg (CG, optional): The card game to fetch set lists for. Defaults to CG.ALL.
        step (int, optional): The number of sets to fetch at once. Defaults to 50.
        **kwargs: Additional keyword arguments to pass to fetch_set_list_pages and fetch_set_lists.

    Returns:
        pd.DataFrame: A DataFrame containing all set lists for the specified card game.

    Raises:
        Any exceptions raised by fetch_set_list_pages() or fetch_set_lists().
    """
    debug = kwargs.get("debug", False)
    sets = fetch_set_list_pages(cg, **kwargs)  # Get list of sets
    keys = sets["Page name"]

    all_set_lists_df = pd.DataFrame(
        columns=["Set", "Card number", "Name", "Rarity", "Print", "Quantity", "Region"]
    )
    total_success = 0
    total_error = 0

    for i in trange(np.ceil(len(keys) / step).astype(int), leave=False):
        success = 0
        error = 0
        if debug:
            tqdm.write(f"Iteration {i}:")

        first = i * step
        last = (i + 1) * step

        set_lists_df, success, error = api.fetch_set_lists(keys[first:last], **kwargs)
        set_lists_df = set_lists_df.merge(sets, on="Page name", how="left").drop(
            "Page name", axis=1
        )
        all_set_lists_df = pd.concat(
            [all_set_lists_df, set_lists_df], ignore_index=True
        )
        total_success += success
        total_error += error

    all_set_lists_df = all_set_lists_df.convert_dtypes()
    all_set_lists_df.sort_values(by=["Set", "Region", "Card number"]).reset_index(
        inplace=True
    )
    print(
        f'{"Total: " if debug else ""}{total_success} set lists received - {total_error} missing'
    )

    return all_set_lists_df


# ======================= #
# Complete execution flow #
# ======================= #


def run_notebooks(
    reports: Union[str, List[str]] = "all",
    progress_handler: Callable = None,
    telegram_first: bool = False,
    suppress_contribs: bool = False,
    **kwargs,
):
    """
    Execute specified Jupyter notebooks in the source directory using Papermill.

    Args:
        reports (Union[str, List[str]]): List of notebooks to execute or 'all' to execute all notebooks in the source directory. Default is 'all'.
        progress_handler (callable): An optional callable to provide progress bar functionality. Default is None.
        telegram_first (bool, optional): Default is False.
        suppress_contribs (bool, optional): Default is False.
        **kwargs: Additional keyword arguments containing secrets key-value pairs to pass to TQDM contrib iterators.

    Returns:
        None
    """
    debug = kwargs.pop("debug", False)

    if reports == "all":
        # Get reports
        reports = sorted(glob.glob("*.ipynb"))
    else:
        reports = [str(reports)] if not isinstance(reports, list) else reports

    if progress_handler:
        external_pbar = progress_handler.pbar(
            iterable=reports, desc="Completion", unit="report", unit_scale=True
        )
    else:
        external_pbar = None

    # Initialize iterators
    iterator = tqdm(
        reports,
        desc="Completion",
        unit="report",
        unit_scale=True,
        dynamic_ncols=True,
    )

    if not suppress_contribs:
        contribs = ["DISCORD", "TELEGRAM"]
        if telegram_first:
            contribs = contribs[::-1]
        secrets = {
            key: value
            for key, value in kwargs.items()
            if (value is not None) and ("TOKEN" in key) or ("CHANNEL_ID") in key
        }
        secrets_file = os.path.join(PARENT_DIR, "assets/secrets.env")
        for contrib in contribs:
            required_secrets = [
                f"{contrib}_" + key if key == "CHANNEL_ID" else key
                for key in [f"{contrib}_TOKEN", f"CHANNEL_ID"]
                if key not in secrets
            ]
            try:
                loaded_secrets = load_secrets(
                    required_secrets, secrets_file=secrets_file, required=True
                )
                secrets = secrets | loaded_secrets

                token = secrets.get(f"{contrib}_TOKEN")
                channel_id = secrets.get(
                    f"{contrib}_CHANNEL_ID", secrets.get("CHANNEL_ID")
                )
                if contrib == "DISCORD":
                    from tqdm.contrib.discord import tqdm as contrib_tqdm

                    channel_id_dict = {"channel_id": channel_id}

                elif contrib == "TELEGRAM":
                    from tqdm.contrib.telegram import tqdm as contrib_tqdm

                    channel_id_dict = {"chat_id": channel_id}

                iterator = contrib_tqdm(
                    reports,
                    desc="Completion",
                    unit="report",
                    unit_scale=True,
                    dynamic_ncols=True,
                    token=token,
                    **channel_id_dict,  # Needed to handle Telegram using chat_ID instaed of channel_ID.
                )

                break
            except:
                pass

    # Create the main logger
    logger = logging.getLogger("papermill")
    logger.setLevel(logging.INFO)

    # Create a StreamHandler and attach it to the logger
    stream_handler = logging.StreamHandler(io.StringIO())
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    stream_handler.addFilter(
        lambda record: record.getMessage().startswith("Ending Cell")
    )
    logger.addHandler(stream_handler)

    # Define a function to update the output variable
    def update_pbar():
        iterator.update((1 / cells))
        if external_pbar:
            external_pbar.update((1 / cells))

    exceptions = []
    for i, report in enumerate(iterator):
        if not report.endswith(".ipynb"):
            report += ".ipynb"

        iterator.n = i
        iterator.last_print_n = i
        iterator.refresh()
        report_name = os.path.basename(report)[:-6]

        with open(report) as f:
            nb = nbformat.read(f, nbformat.NO_CONVERT)
            cells = len(nb.cells)
            # print(f'Number of Cells: {cells}')

        # Attach the update_pbar function to the stream_handler
        stream_handler.flush = update_pbar

        # Update postfix
        tqdm.write(f"Generating {report_name} report")
        iterator.set_postfix(report=report_name)
        if external_pbar:
            external_pbar.set_postfix(report=report_name)

        # execute the notebook with papermill
        os.environ["PM_IN_EXECUTION"] = "True"
        if "yugiquery" in jupyter_client.kernelspec.find_kernel_specs():
            kernel_name = "yugiquery"
        else:
            kernel_name = "python3"

        try:
            pm.execute_notebook(
                report,
                report,
                log_output=True,
                progress_bar=True,
                kernel_name=kernel_name,
            )
        except pm.PapermillExecutionError as e:
            exceptions.append(e)
        finally:
            os.environ.pop("PM_IN_EXECUTION", None)

    # Close the iterator
    iterator.close()
    if external_pbar:
        external_pbar.close()

    # Close the stream_handler
    stream_handler.close()
    # Clear custom handler
    logger.handlers.clear()

    if exceptions:
        combined_message = "\n".join(str(e) for e in exceptions)
        raise Exception(combined_message)


def run(
    reports: Union[str, List[str]] = "all",
    progress_handler=None,
    telegram_first: bool = False,
    suppress_contribs: bool = False,
    cleanup: Union[bool, str] = False,
    dry_run: bool = False,
    **kwargs,
):
    """
    Executes all notebooks in the source directory that match the specified report, updates the page index
    to reflect the last execution timestamp, and clean up redundant data files.

    Args:
        reports (str, optional): The report to generate. Defaults to 'all'.
        progress_handler (function, optional): A progress handler function to report execution progress. Defaults to None.
        telegram_first (bool, optional): Defaults to False.
        suppress_contribs (bool, optional): Defaults to False.
        cleanup (Union[bool,str], optional): whether to cleanup data files after execution. If True, perform cleanup, if False, doesn't perform cleanup. If 'auto', performs cleanup if there are more than 4 data files for each report (assuming one per week). Defaults to 'auto'.
        dry_run (bool, optional): dry_run flag to pass to cleanup_data method call. Defaults to False.
        **kwargs: Additional keyword arguments to pass to run_notebook.

    Returns:
        None: This function does not return a value.
    """
    # Check API status
    if not api.check_API_status():
        if progress_handler:
            progress_handler.exit(API_status=False)
        return

    # Execute all notebooks in the source directory
    try:
        run_notebooks(
            reports=reports,
            progress_handler=progress_handler,
            telegram_first=telegram_first,
            suppress_contribs=suppress_contribs,
            **kwargs,
        )
    except Exception as e:
        raise e
    finally:
        # Update page index to reflect last execution timestamp
        update_index()

    # Cleanup redundant data files
    if cleanup == "auto":
        data_files_count = len(glob.glob(os.path.join(PARENT_DIR, "data/*.bz2")))
        reports_count = len(glob.glob(os.path.join(SCRIPT_DIR, "*.ipynb")))
        if data_files_count / reports_count > 10:
            cleanup_data(dry_run=dry_run)
    elif cleanup:
        cleanup_data(dry_run=dry_run)


# ========= #
# CLI usage #
# ========= #


def auto_or_bool(value):
    if value is None:
        return True
    elif value.lower() == "auto":
        return "auto"
    else:
        return bool(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--reports",
        nargs="+",
        dest="reports",
        default="all",
        type=str,
        required=False,
        help="The report(s) to be generated.",
    )
    parser.add_argument(
        "-t",
        "--telegram-token",
        dest="telegram_token",
        type=str,
        required=False,
        help="Telegram API token.",
    )
    parser.add_argument(
        "-d",
        "--discord-token",
        dest="discord_token",
        type=str,
        required=False,
        help="Discord API token.",
    )
    parser.add_argument(
        "-c",
        "--channel",
        dest="channel_id",
        type=int,
        required=False,
        help="Discord or Telegram Channel/chat ID.",
    )
    parser.add_argument(
        "-s",
        "--suppress-contribs",
        action="store_true",
        required=False,
        help="Disables using TQDM contribs entirely.",
    )
    parser.add_argument(
        "-f",
        "--telegram-first",
        action="store_true",
        required=False,
        help="Force TQDM to try using Telegram as progress bar before Discord.",
    )
    parser.add_argument(
        "--cleanup",
        default="auto",
        type=auto_or_bool,
        nargs="?",
        const=True,
        action="store",
        help="Wether to run the cleanup routine. Options are True, False and 'auto'. Defaults to auto.",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        required=False,
        help="Whether to dry run the cleanup routine. No effect if cleanup is False.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        required=False,
        help="Enables debug flag.",
    )
    args = vars(parser.parse_args())
    # Assures the script is within a git repository before proceesing
    git.assure_repo()
    # Change working directory to script location
    os.chdir(SCRIPT_DIR)
    # Execute the complete workflow
    run(**args)
    # Exit python
    quit()
