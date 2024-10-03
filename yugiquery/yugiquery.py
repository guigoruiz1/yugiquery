#!/usr/bin/env python3

# yugiquery/yugiquery.py

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

# ======= #
# Imports #
# ======= #

# Standard library packages
import argparse
import io
import json
import logging
import os
import re
import time
import urllib.parse as up
import warnings
from ast import literal_eval
from enum import Enum
from pathlib import Path
from typing import (
    Dict,
    List,
    Literal,
    Tuple,
)

# Third-party imports
import arrow
from ipylab import JupyterFrontEnd
from IPython import get_ipython
from IPython.display import Markdown, display
import jupyter_client
import nbformat
from nbconvert import HTMLExporter
from nbconvert.writers import FilesWriter
import numpy as np
import pandas as pd
import papermill as pm
from termcolor import cprint
from tqdm.auto import tqdm, trange
from traitlets.config import Config

# Local application imports
if __package__:
    from .utils import *
else:
    from utils import *

# Overwrite packages with versions specific for jupyter notebook
if dirs.is_notebook:
    from itables import init_notebook_mode

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


#: A dictionary mapping card types to their corresponding properties to query.
card_properties = {
    "monster": [
        "password",
        "card_type",
        "primary",
        "secondary",
        "attribute",
        "monster_type",
        "stars",
        "atk",
        "def",
        "scale",
        "link",
        "arrows",
        "effect_type",
        "archseries",
        "alternate_artwork",
        "edited_artwork",
        "tcg",
        "ocg",
        "date",
    ],
    "st": [
        "password",
        "card_type",
        "property",
        "effect_type",
        "archseries",
        "alternate_artwork",
        "edited_artwork",
        "tcg",
        "ocg",
        "date",
    ],
    "counter": [
        "password",
        "card_type",
        "effect_type",
        "archseries",
        "alternate_artwork",
        "edited_artwork",
        "tcg",
        "ocg",
        "date",
    ],
    "skill": ["card_type", "property", "archseries", "tcg", "date", "speed", "character"],
    "speed": [
        "password",
        "card_type",
        "property",
        "primary",
        "secondary",
        "attribute",
        "monster_type",
        "stars",
        "atk",
        "def",
        "effect_type",
        "archseries",
        "alternate_artwork",
        "edited_artwork",
        "tcg",
        "ocg",
        "date",
        "speed",
    ],
    "rush": [
        "card_type",
        "property",
        "primary",
        "attribute",
        "monster_type",
        "stars",
        "atk",
        "def",
        "effect_type",
        "archseries",
        "date",
        "rush_alt_artwork",
        "rush_edited_artwork",
        "maximum_atk",
        "misc",
    ],
    "bandai": ["card_type", "level", "atk", "def", "number", "monster_type", "rule", "sets", "rarity", "ability", "date"],
}

# =============== #
# Data management #
# =============== #


def generate_changelog(previous_df: pd.DataFrame, current_df: pd.DataFrame, col: str | List[str]) -> pd.DataFrame:
    """
    Generates a changelog DataFrame by comparing two DataFrames based on a specified column.

    Args:
        previous_df (pd.DataFrame): A DataFrame containing the previous version of the data.
        current_df (pd.DataFrame): A DataFrame containing the current version of the data.
        col (str | List[str]): The name of the column to compare the DataFrames on.

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
        print("No changes")

    return changelog


def benchmark(timestamp: arrow.Arrow, report: str | None = None) -> None:
    """
    Records the execution time of a report and saves the data to a JSON file.

    Args:
        timestamp (arrow.Arrow): The timestamp when the report execution began.
        report (str | None, optional): The name of the report being benchmarked. If None, tries obtaining report name from JPY_SESSION_NAME environment variable.

    Returns:
        None
    """
    if report is None:
        path = get_notebook_path()
        report = path.stem if path else "Unnamed"

    now = arrow.utcnow()
    timedelta = now - timestamp
    benchmark_file = dirs.DATA / "benchmark.json"
    data = load_json(benchmark_file)

    # Add the new data to the existing data
    if report not in data:
        data[report] = []
    data[report].append({"ts": now.isoformat(), "average": timedelta.total_seconds(), "weight": 1})
    # Save new data to file
    with open(benchmark_file, "w+") as file:
        json.dump(data, file, indent=4)

    result = git.commit(
        files=[benchmark_file],
        message=f"{report.capitalize()} report benchmarked - {now.isoformat()}",
    )
    print(result)


def condense_changelogs(files: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
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
            r"(\w+)_\w+_(\d{8}T\d{4})Z_(\d{8}T\d{4})Z.bz2",
            Path(file).name,
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
    index = new_changelog.drop(["Modification date", "Version"], axis=1).drop_duplicates(keep="last").index
    new_filename = Path(file).parent.joinpath(
        make_filename(
            report=changelog_name,
            timestamp=arrow.get(last_date),
            previous_timestamp=arrow.get(first_date),
        ),
    )
    return new_changelog.loc[index], new_filename


def condense_benchmark(benchmark: Dict[str, List[Dict[str, str | float]]]) -> Dict[str, List[Dict[str, str | float]]]:
    """
    Condenses a benchmark dictionary by calculating the weighted average and total weight for each key.

    Args:
        benchmark (Dict[str, List[Dict[str, str | float]]]): A dictionary containing benchmark data.

    Returns:
        Dict[str, List[Dict[str, str | float]]]: The condensed benchmark dictionary with updated entries.
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


def cleanup_data(dry_run=False) -> None:
    """
    Cleans up data files, keeping only the most recent file from each month and week.

    Args:
        dry_run (bool): If True, the function will only print the files that would be deleted without actually deleting them. Defaults to False.

    Returns:
        None
    """
    # Benchmark
    benchmark_file = dirs.DATA / "benchmark.json"
    if benchmark_file.is_file():
        benchmark = load_json(benchmark_file)
        new_benchmark = condense_benchmark(benchmark)
        if dry_run:
            print("Benchmark:", new_benchmark)
        else:
            with open(benchmark_file, "w+") as f:
                json.dump(new_benchmark, f, indent=4)

    # Data CSV files
    file_list = list(dirs.DATA.glob("*.bz2"))
    if not file_list:
        return

    # Create a DataFrame
    df = pd.DataFrame(file_list, columns=["Name"])

    # Convert the 'Date' column to a datetime type
    df["Date"] = pd.to_datetime(df["Name"].apply(os.path.getctime), unit="s")

    # Create a new column 'Group' based on the first two elements after splitting the filename
    df["Group"] = df["Name"].apply(lambda x: "_".join(Path(x).name.split("_", 2)[:2]))

    # Group the DataFrame by 'Group' and 'Date' (year and month)
    grouped = df.groupby(["Group", pd.Grouper(key="Date", freq="MS")])

    # Get a list of all the files created on the same month of the same year, separated by whether they contain "changelog"
    same_month_files = {
        "changelog": [group[1]["Name"].tolist() for group in grouped if "changelog" in group[0][0]],
        "data": [group[1]["Name"].tolist() for group in grouped if not "changelog" in group[0][0]],
    }

    # Get a list of all the files created in the last month and split them into weeks
    last_month_files = df[df["Date"] >= df["Date"].max() - pd.Timedelta("1MS")].resample("W", on="Date").first()

    # Separate the last_month_files by whether they contain "changelog"
    last_month_files = {
        "changelog": last_month_files[last_month_files["Group"].str.contains("changelog")]["Name"].tolist(),
        "data": last_month_files[~last_month_files["Group"].str.contains("changelog")]["Name"].tolist(),
    }

    # Remove the last_month_files from the same_month_files
    same_month_files["changelog"] = [
        files for files in same_month_files["changelog"] if files not in last_month_files["changelog"]
    ]
    same_month_files["data"] = [files for files in same_month_files["data"] if files not in last_month_files["data"]]

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
                dirs.DATA / "benchmark.json",
                dirs.DATA / "*bz2",
            ],
            message=f"Data cleanup {arrow.utcnow().isoformat()}",
        )
        print(result)


def load_latest_data(
    name_pattern: str,
    tuple_cols: List[str] = [
        "Secondary type",
        "Effect type",
        "Link Arrows",
        "Archseries",
        "Artwork",
        "Errata",
        "Rarity",
        "Cover card",
    ],
) -> Tuple[pd.DataFrame, arrow.Arrow] | Tuple[None, None]:
    """
    Loads the most recent data file matching the specified name pattern and applies corrections.

    Args:
        name_pattern (str): Data file name pattern to load.
        tuple_cols (List[str]): List of columns containing tuple values to apply literal_eval. Defaults to ["Secondary type", "Effect type", "Link Arrows", "Archseries", "Artwork", "Errata", "Rarity", "Cover card"].

    Returns:
        Tuple[pd.DataFrame, arrow.Arrow]: A tuple containing the loaded dataframe and the timestamp of the file.
    """
    name_pattern = name_pattern.lower()
    files = sorted(
        list(dirs.DATA.glob(f"{name_pattern}_data_*.bz2")),
        key=os.path.getctime,
        reverse=True,
    )

    if files:
        df = pd.read_csv(files[0], dtype=object)
        for col in tuple_cols:
            if col in df:
                try:
                    df[col] = df[col].dropna().apply(literal_eval)
                except:
                    pass

        for col in ["Modification date", "Release"]:
            if col in df:
                df[col] = pd.to_datetime(df[col])

        ts = arrow.get(Path(files[0]).stem.split("_")[-1])
        print(f"{name_pattern.capitalize()} file loaded")
        return df, ts
    else:
        print(f'No "{name_pattern}" files')
        return None, None


# Sets
def merge_set_info(input_df: pd.DataFrame, input_info_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges set information from an input set info DataFrame into an input set list DataFrame based on set and region.

    Args:
        input_df (pd.DataFrame): A pandas DataFrame containing set lists.
        input_info_df (pd.DataFrame): A pandas DataFrame containing set information.

    Returns:
        pd.DataFrame: A pandas DataFrame with set information merged into it.

    Raises:
        ValueError: If the input DataFrame does not contain the required "Set" and "Region" columns.
    """
    required_columns = ["Set", "Region"]
    if not all(col in input_df.columns for col in required_columns):
        raise ValueError('Input DataFrame must contain "Set" and "Region" columns.')

    regions_dict = load_json(dirs.get_asset("json", "regions.json"))

    def get_release_date(row):
        region = regions_dict.get(row["Region"], row["Region"])
        region_col = f"{region} release date"
        if row["Set"] in input_info_df.index and region_col in input_info_df.columns:
            return input_info_df.at[row["Set"], region_col]
        return np.nan

    input_df["Release"] = input_df.apply(get_release_date, axis=1)
    input_df["Release"] = pd.to_datetime(input_df["Release"].astype(str), errors="coerce")

    merged_df = input_df.merge(
        input_info_df.loc[:, :"Cover card"], left_on="Set", right_index=True, how="outer"
    ).reset_index(drop=True)

    print("Set properties merged")
    return merged_df


# Formatters
def format_artwork(row: pd.Series) -> Tuple[str]:
    """
    Formats a row of a dataframe that contains "alternate artworks" and "edited artworks" columns.
    If the "alternate artworks" column(s) in the row contain at least one "True" value, adds "Alternate" to the result tuple.
    If the "edited artworks" column(s) in the row contain at least one "True" value, adds "Edited" to the result tuple.
    Returns the result tuple.
    Args:

        row (pd.Series): A row of a dataframe that contains "alternate artworks" and "edited artworks" columns.

    Returns:
        Tuple[str]: The formatted row as a tuple.
    """
    result = tuple()
    index_str = row.index.str
    if index_str.endswith("alternate artworks").any():
        matching_cols = row.index[index_str.endswith("alternate artworks")]
        if row[matching_cols].any():
            result += ("Alternate",)
    if index_str.endswith("edited artworks").any():
        matching_cols = row.index[index_str.endswith("edited artworks")]
        if row[matching_cols].any():
            result += ("Edited",)
    if result == tuple():
        return np.nan
    else:
        return result


def format_errata(row: pd.Series) -> Tuple[str]:
    """
    Formats errata information from a pandas Series and returns a tuple of errata types.

    Args:
        row (pd.Series): A pandas Series containing errata information for a single card.

    Returns:
        Tuple[str]: Tuple of errata types if any errata information is present in the input Series, otherwise np.nan.
    """
    result = []
    if "Cards with name errata" in row:
        if row["Cards with name errata"]:
            result.append("Name")
    if "Cards with card type errata" in row:
        if row["Cards with card type errata"]:
            result.append("Type")
    if "Card Errata" in row and not result:
        if row["Card Errata"]:
            result.append("Any")
    if result:
        return tuple(sorted(result))
    else:
        return np.nan


def merge_errata(input_df: pd.DataFrame, input_errata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges errata information from an input errata DataFrame into an input DataFrame based on card names.

    Args:
        input_df (pd.DataFrame): A pandas DataFrame containing card information.
        input_errata_df (pd.DataFrame): A pandas DataFrame containing errata information.

    Returns:
        pd.DataFrame: A pandas DataFrame with errata information merged into it.
    """
    if "Name" in input_df.columns:
        errata_series = input_errata_df.apply(format_errata, axis=1).rename("Errata")
        input_df = input_df.merge(
            errata_series,
            left_on="Name",
            right_index=True,
            how="left",
            suffixes=("", " errata"),
        )
    else:
        print('Error! No "Name" column to join errata')

    return input_df


# User collection
def get_collection(file_name: str = "collection") -> None | pd.DataFrame:
    """
    Loads a user collection from a CSV or Excel file.

    Args:
        file_name (str, optional): The name of the collection file to load. Defaults to "collection".

    Returns:
        None | pd.DataFrame: The loaded collection DataFrame if the file is found, otherwise None.
    """
    collection_file = dirs.DATA.joinpath(file_name)
    if collection_file.with_suffix(".xlsx").is_file():
        collection_file = collection_file.with_suffix(".xlsx")
        collections = pd.read_excel(collection_file, sheet_name=None)
        collection_df = pd.concat(
            [df.assign(Collection=key) for key, df in collections.items() if key != "Instructions"], ignore_index=True
        )
        if collection_df["Collection"].nunique() == 1:
            collection_df.drop(["Collection"], axis=1)
    elif collection_file.with_suffix(".csv").is_file():
        collection_file = collection_file.with_suffix(".csv")
        collection_df = pd.read_csv(collection_file)
    else:
        print(f"No {file_name} file found.")
        return None

    print(f"Loaded {collection_file.name}")
    return collection_df


def find_cards(collections: List[pd.DataFrame] | pd.DataFrame, merge_data=False) -> pd.DataFrame:
    """
    Process a DataFrame with card names, numbers and/or passwords and return a DataFrame with the card names, quantities and, optionally, additional card data merged from database.

    Args:
        collections (List[pd.DataFrame] | pd.DataFrame): DataFrame or list of DataFrames with card names, numbers and/or passwords.

    Returns:
        ((List[pd.DataFrame] | pd.DataFrame): DataFrame or list of DataFrames with card names, quantities and, optionally, additional card data merged from database.
    """
    if not isinstance(collections, list):
        collections = [collections]

    card_df, _ = load_latest_data(name_pattern="cards")
    if any(
        ["Card number" in collection_df and not collection_df["Card number"].dropna().empty for collection_df in collections]
    ):
        set_lists_df, _ = load_latest_data(name_pattern="sets")

    for i, collection_df in enumerate(collections):
        collection_df = collection_df.dropna(how="all", axis=1).dropna(how="all", axis=0)
        collection_df = collection_df.assign(match=np.nan).astype(object)

        if "Card number" in collection_df and not collection_df["Card number"].dropna().empty:
            number_keys = collection_df["Card number"].dropna().str.upper().str.strip()

            list_keys = set_lists_df["Card number"].dropna().str.upper().str.strip()
            list_values = set_lists_df.loc[list_keys.index]["Name"]
            key_name_dict = dict(zip(list_keys, list_values))

            missing = number_keys[~number_keys.isin(list_keys)]
            if missing.count() > 0:
                print(
                    "\nUnable to find the following card(s) by number:\n ⏺",
                    "\n ⏺ ".join(missing.sort_values().unique()),
                )

            matches = number_keys.map(lambda x: key_name_dict.get(x, x))
            collection_df.loc[matches.index, "match"] = matches
            collection_df.drop("Card number", axis=1, inplace=True)

        if "Password" in collection_df and not collection_df[collection_df["match"].isna()]["Password"].empty:
            password_keys = collection_df[collection_df["match"].isna()]["Password"].dropna().astype(int)
            list_keys = card_df["Password"].dropna().astype(int)
            list_values = card_df.loc[list_keys.index]["Name"]
            key_name_dict = dict(zip(list_keys, list_values))

            missing = password_keys[~password_keys.isin(list_keys)]
            if missing.count() > 0:
                print(
                    "\nUnable to find the following card(s) by password:\n ⏺",
                    "\n ⏺ ".join(missing.sort_values().unique().astype(str)),
                )

            matches = password_keys.map(lambda x: key_name_dict.get(x, x))
            collection_df.loc[matches.index, "match"] = matches
            collection_df.drop("Password", axis=1, inplace=True)

        if "Name" in collection_df and not collection_df[collection_df["match"].isna()]["Name"].empty:
            name_keys = collection_df["Name"].dropna().str.lower().str.strip()
            list_keys = card_df["Name"].str.lower().str.strip()
            key_name_dict = dict(zip(list_keys, card_df["Name"]))

            missing = collection_df["Name"].dropna()[~name_keys.isin(list_keys)]
            if missing.count() > 0:
                print("\nUnable to find the following card(s) by name:\n ⏺", "\n ⏺ ".join(missing.sort_values().unique()))

            # Map the "Name" column from list_df to collection_df based on the keys
            matches = name_keys.map(lambda x: key_name_dict.get(x, x))
            collection_df.loc[matches.index, "match"] = matches
            collection_df.drop("Name", axis=1, inplace=True)

        collection_df = collection_df.rename(columns={"match": "Name"})
        columns = list(collection_df.columns.difference(["Count"]))
        collection_df = collection_df.dropna(how="all", axis=1).dropna(how="all", axis=0)
        collection_df = collection_df.groupby(columns, dropna=False).sum().reset_index()
        if merge_data:
            collection_df = collection_df.merge(card_df.drop_duplicates(subset="Name"), on="Name", how="left")
        collections[i] = collection_df

    print("\nCollection data processed.")
    if len(collections) == 1:
        return collections[0]
    else:
        return collections


# User decks
def assign_deck(collection_df: pd.DataFrame, deck_df: pd.DataFrame, return_collection: bool = False) -> pd.DataFrame:
    """
    Match deck cards to collection cards and return the resulting DataFrame with card counts appropriately adjusted.

    Args:
        collection_df (pd.DataFrame): DataFrame with the collection card names and quantities.
        deck_df (pd.DataFrame): DataFrame with the deck card names and quantities.
        return_collection (bool): If True, returns the remaining collection cards in the result DataFrame.

    Returns:
        (pd.DataFrame): DataFrame with the card names and quantities.
    """
    # Initialize a list to collect result rows
    result_rows = []

    # Iterate over each row in deck_df
    for index, deck_row in deck_df.iterrows():
        card_name = deck_row["Name"]
        deck_count = deck_row["Count"]
        deck_deck = deck_row["Deck"] if "Deck" in deck_row else np.nan

        # Get sub DataFrame from collection_df where Name matches
        collection_sub_df = collection_df[collection_df["Name"] == card_name].copy()

        # If Deck column exists, sort so that rows with exact Deck match are first, np.nan second
        if "Deck" in collection_sub_df.columns:
            collection_sub_df = collection_sub_df[collection_sub_df["Deck"].isin([deck_deck, np.nan])]
            collection_sub_df = collection_sub_df.sort_values(
                by=["Deck"], ascending=[True]
            )  # Sort by Deck (exact match first)

        # Subtract from the first available row(s) in collection_sub_df until deck_count is depleted
        for sub_index, collection_row in collection_sub_df.iterrows():
            if deck_count <= 0:
                break  # Deck count fulfilled

            available_count = collection_row["Count"]
            subtract_count = min(deck_count, available_count)

            # Subtract and update deck_count
            deck_count -= subtract_count
            collection_sub_df.loc[collection_row.name, "Count"] -= subtract_count

            # Add the collection_row to the result rows
            result_row = {
                "Name": card_name,
                "Count": available_count,  # Remaining count after subtraction
                "Deck": deck_deck,
                "missing": np.nan,
            }
            for col in collection_row.index.difference(result_row.keys()):
                result_row[col] = collection_row[col]

            result_rows.append(result_row)

            # If Count reaches 0, drop the row from collection_df
            if collection_sub_df.loc[collection_row.name, "Count"] <= 0:
                collection_sub_df.drop(collection_row.name, inplace=True)

        # After processing all available cards, handle missing counts
        if deck_count > 0:
            result_row = {"Name": card_name, "Count": 0, "Deck": deck_deck, "missing": deck_count}
            # Create a new row for the deck with Count set to 0 and missing indicating what's left
            result_rows.append(result_row)

    # Convert the collected result rows into a DataFrame
    result_df = pd.DataFrame(result_rows)

    # Append any remaining rows from collection_df that were not used in the deck
    if return_collection:
        result_df = pd.concat(
            [result_df, collection_df[~collection_df["Name"].isin(result_df["Name"])]], ignore_index=True
        ).sort_values(by=["Name", "Deck"])

    # Replace 0 values in missing with NaN
    result_df["missing"] = result_df["missing"].fillna(0)

    return result_df


## Decklists
def read_decklist(file_path: Path | str) -> pd.DataFrame:
    """
    Read a decklist file and return a DataFrame with the card names.

    Args:
        file_path (Path, str): Path to the decklist file.

    Returns:
        (pd.DataFrame): DataFrame with the card names.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    data = []
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        if line.endswith(":"):  # Section header
            current_section = line[:-1].capitalize()
        elif current_section:
            quantity, card_name = line.split("x ", 1)
            quantity = int(quantity)
            data.append({"Name": card_name, "Count": quantity, "Section": current_section, "Deck": file_path.stem})

    df = pd.DataFrame(data)
    return df


def get_decklists(*files: Path | str) -> pd.DataFrame:
    """
    Load decklist files and return a DataFrame with the card names.

    Args:
        files (Path | str): Paths to the decklist files. If not provided, loads all decklist files in the data directory.

    Returns:
        (pd.DataFrame): DataFrame with card names.
    """
    decklist_df = pd.DataFrame()
    if not files:
        files = list(dirs.DATA.glob("*.txt"))

    for file in files:
        temp_df = read_decklist(file)
        decklist_df = pd.concat([decklist_df, temp_df])
        print(f"Loaded {file.name} deck")

    decklist_df.replace({"Section": {"Monster": "Main", "Spell": "Main", "Trap": "Main"}}, inplace=True)
    return decklist_df


# YDK
def read_ydk(file_path: Path | str) -> pd.DataFrame:
    """
    Read a YDK file and return a DataFrame with the card codes.

    Args:
        file_path (Path | str): Path to the YDK file.

    Returns:
        (pd.DataFrame): DataFrame with the card codes.

    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    data = []
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        if line in ["#main", "#extra", "!side"]:
            current_section = line[1:].capitalize()
        elif current_section:
            data.append({"Code": line, "Section": current_section, "Deck": file_path.stem})

    df = pd.DataFrame(data)
    return df


def convert_ydk(ydk_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Convert a DataFrame with YDK card codes to a DataFrame with card names.

    Args:
        ydk_df (pd.DataFrame): DataFrame with YDK card codes.

    Returns:
        (pd.DataFrame | None): DataFrame with card names. If unable to obtain the card data, returns None.
    """
    ygoprodeck_file = dirs.DATA.joinpath("ygoprodeck.json")
    try:
        result = api.fetch_ygoprodeck()
        with open(ygoprodeck_file, "w") as file:
            json.dump(result, file, indent=4)
        ydk_data = pd.DataFrame(result).set_index("id")
    except Exception as e:
        print(e)
        if ygoprodeck_file.is_file():
            ydk_data = pd.read_json(ygoprodeck_file).set_index("id")
        else:
            print("Could not fetch or read ygoprodeck data")
            return None

    ydk_df = ydk_df.copy()

    def get_ydk_card(code) -> float | str:
        code = int(code)
        if code not in ydk_data.index:
            return np.nan
        return ydk_data.loc[code, "name"]

    ydk_df["Name"] = ydk_df["Code"].apply(get_ydk_card)
    not_found = ydk_df[ydk_df["Name"].isna()]["Code"].values
    if len(not_found) > 0:
        print(f"\nUnable to find {len(not_found)} cards:\n ⏺", "\n ⏺ ".join(not_found), "\n")
    ydk_df = ydk_df.drop("Code", axis=1).dropna(subset=["Name"]).reset_index(drop=True)
    ydk_df["Count"] = ydk_df.groupby(["Name", "Section", "Deck"])["Name"].transform("count")
    ydk_df = ydk_df.drop_duplicates().reset_index(drop=True)
    return ydk_df


def get_ydk(*files: Path | str) -> pd.DataFrame | None:
    """
    Load YDK files and return a DataFrame with the card names.

    Args:
        files (Path | str): Paths to YDK files. If not provided, loads all YDK files in the data directory.

    Returns:
        (pd.DataFrame | None): DataFrame with card names. If unable to obtain the card data, returns None.
    """
    ydk_df = pd.DataFrame()
    if not files:
        files = list(dirs.DATA.glob("*.ydk"))
    for file in files:
        temp_df = read_ydk(file)
        ydk_df = pd.concat([ydk_df, temp_df])
        print(f"Loaded {file.name} deck")

    ydk_df = convert_ydk(ydk_df)
    return ydk_df


# =================== #
# Notebook management #
# =================== #


def get_notebook_path() -> Path:
    """
    Gets the path of the current notebook opened in JupyterLab.
    If the path cannot be obtained, returns None.

    Args:
        None

    Returns:
        Path: The path of the current notebook.
    """

    file_path = (
        getattr(get_ipython(), "user_ns", {}).get("__vsc_ipynb_file__")
        or os.environ.get("JPY_SESSION_NAME")
        or os.environ.get("PM_IN_EXECUTION")
        or JupyterFrontEnd().sessions.current_session.get("name")
    )

    return Path(file_path) if file_path else None


def save_notebook() -> None:
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


def export_notebook(
    input_path: str | None = None, output_path: str | None = None, template: str = "auto", no_input: bool = True
) -> None:
    """
    Convert a Jupyter notebook to HTML using nbconvert and save the output to disk.

    Args:
        input_path (str | None, optional): The path to the Jupyter notebook file to convert. If None, gets the notebook path with `get_notebook_path`. Defaults to None.
        output_path (str | None, optional): The path to save the converted HTML file. If None, saves the file to the `REPORTS` directory. Defaults to None.
        template (str, optional): The name of the nbconvert template to use. If "auto", uses "labdynamic" if available, otherwise uses "lab". Defaults to "auto".
        no_input (bool, optional): If True, excludes input cells from the output. Defaults to True.

    Raises:
        ValueError: If no notebook path is provided and cannot be found with `get_notebook_path`.

    Returns:
        None
    """
    if input_path is None:
        input_path = get_notebook_path()
        if input_path is None:
            raise ValueError("No notebook path provided")
        input_path = str(get_notebook_path())
    if output_path is None:
        output_path = str(dirs.REPORTS / Path(input_path).stem)

    if template == "auto":
        template = "labdynamic"

    # Configure the HTMLExporter
    c = Config()
    c.HTMLExporter.extra_template_basedirs = [str(dirs.get_asset("nbconvert")), str(dirs.NBCONVERT)]
    c.HTMLExporter.template_name = template
    if no_input:
        c.TemplateExporter.exclude_output_prompt = True
        c.TemplateExporter.exclude_input = True
        c.TemplateExporter.exclude_input_prompt = True

    # Initialize the HTMLExporter
    html_exporter = HTMLExporter(config=c)

    # Read the notebook content
    with open(input_path, mode="r", encoding="utf-8") as f:
        notebook_content = nbformat.read(f, as_version=4)

    # Convert the notebook to HTML
    logger = logging.getLogger("IPKernelApp")
    logger.setLevel(logging.ERROR)

    (body, resources) = html_exporter.from_notebook_node(notebook_content)
    # Write the output to the specified directory
    writer = FilesWriter()
    writer.write(output=body, resources=resources, notebook_name=output_path)

    logger.setLevel(logging.WARNING)

    print(f"Notebook converted to HTML and saved to {output_path}.html")


# ================ #
# Markdown editing #
# ================ #


def update_index(dry_run: bool = False) -> str:
    """
    Update the index.md and README.md files with a table of links to all HTML reports in the `REPORTS` directory.
    Also update the @REPORT_|_TIMESTAMP@ and @TIMESTAMP@ placeholders in the index.md file with the latest timestamp.
    If the update is successful, commit the changes to Git with a commit message that includes the timestamp.
    If there is no index.md or README.md files in the `ASSETS` directory, print an error message and abort.

    Args:
        dry_run (bool, optional): If True, the function will not commit the changes to Git. Defaults to False.

    Returns:
        str: The result of the Git commit if not `dry_run`, otherwise advisory message.
    """

    index_file_name = "index.md"
    readme_file_name = "README.md"

    index_input_path = dirs.get_asset("markdown", index_file_name)
    readme_input_path = dirs.get_asset("markdown", readme_file_name)
    index_output_path = dirs.WORK / index_file_name
    readme_output_path = dirs.WORK / readme_file_name

    timestamp = arrow.utcnow()
    try:
        with open(index_input_path, encoding="utf-8") as f:
            index = f.read()

        with open(readme_input_path, encoding="utf-8") as f:
            readme = f.read()
    except:
        raise FileNotFoundError('Missing template files in "assets"')

    reports = sorted(dirs.REPORTS.glob("*.html"))
    rows = []
    for report in reports:
        rows.append(
            f"[{Path(report).stem}]({report.relative_to(dirs.WORK)}) | {pd.to_datetime(report.stat().st_mtime,unit='s', utc=True).strftime('%d/%m/%Y %H:%M %Z')}"
        )
    table = " |\n| ".join(rows)

    index = index.replace(f"@REPORT_|_TIMESTAMP@", table)
    index = index.replace(f"@TIMESTAMP@", timestamp.strftime("%d/%m/%Y %H:%M %Z"))

    readme = readme.replace(f"@REPORT_|_TIMESTAMP@", table)
    readme = readme.replace(f"@TIMESTAMP@", timestamp.strftime("%d/%m/%Y %H:%M %Z"))

    if dry_run:
        return "Dry run - README and index updated"
    else:
        with open(index_output_path, "w+", encoding="utf-8") as o:
            print(index, file=o)
        with open(readme_output_path, "w+", encoding="utf-8") as o:
            print(readme, file=o)

        result = git.commit(
            files=[index_output_path, readme_output_path],
            message=f"Index and README timestamp update - {timestamp.isoformat()}",
        )
        return result


def header(name: str | None = None) -> Markdown:
    """
    Generates a Markdown header with a timestamp and the name of the notebook (if provided).
    If there is no header.md file in the `ASSETS` directory, prints an error message and returns None.

    Args:
        name (str | None, optional): The name of the notebook. If None, attempts to extract the name from the environment variable JPY_SESSION_NAME. Defaults to None.

    Returns:
        Markdown: The generated Markdown header.
    """
    if name is None:
        path = get_notebook_path()
        name = path.stem if path else "Unnamed"

    header_path = dirs.get_asset("markdown", "header.md")
    try:
        with open(header_path, encoding="utf-8") as f:
            header = f.read()
    except:
        print('Missing template file in "assets". Aborting...')
        return None

    header = header.replace(
        "@TIMESTAMP@",
        arrow.utcnow().strftime("%d/%m/%Y %H:%M %Z"),
    )
    header = header.replace("@NOTEBOOK@", name)
    return Markdown(header)


def footer(timestamp: arrow.Arrow | None = None) -> Markdown:
    """
    Generates a Markdown footer with a timestamp.
    If there is no footer.md file in the `ASSETS` directory, prints error message and  an returns None.

    Args:
        timestamp (arrow.Arrow | None, optional): The timestamp to use. If None, uses the current time. Defaults to None.

    Returns:
        Markdown: The generated Markdown footer.
    """
    footer_path = dirs.get_asset("markdown", "footer.md")
    try:
        with open(footer_path, encoding="utf-8") as f:
            footer = f.read()
    except:
        print('Missing template file in "assets". Aborting...')
        return None

    now = arrow.utcnow()
    footer = footer.replace("@TIMESTAMP@", now.strftime("%d/%m/%Y %H:%M %Z"))

    return Markdown(footer)


# ================== #
# API Query Wrappers #
# ================== #


# Assets JSON dictionary updating
def update_rarities(save: bool = True) -> Dict[str, str]:
    """
    Updates the rarities.json file by fetching redirects and backlinks for the rarity codes and names.

    Args:
        save (bool, optional): Whether to save the new dictionary. Defaults to True.

    Returns:
        Dict[str, str]: The updated rarities dictionary.
    """
    rarities_file = dirs.get_asset("json", "rarities.json")
    rarity_dict = load_json(rarities_file)

    codes = list(rarity_dict.keys())
    names = list(rarity_dict.values())
    new_rarity_dict = api.fetch_redirect_dict(codes=codes, names=names, category="Rarities", namespace=0)

    rarity_dict = rarity_dict | new_rarity_dict

    if save:
        with open(rarities_file, "w+") as file:
            json.dump(rarity_dict, file, indent=4)

    return rarity_dict


def update_regions(save: bool = True) -> Dict[str, str]:
    """
    Updates the regions.json file by fetching backlinks and redirects for the region names and codes.
    Note: The category "Termonology" holds the regions in the wiki.

    Args:
        save (bool, optional): Whether to save the new dictionary. Defaults to True.

    Returns:
        Dict[str, str]: The updated regions dictionary.
    """
    regions_file = dirs.get_asset("json", "regions.json")
    regions_dict = load_json(regions_file)

    names = list(regions_dict.values())
    new_regions_dict = api.fetch_redirect_dict(names=names, category="Terminology", namespace=0)

    regions_dict = regions_dict | new_regions_dict
    if save:
        with open(regions_file, "w+") as file:
            json.dump(regions_dict, file, indent=4)

    return regions_dict


# Query builder
def card_query(*args, **kwargs) -> str:
    """
    Builds a query string to be passed to the yugipedia Wiki API for a card search query.

    Args:
        default (bool, optional): The default card query string, containing all properties for Monster, Spell and Trap cards. Defaults to False.
        *args: Properties to get from the `prop_dict` or to use directly as query string if not in `prop_dict`.
        **kwargs: Properties to include, if True, or remove, if False, from the query string.

    Raises:
        ValueError: If default is not a valid card type.

    Returns:
        str: A string containing the arguments to be passed to the API for the card search query.
    """
    default_properties = [
        "password",
        "card_type",
        "property",
        "primary",
        "secondary",
        "attribute",
        "monster_type",
        "stars",
        "atk",
        "def",
        "scale",
        "link",
        "arrows",
        "effect_type",
        "archseries",
        "alternate_artwork",
        "edited_artwork",
        "tcg",
        "ocg",
        "date",
    ]

    # Card properties dictionary
    property_dict = {
        "password": "Password",
        "card_type": "Card type",
        "property": "Property",
        "primary": "Primary type",
        "secondary": "Secondary type",
        "attribute": "Attribute",
        "monster_type": "Type=Monster type",
        "stars": "Stars string=Level/Rank",
        "atk": "ATK string=ATK",
        "def": "DEF string=DEF",
        "scale": "Pendulum Scale",
        "link": "Link Rating=Link",
        "arrows": "Link Arrows",
        "effect_type": "Effect type",
        "archseries": "Archseries",
        "alternate_artwork": "Category:OCG/TCG cards with alternate artworks",
        "edited_artwork": "Category:OCG/TCG cards with edited artworks",
        "tcg": "TCG status",
        "ocg": "OCG status",
        "date": "Modification date",
        "image_URL": "Card image",
        "misc": "Misc",
        "summoning": "Summoning",
        # Speed duel specific
        "speed": "TCG Speed Duel status",
        "character": "Character",
        # Rush duel specific
        "rush_alt_artwork": "Category:Rush Duel cards with alternate artworks",
        "rush_edited_artwork": "Category:Rush Duel cards with edited artworks",
        "maximum_atk": "MAXIMUM ATK",
        # Bandai specific
        "level": "Level",
        "number": "Bandai number=Card number",
        "rule": "Bandai rule=Rule",
        "sets": "Sets=Set",
        "rarity": "Rarity",
        "ability": "Ability",
        # Deprecated - Use for debuging
        "category": "category",
    }
    # Initialize string
    search_string = "|?English%20name=Name"

    # Initialize props list
    default = kwargs.pop("default", False)
    props = set(default_properties) if default else set()

    # Add args to props
    props.update(args)

    # Handle kwargs
    for key, value in kwargs.items():
        if value and key not in props:
            props.add(key)
        elif not value and key in props:
            props.discard(key)

    # Build the search string
    for prop in props:
        search_string += f"|?{up.quote(property_dict.get(prop, prop))}"

    return search_string


# Bandai
def fetch_bandai(bandai_query: str | None = None, limit: int = 200, **kwargs) -> pd.DataFrame:
    """
    Fetch Bandai cards.

    Args:
        bandai_query (str | None, optional): A string representing a SMW query to search for. Defaults to None.
        limit (int, optional): An integer that represents the maximum number of results to fetch. Defaults to 200.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pandas.DataFrame: A pandas DataFrame object containing the properties of the fetched Bandai cards.
    """
    debug = check_debug(kwargs.get("debug", False))

    concept = "[[Medium::Bandai]]"
    if bandai_query is None:
        bandai_query = card_query(*card_properties["bandai"])

    print(f"Downloading bandai cards")
    bandai_df = api.fetch_properties(concept, bandai_query, step=limit, limit=limit, **kwargs)
    if "Monster type" in bandai_df:
        bandai_df["Monster type"] = bandai_df["Monster type"].dropna().apply(lambda x: x.split("(")[0])  # Temporary
    if debug:
        print("- Total")

    print(f"{len(bandai_df.index)} results\n")

    time.sleep(0.5)

    return bandai_df


# Cards
def fetch_st(
    st_query: str | None = None,
    st: str = "both",
    cg: CG = CG.ALL,
    step: int = 500,
    limit: int = 5000,
    **kwargs,
) -> pd.DataFrame:
    """
    Fetch spell or trap cards based on query and properties of the cards.

    Args:
        st_query (str | None, optional): A string representing a SMW query to search for. Defaults to None.
        st (str, optional): A string representing the type of cards to fetch, either "spell", "trap", "both", or "all". Defaults to "both".
        cg (CG, optional): An Enum that represents the card game to fetch cards from. Defaults to CG.ALL.
        step (int, optional): An integer that represents the number of results to fetch at a time. Defaults to 500.
        limit (int, optional): An integer that represents the maximum number of results to fetch. Defaults to 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pandas.DataFrame: A pandas DataFrame object containing the properties of the fetched spell/trap cards.

    Raises:
        ValueError: Raised if the "st" argument is not one of "spell", "trap", "both", or "all".
        ValueError: Raised if the "cg" argument is not a valid CG.
    """
    debug = check_debug(kwargs.get("debug", False))
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

    if st_query is None:
        st_query = card_query(*card_properties["st"])

    print(f"Downloading {st}s")
    st_df = api.fetch_properties(concept, st_query, step=step, limit=limit, **kwargs)

    if debug:
        print("- Total")

    print(f"{len(st_df.index)} results\n")

    return st_df


def fetch_monster(
    monster_query: str | None = None,
    cg: CG = CG.ALL,
    step: int = 500,
    limit: int = 5000,
    exclude_token=True,
    **kwargs,
) -> pd.DataFrame:
    """
    Fetch monster cards based on query and properties of the cards.

    Args:
        monster_query (str | None, optional): A string representing a SMW query to search for. Defaults to None.
        cg (CG, optional): An Enum that represents the card game to fetch cards from. Defaults to CG.ALL.
        step (int, optional): An integer that represents the number of results to fetch at a time. Defaults to 500.
        limit (int, optional): An integer that represents the maximum number of results to fetch. Defaults to 5000.
        exclude_token (bool, optional): A boolean that determines whether to exclude Monster Tokens or not. Defaults to True.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pandas.DataFrame: A pandas DataFrame object containing the properties of the fetched monster cards.

    Raises:
        ValueError: Raised if the "cg" argument is not a valid CG.
    """
    debug = check_debug(kwargs.get("debug", False))
    valid_cg = cg.value
    attributes = ["DIVINE", "LIGHT", "DARK", "WATER", "EARTH", "FIRE", "WIND", "?"]
    if monster_query is None:
        monster_query = card_query(*card_properties["monster"])

    print("Downloading monsters")
    monster_df = pd.DataFrame()
    iterator = tqdm(
        attributes,
        leave=False,
        unit="attribute",
        dynamic_ncols=(not dirs.is_notebook),
        disable=("PM_IN_EXECUTION" in os.environ),
    )
    for att in iterator:
        iterator.set_description(att)
        if debug:
            tqdm.write(f"- {att}")

        concept = f"[[Concept:CG%20monsters]][[Attribute::{att}]]"

        if valid_cg != "CG":
            concept += f"[[Medium::{valid_cg}]]"

        temp_df = api.fetch_properties(concept, monster_query, step=step, limit=limit, iterator=iterator, **kwargs)
        monster_df = pd.concat([monster_df, temp_df], ignore_index=True, axis=0)

    if exclude_token and "Primary type" in monster_df:
        monster_df = monster_df[monster_df["Primary type"] != "Monster Token"].reset_index(drop=True)

    if debug:
        print("- Total")

    print(f"{len(monster_df.index)} results\n")

    return monster_df


# Non deck cards
def fetch_token(token_query: str | None = None, cg=CG.ALL, step: int = 500, limit: int = 5000, **kwargs) -> pd.DataFrame:
    """
    Fetch token cards based on query and properties of the cards.

    Args:
        token_query (str | None, optional): A string representing a SWM query to search for. Defaults to None.
        step (int, optional): An integer that represents the number of results to fetch at a time. Defaults to 500.
        limit (int, optional): An integer that represents the maximum number of results to fetch. Defaults to 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pandas.DataFrame: A pandas DataFrame object containing the properties of the fetched token cards.

    Raises:
        ValueError: Raised if the "cg" argument is not a valid CG.
    """

    valid_cg = cg.value

    concept = f"[[Category:Tokens]]"
    if valid_cg != "CG":
        concept += f"[[Category:{valid_cg}%20cards]]"
    else:
        concept += "[[Category:TCG%20cards||OCG%20cards]]"

    if token_query is None:
        token_query = card_query(*card_properties["monster"])

    print("Downloading tokens")
    token_df = api.fetch_properties(concept, token_query, step=step, limit=limit, **kwargs)

    print(f"{len(token_df.index)} results\n")

    return token_df


def fetch_counter(counter_query: str | None = None, cg=CG.ALL, step: int = 500, limit: int = 5000, **kwargs) -> pd.DataFrame:
    """
    Fetch counter cards based on query and properties of the cards.

    Args:
        counter_query (str | None, optional): A string representing a SMW query to search for. Defaults to None.
        step (int, optional): An integer that represents the number of results to fetch at a time. Defaults to 500.
        limit (int, optional): An integer that represents the maximum number of results to fetch. Defaults to 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pandas.DataFrame: A pandas DataFrame object containing the properties of the fetched counter cards.

    Raises:
        ValueError: Raised if the "cg" argument is not a valid CG.
    """
    valid_cg = cg.value

    concept = f"[[Category:Counters]][[Page%20type::Card%20page]]"
    if valid_cg != "CG":
        concept += f"[[Medium::{valid_cg}]]"

    if counter_query is None:
        counter_query = card_query(*card_properties["counter"])

    print("Downloading counters")
    counter_df = api.fetch_properties(concept, counter_query, step=step, limit=limit, **kwargs)

    print(f"{len(counter_df.index)} results\n")

    return counter_df


# Alternative formats
def fetch_speed(speed_query: str | None = None, step: int = 500, limit: int = 5000, **kwargs) -> pd.DataFrame:
    """
    Fetches TCG Speed Duel cards from the yugipedia Wiki API.

    Args:
        speed_query (str | None, optional):  A string representing a SMW query to search for. Defaults to None.
        step (int, optional): The number of results to fetch in each API call. Defaults to 500.
        limit (int, optional): The maximum number of results to fetch. Defaults to 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        A pandas DataFrame containing the fetched TCG Speed Duel cards.
    """
    debug = check_debug(kwargs.get("debug", False))

    concept = "[[Category:TCG Speed Duel cards]]"
    if speed_query is None:
        speed_query = card_query(*card_properties["speed"])

    print(f"Downloading Speed duel cards")
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


def fetch_skill(skill_query: str | None = None, step: int = 500, limit: int = 5000, **kwargs) -> pd.DataFrame:
    """
    Fetches skill cards from the yugipedia Wiki API.

    Args:
        skill_query (str | None, optional): A string representing a SMW query to search for. Defaults to None.
        step (int, optional): The number of results to fetch in each API call. Defaults to 500.
        limit (int, optional): The maximum number of results to fetch. Defaults to 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        A pandas DataFrame containing the fetched skill cards.
    """

    concept = "[[Category:Skill%20Cards]][[Card type::Skill Card]]"
    if skill_query is None:
        skill_query = card_query(*card_properties["skill"])

    print("Downloading skill cards")
    skill_df = api.fetch_properties(concept, skill_query, step=step, limit=limit, **kwargs)

    print(f"{len(skill_df.index)} results\n")

    return skill_df


def fetch_rush(rush_query: str | None = None, step: int = 500, limit: int = 5000, **kwargs) -> pd.DataFrame:
    """
    Fetches Rush Duel cards from the Yu-Gi-Oh! Wikia API.

    Args:
        rush_query (str | None, optional): A search query to filter the results. If not provided, it defaults to "rush".
        step (int, optional): The number of results to fetch in each API call. Defaults to 500.
        limit (int, optional): The maximum number of results to fetch. Defaults to 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        A pandas DataFrame containing the fetched Rush Duel cards.
    """
    concept = f"[[Category:Rush%20Duel%20cards]][[Medium::Rush%20Duel]]"
    if rush_query is None:
        rush_query = card_query(*card_properties["rush"])

    print("Downloading Rush Duel cards")
    rush_df = api.fetch_properties(concept, rush_query, step=step, limit=limit, **kwargs)

    print(f"{len(rush_df.index)} results\n")

    return rush_df


# Unusable cards
def fetch_unusable(
    query: str | None = None,
    cg: CG = CG.ALL,
    filter=True,
    step: int = 500,
    limit: int = 5000,
    **kwargs,
) -> pd.DataFrame:
    """
    Fetch unusable cards based on query and properties of the cards. Unusable cards include "Strategy cards", "Tip cards",
    "Card Checklists", etc, which are not actual cards. The filter option enables filtering those out and keeping only cards
    such as Duelist Kingdom "Ticket cards", old video-game promo "Character cards" and "Non-game cards" which have the layout
    of a real card, such as "Everyone's King". This criteria is not free of ambiguity.

    Args:
        query (str | None, optional): A string representing a SMW query to search for. Defaults to None.
        cg (CG, optional): An Enum that represents the card game to fetch cards from. Defaults to CG.ALL.
        filter (bool, optional): Keep only "Character Cards", "Non-game cards" and "Ticket Cards".
        step (int, optional): An integer that represents the number of results to fetch at a time. Defaults to 500.
        limit (int, optional): An integer that represents the maximum number of results to fetch. Defaults to 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pandas.DataFrame: A pandas DataFrame object containing the properties of the fetched spell/trap cards.

    Raises:
        ValueError: Raised if the "cg" argument is not a valid CG.

    """
    debug = check_debug(kwargs.get("debug", False))
    concept = "[[Category:Unusable cards]]"

    valid_cg = cg.value
    if valid_cg == "CG":
        concept = "OR".join([concept + f"[[{s} status::+]]" for s in ["TCG", "OCG"]])
    else:
        concept += f"[[{valid_cg} status::+]]"

    concept = up.quote(concept)

    if query is None:
        query = card_query(default=True)

    print(f"Downloading unusable cards")
    unusable_df = api.fetch_properties(concept, query, step=step, limit=limit, **kwargs)

    if filter and "Card type" in unusable_df:
        unusable_df = unusable_df[
            unusable_df["Card type"].isin(["Character Card", "Non-game card", "Ticket Card"])
        ].reset_index(drop=True)

    unusable_df.dropna(how="all", axis=1, inplace=True)

    if debug:
        print("- Total")

    print(f"{len(unusable_df.index)} results\n")

    return unusable_df


# Extra properties
def fetch_errata(errata: str = "all", step: int = 500, **kwargs) -> pd.DataFrame:
    """
    Fetches errata information from the yuipedia Wiki API.

    Args:
        errata (str): The type of errata information to fetch. Valid values are 'name', 'type', and 'all'. Defaults to 'all'.
        step (int): The number of results to fetch in each API call. Defaults to 500.
        **kwargs: Additional keyword arguments to pass to fetch_categorymembers.

    Returns:
        pandas.DataFrame: A pandas DataFrame containing a boolean table indicating whether each card has errata information for the specified type.
    """
    debug = check_debug(kwargs.get("debug", False))
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
        dynamic_ncols=(not dirs.is_notebook),
        disable=("PM_IN_EXECUTION" in os.environ),
    )
    for cat in iterator:
        desc = cat.split("Category:")[-1]
        iterator.set_description(desc)
        if debug:
            tqdm.write(f"- {cat}")

        temp = api.fetch_categorymembers(cat, namespace=3010, step=step, iterator=iterator, debug=debug)
        errata_data = temp["title"].apply(lambda x: x.split("Card Errata:")[-1])
        errata_series = pd.Series(data=True, index=errata_data, name=desc)
        errata_df = pd.concat([errata_df, errata_series], axis=1).astype("boolean").fillna(False).sort_index()

    if debug:
        print("- Total")

    print(f"{len(errata_df.index)} results\n")
    return errata_df


# Sets
def fetch_set_list_pages(cg: CG = CG.ALL, step: int = 500, limit=5000, **kwargs) -> pd.DataFrame:
    """
    Fetches a list of 'Set Card Lists' pages from the yugipedia Wiki API.

    Args:
        cg (CG): A member of the CG enum representing the card game for which set lists are being fetched.
        step (int): The number of pages to fetch in each API request.
        limit (int): The maximum number of pages to fetch.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pd.DataFrame: A DataFrame containing the titles of the set list pages.

    Raises:
        ValueError: Raised if the "cg" argument is not a valid CG.

    """
    debug = check_debug(kwargs.get("debug", False))
    valid_cg = cg.value
    if valid_cg == "CG":
        category = ["TCG Set Card Lists", "OCG Set Card Lists"]
    else:
        category = f"{valid_cg}%20Set%20Card%20Lists"

    print("Downloading list of 'Set Card Lists' pages")
    set_list_pages = pd.DataFrame()
    iterator = tqdm(
        category,
        leave=False,
        unit="category",
        dynamic_ncols=(not dirs.is_notebook),
        disable=("PM_IN_EXECUTION" in os.environ),
    )
    for cat in iterator:
        iterator.set_description(cat.split("Category:")[-1])
        temp = api.fetch_categorymembers(cat, namespace=None, step=step, iterator=iterator, debug=debug)
        sub_categories = pd.DataFrame(temp)["title"]
        sub_iterator = tqdm(
            sub_categories,
            leave=False,
            unit="subcategory",
            dynamic_ncols=(not dirs.is_notebook),
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


def fetch_all_set_lists(cg: CG = CG.ALL, step: int = 40, **kwargs) -> pd.DataFrame:
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
    debug = check_debug(kwargs.get("debug", False))
    sets = fetch_set_list_pages(cg, **kwargs)  # Get list of sets
    keys = sets["Page name"]

    all_set_lists_df = pd.DataFrame(columns=["Set", "Card number", "Name", "Rarity", "Print", "Quantity", "Region"])
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
        set_lists_df = set_lists_df.merge(sets, on="Page name", how="left").drop("Page name", axis=1)
        all_set_lists_df = pd.concat([all_set_lists_df, set_lists_df], ignore_index=True)
        total_success += success
        total_error += error

    all_set_lists_df = all_set_lists_df.convert_dtypes()
    all_set_lists_df.sort_values(by=["Set", "Region", "Card number"]).reset_index(inplace=True)
    print(f'{"Total: " if debug else ""}{total_success} set lists received - {total_error} missing')

    return all_set_lists_df


# ======================= #
# Complete execution flow #
# ======================= #


# TODO: Propagate the debug flag to notebooks
def run_notebooks(
    reports: str | list[str],
    external_pbar: tqdm | None = None,
    discord: bool | argparse.Namespace = False,
    telegram: bool | argparse.Namespace = False,
    dry_run: bool = False,
    debug: bool = False,
) -> None:
    """
    Execute specified Jupyter notebooks using Papermill.

    Args:
        reports (str | List[str]): List of notebooks to execute.
        external_pbar (tqdm | None, optional): An external tqdm progress bar to update. Defaults to None.
        discord (bool | argparse.Namespace, optional): Discord configuration, either as a boolean or argparse.Namespace. Default is False.
        telegram (bool | argparse.Namespace, optional): Telegram configuration, either as a boolean or argparse.Namespace. Default is False.
        dry_run (bool, optional): Whether to run in dry run mode. Default is False.
        debug (bool, optional): Whether to enable debug mode. Default is False.

    Returns:
        None

    Raises:
        Exception: Raised if any exceptions occur during notebook execution.
    """

    contribs = {"discord": discord, "telegram": telegram}

    # Initialize iterators
    warnings.filterwarnings("ignore", message=".*clamping frac to range.*")
    pbars = []

    pbar_kwargs = dict(
        iterable=reports,
        unit="report",
        unit_scale=True,
        dynamic_ncols=(not dirs.is_notebook),
        delay=1,
        desc="Completion",
    )

    warnings.filterwarnings("ignore", message=".*clamping frac to range.*")
    if external_pbar:
        pbars.append(external_pbar(position=0, **pbar_kwargs))
    else:
        pbars.append(
            tqdm(
                position=0,
                **pbar_kwargs,
            )
        )

    # Helper to handle each contrib
    def setup_contrib(contrib: str) -> tqdm | None:
        contrib_upper = contrib.upper()
        contrib_value = contribs.get(contrib)
        if contrib_upper == "DISCORD":
            ch_key = "CHANNEL_ID"
        elif contrib_upper == "TELEGRAM":
            ch_key = "CHAT_ID"
        if contrib_value is True:

            required_secrets = [f"{contrib_upper}_TOKEN", f"{contrib_upper}_{ch_key}"]
            try:
                secrets = load_secrets(
                    required_secrets,
                    secrets_file=dirs.secrets_file,
                    required=True,
                )
                tkn = secrets.get(required_secrets[0])
                ch = secrets.get(required_secrets[1])
            except:
                cprint(text=f"Missing {contrib} secrets. Ignoring...", color="yellow")
                return
        elif isinstance(contrib_value, argparse.Namespace):
            tkn = contrib_value.tkn
            ch = contrib_value.ch
        else:
            return

        try:
            if contrib_upper == "DISCORD":
                from tqdm.contrib.discord import tqdm as contrib_tqdm
            elif contrib_upper == "TELEGRAM":
                from tqdm.contrib.telegram import tqdm as contrib_tqdm
            else:
                cprint(text=f"Unsupported contrib: {contrib}. Ignoring...", color="yellow")

            return contrib_tqdm(
                token=tkn,
                file=open(file=os.devnull, mode="w"),
                **{ch_key: ch},
                **pbar_kwargs,
            )
        except:
            pass

    # Iterate over potential contrib names
    for contrib in contribs:
        pbar = setup_contrib(contrib)
        if pbar:
            pbars.append(pbar)

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
            # print(f'Number of Cells: {cells}')

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
        if "yugiquery" in jupyter_client.kernelspec.find_kernel_specs():
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

    tqdm.write("\nExecution completed")  # Empty character for better readability
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
    reports: str | List[str] = "all",
    progress_handler: ProgressHandler | None = None,
    cleanup: bool | Literal["auto"] = "auto",
    dry_run: bool = False,
    squash: bool = True,
    discord: bool | argparse.Namespace = False,
    telegram: bool | argparse.Namespace = False,
    debug: bool = False,
) -> None:
    """
    Executes all notebooks in the user and package `NOTEBOOKS` directories that match the specified report, updates the page index
    to reflect the last execution timestamp, and clean up redundant data files.

    Args:
        reports (str | List[str], optional): The report to generate. Defaults to 'all'.
        progress_handler (ProgressHandler | None, optional): An optional ProgressHandler instance to report execution progress. Defaults to None.
        cleanup (bool | Literal["auto"], optional): whether to cleanup data files after execution. If True, perform cleanup, if False, doesn't perform cleanup. If 'auto', performs cleanup if there are more than 4 data files for each report (assuming one per week). Defaults to 'auto'.
        dry_run (bool, optional): dry_run flag to pass to cleanup_data method call. Defaults to False.
        squash (bool, optional): squash commits after execution. Defaults to True.
        discord (bool | argparse.Namespace, optional): Discord configuration, either as a boolean or argparse.Namespace. Default is False.
        telegram (bool | argparse.Namespace, optional): Telegram configuration, either as a boolean or argparse.Namespace. Default is False.
        debug (bool, optional): Whether to enable debug mode. Default is False.

    Raises:
        Exception: Raised if any exceptions occur during notebook execution.

    Returns:
        None: This function does not return a value.
    """
    # debug = check_debug(debug)

    reports = dirs.find_notebooks(reports)

    # Check API status
    api_status = api.check_status()
    if progress_handler:
        progress_handler.send(API_status=api_status)
    if not api_status:
        return

    # Get the current commit hash
    start_commit = git.get_repo().head.commit

    # TODO: Error handling
    lock("run")

    # Execute notebooks
    try:
        if len(reports) > 0:
            run_notebooks(
                reports=reports,
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
        raise e
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
    # TODO: Check error handling
    if cleanup == "auto":
        data_files_count = len(list(dirs.DATA.glob("*.bz2")))
        reports_count = len(list(dirs.REPORTS.glob("*.html")))
        cleanup = data_files_count / max(reports_count, 1) > 10
    if cleanup:
        cleanup_data(dry_run=dry_run)

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

    # TODO: Error handling
    unlock("run")


# ========= #
# CLI usage #
# ========= #


def main(args):
    # Assures the script is within a git repository before proceesing
    _ = git.ensure_repo()
    # Execute the complete workflow
    run(
        reports=args.reports,
        cleanup=args.cleanup,
        dry_run=args.dryrun,
        discord=args.discord,
        telegram=args.telegram,
        debug=args.debug,
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
        "--debug",
        action="store_true",
        required=False,
        help="run in debug mode",
    )
    debug_group.add_argument("-p", "--paths", action="store_true", help="print YugiQuery paths and exit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    set_parser(parser)
    args = parser.parse_args()

    if args.paths:
        dirs.print()
        exit()

    main(args)
