# yugiquery/core/data.py

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import os
from ast import literal_eval
from pathlib import Path
from typing import List, Literal, Tuple, overload

# --- Imports: Third-Party --- #
import arrow
import re
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# --- Imports: Local Application --- #
from .. import api
from ..utils import dirs, load_json, LoggerConfig, filename_ts_fmt, get_notebook_path

# --- Logger Setup --- #
logger = LoggerConfig.get_logger()

# --- Data File Loading and Normalization --- #


def load_latest(
    name_pattern: str,
    type: str = "data",
) -> Tuple[pd.DataFrame | None, arrow.Arrow | None]:
    """
    Loads the latest file matching the given name pattern and type, and attempts to parse specified columns as tuples.
    Returns the loaded DataFrame and the timestamp extracted from the filename. If no matching file is found, returns (None, None).
    If type is "changelog", the timestamp returned refers to the end of the period covered by the changelog i.e. the "to" date.

    Args:
        name_pattern (str): The pattern to match in the filename (e.g., "cards")
        type (str, optional): The type of file to look for, either "data" or "changelog". Defaults to "data".
        tuple_cols (List[str], optional): Additional list of column names to attempt to parse as tuples. Defaults to [].

    Returns:
        Tuple[pd.DataFrame | None, arrow.Arrow | None]: A tuple containing the loaded DataFrame (or None if not found) and the timestamp (or None if not found).
    """
    name_pattern = name_pattern.lower()
    files = sorted(
        list(dirs.DATA.glob(f"{name_pattern}_{type}_*.bz2")),
        key=os.path.getctime,
        reverse=True,
    )

    if files:
        df = pd.read_csv(files[0], dtype=object, keep_default_na=False, na_values="")
        tuple_pattern = re.compile(r"^\(.*['\"].*\)$")
        for col in df.columns:
            first_val = df[col].dropna().astype(str).iloc[0] if not df[col].dropna().empty else None
            if first_val and tuple_pattern.match(first_val):
                try:
                    df[col] = df[col].dropna().apply(literal_eval)
                except (ValueError, SyntaxError):
                    pass

        for col in df.filter(regex="(?i)(date|time|release|debut)").columns:
            df[col] = pd.to_datetime(df[col])
        with logging_redirect_tqdm():
            logger.info("%s file loaded from %s.", name_pattern.capitalize(), files[0])
        if dirs.is_notebook:
            nbpath = get_notebook_path()
            nbpath = nbpath.parent if nbpath else dirs.WORK
            relpath = Path(os.path.relpath(files[0], nbpath)).as_posix()
            display(Markdown(f"{name_pattern.capitalize()} {type} loaded from [{relpath}]({relpath})"))
        else:
            relpath = Path(os.path.relpath(files[0], dirs.WORK)).as_posix()
            tqdm.write(f"{name_pattern.capitalize()} {type} loaded from {relpath}")

        ts = arrow.get(Path(files[0]).stem.split("_")[-1])
        return df, ts

    logger.warning('No file matching pattern "%s" found.', name_pattern)
    return None, None


def load_changelog_for(name: str, timestamp: str | arrow.Arrow | None) -> pd.DataFrame | None:
    """
    Load the changelog covering the given timestamp for the specified name.
    If timestamp is None, loads the latest changelog available for the given name.
    Otherwise, finds the changelog file whose period covers the timestamp, or the latest one before it if none covers it.
    The changelog files are expected to be named in the format "{name}_changelog_{from}_{to}.bz2", where "from" and "to" are timestamps in "YYYYMMDDTHHmmZ" format.

    Args:
        name (str): The base name of the data file (e.g., 'bandai', 'cards', etc.)
        timestamp (str | arrow.Arrow | None): The timestamp string or Arrow object (e.g., '20250301T1551Z').
            If None, loads the latest changelog available for the given name.

    Returns:
        pd.DataFrame | None: The loaded changelog DataFrame (or None if not found).
    """
    if timestamp is None:
        df, _ = load_latest(name, type="changelog")
        return df

    ts = timestamp if isinstance(timestamp, arrow.Arrow) else arrow.get(timestamp)

    # Find all changelog files for this dataset
    changelog_files = list(dirs.DATA.glob(f"{name}_changelog_*.bz2"))
    if not changelog_files:
        logger.info("No changelog found for %s", name)
        return None

    changelog_periods = []
    for file in changelog_files:
        parts = Path(file).stem.split("_")
        if len(parts) < 4 or parts[0] != name or parts[1] != "changelog":
            continue
        from_ts, to_ts = arrow.get(parts[-2]), arrow.get(parts[-1])
        changelog_periods.append((file, from_ts, to_ts))

    if not changelog_periods:
        logger.info("No valid changelog periods found for %s", name)
        return None

    # Sort by to_ts descending to prioritize most recent coverage
    changelog_periods.sort(key=lambda x: x[2], reverse=True)

    # Pick the changelog whose range includes ts, or the latest one before ts
    for file, from_ts, to_ts in changelog_periods:
        if from_ts <= ts <= to_ts or to_ts <= ts:
            changelog_file = file
            break
    else:
        logger.info("No changelog found for %s at or before %s", name, ts)
        return None

    df = pd.read_csv(changelog_file, dtype=object, keep_default_na=False, na_values="")
    if dirs.is_notebook:
        nbpath = get_notebook_path()
        nbpath = nbpath.parent if nbpath else dirs.WORK
        relpath = Path(os.path.relpath(changelog_file, nbpath)).as_posix()
        display(Markdown(f"Changelog loaded from [{relpath}]({relpath}) for {name.capitalize()} data"))
    else:
        relpath = Path(os.path.relpath(changelog_file, dirs.WORK)).as_posix()
        tqdm.write(f"Changelog loaded from {relpath} for {name.capitalize()} data")
    logger.info("Changelog loaded from %s for %s %s", relpath, name.capitalize(), filename_ts_fmt(to_ts))
    return df


# --- Set and Card Dataset Merges --- #


def merge_set_info(input_df: pd.DataFrame, input_info_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge set metadata into set list data by set and region.
    The input dataframe must contain "Set" and "Region" columns. The function will look up release dates based on the region and merge additional set information from the input_info_df, which should be indexed by set name.

    Args:
        input_df (pd.DataFrame): DataFrame containing at least "Set" and "Region" columns.
        input_info_df (pd.DataFrame): DataFrame indexed by set name containing set metadata, including release dates by region.
    Returns:
        pd.DataFrame: The input dataframe merged with set metadata and release dates.
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

    logger.info("Set properties merged")
    return merged_df


def merge_set_to_cards(*card_df, set_df) -> pd.DataFrame:
    """
    Merge set lists into one or more card information dataframes.
    The function takes one or more card dataframes and a set dataframe, and merges them based on card names. It creates an "index" column by normalizing card names (lowercasing and removing "#") in both the card and set dataframes, then performs an inner merge on this index. The resulting dataframe is cleaned up by dropping the index and renaming columns appropriately.

    Args:
        *card_df: One or more DataFrames containing card information, each with a "Name" column.
        set_df (pd.DataFrame): DataFrame containing set information, with a "Name" column.
    Returns:
        pd.DataFrame: A merged DataFrame containing card information enriched with set data, matched by normalized card names.
    """
    full_df = pd.concat(card_df).drop_duplicates(ignore_index=True)
    full_df["index"] = full_df["Name"].str.lower().str.replace("#", "")
    set_df = set_df.copy()
    set_df["index"] = set_df["Name"].str.lower().str.replace("#", "")

    full_df = full_df.merge(set_df, how="inner", on="index")
    full_df = full_df.convert_dtypes()
    full_df["Name"] = full_df["Name_x"].fillna(full_df["Name_y"])
    full_df.drop(["index", "Name_x", "Name_y"], axis=1, inplace=True)
    full_df.rename(
        columns={
            "Page URL_x": "Card page URL",
            "Page URL_y": "Set page URL",
            "Modification date_x": "Card modification date",
            "Modification date_y": "Set modification date",
        },
        inplace=True,
    )
    full_df = full_df[np.append(full_df.columns[-1:], full_df.columns[:-1])]
    return full_df


def _format_errata(row: pd.Series) -> Tuple[str, ...] | float:
    """Helper function to format errata information into a tuple of affected fields."""
    result = []
    if "Cards with name errata" in row and row["Cards with name errata"]:
        result.append("Name")
    if "Cards with card type errata" in row and row["Cards with card type errata"]:
        result.append("Type")
    if "Card Errata" in row and not result and row["Card Errata"]:
        result.append("Any")
    if result:
        return tuple(sorted(result))
    return np.nan


def merge_errata(input_df: pd.DataFrame, input_errata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge errata information into the input DataFrame by card name.
    The function checks if the input DataFrame contains a "Name" column, then applies the _format_errata helper function to the input_errata_df to create a Series of errata information. This Series is merged into the input DataFrame based on the "Name" column, resulting in an "Errata" column that indicates which fields are affected by errata for each card.

    Args:
        input_df (pd.DataFrame): DataFrame containing card information, must include a "Name" column.
        input_errata_df (pd.DataFrame): DataFrame containing errata information, with columns indicating which cards have name or type errata.

    Returns:
        pd.DataFrame: The input DataFrame merged with errata information, containing an "Errata" column that specifies which fields are affected by errata for each card.
    """
    if "Name" in input_df.columns:
        errata_series: pd.Series = input_errata_df.apply(_format_errata, axis=1)
        input_df = input_df.merge(
            errata_series.rename("Errata"),
            left_on="Name",
            right_index=True,
            how="left",
            suffixes=("", " errata"),
        )
    else:
        logger.error('No "Name" column to join errata')

    return input_df


# --- Collection Loading --- #


def get_collection(file_name: str = "collection") -> None | pd.DataFrame:
    """
    Load a user collection from CSV or Excel.
    The function looks for a file with the specified name in the data directory, first checking for an Excel file and then a CSV file. If an Excel file is found, it loads all sheets and concatenates them into a single DataFrame with an additional "Collection" column indicating the sheet name. If a CSV file is found, it loads it directly into a DataFrame. If no file is found, it logs a warning and returns None.

    Args:
        file_name (str, optional): The base name of the collection file (without extension). Defaults to "collection".
    Returns:
        pd.DataFrame | None: The loaded collection DataFrame if a file is found, otherwise None.
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
        logger.warning("No %s file found.", file_name)
        return None

    collection_df = collection_df.convert_dtypes(convert_string=False)

    logger.info("Loaded %s.", collection_file.name)
    print(f"Loaded {collection_file.name}.")
    return collection_df


# --- Card Matching Pipeline --- #


def find_cards(list_df: pd.DataFrame, card_data: bool = False, set_data: bool = False) -> pd.DataFrame:
    """
    Match a card list against latest datasets and optionally enrich with card data.
    The function takes an input DataFrame containing a list of cards, and attempts to match each card against the latest card and set datasets. It first checks if the input DataFrame is empty and returns it if so. Then it loads the necessary reference data based on the columns present in the input DataFrame. The matching process is performed in three stages: first by "Card number" using set data, then by "Password" using card data, and finally by "Name" using either card or set data. After matching, the function finalizes the matched cards by grouping and summing counts, and optionally merging with card data for enrichment.

    Args:
        list_df (pd.DataFrame): DataFrame containing a list of cards to match, with columns such as "Name", "Card number", or "Password".
        card_data (bool, optional): Whether to load and merge card data for enrichment. Defaults to False.
        set_data (bool, optional): Whether to load and merge set data for matching by card number. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the matched cards, enriched with card data if requested.
    """
    if list_df.empty:
        logger.warning("List empty. Ignoring.")
        return list_df

    # Load reference data based on what's in the input dataframe
    card_df = _load_card_data(list_df, card_data)
    set_lists_df = _load_set_data(list_df)

    if card_df is None and set_lists_df is None:
        raise FileNotFoundError("No card or set lists data files found.")

    logger.info("Finding cards in database...")
    print("Finding cards in database...")

    # Prepare list dataframe for matching
    original_cols = list_df.columns
    list_df = list_df.dropna(how="all").assign(match=np.nan).astype(object)

    # Match cards using available keys in priority order
    _match_cards_by_card_number(list_df, original_cols, set_lists_df, set_data)
    _match_cards_by_password(list_df, original_cols, card_df)
    _match_cards_by_name(list_df, original_cols, card_df, set_lists_df)

    # Finalize and return
    list_df = _finalize_matched_cards(list_df, card_df, card_data)

    logger.info("%d out of %d cards found.", list_df[list_df["Name"].notna()]["Count"].sum(), list_df["Count"].sum())
    print(f"{list_df[list_df['Name'].notna()]['Count'].sum()} out of {list_df['Count'].sum()} cards found.")

    return list_df


def _load_card_data(list_df: pd.DataFrame, card_data: bool) -> pd.DataFrame | None:
    """
    Helper function to load card data if "Name" or "Password" columns are present in the input DataFrame of find_cards.

    Args:
        list_df (pd.DataFrame): The input DataFrame containing the card list to be matched.
        card_data (bool): Flag indicating whether to load card data for enrichment.

    Returns:
        pd.DataFrame | None: The loaded card data DataFrame if relevant columns are present and card_data is True, otherwise None.
    """
    if card_data or any(col in list_df and not list_df[col].dropna().empty for col in ["Name", "Password"]):
        card_df, _ = load_latest(name_pattern="cards")
        if card_df is not None:
            card_df.sort_values(by=["Name", "Primary type", "Property"], ignore_index=True, inplace=True)
        return card_df
    return None


def _load_set_data(list_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Helper function to load set lists data if "Card number" column is present in the input DataFrame of find_cards.

    Args:
        list_df (pd.DataFrame): The input DataFrame containing the card list to be matched.

    Returns:
        pd.DataFrame | None: The loaded set lists DataFrame if "Card number" column is present and not empty, otherwise None.
    """
    if "Card number" in list_df and not list_df["Card number"].dropna().empty:
        set_lists_df, _ = load_latest(name_pattern="sets")
        if set_lists_df is not None:
            set_lists_df = (
                set_lists_df.sort_values(by="Release")
                .drop_duplicates(subset="Card number", keep="first")
                .dropna(subset=["Card number"])
            )
        return set_lists_df
    return None


def _match_cards_by_card_number(
    list_df: pd.DataFrame, original_cols: pd.Index, set_lists_df: pd.DataFrame | None, set_data: bool
) -> None:
    """
    Helper function to match cards by "Card number" using set data if the column is present in the input DataFrame of find_cards.

    Args:
        list_df (pd.DataFrame): The input DataFrame containing the card list to be matched, with a "match" column for storing results.
        original_cols (pd.Index): The original columns of the input DataFrame before processing.
        set_lists_df (pd.DataFrame | None): The loaded set lists DataFrame to be used for matching by card number, or None if not available.
        set_data (bool): Flag indicating whether to use set data for matching by card number.

    Returns:
        None: The function updates the input list_df in place with matched card names based on card number.
    """
    if "Card number" not in original_cols or set_lists_df is None:
        return

    if list_df["Card number"].dropna().empty:
        return

    if set_data:
        _merge_set_data(list_df, set_lists_df)
    else:
        _merge_with_keys(list_df, "Card number", set_lists_df, "Card number", "Name")


def _merge_set_data(df: pd.DataFrame, set_lists_df: pd.DataFrame | None) -> None:
    """
    Helper function to merge set data into the input DataFrame for matching by "Card number".

    Args:
        df (pd.DataFrame): The input DataFrame containing the card list to be matched, with a "match" column for storing results.
        set_lists_df (pd.DataFrame | None): The loaded set lists DataFrame to be used for merging, or None if not available.

    Returns:
        None: The function updates the input df in place by merging set data based on "Card number" and logs any missing card numbers that could not be matched.
    """
    if set_lists_df is None:
        return

    df["Card number"] = df["Card number"].str.upper()
    extra_cols = set_lists_df.columns.difference(df.columns).union(["Card number", "Name"])
    merged_result = df.merge(set_lists_df[extra_cols], on="Card number", how="left")
    merged_result["match"] = merged_result["Name_y"] if "Name_y" in merged_result else merged_result["Name"]
    merged_result.rename({"Name_x": "Name"}, axis=1, inplace=True, errors="ignore")
    merged_result.drop(columns=["Name_y"], inplace=True, errors="ignore")

    # Log missing card numbers
    missing = (
        merged_result["Card number"][~merged_result["Card number"].isin(set_lists_df["Card number"])]
        .dropna()
        .sort_values()
        .unique()
        .astype(str)
    )
    if len(missing) > 0:
        logger.warning(
            'Unable to find the following %d card(s) by "Card number":\n * %s',
            len(missing),
            "\n * ".join(missing),
        )

    # Update input dataframe columns
    for col in merged_result.columns:
        df[col] = merged_result[col]


def _match_cards_by_password(list_df: pd.DataFrame, original_cols: pd.Index, card_df: pd.DataFrame | None) -> None:
    """
    Helper function to match cards by "Password" using card data if the column is present in the input DataFrame of find_cards.

    Args:
        list_df (pd.DataFrame): The input DataFrame containing the card list to be matched, with a "match" column for storing results.
        original_cols (pd.Index): The original columns of the input DataFrame before processing.
        card_df (pd.DataFrame | None): The loaded card data DataFrame to be used for matching by password, or None if not available.

    Returns:
        None: The function updates the input list_df in place with matched card names based on password.
    """
    if "Password" not in original_cols or card_df is None or list_df["match"].notna().all():
        return

    _merge_with_keys(list_df, "Password", card_df, "Password", "Name")


def _match_cards_by_name(
    list_df: pd.DataFrame,
    original_cols: pd.Index,
    card_df: pd.DataFrame | None,
    set_lists_df: pd.DataFrame | None,
) -> None:
    """
    Helper function to match cards by "Name" using either card or set data if the column is present in the input DataFrame of find_cards.

    Args:
        list_df (pd.DataFrame): The input DataFrame containing the card list to be matched, with a "match" column for storing results.
        original_cols (pd.Index): The original columns of the input DataFrame before processing.
        card_df (pd.DataFrame | None): The loaded card data DataFrame to be used for matching by name, or None if not available.
        set_lists_df (pd.DataFrame | None): The loaded set lists DataFrame to be used for matching by name, or None if not available.

    Returns:
        None: The function updates the input list_df in place with matched card names based on name.
    """
    if "Name" not in original_cols or list_df["match"].notna().all():
        return

    ref_df = card_df if card_df is not None else set_lists_df

    if ref_df is not None:
        _merge_with_keys(list_df, "Name", ref_df, "Name", "Name")

    # Try ygoprodeck for old card names as fallback
    if list_df["match"].isna().any():
        try:
            ydk_data = api.get_ygoprodeck()
            ydk_data["Old name"] = ydk_data["misc_info"].apply(
                lambda x: tuple(y["beta_name"] for y in x if "beta_name" in y)
            )
            _merge_with_keys(
                list_df,
                "Name",
                ydk_data.explode("Old name"),
                "Old name",
                "name",
            )
        except Exception as e:
            logger.warning("Unable to get old names from ygoprodeck: %s", e)


def _merge_with_keys(df: pd.DataFrame, key_col: str, ref_df: pd.DataFrame, ref_key: str, ref_val: str) -> pd.DataFrame:
    """
    Helper function to merge reference data into the input DataFrame based on a specified key column and reference key-value pair.

    Args:
        df (pd.DataFrame): The input DataFrame containing the card list to be matched, with a "match" column for storing results.
        key_col (str): The column name in the input DataFrame to be used as the key for merging.
        ref_df (pd.DataFrame): The reference DataFrame containing the data to merge.
        ref_key (str): The column name in the reference DataFrame to be used as the key for merging.
        ref_val (str): The column name in the reference DataFrame to be used as the value for merging.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    if ref_df is None:
        return df

    keys = df[df["match"].isna()][key_col].dropna()
    ref_df = ref_df.dropna(subset=ref_key)
    list_keys = ref_df[ref_key]

    # Normalize keys for comparison (case-insensitive for names, numeric for passwords)
    if key_col != "Password":
        keys = keys.str.lower().str.strip()
        list_keys = list_keys.str.lower().str.strip()
    else:
        keys = keys.astype(int)
        list_keys = list_keys.astype(int)

    # Create mapping and identify missing cards
    key_name_dict = dict(zip(list_keys, ref_df[ref_val]))
    missing = df.loc[keys[~keys.isin(list_keys)].index, key_col].sort_values().unique().astype(str)

    if len(missing) > 0:
        logger.warning(
            'Unable to find the following %d card(s) by "%s":\n * %s',
            len(missing),
            ref_key,
            "\n * ".join(missing),
        )

    # Apply matches to dataframe
    matches = keys.map(key_name_dict.get)
    df.loc[matches.index, "match"] = matches
    return df


def _finalize_matched_cards(list_df: pd.DataFrame, card_df: pd.DataFrame | None, card_data: bool) -> pd.DataFrame:
    """
    Helper function to finalize the matched cards by grouping and summing counts, and optionally merging with card data for enrichment.

    Args:
        list_df (pd.DataFrame): The input DataFrame containing the card list with a "match" column for matched card names and a "Count" column for quantities.
        card_df (pd.DataFrame | None): The loaded card data DataFrame to be used for enrichment, or None if not available.
        card_data (bool): Flag indicating whether to merge with card data for enrichment.

    Returns:
        pd.DataFrame: The finalized DataFrame containing matched cards, grouped and summed by name, and enriched with card data if requested.
    """
    # Drop original key columns and rename match to Name
    list_df.drop(columns=["Card number", "Password", "Name"], inplace=True, errors="ignore")
    list_df.rename(columns={"match": "Name"}, inplace=True)

    # Group by all columns except Count and sum counts
    list_df = list_df.groupby(list_df.columns.difference(["Count"]).tolist(), dropna=False).sum().reset_index()

    # Merge with card data if requested
    if card_data and card_df is not None:
        list_df = list_df[list_df.columns.difference(card_df.columns).union(["Name"])].merge(
            card_df.drop_duplicates(subset="Name", keep="first"), on="Name", how="left"
        )

    # Ensure proper dtypes and sorting
    list_df = list_df.convert_dtypes(convert_string=False).sort_values(by=["Name", "Count"], ignore_index=True)
    list_df["Count"] = list_df["Count"].astype(int)

    return list_df


# --- Release Timeline Utilities --- #


def get_releases_by(df, column=None, operation="debut", numeric=False, crosstab=False) -> pd.DataFrame:
    """
    Get release dates grouped by column using a selectable operation among "debut", "last", "first", or "all".
    By default, it returns the debut release date for each group, but it can also return the last release date, the first release date, or all unique release dates for each group.
    If a column is specified, it groups by that column and "Name", otherwise it groups by "Name" alone.
    The function also attempts to convert the grouping column to numeric if numeric is True, and can return a crosstab of release dates by the grouping column if crosstab is True.

    Args:
        df (pd.DataFrame): DataFrame containing card data with a "Release" column and optionally a column to group by.
        column (str, optional): The column name to group by (e.g., "Primary type"). If None, groups by "Name". Defaults to None.
        operation (str, optional): The operation to perform on release dates. Options are "debut" (default), "last", "first", or "all". Defaults to "debut".
        numeric (bool, optional): Whether to attempt to convert the grouping column to numeric. Defaults to False.
        crosstab (bool, optional): Whether to return a crosstab of release dates by the grouping column instead of a DataFrame with release dates. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing release dates grouped by the specified column and operation, or a crosstab if crosstab is True.
    """

    if column is None:
        group_cols = ["Name"]
    else:
        group_cols = [column, "Name"]

    if operation == "all":
        df = df[df["Release"].notna()]
        df = df.explode(column) if column else df
        result = df.groupby(group_cols)["Release"].unique().explode()
    elif operation == "debut":
        df = df.explode(column) if column else df
        result = df.groupby(group_cols)[df.filter(regex="(?i)(debut)").columns].min().min(axis=1)
    elif operation in ["last", "first"]:
        df = df[df["Release"].notna()]
        df = df.explode(column) if column else df
        agg_func = "max" if operation == "last" else "min"
        result = df.groupby(group_cols)["Release"].agg(agg_func)
    else:
        raise ValueError("Invalid operation. Choose from 'debut', 'last', or 'first'.")

    operation = operation.capitalize()
    if operation != "Debut":
        operation = f"{operation} release"
        if operation.startswith("All"):
            operation += "s"

    result = result.rename(operation).reset_index().drop("Name", axis=1).sort_values(by=operation)
    if column is None:
        result = result[operation]
    else:
        numeric = pd.to_numeric(result[column], errors="coerce")
        if len(numeric.dropna()) > 0:
            result[column] = numeric

        if crosstab:
            result = pd.crosstab(result[operation], result[column])

    return result


# --- Data Selectors --- #


def select_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a DataFrame to select cards with a valid "Level" attribute, excluding "Xyz Monster" and "Link Monster" types.

    Args:
        df (pd.DataFrame): DataFrame containing card data with columns such as "Primary type" and either "Level/Rank/Link" or "Level".

    Raises:
        ValueError: If neither "Level/Rank/Link", "Level/Rank" nor "Level" columns are present in the DataFrame.

    Returns:
        pd.DataFrame: Filtered DataFrame with a "Level" column, containing only rows with non-null level values and excluding "Xyz Monster" and "Link Monster" types.
    """
    filter = (df["Primary type"] != "Xyz Monster") & (df["Primary type"] != "Link Monster")
    if "Level/Rank/Link" in df:
        return (
            df[filter]
            .dropna(subset=["Level/Rank/Link"])
            .rename(columns={"Level/Rank/Link": "Level"})
            .dropna(how="all", axis=1)
        )
    elif "Level/Rank" in df:
        return df[filter].dropna(subset=["Level/Rank"]).rename(columns={"Level/Rank": "Level"}).dropna(how="all", axis=1)
    elif "Level" in df:
        return df[filter].dropna(subset=["Level"]).dropna(how="all", axis=1)
    else:
        raise ValueError("No Level, Level/Rank or Level/Rank/Link columns found")


def select_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a DataFrame to select cards with a valid "Rank" attribute, including only "Xyz Monster" types.

    Args:
        df (pd.DataFrame): DataFrame containing card data with columns such as "Primary type" and either "Level/Rank/Link", "Level/Rank" or "Rank".
    Raises:
        ValueError: If neither "Level/Rank/Link", "Level/Rank" nor "Rank" columns are present in the DataFrame.
    Returns:
        pd.DataFrame: Filtered DataFrame with a "Rank" column, containing only rows
    """
    filter = df["Primary type"] == "Xyz Monster"
    if "Level/Rank/Link" in df:
        return (
            df[filter]
            .dropna(subset=["Level/Rank/Link"])
            .rename(columns={"Level/Rank/Link": "Rank"})
            .dropna(how="all", axis=1)
        )
    elif "Level/Rank" in df:
        return df[filter].dropna(subset=["Level/Rank"]).rename(columns={"Level/Rank": "Rank"}).dropna(how="all", axis=1)
    elif "Rank" in df:
        return df[filter].dropna(subset=["Rank"]).dropna(how="all", axis=1)
    else:
        raise ValueError("No Rank, Level/Rank or Level/Rank/Link columns found")


def select_stars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a DataFrame to select cards with a valid "Level" or "Rank" attribute, excluding "Link Monster" type, and renames the column to "Stars".

    Args:
        df (pd.DataFrame): DataFrame containing card data with columns such as "Primary type" and either "Level/Rank/Link", "Level/Rank", "Level", "Rank" or "Stars".

    Raises:
        ValueError: If neither "Level/Rank/Link", "Level/Rank", "Level", "Rank" nor "Stars" columns are present in the DataFrame.

    Returns:
        pd.DataFrame: Filtered DataFrame with a "Stars" column, containing only rows with non-null level values and excluding "Xyz Monster" and "Link Monster" types.
    """
    filter = (df["Primary type"] != "Xyz Monster") & (df["Primary type"] != "Link Monster")
    if "Level/Rank/Link" in df:
        return (
            df[filter]
            .dropna(subset=["Level/Rank/Link"])
            .rename(columns={"Level/Rank/Link": "Stars"})
            .dropna(how="all", axis=1)
        )
    elif "Level/Rank" in df:
        return df[filter].dropna(subset=["Level/Rank"]).rename(columns={"Level/Rank": "Stars"}).dropna(how="all", axis=1)
    elif "Level" in df:
        return df[filter].dropna(subset=["Level"]).rename(columns={"Level": "Stars"}).dropna(how="all", axis=1)
    elif "Rank" in df:
        return df[filter].dropna(subset=["Rank"]).rename(columns={"Rank": "Stars"}).dropna(how="all", axis=1)
    elif "Stars" in df:
        return df[filter].dropna(subset=["Stars"]).dropna(how="all", axis=1)
    else:
        raise ValueError("No Level, Rank, Level/Rank, Level/Rank/Link or Stars columns found")


def select_link(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a DataFrame to select cards with a valid "Link" attribute, including only "Link Monster" types.

    Args:
        df (pd.DataFrame): DataFrame containing card data with columns such as "Primary type" and either "Level/Rank/Link" or "Link".

    Raises:
        ValueError: If neither "Level/Rank/Link" nor "Link" columns are present in the DataFrame.

    Returns:
        pd.DataFrame: Filtered DataFrame with a "Link" column, containing only rows with non-null link values and including only "Link Monster" types.
    """
    filter = df["Primary type"] == "Link Monster"
    if "Level/Rank/Link" in df:
        return (
            df[filter]
            .dropna(subset=["Level/Rank/Link"])
            .rename(columns={"Level/Rank/Link": "Link"})
            .dropna(how="all", axis=1)
        )
    elif "Link" in df:
        return df[filter].dropna(subset=["Level/Rank/Link"]).dropna(how="all", axis=1)
    else:
        raise ValueError("No Link or Level/Rank/Link columns found")


def select_pendulum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a DataFrame to select cards with a valid "Pendulum Scale" attribute, including only "Pendulum Monster" types.

    Args:
        df (pd.DataFrame): DataFrame containing card data with columns such as "Primary type" and "Pendulum Scale".

    Raises:
        ValueError: If "Pendulum Scale" column is not present in the DataFrame.

    Returns:
        pd.DataFrame: Filtered DataFrame with a "Pendulum Scale" column, containing only rows with non-null pendulum scale values and including only "Pendulum Monster" types.
    """
    if "Pendulum Scale" in df:
        return df.dropna(subset=["Pendulum Scale"]).dropna(how="all", axis=1)
    else:
        raise ValueError("No Pendulum Scale column found")


def select_unusable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a DataFrame to select cards that are marked as "Unusable" in the "Card status" column.

    Args:
        df (pd.DataFrame): DataFrame containing card data with a "Card status" column.

    Raises:
        ValueError: If "Card status" column is not present in the DataFrame.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only rows where "Card status" is "Unusable".
    """
    if "Card type" in df:
        return df[
            ~df["Card type"].apply(
                lambda x: any(i in x for i in ["Monster Card", "Spell Card", "Trap Card", "Monster Token", "Counter"])
            )
        ].dropna(how="all", axis=1)
    else:
        raise ValueError("No Card type column found")


def select_token_counter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a DataFrame to select cards that are marked as "Token" or "Counter" in the "Card type" column.

    Args:
        df (pd.DataFrame): DataFrame containing card data with a "Card type" column.

    Raises:
        ValueError: If "Card type" column is not present in the DataFrame.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only rows where "Card type" is "Monster Token" or "Counter".
    """
    if "Card type" in df:
        return df[df["Card type"].apply(lambda x: any(i in x for i in ["Monster Token", "Counter"]))].dropna(
            how="all", axis=1
        )
    else:
        raise ValueError("No Card type column found")
