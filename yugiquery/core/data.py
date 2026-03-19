# yugiquery/core/data.py

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import os
from ast import literal_eval
from pathlib import Path
from typing import List, Literal, Tuple, overload

# --- Imports: Third-Party --- #
import arrow
import numpy as np
import pandas as pd

# --- Imports: Local Application --- #
from .. import api
from ..utils import dirs, load_json, LoggerConfig

# --- Logger Setup --- #
logger = LoggerConfig.get_logger()

# --- Data File Loading and Normalization --- #


@overload
def load_latest_data(
    name_pattern: str,
    tuple_cols: List[str] = ...,
    return_ts: Literal[False] = False,
) -> pd.DataFrame | None: ...


@overload
def load_latest_data(
    name_pattern: str,
    tuple_cols: List[str] = ...,
    return_ts: Literal[True] = True,
) -> Tuple[pd.DataFrame | None, arrow.Arrow | None]: ...


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
    return_ts: bool = False,
) -> pd.DataFrame | None | Tuple[pd.DataFrame | None, arrow.Arrow | None]:
    """
    Loads the most recent data file matching the specified name pattern and applies corrections.
    """
    name_pattern = name_pattern.lower()
    files = sorted(
        list(dirs.DATA.glob(f"{name_pattern}_data_*.bz2")),
        key=os.path.getctime,
        reverse=True,
    )

    if files:
        df = pd.read_csv(files[0], dtype=object, keep_default_na=False, na_values="")
        for col in tuple_cols:
            if col in df:
                try:
                    df[col] = df[col].dropna().apply(literal_eval)
                except (ValueError, SyntaxError):
                    pass

        for col in df.filter(regex="(?i)(date|time|release|debut)").columns:
            df[col] = pd.to_datetime(df[col])

        logger.info("%s file loaded.", name_pattern.capitalize())
        print(f"{name_pattern.capitalize()} file loaded.")
        if return_ts:
            ts = arrow.get(Path(files[0]).stem.split("_")[-1])
            return df, ts
        return df

    logger.warning('No file matching pattern "%s" found.', name_pattern)
    if return_ts:
        return None, None
    return None


# --- Set and Card Dataset Merges --- #


def merge_set_info(input_df: pd.DataFrame, input_info_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge set metadata into set list data by set and region.
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
    print("Set properties merged")
    return merged_df


def merge_set_to_cards(*card_df, set_df) -> pd.DataFrame:
    """
    Merge set lists into one or more card information dataframes.
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
    """Merge errata information into the input DataFrame by card name."""
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
    """Load a user collection from CSV or Excel."""
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
    """Match a card list against latest datasets and optionally enrich with card data."""
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
    """Load card data if needed."""
    if card_data or any(col in list_df and not list_df[col].dropna().empty for col in ["Name", "Password"]):
        card_df = load_latest_data(name_pattern="cards")
        if card_df is not None:
            card_df.sort_values(by=["Name", "Primary type", "Property"], ignore_index=True, inplace=True)
        return card_df
    return None


def _load_set_data(list_df: pd.DataFrame) -> pd.DataFrame | None:
    """Load set list data if card numbers are present."""
    if "Card number" in list_df and not list_df["Card number"].dropna().empty:
        set_lists_df = load_latest_data(name_pattern="sets")
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
    """Match cards by card number from set data."""
    if "Card number" not in original_cols or set_lists_df is None:
        return

    if list_df["Card number"].dropna().empty:
        return

    if set_data:
        _merge_set_data(list_df, set_lists_df)
    else:
        _merge_with_keys(list_df, "Card number", set_lists_df, "Card number", "Name")


def _merge_set_data(df: pd.DataFrame, set_lists_df: pd.DataFrame | None) -> None:
    """Merge set data by card number when set_data flag is true. Updates df in place."""
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
    """Match remaining unmatched cards by password."""
    if "Password" not in original_cols or card_df is None or list_df["match"].notna().all():
        return

    _merge_with_keys(list_df, "Password", card_df, "Password", "Name")


def _match_cards_by_name(
    list_df: pd.DataFrame,
    original_cols: pd.Index,
    card_df: pd.DataFrame | None,
    set_lists_df: pd.DataFrame | None,
) -> None:
    """Match remaining unmatched cards by name, including old card names."""
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
    """Merge cards by key column (Card number, Password, or Name) against reference dataframe."""
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
    """Clean up and finalize the matched cards dataframe."""
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
    """Get release dates grouped by column using a selectable operation."""

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
        ValueError: If neither "Level/Rank/Link" nor "Level" columns are present in the DataFrame.

    Returns:
        pd.DataFrame: Filtered DataFrame with a "Level" column, containing only rows with non-null level values and excluding "Xyz Monster" and "Link Monster" types.
    """
    filter = (df["Primary type"] != "Xyz Monster") & (df["Primary type"] != "Link Monster")
    if "Level/Rank/Link" in df:
        return df[filter].dropna(subset=["Level/Rank/Link"]).rename(columns={"Level/Rank/Link": "Level"})
    elif "Level" in df:
        return df[filter].dropna(subset=["Level/Rank/Link"])
    else:
        raise ValueError("No Level or Level/Rank/Link columns found")


def select_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a DataFrame to select cards with a valid "Rank" attribute, including only "Xyz Monster" types.

    Args:
        df (pd.DataFrame): DataFrame containing card data with columns such as "Primary type" and either "Level/Rank/Link" or "Rank".
    Raises:
        ValueError: If neither "Level/Rank/Link" nor "Rank" columns are present in the DataFrame.
    Returns:
        pd.DataFrame: Filtered DataFrame with a "Rank" column, containing only rows
    """
    filter = df["Primary type"] == "Xyz Monster"
    if "Level/Rank/Link" in df:
        return df[filter].dropna(subset=["Level/Rank/Link"]).rename(columns={"Level/Rank/Link": "Rank"})
    elif "Rank" in df:
        return df[filter].dropna(subset=["Level/Rank/Link"])
    else:
        raise ValueError("No Rank or Level/Rank/Link columns found")


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
        return df[filter].dropna(subset=["Level/Rank/Link"]).rename(columns={"Level/Rank/Link": "Link"})
    elif "Link" in df:
        return df[filter].dropna(subset=["Level/Rank/Link"])
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
        return df.dropna(subset=["Pendulum Scale"])
    else:
        raise ValueError("No Pendulum Scale column found")
