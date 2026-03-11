# yugiquery/core/data.py

# -*- coding: utf-8 -*-

import logging
import os
from ast import literal_eval
from pathlib import Path
from typing import List, Literal, Tuple, overload

import arrow
import numpy as np
import pandas as pd

from ..utils import dirs, load_json
from .decks import get_ygoprodeck

logger = logging.getLogger(__name__)


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
        if return_ts:
            ts = arrow.get(Path(files[0]).stem.split("_")[-1])
            return df, ts
        return df

    logger.warning('No file matching pattern "%s" found.', name_pattern)
    if return_ts:
        return None, None
    return None


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
    return collection_df


def find_cards(list_df: pd.DataFrame | pd.DataFrame, card_data: bool = False, set_data: bool = False) -> pd.DataFrame:
    """Match a card list against latest datasets and optionally enrich with card data."""
    if list_df.empty:
        logger.warning("List empty. Ignoring.")
        return list_df

    card_df = None
    set_lists_df = None

    if card_data or any(col in list_df and not list_df[col].dropna().empty for col in ["Name", "Password"]):
        card_df = load_latest_data(name_pattern="cards")
        if card_df is not None:
            card_df.sort_values(by=["Name", "Primary type", "Property"], ignore_index=True, inplace=True)
    if "Card number" in list_df and not list_df["Card number"].dropna().empty:
        set_lists_df = load_latest_data(name_pattern="sets")
        if set_lists_df is not None:
            set_lists_df = (
                set_lists_df.sort_values(by="Release")
                .drop_duplicates(subset="Card number", keep="first")
                .dropna(subset=["Card number"])
            )

    if card_df is None and set_lists_df is None:
        raise FileNotFoundError("No card or set lists data files found.")

    logger.info("Finding cards in database...")

    def merge_set_data(df: pd.DataFrame) -> pd.DataFrame:
        if set_lists_df is None:
            return df
        df["Card number"] = df["Card number"].str.upper()
        extra_cols = set_lists_df.columns.difference(df.columns).union(["Card number", "Name"])
        merged_df = df.merge(set_lists_df[extra_cols], on="Card number", how="left")
        merged_df["match"] = merged_df["Name_y"] if "Name_y" in merged_df else merged_df["Name"]
        merged_df.rename({"Name_x": "Name"}, axis=1, inplace=True, errors="ignore")
        merged_df.drop(columns=["Name_y"], inplace=True, errors="ignore")
        missing = (
            merged_df["Card number"][~merged_df["Card number"].isin(set_lists_df["Card number"])]
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
        return merged_df

    def merge_with_keys(df: pd.DataFrame, key_col: str, ref_df: pd.DataFrame, ref_key: str, ref_val: str) -> pd.DataFrame:
        if ref_df is None:
            return df
        keys = df[df["match"].isna()][key_col].dropna()
        ref_df = ref_df.dropna(subset=ref_key)
        list_keys = ref_df[ref_key]
        if key_col != "Password":
            keys = keys.str.lower().str.strip()
            list_keys = list_keys.str.lower().str.strip()
        else:
            keys = keys.astype(int)
            list_keys = list_keys.astype(int)

        key_name_dict = dict(zip(list_keys, ref_df[ref_val]))
        missing = df.loc[keys[~keys.isin(list_keys)].index, key_col].sort_values().unique().astype(str)
        if len(missing) > 0:
            logger.warning(
                'Unable to find the following %d card(s) by "%s":\n * %s',
                len(missing),
                ref_key,
                "\n * ".join(missing),
            )
        matches = keys.map(key_name_dict.get)
        df.loc[matches.index, "match"] = matches
        return df

    original_cols = list_df.columns
    list_df = list_df.dropna(how="all").assign(match=np.nan).astype(object)
    if "Card number" in original_cols and not list_df["Card number"].dropna().empty and set_lists_df is not None:
        list_df = (
            merge_set_data(list_df)
            if set_data
            else merge_with_keys(
                df=list_df,
                key_col="Card number",
                ref_df=set_lists_df,
                ref_key="Card number",
                ref_val="Name",
            )
        )

    if "Password" in original_cols and not list_df["match"].notna().all() and card_df is not None:
        list_df = merge_with_keys(df=list_df, key_col="Password", ref_df=card_df, ref_key="Password", ref_val="Name")

    if "Name" in original_cols and not list_df["match"].notna().all() and (card_df is not None or set_lists_df is not None):
        ref_df = card_df if card_df is not None else set_lists_df

        if ref_df is not None:
            list_df = merge_with_keys(df=list_df, key_col="Name", ref_df=ref_df, ref_key="Name", ref_val="Name")

        if list_df["match"].isna().any():
            try:
                ydk_data = get_ygoprodeck()
                ydk_data["Old name"] = ydk_data["misc_info"].apply(
                    lambda x: tuple(y["beta_name"] for y in x if "beta_name" in y)
                )
                list_df = merge_with_keys(
                    df=list_df,
                    key_col="Name",
                    ref_df=ydk_data.explode("Old name"),
                    ref_key="Old name",
                    ref_val="name",
                )
            except Exception as e:
                logger.warning("Unable to get old names from ygoprodeck: %s", e)

    list_df.drop(columns=["Card number", "Password", "Name"], inplace=True, errors="ignore")
    list_df.rename(columns={"match": "Name"}, inplace=True)
    list_df = list_df.groupby(list_df.columns.difference(["Count"]).tolist(), dropna=False).sum().reset_index()

    if card_data and card_df is not None:
        list_df = list_df[list_df.columns.difference(card_df.columns).union(["Name"])].merge(
            card_df.drop_duplicates(subset="Name", keep="first"), on="Name", how="left"
        )

    list_df = list_df.convert_dtypes(convert_string=False).sort_values(by=["Name", "Count"], ignore_index=True)
    list_df["Count"] = list_df["Count"].astype(int)
    logger.info("%d out of %d cards found.", list_df[list_df["Name"].notna()]["Count"].sum(), list_df["Count"].sum())

    return list_df


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
