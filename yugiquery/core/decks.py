# yugiquery/core/decks.py

# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import api
from ..utils import dirs


def read_decklist(file_path: Path | str) -> pd.DataFrame:
    """
    Read a decklist file and return a DataFrame with the card names.

    Args:
        file_path (Path, str): Path to the decklist file.

    Returns:
        (pd.DataFrame): DataFrame with the card names.
    """
    file_path = Path(file_path)

    with open(file_path, "r") as file:
        lines = file.readlines()

    data = []
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.endswith(":"):
            current_section = line[:-1].capitalize()
        elif current_section:
            quantity, card_name = line.split("x ", 1)
            quantity = int(quantity)
            data.append(
                {
                    "Name": card_name,
                    "Count": quantity,
                    "Section": current_section,
                    "Deck": file_path.stem.replace("_", " ").capitalize(),
                }
            )

    df = pd.DataFrame(data).convert_dtypes(convert_string=False)
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
        files = tuple(dirs.DATA.glob("*.txt"))

    for file in files:
        file = Path(file)
        temp_df = read_decklist(file)
        decklist_df = pd.concat([decklist_df, temp_df])
        print(f"Loaded {file.stem} deck.")

    decklist_df.replace({"Section": {"Monster": "Main", "Spell": "Main", "Trap": "Main"}}, inplace=True)
    return decklist_df


def assign_deck(collection_df: pd.DataFrame, deck_df: pd.DataFrame, return_collection: bool = False) -> pd.DataFrame:
    """Match deck cards to collection cards and adjust card counts."""
    result_rows = []

    for _, deck_row in deck_df.iterrows():
        card_name = deck_row["Name"]
        deck_count = deck_row["Count"]
        deck_deck = deck_row["Deck"] if "Deck" in deck_row else np.nan

        collection_sub_df = collection_df[collection_df["Name"] == card_name].copy()

        if "Deck" in collection_sub_df.columns:
            deck_mask = collection_sub_df["Deck"].eq(deck_deck) | collection_sub_df["Deck"].isna()
            collection_sub_df = collection_sub_df[deck_mask]
            collection_sub_df = collection_sub_df.sort_values(by=["Deck"], key=lambda s: s.isna())

        for _, collection_row in collection_sub_df.iterrows():
            if deck_count <= 0:
                break

            available_count = collection_row["Count"]
            subtract_count = min(deck_count, available_count)

            deck_count -= subtract_count
            collection_sub_df.loc[collection_row.name, "Count"] -= subtract_count

            result_row = {
                "Name": card_name,
                "Count": available_count,
                "Deck": deck_deck,
                "missing": np.nan,
            }
            for col in collection_row.index.difference(result_row.keys()):
                result_row[col] = collection_row[col]

            result_rows.append(result_row)

            if collection_sub_df.loc[collection_row.name, "Count"] <= 0:
                collection_sub_df.drop(collection_row.name, inplace=True)

        if deck_count > 0:
            result_row = {"Name": card_name, "Count": 0, "Deck": deck_deck, "missing": deck_count}
            result_rows.append(result_row)

    result_df = pd.DataFrame(result_rows)

    if return_collection:
        result_df = pd.concat(
            [result_df, collection_df[~collection_df["Name"].isin(result_df["Name"])]], ignore_index=True
        ).sort_values(by=["Name", "Deck"])

    result_df["missing"] = result_df["missing"].fillna(0)

    return result_df


def check_limits(deck_df: pd.DataFrame) -> pd.DataFrame:
    """Check decklist limits for each format (e.g. TCG and OCG)."""
    formats = deck_df.filter(like="status").columns.tolist()
    forbidden = (deck_df[formats] == "Forbidden").any(axis=1)
    limited = (deck_df[formats] == "Limited").any(axis=1) & (deck_df["Count"] > 1)
    semi_limited = (deck_df[formats] == "Semi-Limited").any(axis=1) & (deck_df["Count"] > 2)

    melted_df = deck_df[forbidden | limited | semi_limited].melt(
        id_vars=["Name", "Deck", "Section", "Count"],
        value_vars=formats,
        var_name="Format",
        value_name="Status",
    )

    melted_df["Status"] = melted_df["Status"].apply(lambda x: np.nan if x == "Unlimited" else x)
    melted_df = melted_df.dropna(subset=["Status"])
    melted_df = melted_df.assign(Value=melted_df["Format"].str.replace(" status", ""))

    result = melted_df.pivot_table(
        index=["Name", "Deck", "Section", "Count"],
        columns="Status",
        values="Value",
        aggfunc=lambda x: "/".join(sorted(set(x))),
    )
    return result


def get_ygoprodeck() -> pd.DataFrame:
    """
    Fetch the YGOProDeck data from the API or local file.

    Returns:
        pd.DataFrame: A DataFrame of the YGOProDeck data.

    Raises:
        Exception: Exceptions raised by api.fetch_ygoprodeck.
    """
    ygoprodeck_file = dirs.DATA.joinpath("ygoprodeck.json")
    try:
        result = api.fetch_ygoprodeck()
        with open(ygoprodeck_file, "w+") as file:
            json.dump(result, file, indent=4)
        ydk_data = pd.DataFrame(result).set_index("id")
    except Exception as e:
        if ygoprodeck_file.is_file():
            print("Unable to fetch ygoprodeck data. Using local file.")
            print(e)
            ydk_data = pd.read_json(ygoprodeck_file).set_index("id")
        else:
            print("Unable to obtain ygoprodeck data")
            raise

    return ydk_data


def read_ydk(file_path: Path | str) -> pd.DataFrame:
    """
    Read a YDK file and return a DataFrame with the card codes.

    Args:
        file_path (Path | str): Path to the YDK file.

    Returns:
        (pd.DataFrame): DataFrame with the card codes.
    """
    file_path = Path(file_path)
    with open(file_path, "r") as file:
        lines = file.readlines()

    data = []
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line in ["#main", "#extra", "!side"]:
            current_section = line[1:].capitalize()
        elif current_section:
            data.append({"Code": line, "Section": current_section, "Deck": file_path.stem.replace("_", " ").capitalize()})

    df = pd.DataFrame(data).convert_dtypes(convert_string=False)
    return df


def convert_ydk(ydk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a DataFrame with YDK card codes to a DataFrame with card names.

    Args:
        ydk_df (pd.DataFrame): DataFrame with YDK card codes.

    Returns:
        (pd.DataFrame): DataFrame with card names. If unable to obtain the card data, returns input DataFrame.
    """
    try:
        ydk_data = get_ygoprodeck()
    except Exception as e:
        print(e)
        return ydk_df

    ydk_df = ydk_df.copy()

    def get_ydk_card(code) -> Any:
        code = int(code)
        if code not in ydk_data.index:
            return np.nan
        return ydk_data.at[code, "name"]

    ydk_df["Name"] = ydk_df["Code"].apply(get_ydk_card)
    not_found = ydk_df[ydk_df["Name"].isna()]
    if len(not_found) > 0:
        print()
        for deck in not_found["Deck"].unique():
            print(f"Unable to find {len(not_found[not_found['Deck'] == deck])} cards in {deck}:")
            print(" *", "\n * ".join(not_found[not_found["Deck"] == deck]["Code"].astype(str).unique()), "\n")

    ydk_df = ydk_df.drop("Code", axis=1).dropna(subset=["Name"]).reset_index(drop=True)
    ydk_df["Count"] = ydk_df.groupby(["Name", "Section", "Deck"])["Name"].transform("count").astype(int)
    ydk_df = ydk_df.drop_duplicates().reset_index(drop=True)
    return ydk_df


def get_ydk(*files: Path | str) -> pd.DataFrame:
    """
    Load YDK files and return a DataFrame with the card names.

    Args:
        files (Path | str): Paths to YDK files. If not provided, loads all YDK files in the data directory.

    Returns:
        (pd.DataFrame): DataFrame with card names. If unable to obtain the card data, returns raw YDK DataFrame.
    """
    ydk_df = pd.DataFrame()
    if not files:
        files = tuple(dirs.DATA.glob("*.ydk"))
    for file in files:
        file = Path(file)
        temp_df = read_ydk(file)
        ydk_df = pd.concat([ydk_df, temp_df])
        print(f"Loaded {file.stem} deck.")

    if not ydk_df.empty:
        ydk_df = convert_ydk(ydk_df)
    return ydk_df
