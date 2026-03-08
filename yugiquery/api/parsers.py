# yugiquery/api/parsers.py

# -*- coding: utf-8 -*-

# ======= #
# Imports #
# ======= #

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests


arrows_dict: Dict[str, str] = {
    "Middle-Left": "←",
    "Middle-Right": "→",
    "Top-Left": "↖",
    "Top-Center": "↑",
    "Top-Right": "↗",
    "Bottom-Left": "↙",
    "Bottom-Center": "↓",
    "Bottom-Right": "↘",
}


# ========== #
# Formatting #
# ========== #


def _format_df(input_df: pd.DataFrame, include_all: bool = False) -> pd.DataFrame:
    """
    Formats a dataframe containing card information.
    Returns a new dataframe with specific columns extracted and processed.

    Args:
        input_df (pd.DataFrame): The input dataframe to format.
        include_all (bool, optional): If True, include all unspecified columns in the output dataframe. Default is False.

    Returns:
        pd.DataFrame: The formatted dataframe.
    """
    df = pd.DataFrame(index=input_df.index)

    # Column name: multiple values
    individual_cols = {
        "Name": False,
        "Password": False,
        "Card type": False,
        "Property": False,
        "Card image": False,
        "Archseries": True,
        "Misc": True,
        "Category": True,
        "Summoning": True,
        # Monster card specific columns
        "Attribute": False,
        "Primary type": True,
        "Secondary type": True,
        "Monster type": False,
        "Effect type": True,
        "DEF": False,
        "Pendulum Scale": False,
        "Link": False,
        "Rank": False,
        # Skill card specific columns
        "Character": False,
        # Rush duel specific columns
        # Set specific columns
        "Set": False,
        "Card number": False,
        "Series": False,
        "Set type": False,
        "Cover card": True,
        # Bandai specific columns
        "Ability": False,
        "Rule": False,
    }
    for col, multi in individual_cols.items():
        if col in input_df.columns:
            extracted_col = input_df[col].apply(_extract_fulltext, multiple=multi)
            # Primary type classification
            if col == "Primary type":
                df[col] = extracted_col.apply(_extract_primary_type)
            elif col == "Misc":
                # Rush specific
                df = df.join(extracted_col.apply(_extract_misc))
            else:
                df[col] = extracted_col

    # Link arrows styling
    if "Link Arrows" in input_df.columns:
        df["Link Arrows"] = input_df["Link Arrows"].apply(
            lambda x: (tuple([arrows_dict[i] for i in sorted(x)]) if len(x) > 0 else np.nan)
        )

    # Columns with matching name pattern: extraction function
    filter_cols = {
        "ATK": True,
        "Level": True,
        " status": True,
        " Material": True,
        "Page ": False,
    }
    for col, extract in filter_cols.items():
        col_matches = input_df.filter(like=col).columns
        if len(col_matches) > 0:
            extracted_cols = input_df[col_matches].map(_extract_fulltext if extract else lambda x: x)
            if col == " Material":
                df["Materials"] = extracted_cols.apply(lambda x: tuple(elem for tup in x for elem in tup), axis=1)
            else:
                df = df.join(extracted_cols)

    # Category boolean columns for merging into tuple
    category_bool_cols = {
        "Artwork": ".*[aA]rtworks$",
    }
    for col, cat in category_bool_cols.items():
        col_matches = input_df.filter(regex=cat).columns
        if len(col_matches) > 0:
            cat_bool = input_df[col_matches].map(_extract_category_bool)
            # Artworks extraction
            if col == "Artwork":
                df[col] = cat_bool.apply(_extract_artwork, axis=1)
            else:
                df[col] = cat_bool

    # Date columns concatenation
    if len(input_df.filter(regex="(?i)(date|time|release|debut)").columns) > 0:
        df = df.join(
            input_df.filter(regex="(?i)(date|time|release|debut)").map(
                lambda x: (
                    pd.to_datetime(pd.to_numeric(x[0]["timestamp"]), unit="s", errors="coerce") if len(x) > 0 else np.nan
                )
            )
        )

    # Include other unspecified columns
    if include_all:
        df = df.join(input_df[input_df.columns.difference(df.columns)].map(_extract_fulltext, multiple=True))

    return df


def _extract_results(response: requests.Response) -> pd.DataFrame:
    """
    Extracts the relevant data from the response object and returns it as a Pandas DataFrame.

    Args:
        response (requests.Response): The response object obtained from making a GET request to the Yu-Gi-Oh! Wiki API.

    Returns:
        pd.DataFrame: A DataFrame containing the relevant data extracted from the response object.
    """
    json = response.json()
    df = pd.DataFrame(json["query"]["results"]).transpose()
    if "printouts" in df:
        df = pd.DataFrame(df["printouts"].values.tolist(), index=df["printouts"].keys())
        page_url = pd.DataFrame(json["query"]["results"]).transpose()["fullurl"].rename("Page URL")
        page_name = (
            pd.DataFrame(json["query"]["results"]).transpose()["fulltext"].rename("Page name")
        )  # Not necessarily same as card name (Used to merge errata)
        df = pd.concat([df, page_name, page_url], axis=1)

    return df


def _extract_fulltext(element: List[Dict[str, Any] | str], multiple: bool = False) -> str | Tuple[str, ...] | float:
    """
    Extracts fulltext from a list of dictionaries or strings.
    If multiple is True, returns a sorted tuple of all fulltexts.
    Otherwise, returns the first fulltext found, with leading/trailing whitespaces removed.
    If the input list is empty, returns np.nan.

    Args:
        element (List[Dict[str, Any] | str]): A list of dictionaries or strings to extract fulltext from.
        multiple (bool, optional): If True, return a tuple of all fulltexts. Otherwise, return the first fulltext. Default is False.

    Returns:
        str or Tuple[str, ...] or np.nan: The extracted fulltext(s).
    """

    def clean_text(text: str) -> str:
        # Regex to remove any substring of the form (* Archetype/Series)
        cleaned = re.sub(r"\(.*\s(?:Archetype|Series)\)", "", text)
        return cleaned.strip("\u200e")

    if len(element) > 0:
        if isinstance(element[0], int):
            return str(element[0])
        elif (
            isinstance(element[0], dict) and "fulltext" in element[0]
        ):  # If one is, all are expected to be dicts with "fulltext" key
            if multiple:
                return tuple(sorted([clean_text(str(i["fulltext"])) for i in element]))  # type: ignore
            else:
                return clean_text(element[0]["fulltext"])
        else:
            if multiple:
                return tuple(sorted([clean_text(str(i)) for i in element]))
            else:
                return clean_text(str(element[0]))
    else:
        return np.nan


def _extract_category_bool(element: List[str]) -> float | bool:
    """
    Extracts a boolean value from a list of strings that represent a boolean value.
    If the first string in the list is "t", returns True.
    If the first string in the list is "f", returns False.
    Otherwise, returns np.nan.

    Args:
        element (List[str]): The list of strings to extract a boolean value from.

    Returns:
        bool | np.nan: The extracted boolean value.
    """
    if len(element) > 0:
        if element[0] == "f":
            return False
        elif element[0] == "t":
            return True

    return np.nan


def _extract_primary_type(element: str | List[str] | Tuple[str]) -> str | List[str]:
    """
    Extracts the primary type of a card.
    If the input is a list or tuple, removes "Pendulum Monster" and "Maximum Monster" from the list.
    If the input is a list or tuple with only one element, returns that element.
    If the input is a list or tuple with multiple elements, returns the first element that is not "Effect Monster".
    Otherwise, returns the input.

    Args:
        element (str | List[str] | Tuple[str]): The type(s) to extract the primary type from.

    Returns:
        str | List[str]: The extracted primary type(s).
    """
    if isinstance(element, list) or isinstance(element, tuple):
        if "Monster Token" in element:
            return "Monster Token"
        else:
            element = [z for z in element if (z != "Pendulum Monster") and (z != "Maximum Monster")]
            if len(element) == 1 and "Effect Monster" in element:
                return "Effect Monster"
            elif len(element) > 0:
                return [z for z in element if z != "Effect Monster"][0]
            else:
                return "???"

    return element


def _extract_misc(element: str | List[str] | Tuple[str]) -> pd.Series:
    """
    Extracts the misc properties of a card.
    Checks whether the input contains the values "Legend Card" or "Requires Maximum Mode" and creates a boolean table.

    Args:
        element (str | List[str] | Tuple[str]): The Misc values to generate the boolean table from.

    Returns:
        pd.Series: A pandas Series of boolean values indicating whether "Legend Card" and "Requires Maximum Mode" are present in the input.
    """
    if isinstance(element, list) or isinstance(element, tuple):
        return pd.Series(
            [val in element for val in ["Legend Card", "Requires Maximum Mode"]],
            index=["Legend", "Maximum mode"],
        )
    else:
        return pd.Series([False, False], index=["Legend", "Maximum mode"])


def _extract_artwork(row: pd.Series) -> float | Tuple[str, ...]:
    """
    Formats a row in a dataframe that contains "alternate artworks" and "edited artworks" columns.
    If the "alternate artworks" column in a row contain at least one "True" value, adds "Alternate" to the result tuple.
    If the "edited artworks" column in a row contain at least one "True" value, adds "Edited" to the result tuple.
    Returns the resulting tuples.

    Args:
        row (pd.Series): Row in a dataframe that may contain "alternate artworks" and/or "edited artworks" columns.

    Returns:
        Tuple[str, ...]: The formatted row as a tuple.
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
