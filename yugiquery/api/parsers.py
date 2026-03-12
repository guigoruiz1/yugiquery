# yugiquery/api/parsers.py

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import re
import logging
from typing import Any, Dict, List, Tuple

# --- Imports: Third-Party --- #
import numpy as np
import pandas as pd
import requests
import wikitextparser as wtp


logger = logging.getLogger(__name__)

# --- Arrows Dictionary --- #
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

# --- Formatting Functions --- #


def format_df(input_df: pd.DataFrame, include_all: bool = False) -> pd.DataFrame:
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


def extract_results(response: requests.Response) -> pd.DataFrame:
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


# --- Helper Functions --- #


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


# --- Set List Parsing Utilities --- #


def _process_list_extras(df: pd.DataFrame) -> pd.DataFrame | None:
    """Extract extra parameters from '//' comments in df.

    Parses descriptions like 'description::Token Name' and 'print::value' from the list.

    Args:
        df: The raw parsed list DataFrame containing potential // comments.

    Returns:
        DataFrame with extracted extra parameters indexed by original row, or None if nothing found.
    """
    # Handle extra parameters passed as "// descriptions"
    extra = df.map(lambda x: (x.split("//")[1] if isinstance(x, str) and "//" in x else None)).dropna(how="all")

    if extra.empty:
        return None

    extra = extra.stack().droplevel(1, axis=0).dropna()
    extra_lines = pd.DataFrame()
    for extra_idx, extra_value in extra.items():
        if isinstance(extra_value, str) and "::" in extra_value:
            col, val = extra_value.split("::")
            # Strip and process col and val to extract desired values
            col = col.strip().strip("@").lower()
            val = val.strip().strip("(").strip(")").split("]]")[0].split("[[")[-1]
            extra_lines.loc[extra_idx, col] = val

    extra_lines = extra_lines.dropna(how="all")
    return extra_lines if not extra_lines.empty else None


def _clean_list_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean df by removing '//' comments and normalizing whitespace.

    Args:
        df: The raw parsed list DataFrame.

    Returns:
        Cleaned DataFrame with comments removed and whitespace normalized.
    """
    df = df.map(lambda x: x.split("//")[0] if isinstance(x, str) and "//" in x else x)
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace(
        to_replace=r"^\s*$|^@.*$",
        value=None,
        regex=True,
        inplace=True,
    )
    return df


def _parse_set_list_args(
    template_arguments, rarities: Dict[str, str]
) -> Tuple[Dict[str, Any | None], pd.DataFrame | None, pd.DataFrame | None]:
    """Parse all arguments from a 'set list' template.

    Extracts metadata (region, rarity, print, qty, description, options) and the set list data.

    Args:
        template_arguments: Template arguments to parse.
        rarities: Mapping of rarity codes to full names.

    Returns:
        Tuple of:
        - meta dict with keys: region, rarity, card_print, qty, desc, opt
        - df (DataFrame of parsed set data, or None if not found)
        - extras (DataFrame of extra parameters, or None if not found)
    """
    meta: Dict[str, Any | None] = {
        "region": None,
        "rarity": None,
        "card_print": None,
        "qty": None,
        "desc": None,
        "opt": None,
    }
    df: pd.DataFrame | None = None
    extras: pd.DataFrame | None = None

    for argument in template_arguments:
        if "region=" in argument:
            meta["region"] = argument.value

        elif "rarities=" in argument:
            meta["rarity"] = tuple(
                rarities.get(
                    (
                        i[0].upper() + i[1:]
                        if i[0].islower()
                        else i
                        # Correct lower case acronyms (Example: c->C for common)
                    ).strip(),
                    i.strip(),
                )
                for i in (argument.value).split(",")
            )

        elif "print=" in argument:
            meta["card_print"] = argument.value

        elif "qty=" in argument:
            meta["qty"] = argument.value

        elif "description=" in argument:
            meta["desc"] = argument.value

        elif "options=" in argument:
            meta["opt"] = argument.value

        else:
            # This is the set list data
            set_list = argument.value[1:-1]
            lines = set_list.split("\n")

            df = pd.DataFrame([x.split(";") for x in lines])
            df = df[~df[0].str.contains("!:")]

            # Extract extra parameters before final cleaning
            extras = _process_list_extras(df)
            # Clean df (remove comments, normalize whitespace)
            df = _clean_list_df(df)

    return meta, df, extras


def _build_set_df(
    df: pd.DataFrame,
    extras: pd.DataFrame | None,
    meta: Dict[str, Any | None],
    rarities: Dict[str, str],
    columns: List[str],
) -> pd.DataFrame:
    """Build the final set DataFrame from parsed template data.

    Constructs columns: Name, Card number, Rarity, Print, Quantity, then merges with extras.

    Args:
        df: Cleaned DataFrame with raw set data.
        extras: DataFrame with extracted extra parameters, or None.
        meta: Dict with keys: opt, card_print, qty, rarity, desc.
        rarities: Mapping of rarity codes to full names.
        columns: Column names for the output DataFrame.

    Returns:
        DataFrame with all columns populated according to template logic.
    """
    result = pd.DataFrame(columns=columns)
    opt = meta["opt"]
    card_print = meta["card_print"]
    qty = meta["qty"]
    template_rarity = meta["rarity"]
    desc = meta["desc"]

    noabbr = opt == "noabbr"
    result["Name"] = df[1 - noabbr].apply(lambda x: (x.strip("\u200e").split(" (")[0] if isinstance(x, str) else x))

    if not noabbr and len(df.columns) > 1:
        result["Card number"] = df[0]

    if len(df.columns) > (2 - noabbr):  # and rare in str
        result["Rarity"] = df[2 - noabbr].apply(
            lambda x: (
                tuple([rarities.get(y.strip(), y.strip()) for y in x.split(",")])
                if isinstance(x, str) and "description::" not in x
                else template_rarity
            )
        )
    else:
        result["Rarity"] = pd.Series([template_rarity] * len(result.index), index=result.index)

    if len(df.columns) > (3 - noabbr):
        if card_print is not None:  # and new/reprint in str
            result["Print"] = df[3 - noabbr].apply(lambda x: (card_print if (card_print and x is None) else x))

            if len(df.columns) > (4 - noabbr) and qty:
                result["Quantity"] = df[4 - noabbr].apply(lambda x: x if x is not None else qty)

        elif qty:
            result["Quantity"] = df[3 - noabbr].apply(lambda x: x if x is not None else qty)

    # Handle token name and print in description
    if extras is not None:
        for row in result.index:
            # Handle token name in description
            if "description" in extras and row in extras["description"].dropna().index:
                name_value = result.at[row, "Name"]
                desc_value = extras.at[row, "description"]
                if (
                    isinstance(name_value, str)
                    and isinstance(desc_value, str)
                    and "Token" in name_value
                    and "Token" in desc_value
                ):
                    result.at[row, "Name"] = desc_value

            # Handle print in description
            if "print" in extras and row in extras["print"].dropna().index:
                result.at[row, "Print"] = extras.at[row, "print"]
    else:
        # TODO: Test
        # Use template-level values as fallback
        for row in result.index:
            name_value = result.at[row, "Name"]
            print_value = result.at[row, "Print"]

            # Handle token name from template description
            if isinstance(name_value, str) and isinstance(desc, str):
                if "Token" in name_value and "Token" in desc:
                    result.at[row, "Name"] = desc

            # Handle print from template card_print or description
            if pd.isna(print_value):
                if isinstance(desc, str) and "print" in desc.lower():
                    result.at[row, "Print"] = desc

    return result


def _extract_set_title(parsed_templates, page_name: str) -> str:
    """Extract set title from templates or fallback to page name.

    Searches for 'set page header' template with 'set=' argument.

    Args:
        parsed_templates: List of parsed templates from wikitextparser.
        page_name: Fallback page name if no title found in templates.

    Returns:
        Set title (or derived from page_name if not found).
    """
    title = None
    for template in parsed_templates:
        if template.name.lower() == "set page header":
            for argument in template.arguments:
                if "set=" in argument:
                    title = argument.value
                    break
            if title:
                break

    if not title:
        title = page_name.split("Lists:")[1]

    return title


def process_content(content: Dict, rarities: Dict[str, str], columns: List[str]) -> Tuple[pd.DataFrame | None, int, int]:
    """Process a single page content for set lists.

    Searches for 'set list' templates and builds DataFrames.

    Args:
        content: Page content dict with 'revisions' key.
        rarities: Mapping of rarity codes to full names.
        columns: Column names for output DataFrame.
    Returns:
        Tuple of (combined_df or None, success_count, error_count).
    """
    combined = None
    success = 0
    error = 0

    raw = content["revisions"][0]["*"]
    parsed = wtp.parse(raw)
    page_name = content["title"]

    # Extract set title first
    title = _extract_set_title(parsed.templates, page_name)

    # Process set list templates
    for template in parsed.templates:
        if template.name.lower() == "set list":
            meta, df, extras = _parse_set_list_args(template.arguments, rarities)

            if df is None:
                error += 1
                logger.debug('Error! Unable to parse template for "%s"', page_name)
                continue

            result = _build_set_df(df, extras, meta, rarities, columns)

            result["Set"] = re.sub(pattern=r"\(\w{3}-\w{2}\)\s*$", repl="", string=title).strip()
            result["Region"] = meta["region"].upper() if meta["region"] else None
            result["Page name"] = page_name

            if combined is None:
                combined = result
            else:
                combined = pd.concat([combined, result], ignore_index=True)

            success += 1

    return combined, success, error
