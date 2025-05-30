# yugiquery/utils/api.py

# -*- coding: utf-8 -*-

# =============== #
# API call module #
# =============== #

# ======= #
# Imports #
# ======= #

# Standard library imports
import asyncio
import os
import re
import socket
import time
from types import SimpleNamespace
from typing import (
    Any,
    Dict,
    List,
    Tuple,
)
import urllib.parse as up

# Third-party imports
import aiohttp
import numpy as np
import pandas as pd
import requests
from termcolor import cprint
from tqdm.auto import tqdm, trange
import wikitextparser as wtp


# Local application imports
from .helpers import *
from .dirs import dirs
from ..metadata import __title__, __url__, __version__

# Import Halo according to the environment
if dirs.is_notebook:
    from halo import HaloNotebook as Halo
else:
    from halo import Halo

# ============ #
# Dictionaries #
# ============ #

URLS: SimpleNamespace = SimpleNamespace(
    base="https://yugipedia.com/api.php",
    media="https://ms.yugipedia.com/",
    revisions_action="?action=query&format=json&prop=revisions&rvprop=content&titles=",
    ask_action="?action=ask&format=json&query=",
    askargs_action="?action=askargs&format=json&conditions=",
    categorymembers_action="?action=query&format=json&list=categorymembers&cmdir=desc&cmsort=timestamp&cmtitle=Category:",
    redirects_action="?action=query&format=json&redirects=True&titles=",
    backlinks_action="?action=query&format=json&list=backlinks&blfilterredir=redirects&bltitle=",
    images_action="?action=query&prop=images&format=json&titles=",
    ygoprodeck="https://db.ygoprodeck.com/api/v7/cardinfo.php",
    headers={"User-Agent": f"{__title__} v{__version__} - {__url__}"} | load_json(dirs.get_asset("json", "headers.json")),
)
"""A mapping of yugipedia API URLs with HTTP headers dinamically loaded from the headers.json file in the assets directory.

:meta hide-value:

"""

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
"""A dictionary mapping link arrow positions to their corresponding Unicode characters.

:meta hide-value:

"""


# ========= #
# Functions #
# ========= #


# YGOPRODECK
def fetch_ygoprodeck(misc=True) -> List[Dict[str, Any]]:
    """
    Fetch the card data from ygoprodeck.com.

    Returns:
        (List[Dict[str, Any]]): List of card data.

    Raises:
        requests.exceptions.HTTPError: If an HTTP error occurs while fetching the data.
    """
    ydk_url = URLS.ygoprodeck
    if misc:
        ydk_url += "?misc=yes"
    response = requests.get(ydk_url)
    response.raise_for_status()
    result = response.json()
    return result["data"]


def check_status() -> bool:
    """
    Checks if the API is running and reachable by making a query to retrieve site information. If the API is up and running, returns True. If the API is down or unreachable, returns False and prints an error message with details.

    Returns:
        bool: True if the API is up and running, False otherwise.
    """
    params = {
        "action": "query",
        "meta": "siteinfo",
        "siprop": "general",
        "format": "json",
    }

    try:
        response = requests.get(URLS.base, params=params, headers=URLS.headers)
        response.raise_for_status()
        cprint(text=f"{URLS.base} is up and running {response.json()['query']['general']['generator']}", color="green")
        return True
    except requests.exceptions.RequestException as err:
        cprint(text=f"{URLS.base} is not alive", color="red")
        print(err)
        domain = up.urlparse(URLS.base).netloc
        port = 443

        try:
            socket.create_connection((domain, port), timeout=2)
            cprint(text=f"{domain} is reachable", color="yellow")
        except OSError as err:
            cprint(text=f"{domain} is not reachable", color="red")
            print(err)

        return False


def fetch_categorymembers(
    category: str,
    namespace: int | None = None,
    step: int = 500,
    iterator: tqdm | None = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Fetches members of a category from the API by making iterative requests with a specified step size until all members are retrieved.

    Args:
        category (str): The category to retrieve members for.
        namespace (int| None, optional): The namespace ID to filter the members by. Defaults to None (no namespace).
        step (int, optional): The number of members to retrieve in each request. Defaults to 500.
        iterator (tqdm.std.tqdm | None, optional): A tqdm iterator to display progress updates. Defaults to None.
        debug (bool, optional): If True, prints the URL of each request for debugging purposes. Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing the members of the category.
    """
    debug = check_debug(debug)
    params = {"cmlimit": step, "cmnamespace": namespace}

    lastContinue = {}
    all_results = []
    i = 0
    with Halo(
        text="Fetching category members...",
        spinner="line",
        enabled=("PM_IN_EXECUTION" not in os.environ),
    ) as spinner:
        try:
            while True:
                if iterator is None:
                    spinner.text = f"Fetching category members... Iteration {i+1}"
                else:
                    iterator.set_postfix(it=i + 1)

                params = params.copy()
                params.update(lastContinue)
                response = requests.get(
                    URLS.base + URLS.categorymembers_action + category,
                    params=params,
                    headers=URLS.headers,
                )
                if debug:
                    tqdm.write("\n" + response.url)
                if response.status_code != 200:
                    spinner.fail(f"HTTP error code {response.status_code}")
                    break

                result = response.json()
                if "error" in result:
                    spinner.fail(result["error"]["info"])
                    # raise Exception(result['error']['info'])
                if "warnings" in result:
                    spinner.warn(result["warnings"])
                    # print(result['warnings'])
                if "query" in result:
                    all_results += result["query"]["categorymembers"]
                    if debug:
                        tqdm.write(f"\nIteration {i+1}: {len(result['query']['categorymembers'])} results")
                if "continue" not in result:
                    spinner.succeed("Fetch completed")
                    break
                lastContinue = result["continue"]
                i += 1

            if "PM_IN_EXECUTION" not in os.environ:
                time.sleep(0.5)

        except (KeyboardInterrupt, SystemExit):
            spinner.fail("Execution interrupted.")
            if "PM_IN_EXECUTION" not in os.environ:
                time.sleep(0.5)
            raise

        if dirs.is_notebook:
            spinner.output.close()

    results_df = pd.DataFrame(all_results)
    return results_df


def fetch_properties(
    condition: str,
    query: str,
    step: int = 500,
    limit: int = 5000,
    iterator: tqdm | None = None,
    include_all: bool = False,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Fetches properties from the API by making iterative requests with a specified step size until a specified limit is reached.

    Args:
        condition (str): The query condition to filter the properties by.
        query (str): The query to retrieve the properties.
        step (int, optional): The number of properties to retrieve in each request. Defaults to 500.
        limit (int, optional): The maximum number of properties to retrieve. Defaults to 5000.
        iterator (tqdm.std.tqdm | None, optional): A tqdm iterator to display progress updates. Defaults to None.
        include_all (bool, optional): If True, includes all properties in the DataFrame. If False, includes only properties that have values. Defaults to False.
        debug (bool, optional): If True, prints the URL of each request for debugging purposes. Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing the properties matching the query and condition.
    """
    debug = check_debug(debug)
    df = pd.DataFrame()
    i = 0
    complete = False
    with Halo(
        text="Fetching properties...",
        spinner="line",
        enabled=("PM_IN_EXECUTION" not in os.environ),
    ) as spinner:
        try:
            while not complete:
                if iterator is None:
                    # spinner.clear()
                    spinner.text = f"Fetching properties... Iteration {i+1}"
                else:
                    iterator.set_postfix(it=i + 1)

                response = requests.get(
                    url=URLS.base + URLS.ask_action + condition + query + f"|limit%3D{step}|offset={i*step}|order%3Dasc",
                    headers=URLS.headers,
                )
                if debug:
                    tqdm.write("\n" + response.url)
                if response.status_code != 200:
                    spinner.fail(f"HTTP error code {response.status_code}")
                    break

                result = extract_results(response)
                formatted_df = format_df(input_df=result, include_all=include_all)
                df = pd.concat([df, formatted_df], ignore_index=True, axis=0)

                if debug:
                    tqdm.write(f"\nIteration {i+1}: {len(formatted_df.index)} results")

                if len(formatted_df.index) < step or (i + 1) * step >= limit:
                    spinner.succeed("Fetch completed")
                    complete = True
                else:
                    i += 1

            if "PM_IN_EXECUTION" not in os.environ:
                time.sleep(0.5)

        except (KeyboardInterrupt, SystemExit):
            spinner.fail("Execution interrupted.")
            if "PM_IN_EXECUTION" not in os.environ:
                time.sleep(0.5)
            raise

        if dirs.is_notebook:
            spinner.output.close()

    return df


def fetch_redirects(*titles: str) -> Dict[str, str]:
    """
    Fetches redirects for a list of page titles.

    Args:
        titles (str): Multiple title strings.

    Returns:
        Dict[str, str]: A dictionary mapping source titles to their corresponding redirect targets.
    """
    results = {}
    iterator = trange(np.ceil(len(titles) / 50).astype(int), desc="Redirects", leave=False)
    for i in iterator:
        first = i * 50
        last = (i + 1) * 50
        target_titles = "|".join(titles[first:last])
        response = requests.get(
            url=URLS.base + URLS.redirects_action + target_titles,
            headers=URLS.headers,
        ).json()
        redirects = response["query"]["redirects"]
        for redirect in redirects:
            results[redirect.get("from", "")] = redirect.get("to", "")

    return results


def fetch_backlinks(*titles: str) -> Dict[str, str]:
    """
    Fetches backlinks for a list of page titles.

    Args:
        titles (str): Multiple title strings.

    Returns:
        Dict[str, str]: A dictionary mapping backlink titles to their corresponding target titles.
    """
    results = {}
    iterator = tqdm(titles, dynamic_ncols=(not dirs.is_notebook), desc="Backlinks", leave=False)
    for target_title in iterator:
        iterator.set_postfix(title=target_title)
        response = requests.get(
            url=URLS.base + URLS.backlinks_action + target_title,
            headers=URLS.headers,
        ).json()
        backlinks = response["query"]["backlinks"]
        for backlink in backlinks:
            if re.match(pattern=r"^[a-zA-Z]+$", string=backlink["title"]) and backlink["title"] not in target_title.split(
                " "
            ):
                results[backlink["title"]] = target_title

    return results


# Wrapper for dictionaries
def fetch_redirect_dict(
    codes: str | List[str] = [], names: str | List[str] = [], category: str = "", **kwargs
) -> Dict[str, str]:
    """
    Fetches a dictionary mapping rarity codes to their corresponding names by searching for backlinks and redirects.

    Args:
        names (str | List[str], optional): A list of names, i.e. "Super Rare" to search for a backling.
        codes (str | List[str], optional): A list of codes, i.e. "SR" to search for a redirect.
        category (str, optional): A category to search for backlinks. Defaults to empty.
        **kwargs: Additional keyword arguments to pass to the fetch_categorymembers

    Returns:
        Dict[str, str]: A dictionary mapping codes to their corresponding names.

    """
    if isinstance(codes, str):
        codes = [codes]
    if isinstance(names, str):
        names = [names]
    if category:
        names.extend(fetch_categorymembers(category=category, namespace=0, **kwargs)["title"])

    backlinks = fetch_backlinks(*names)
    redirects = fetch_redirects(*codes)
    return redirects | backlinks


def fetch_set_info(*sets: str, extra_info: List[str] = [], step: int = 15, debug: bool = False) -> pd.DataFrame:
    """
    Fetches information for a list of sets.

    Args:
        sets (str | List[str]): Multiple set names to fetch information for.
        extra_info (List[str], optional): A list of additional information to fetch for each set. Defaults to an empty list.
        step (int, optional): The number of sets to fetch information for at once. Defaults to 15.
        debug (bool, optional): If True, prints debug information. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing information for all sets in the list.

    Raises:
        Any exceptions raised by requests.get().
    """
    debug = check_debug(debug)
    if debug:
        print(f"{len(sets)} sets requested")

    regions_dict = load_json(dirs.get_asset("json", "regions.json"))
    # Info to ask
    info = extra_info + ["Series", "Set type", "Cover card"]
    # Release to ask
    release = [i + " release date" for i in set(regions_dict.values())]
    # Ask list
    ask = up.quote(string="|".join(np.append(info, release)))

    # Get set info
    set_info_df = pd.DataFrame()
    for i in trange(np.ceil(len(sets) / step).astype(int), leave=False):
        first = i * step
        last = (i + 1) * step
        titles = up.quote(string="]]OR[[".join(sets[first:last]))
        response = requests.get(
            url=URLS.base + URLS.askargs_action + titles + f"&printouts={ask}",
            headers=URLS.headers,
        )
        formatted_response = extract_results(response)
        formatted_response.drop(
            "Page name", axis=1, inplace=True
        )  # Page name not needed - no set errata, set name same as page name
        formatted_df = format_df(input_df=formatted_response, include_all=(True if extra_info else True))
        if debug:
            tqdm.write(f"Iteration {i}\n{len(formatted_df)} set properties downloaded - {step-len(formatted_df)} errors")
            tqdm.write("-------------------------------------------------")

        set_info_df = pd.concat([set_info_df, formatted_df.dropna(axis=1, how="all")])

    set_info_df = set_info_df.convert_dtypes()
    set_info_df.sort_index(inplace=True)

    print(f'{"Total:" if debug else ""}{len(set_info_df)} set properties received - {len(sets)-len(set_info_df)} errors')

    return set_info_df


# TODO: Refactor
# TODO: Translate region code?
def fetch_set_lists(
    *titles: str, debug: bool = False
) -> None | Tuple[pd.DataFrame, int, int]:  # Separate formating function
    """
    Fetches card set lists from a list of page titles.

    Args:
        titles (str): Multiple page titles from which to fetch set lists.
        debug (bool, optional): If True, prints debug information. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, int, int]: A DataFrame containing the parsed card set lists, the number of successful requests, and the number of failed requests.
    """
    debug = check_debug(debug)
    if debug:
        print(f"{len(titles)} sets requested")

    titles = up.quote(string="|".join(titles))
    rarity_dict = load_json(dirs.get_asset("json", "rarities.json"))
    set_lists_df = pd.DataFrame(
        columns=[
            "Set",
            "Card number",
            "Name",
            "Rarity",
            "Print",
            "Quantity",
            "Region",
            "Page name",
        ]
    )
    success = 0
    error = 0

    response = requests.get(
        url=URLS.base + URLS.revisions_action + titles,
        headers=URLS.headers,
    )
    if debug:
        print(response.url)
    try:
        json = response.json()
    except:
        print(response.url)
        return

    contents = json["query"]["pages"].values()

    for content in contents:
        if "revisions" in content.keys():
            title = None
            raw = content["revisions"][0]["*"]
            parsed = wtp.parse(raw)
            for template in parsed.templates:
                if template.name.lower() == "set page header":
                    for argument in template.arguments:
                        if "set=" in argument:
                            title = argument.value
                if template.name.lower() == "set list":
                    set_df = pd.DataFrame(columns=set_lists_df.columns)
                    page_name = content["title"]

                    region = None
                    rarity = None
                    card_print = None
                    qty = None
                    desc = None
                    opt = None
                    list_df = None
                    extra_df = None

                    for argument in template.arguments:
                        if "region=" in argument:
                            region = argument.value
                            # if region = 'ES': # Remove second identifier for spanish
                            #     region = 'SP'

                        elif "rarities=" in argument:
                            rarity = tuple(
                                rarity_dict.get(
                                    (
                                        i[0].upper() + i[1:] if i[0].islower() else i
                                    ).strip(),  # Correct lower case accronymns (Example: c->C for common)
                                    i.strip(),
                                )
                                for i in (argument.value).split(",")
                            )

                        elif "print=" in argument:
                            card_print = argument.value

                        elif "qty=" in argument:
                            qty = argument.value

                        elif "description=" in argument:
                            desc = argument.value

                        elif "options=" in argument:
                            opt = argument.value

                        else:
                            set_list = argument.value[1:-1]
                            lines = set_list.split("\n")

                            list_df = pd.DataFrame([x.split(";") for x in lines])
                            list_df = list_df[~list_df[0].str.contains("!:")]

                            # Handle extra parameters passed as "// descriptions"
                            extra = list_df.map(
                                lambda x: (x.split("//")[1] if isinstance(x, str) and "//" in x else None)
                            ).dropna(how="all")
                            if not extra.empty:
                                extra = extra.stack().droplevel(1, axis=0)
                                extra_lines = pd.DataFrame()
                                for extra_idx, extra_value in extra.items():
                                    if "::" in extra_value:
                                        col, val = extra_value.split("::")
                                        # Strip and process col and val to extract desired values
                                        col = col.strip().strip("@").lower()
                                        val = val.strip().strip("(").strip(")").split("]]")[0].split("[[")[-1]
                                        extra_lines.loc[extra_idx, col] = val

                                extra_lines = extra_lines.dropna(how="all")
                                if not extra_lines.empty:
                                    extra_df = extra_lines
                            ###

                            list_df = list_df.map(lambda x: x.split("//")[0] if x is not None else x)

                            list_df = list_df.map(lambda x: x.strip() if x is not None else x)
                            list_df.replace(
                                to_replace=r"^\s*$|^@.*$",
                                value=None,
                                regex=True,
                                inplace=True,
                            )

                    if list_df is None:
                        error += 1
                        if debug:
                            cprint(text=f'Error! Unable to parse template for "{page_name}"', color="red")
                        continue

                    noabbr = opt == "noabbr"
                    set_df["Name"] = list_df[1 - noabbr].apply(
                        lambda x: (x.strip("\u200e").split(" (")[0] if x is not None else x)
                    )

                    if not noabbr and len(list_df.columns > 1):
                        set_df["Card number"] = list_df[0]

                    if len(list_df.columns) > (2 - noabbr):  # and rare in str
                        set_df["Rarity"] = list_df[2 - noabbr].apply(
                            lambda x: (
                                tuple([rarity_dict.get(y.strip(), y.strip()) for y in x.split(",")])
                                if x is not None and "description::" not in x
                                else rarity
                            )
                        )

                    else:
                        set_df["Rarity"] = [rarity for _ in set_df.index]

                    if len(list_df.columns) > (3 - noabbr):
                        if card_print is not None:  # and new/reprint in str
                            set_df["Print"] = list_df[3 - noabbr].apply(
                                lambda x: (card_print if (card_print and x is None) else x)
                            )

                            if len(list_df.columns) > (4 - noabbr) and qty:
                                set_df["Quantity"] = list_df[4 - noabbr].apply(lambda x: x if x is not None else qty)

                        elif qty:
                            set_df["Quantity"] = list_df[3 - noabbr].apply(lambda x: x if x is not None else qty)

                    if not title:
                        title = page_name.split("Lists:")[1]

                    # Handle token name and print in description
                    if extra_df is not None:
                        for row in set_df.index:
                            # Handle token name in description
                            if "description" in extra_df and row in extra_df["description"].dropna().index:
                                if (
                                    set_df.at[row, "Name"] is not None
                                    and "Token" in set_df.at[row, "Name"]
                                    and "Token" in extra_df.at[row, "description"]
                                ):
                                    set_df.at[row, "Name"] = extra_df.at[row, "description"]

                            # Handle print in description
                            if "print" in extra_df and row in extra_df["print"].dropna().index:
                                set_df.at[row, "Print"] = extra_df.at[row, "print"]
                    ###

                    set_df["Set"] = re.sub(pattern=r"\(\w{3}-\w{2}\)\s*$", repl="", string=title).strip()
                    set_df["Region"] = region.upper()
                    set_df["Page name"] = page_name
                    set_lists_df = (
                        pd.concat([set_lists_df, set_df], ignore_index=True).infer_objects(copy=False).fillna(np.nan)
                    )
                    success += 1

        else:
            error += 1
            if debug:
                cprint(text=f"Error! No content for \"{content['title']}\"", color="red")

    if debug:
        print(f"{success} set lists received - {error} missing")
        print("-------------------------------------------------")

    return set_lists_df, success, error


# Images
def fetch_page_images(*titles: str, imlimit: int = 500) -> List[str]:
    """
    Fetches images from the MediaWiki API.

    Args:
        titles (str): Multiple page titles for which to fetch image file names.
        imlimit (int, optional): The maximum number of images to fetch. Defaults to 500.

    Returns:
        pd.Series: A Series containing the image file names.
    """
    titles = up.quote("|".join(titles))
    response = requests.get(
        url=URLS.base + URLS.images_action + titles + f"&imlimit={imlimit}",
        headers=URLS.headers,
    ).json()

    # Extract the pages data from the response
    pages = response["query"]["pages"]

    # Transform the data to remove "File:" prefix and filter out SVG files
    images_list = [
        y["title"].lstrip("File:") for page in pages.values() for y in page.get("images", []) if "svg" not in y["title"]
    ]

    return pd.Series(images_list).drop_duplicates()


# ========== #
# Formatting #
# ========== #


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
            extracted_col = input_df[col].apply(extract_fulltext, multiple=multi)
            # Primary type classification
            if col == "Primary type":
                df[col] = extracted_col.apply(extract_primary_type)
            elif col == "Misc":
                # Rush specific
                df = df.join(extracted_col.apply(extract_misc))
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
            extracted_cols = input_df[col_matches].map(extract_fulltext if extract else lambda x: x)
            if col == " Material":
                df["Materials"] = extracted_cols.apply(lambda x: tuple(elem for tup in col for elem in tup), axis=1)
            else:
                df = df.join(extracted_cols)

    # Category boolean columns for merging into tuple
    category_bool_cols = {
        "Artwork": ".*[aA]rtworks$",
    }
    for col, cat in category_bool_cols.items():
        col_matches = input_df.filter(regex=cat).columns
        if len(col_matches) > 0:
            cat_bool = input_df[col_matches].map(extract_category_bool)
            # Artworks extraction
            if col == "Artwork":
                df[col] = cat_bool.apply(extract_artwork, axis=1)
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
        df = df.join(input_df[input_df.columns.difference(df.columns)].map(extract_fulltext, multiple=True))

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


# Receives a list of dictionaries or strings from an element of a series or detaframe
def extract_fulltext(element: List[Dict[str, Any] | str], multiple: bool = False) -> str | Tuple[str] | float:
    """
    Extracts fulltext from a list of dictionaries or strings.
    If multiple is True, returns a sorted tuple of all fulltexts.
    Otherwise, returns the first fulltext found, with leading/trailing whitespaces removed.
    If the input list is empty, returns np.nan.

    Args:
        element (List[Dict[str, Any] | str]): A list of dictionaries or strings to extract fulltext from.
        multiple (bool, optional): If True, return a tuple of all fulltexts. Otherwise, return the first fulltext. Default is False.

    Returns:
        str or Tuple[str] or np.nan: The extracted fulltext(s).
    """

    def clean_text(text: str) -> str:
        # Regex to remove any substring of the form (* Archetype/Series)
        cleaned = re.sub(r"\(.*\s(?:Archetype|Series)\)", "", text)
        return cleaned.strip("\u200e")

    if len(element) > 0:
        if isinstance(element[0], int):
            return str(element[0])
        elif "fulltext" in element[0]:
            if multiple:
                return tuple(sorted([clean_text(i["fulltext"]) for i in element]))
            else:
                return clean_text(element[0]["fulltext"])
        else:
            if multiple:
                return tuple(sorted([clean_text(i) for i in element]))
            else:
                return clean_text(element[0])
    else:
        return np.nan


# Cards
# Receives a list of strings from an element of a series
def extract_category_bool(element: List[str]) -> float | bool:
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


# Receives a list/tuple of strings or string from an element of a series
def extract_primary_type(element: str | List[str] | Tuple[str]) -> str | List[str]:
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


# Receives a list/tuple of strings or string from an element of a series
def extract_misc(element: str | List[str] | Tuple[str]) -> pd.Series:
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


# Receives a series representing a row in a dataframe
def extract_artwork(row: pd.Series) -> float | Tuple[str]:
    """
    Formats a row in a dataframe that contains "alternate artworks" and "edited artworks" columns.
    If the "alternate artworks" column in a row contain at least one "True" value, adds "Alternate" to the result tuple.
    If the "edited artworks" column in a row contain at least one "True" value, adds "Edited" to the result tuple.
    Returns the resulting tuples.

    Args:
        row (pd.Series): Row in a dataframe that may contain "alternate artworks" and/or "edited artworks" columns.

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


# =========== #
# Downloading #
# =========== #


# TODO: Refactor, move somewhere else
async def download_media(
    *file_names: str,
    output_path: str = "media",
    max_tasks: int = 10,
) -> pd.DataFrame:
    """
    Downloads a set of files given their names and saves them to a specified folder.
    Returns a DataFrame listing file names and URLs that failed to download.

    Args:
        file_names (str): Multiple names of the media files to be downloaded.
        output_path (str, optional): The path to the folder where the downloaded files will be saved. Defaults to "media".
        max_tasks (int, optional): The maximum number of files to download at once. Defaults to 10.

    Returns:
        pandas.DataFrame: A DataFrame with columns "file_name", "url" and "success" for each download.
    """
    # Prepare URLs from file names
    file_names = pd.Series(file_names)
    file_names_md5 = file_names.apply(md5)
    urls = file_names_md5.apply(lambda x: f"/{x[0]}/{x[0]}{x[1]}/") + file_names
    download_results = []

    # Download media from URL
    async def download_file(session, url, save_folder, semaphore, pbar):
        async with semaphore:
            save_name = url.split("/")[-1]
            save_file = Path(save_folder).joinpath(save_name)
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise ValueError(f"URL {url} returned status code {response.status}")
                    total_size = int(response.headers.get("Content-Length", 0))
                    progress = tqdm(
                        unit="B",
                        total=total_size,
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=save_name,
                        leave=False,
                        dynamic_ncols=(not dirs.is_notebook),
                        disable=("PM_IN_EXECUTION" in os.environ),
                    )

                    # Remove existing file if already exists to ensure a fresh download
                    if save_file.is_file():
                        save_file.unlink()

                    # Write downloaded content in chunks
                    with open(save_file, "wb") as f:
                        while True:
                            chunk = await response.content.read(1024)
                            if not chunk:
                                break
                            f.write(chunk)
                            progress.update(len(chunk))
                    progress.close()
                download_results.append({"file_name": save_name, "url": URLS.media + url, "success": True})
            except Exception as e:
                # Cleanup if any error occurs and log the failure
                if save_file.is_file():
                    save_file.unlink()
                download_results.append({"file_name": save_name, "url": URLS.media + url, "success": False})
                tqdm.write(f"Failed to download {save_name}: {e}")
            finally:
                pbar.update()

    # Parallelize file downloads
    semaphore = asyncio.Semaphore(max_tasks)
    async with aiohttp.ClientSession(base_url=URLS.media, headers=URLS.headers) as session:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        with tqdm(
            total=len(urls),
            unit="file",
            dynamic_ncols=(not dirs.is_notebook),
            disable=("PM_IN_EXECUTION" in os.environ),
        ) as pbar:
            tasks = [
                download_file(
                    session=session,
                    url=url,
                    save_folder=output_path,
                    semaphore=semaphore,
                    pbar=pbar,
                )
                for url in urls
            ]
            # Run tasks as they complete to handle failures gracefully
            await asyncio.gather(*tasks, return_exceptions=True)

    # Return failed downloads as a DataFrame
    return pd.DataFrame(download_results)
