# yugiquery/api/client.py

# -*- coding: utf-8 -*-

# =============== #
# API call module #
# =============== #

# ======= #
# Imports #
# ======= #

# Standard library imports
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
import numpy as np
import pandas as pd
import requests
from termcolor import cprint
from tqdm.auto import tqdm, trange
import wikitextparser as wtp

# Local application imports
from .. import utils
from ..metadata import __title__, __url__, __version__
from .parsers import _extract_results, _format_df

# Import Halo according to the environment
if utils.dirs.is_notebook:
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
    pageimages_action="?action=query&format=json&prop=pageimages&piprop=original&titles=",
    images_action="?action=query&format=json&prop=images&titles=",
    ygoprodeck="https://db.ygoprodeck.com/api/v7/cardinfo.php",
    headers={"User-Agent": f"{__title__} v{__version__} - {__url__}"}
    | utils.load_json(utils.dirs.get_asset("json", "headers.json")),
)
"""A mapping of yugipedia API URLs with HTTP headers dinamically loaded from the headers.json file in the assets directory.

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
    debug = utils.check_debug(debug)
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
    debug = utils.check_debug(debug)
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

                result = _extract_results(response)
                formatted_df = _format_df(input_df=result, include_all=include_all)
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
    iterator = tqdm(titles, dynamic_ncols=(not utils.dirs.is_notebook), desc="Backlinks", leave=False)
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
    debug = utils.check_debug(debug)
    if debug:
        print(f"{len(sets)} sets requested")

    regions_dict = utils.load_json(utils.dirs.get_asset("json", "regions.json"))
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
        formatted_response = _extract_results(response)
        formatted_response.drop(
            "Page name", axis=1, inplace=True
        )  # Page name not needed - no set errata, set name same as page name
        formatted_df = _format_df(input_df=formatted_response, include_all=(True if extra_info else False))
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
    debug = utils.check_debug(debug)
    if debug:
        print(f"{len(titles)} sets requested")

    titles_str = up.quote(string="|".join(titles))
    rarity_dict = utils.load_json(utils.dirs.get_asset("json", "rarities.json"))
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
        url=URLS.base + URLS.revisions_action + titles_str,
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
                                        i[0].upper() + i[1:]
                                        if i[0].islower()
                                        else i
                                        # Correct lower case accronymns (Example: c->C for common)
                                    ).strip(),
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
                                if not extra_lines.empty:
                                    extra_df = extra_lines
                            ###

                            list_df = list_df.map(lambda x: x.split("//")[0] if isinstance(x, str) and "//" in x else x)
                            list_df = list_df.map(lambda x: x.strip() if isinstance(x, str) else x)
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
                        lambda x: (x.strip("\u200e").split(" (")[0] if isinstance(x, str) else x)
                    )

                    if not noabbr and len(list_df.columns) > 1:
                        set_df["Card number"] = list_df[0]

                    if len(list_df.columns) > (2 - noabbr):  # and rare in str
                        set_df["Rarity"] = list_df[2 - noabbr].apply(
                            lambda x: (
                                tuple([rarity_dict.get(y.strip(), y.strip()) for y in x.split(",")])
                                if isinstance(x, str) and "description::" not in x
                                else rarity
                            )
                        )

                    else:
                        set_df["Rarity"] = pd.Series([rarity] * len(set_df.index), index=set_df.index)

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
                                name_value = set_df.at[row, "Name"]
                                desc_value = extra_df.at[row, "description"]
                                if (
                                    isinstance(name_value, str)
                                    and isinstance(desc_value, str)
                                    and "Token" in name_value
                                    and "Token" in desc_value
                                ):
                                    set_df.at[row, "Name"] = desc_value

                            # Handle print in description
                            if "print" in extra_df and row in extra_df["print"].dropna().index:
                                set_df.at[row, "Print"] = extra_df.at[row, "print"]
                    else:
                        # TODO: Test
                        # Use template-level values as fallback
                        for row in set_df.index:
                            name_value = set_df.at[row, "Name"]
                            print_value = set_df.at[row, "Print"]

                            # Handle token name from template description
                            if isinstance(name_value, str) and isinstance(desc, str):
                                if "Token" in name_value and "Token" in desc:
                                    set_df.at[row, "Name"] = desc

                            # Handle print from template card_print or description
                            if pd.isna(print_value):
                                if isinstance(desc, str) and "print" in desc.lower():
                                    set_df.at[row, "Print"] = desc
                    ###

                    set_df["Set"] = re.sub(pattern=r"\(\w{3}-\w{2}\)\s*$", repl="", string=title).strip()
                    set_df["Region"] = region.upper() if region else None
                    set_df["Page name"] = page_name
                    set_lists_df = pd.concat([set_lists_df, set_df], ignore_index=True).infer_objects().fillna(np.nan)
                    success += 1

        else:
            error += 1
            if debug:
                cprint(text=f"Error! No content for \"{content['title']}\"", color="red")

    if debug:
        print(f"{success} set lists received - {error} missing")
        print("-------------------------------------------------")

    return set_lists_df, success, error
