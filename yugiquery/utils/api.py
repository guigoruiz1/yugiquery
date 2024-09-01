# yugiquery/utils/api.py

# -*- coding: utf-8 -*-

# =============== #
# API call module #
# =============== #

import aiohttp
import asyncio
import requests
import socket
import time
import numpy as np
import pandas as pd
import urllib.parse as up
import wikitextparser as wtp
from .helpers import *

# Variables
http_headers = {"User-Agent": "Yugiquery/1.0 - https://guigoruiz1.github.io/yugiquery/"}
base_url = "https://yugipedia.com/api.php"
media_url = "https://ws.yugipedia.com/"
revisions_query_action = (
    "?action=query&format=json&prop=revisions&rvprop=content&titles="
)

ask_query_action = "?action=ask&format=json&query="
askargs_query_action = "?action=askargs&format=json&conditions="
categorymembers_query_action = "?action=query&format=json&list=categorymembers&cmdir=desc&cmsort=timestamp&cmtitle=Category:"
redirects_query_action = "?action=query&format=json&redirects=True&titles="
backlinks_query_action = (
    "?action=query&format=json&list=backlinks&blfilterredir=redirects&bltitle="
)


arrows_dict = {
    "Middle-Left": "←",
    "Middle-Right": "→",
    "Top-Left": "↖",
    "Top-Center": "↑",
    "Top-Right": "↗",
    "Bottom-Left": "↙",
    "Bottom-Center": "↓",
    "Bottom-Right": "↘",
}


# Methods
def check_status():
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
        response = requests.get(base_url, params=params, headers=http_headers)
        response.raise_for_status()
        print(
            f"{base_url} is up and running {response.json()['query']['general']['generator']}"
        )
        return True
    except requests.exceptions.RequestException as err:
        print(f"{base_url} is not alive: {err}")
        domain = up.urlparse(base_url).netloc
        port = 443

        try:
            socket.create_connection((domain, port), timeout=2)
            print(f"{domain} is reachable")
        except OSError as err:
            print(f"{domain} is not reachable: {err}")

        return False


def fetch_categorymembers(
    category: str,
    namespace: int = 0,
    step: int = 500,
    iterator=None,
    debug: bool = False,
):
    """
    Fetches members of a category from the API by making iterative requests with a specified step size until all members are retrieved.

    Args:
        category (str): The category to retrieve members for.
        namespace (int, optional): The namespace ID to filter the members by. Defaults to 0 (main namespace).
        step (int, optional): The number of members to retrieve in each request. Defaults to 500.
        iterator (tqdm.std.tqdm, optional): A tqdm iterator to display progress updates. Defaults to None.
        debug (bool, optional): If True, prints the URL of each request for debugging purposes. Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing the members of the category.
    """
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
                    f"{base_url}{categorymembers_query_action}{category}",
                    params=params,
                    headers=http_headers,
                )
                if debug:
                    print(response.url)
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

        spinner.output.close()

    results_df = pd.DataFrame(all_results)
    return results_df


def fetch_properties(
    condition: str,
    query: str,
    step: int = 500,
    limit: int = 5000,
    iterator=None,
    include_all: bool = False,
    debug: bool = False,
):
    """
    Fetches properties from the API by making iterative requests with a specified step size until a specified limit is reached.

    Args:
        condition (str): The query condition to filter the properties by.
        query (str): The query to retrieve the properties.
        step (int, optional): The number of properties to retrieve in each request. Defaults to 500.
        limit (int, optional): The maximum number of properties to retrieve. Defaults to 5000.
        iterator (tqdm.std.tqdm, optional): A tqdm iterator to display progress updates. Defaults to None.
        include_all (bool, optional): If True, includes all properties in the DataFrame. If False, includes only properties that have values. Defaults to False.
        debug (bool, optional): If True, prints the URL of each request for debugging purposes. Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing the properties matching the query and condition.
    """
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
                    f"{base_url}{ask_query_action}{condition}{query}|limit%3D{step}|offset={i*step}|order%3Dasc",
                    headers=http_headers,
                )
                if debug:
                    print(response.url)
                if response.status_code != 200:
                    spinner.fail(f"HTTP error code {response.status_code}")
                    break

                result = extract_results(response)
                formatted_df = format_df(result, include_all=include_all)
                df = pd.concat([df, formatted_df], ignore_index=True, axis=0)

                if debug:
                    tqdm.write(f"Iteration {i+1}: {len(formatted_df.index)} results")

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

        spinner.output.close()

    return df


def fetch_redirects(titles: List[str]):
    """
    Fetches redirects for a list of page titles.

    Args:
        titles (List[str]): A list of titles.

    Returns:
        Dict[str, str]: A dictionary mapping source titles to their corresponding redirect targets.
    """
    results = {}
    iterator = trange(
        np.ceil(len(titles) / 50).astype(int), desc="Redirects", leave=False
    )
    for i in iterator:
        first = i * 50
        last = (i + 1) * 50
        target_titles = "|".join(titles[first:last])
        response = requests.get(
            base_url + redirects_query_action + target_titles, headers=http_headers
        ).json()
        redirects = response["query"]["redirects"]
        for redirect in redirects:
            results[redirect.get("from", "")] = redirect.get("to", "")

    return results


def fetch_backlinks(titles: List[str]):
    """
    Fetches backlinks for a list of page titles.

    Args:
        titles (List[str]): A list of titles.

    Returns:
        Dict[str, str]: A dictionary mapping backlink titles to their corresponding target titles.
    """
    results = {}
    iterator = tqdm(titles, desc="Backlinks", leave=False)
    for target_title in iterator:
        iterator.set_postfix(title=target_title)
        response = requests.get(
            base_url + backlinks_query_action + target_title, headers=http_headers
        ).json()
        backlinks = response["query"]["backlinks"]
        for backlink in backlinks:
            if re.match(r"^[a-zA-Z]+$", backlink["title"]) and backlink[
                "title"
            ] not in target_title.split(" "):
                results[backlink["title"]] = target_title

    return results


def fetch_set_info(
    sets: List[str], extra_info: List[str] = [], step: int = 15, **kwargs
):
    """
    Fetches information for a list of sets.

    Args:
        sets (List[str]): A list of set names to fetch information for.
        extra_info (List[str], optional): A list of additional information to fetch for each set. Defaults to an empty list.
        step (int, optional): The number of sets to fetch information for at once. Defaults to 15.
        **kwargs: Additional keyword arguments.

    Returns:
        pd.DataFrame: A DataFrame containing information for all sets in the list.

    Raises:
        Any exceptions raised by requests.get().
    """
    debug = kwargs.get("debug", False)
    if debug:
        print(f"{len(titles)} sets requested")

    regions_dict = load_json(os.path.join(SCRIPT_DIR, "assets/json/regions.json"))
    # Info to ask
    info = extra_info + ["Series", "Set type", "Cover card"]
    # Release to ask
    release = [i + " release date" for i in set(regions_dict.values())]
    # Ask list
    ask = up.quote("|".join(np.append(info, release)))

    # Get set info
    set_info_df = pd.DataFrame()
    for i in trange(np.ceil(len(sets) / step).astype(int), leave=False):
        first = i * step
        last = (i + 1) * step
        titles = up.quote("]]OR[[".join(sets[first:last]))
        response = requests.get(
            f"{base_url}{askargs_query_action}{titles}&printouts={ask}",
            headers=http_headers,
        )
        formatted_response = extract_results(response)
        formatted_response.drop(
            "Page name", axis=1, inplace=True
        )  # Page name not needed - no set errata, set name same as page name
        formatted_df = format_df(
            formatted_response, include_all=(True if extra_info else True)
        )
        if debug:
            tqdm.write(
                f"Iteration {i}\n{len(formatted_df)} set properties downloaded - {step-len(formatted_df)} errors"
            )
            tqdm.write("-------------------------------------------------")

        set_info_df = pd.concat([set_info_df, formatted_df.dropna(axis=1, how="all")])

    set_info_df = set_info_df.convert_dtypes()
    set_info_df.sort_index(inplace=True)

    print(
        f'{"Total:" if debug else ""}{len(set_info_df)} set properties received - {len(sets)-len(set_info_df)} errors'
    )

    return set_info_df


def fetch_set_lists(titles: List[str], **kwargs):  # Separate formating function
    """
    Fetches card set lists from a list of page titles.

    Args:
        titles (List[str]): A list of page titles from which to fetch set lists.
        **kwargs: Additional keyword arguments.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed card set lists.
    """
    debug = kwargs.get("debug", False)
    if debug:
        print(f"{len(titles)} sets requested")

    titles = up.quote("|".join(titles))
    rarity_dict = load_json(os.path.join(SCRIPT_DIR, "assets/json/rarities.json"))
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
        f"{base_url}{revisions_query_action}{titles}", headers=http_headers
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
                if template.name == "Set page header":
                    for argument in template.arguments:
                        if "set=" in argument:
                            title = argument.value
                if template.name == "Set list":
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
                                lambda x: (
                                    x.split("//")[1]
                                    if isinstance(x, str) and "//" in x
                                    else None
                                )
                            ).dropna(how="all")
                            if not extra.empty:
                                extra = extra.stack().droplevel(1, axis=0)
                                extra_lines = pd.DataFrame()
                                for extra_idx, extra_value in extra.items():
                                    if "::" in extra_value:
                                        col, val = extra_value.split("::")
                                        # Strip and process col and val to extract desired values
                                        col = col.strip().strip("@").lower()
                                        val = (
                                            val.strip()
                                            .strip("(")
                                            .strip(")")
                                            .split("]]")[0]
                                            .split("[[")[-1]
                                        )
                                        extra_lines.loc[extra_idx, col] = val

                                extra_lines = extra_lines.dropna(how="all")
                                if not extra_lines.empty:
                                    extra_df = extra_lines
                            ###

                            list_df = list_df.map(
                                lambda x: x.split("//")[0] if x is not None else x
                            )

                            list_df = list_df.map(
                                lambda x: x.strip() if x is not None else x
                            )
                            list_df.replace(
                                r"^\s*$|^@.*$", None, regex=True, inplace=True
                            )

                    if list_df is None:
                        error += 1
                        if debug:
                            print(f'Error! Unable to parse template for "{page_name}"')
                        continue

                    noabbr = opt == "noabbr"
                    set_df["Name"] = list_df[1 - noabbr].apply(
                        lambda x: (
                            x.strip("\u200e").split(" (")[0] if x is not None else x
                        )
                    )

                    if not noabbr and len(list_df.columns > 1):
                        set_df["Card number"] = list_df[0]

                    if len(list_df.columns) > (2 - noabbr):  # and rare in str
                        set_df["Rarity"] = list_df[2 - noabbr].apply(
                            lambda x: (
                                tuple(
                                    [
                                        rarity_dict.get(y.strip(), y.strip())
                                        for y in x.split(",")
                                    ]
                                )
                                if x is not None and "description::" not in x
                                else rarity
                            )
                        )

                    else:
                        set_df["Rarity"] = [rarity for _ in set_df.index]

                    if len(list_df.columns) > (3 - noabbr):
                        if card_print is not None:  # and new/reprint in str
                            set_df["Print"] = list_df[3 - noabbr].apply(
                                lambda x: (
                                    card_print if (card_print and x is None) else x
                                )
                            )

                            if len(list_df.columns) > (4 - noabbr) and qty:
                                set_df["Quantity"] = list_df[4 - noabbr].apply(
                                    lambda x: x if x is not None else qty
                                )

                        elif qty:
                            set_df["Quantity"] = list_df[3 - noabbr].apply(
                                lambda x: x if x is not None else qty
                            )

                    if not title:
                        title = page_name.split("Lists:")[1]

                    # Handle token name and print in description
                    if extra_df is not None:
                        for row in set_df.index:
                            # Handle token name in description
                            if (
                                "description" in extra_df
                                and row in extra_df["description"].dropna().index
                            ):
                                if (
                                    set_df.at[row, "Name"] is not None
                                    and "Token" in set_df.at[row, "Name"]
                                    and "Token" in extra_df.at[row, "description"]
                                ):
                                    set_df.at[row, "Name"] = extra_df.at[
                                        row, "description"
                                    ]

                            # Handle print in description
                            if (
                                "print" in extra_df
                                and row in extra_df["print"].dropna().index
                            ):
                                set_df.at[row, "Print"] = extra_df.at[row, "print"]
                    ###

                    set_df["Set"] = re.sub(r"\(\w{3}-\w{2}\)\s*$", "", title).strip()
                    set_df["Region"] = region.upper()
                    set_df["Page name"] = page_name
                    set_lists_df = (
                        pd.concat([set_lists_df, set_df], ignore_index=True)
                        .infer_objects(copy=False)
                        .fillna(np.nan)
                    )
                    success += 1

        else:
            error += 1
            if debug:
                print(f"Error! No content for \"{content['title']}\"")

    if debug:
        print(f"{success} set lists received - {error} missing")
        print("-------------------------------------------------")

    return set_lists_df, success, error


# Images
async def download_images(
    file_names: pd.DataFrame, save_folder: str = "../images/", max_tasks: int = 10
):
    """
    Downloads a set of images given their names and saves them to a specified folder.

    Args:
        file_names (pandas.DataFrame): A DataFrame containing the names of the image files to be downloaded.
        save_folder (str): The path to the folder where the downloaded images will be saved. Defaults to "../images/".
        max_tasks (int): The maximum number of images to download at once. Defaults to 10.

    Returns:
        None
    """
    # Prepare URL from file names
    file_names_md5 = file_names.apply(md5)
    urls = file_names_md5.apply(lambda x: f"/{x[0]}/{x[0]}{x[1]}/") + file_names

    # Download image from URL
    async def download_image(session, url, save_folder, semaphore, pbar):
        async with semaphore:
            async with session.get(url) as response:
                save_name = url.split("/")[-1]
                if response.status != 200:
                    raise ValueError(
                        f"URL {url} returned status code {response.status}"
                    )
                total_size = int(response.headers.get("Content-Length", 0))
                progress = tqdm(
                    unit="B",
                    total=total_size,
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=save_name,
                    leave=False,
                    disable=("PM_IN_EXECUTION" in os.environ),
                )
                if os.path.isfile(f"{save_folder}/{save_name}"):
                    os.remove(f"{save_folder}/{save_name}")
                while True:
                    chunk = await response.content.read(1024)
                    if not chunk:
                        break
                    progress.update(len(chunk))
                    with open(f"{save_folder}/{save_name}", "ab") as f:
                        f.write(chunk)
                progress.close()
                return save_name

    # Parallelize image downloads
    semaphore = asyncio.Semaphore(max_tasks)
    async with aiohttp.ClientSession(
        base_url="https://ms.yugipedia.com/", headers=http_headers
    ) as session:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        with tqdm(
            total=len(urls),
            unit_scale=True,
            unit="file",
            disable=("PM_IN_EXECUTION" in os.environ),
        ) as pbar:
            tasks = [
                download_image(session, url, save_folder, semaphore, pbar)
                for url in urls
            ]
            for task in asyncio.as_completed(tasks):
                pbar.update()
                await task


# ========== #
# Formatting #
# ========== #


def extract_results(response: requests.Response):
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
        page_url = (
            pd.DataFrame(json["query"]["results"])
            .transpose()["fullurl"]
            .rename("Page URL")
        )
        page_name = (
            pd.DataFrame(json["query"]["results"])
            .transpose()["fulltext"]
            .rename("Page name")
        )  # Not necessarily same as card name (Used to merge errata)
        df = pd.concat([df, page_name, page_url], axis=1)

    return df


def extract_fulltext(x: List[Union[Dict[str, Any], str]], multiple: bool = False):
    """
    Extracts fulltext from a list of dictionaries or strings.
    If multiple is True, returns a sorted tuple of all fulltexts.
    Otherwise, returns the first fulltext found, with leading/trailing whitespaces removed.
    If the input list is empty, returns np.nan.

    Args:
        x (List[Union[Dict[str, Any], str]]): A list of dictionaries or strings to extract fulltext from.
        multiple (bool): If True, return a tuple of all fulltexts. Otherwise, return the first fulltext. Default is False.

    Returns:
        str or Tuple[str] or np.nan: The extracted fulltext(s).
    """
    if len(x) > 0:
        if isinstance(x[0], int):
            return str(x[0])
        elif "fulltext" in x[0]:
            if multiple:
                return tuple(sorted([i["fulltext"] for i in x]))
            else:
                return x[0]["fulltext"].strip("\u200e")
        else:
            if multiple:
                return tuple(sorted(x))
            else:
                return x[0].strip("\u200e")
    else:
        return np.nan


def format_df(input_df: pd.DataFrame, include_all: bool = False):
    """
    Formats a dataframe containing card information.
    Returns a new dataframe with specific columns extracted and processed.

    Args:
        input_df (pd.DataFrame): The input dataframe to format.
        include_all (bool): If True, include all unspecified columns in the output dataframe. Default is False.

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
            lambda x: (
                tuple([arrows_dict[i] for i in sorted(x)]) if len(x) > 0 else np.nan
            )
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
            extracted_cols = input_df[col_matches].map(
                extract_fulltext if extract else lambda x: x
            )
            if col == " Material":
                df["Materials"] = extracted_cols.apply(
                    lambda x: tuple(elem for tup in col for elem in tup), axis=1
                )
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
    if len(input_df.filter(like=" date").columns) > 0:
        df = df.join(
            input_df.filter(like=" date").map(
                lambda x: (
                    pd.to_datetime(
                        pd.to_numeric(x[0]["timestamp"]), unit="s", errors="coerce"
                    )
                    if len(x) > 0
                    else np.nan
                )
            )
        )

    # Include other unspecified columns
    if include_all:
        df = df.join(
            input_df[input_df.columns.difference(df.columns)].map(
                extract_fulltext, multiple=True
            )
        )

    return df


# Cards
def extract_artwork(row: pd.Series):
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


def extract_primary_type(x: Union[str, List[str], Tuple[str]]):
    """
    Extracts the primary type of a card.
    If the input is a list or tuple, removes "Pendulum Monster" and "Maximum Monster" from the list.
    If the input is a list or tuple with only one element, returns that element.
    If the input is a list or tuple with multiple elements, returns the first element that is not "Effect Monster".
    Otherwise, returns the input.

    Args:
        x (Union[str, List[str], Tuple[str]]): The input type(s) to extract the primary type from.

    Returns:
        Union[str, List[str]]: The extracted primary type(s).
    """
    if isinstance(x, list) or isinstance(x, tuple):
        if "Monster Token" in x:
            return "Monster Token"
        else:
            x = [z for z in x if (z != "Pendulum Monster") and (z != "Maximum Monster")]
            if len(x) == 1 and "Effect Monster" in x:
                return "Effect Monster"
            elif len(x) > 0:
                return [z for z in x if z != "Effect Monster"][0]
            else:
                return "???"

    return x


def extract_misc(x: Union[str, List[str], Tuple[str]]):
    """
    Extracts the misc properties of a card.
    Checks whether the input contains the values "Legend Card" or "Requires Maximum Mode" and creates a boolean table.

    Args:
        x (Union[str, List[str], Tuple[str]]): The Misc values to generate the boolean table from.

    Returns:
        pd.Series: A pandas Series of boolean values indicating whether "Legend Card" and "Requires Maximum Mode" are present in the input.
    """
    if isinstance(x, list) or isinstance(x, tuple):
        return pd.Series(
            [val in x for val in ["Legend Card", "Requires Maximum Mode"]],
            index=["Legend", "Maximum mode"],
        )
    else:
        return pd.Series([False, False], index=["Legend", "Maximum mode"])


def extract_category_bool(x: List[str]):
    """
    Extracts a boolean value from a list of strings that represent a boolean value.
    If the first string in the list is "t", returns True.
    If the first string in the list is "f", returns False.
    Otherwise, returns np.nan.

    Args:
        x (List[str]): The input list of strings to extract a boolean value from.

    Returns:
        Union[bool, np.nan]: The extracted boolean value.
    """
    if len(x) > 0:
        if x[0] == "f":
            return False
        elif x[0] == "t":
            return True

    return np.nan
