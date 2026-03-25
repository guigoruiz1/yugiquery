# yugiquery/api/client.py

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import asyncio
import os
import re
import socket
import time
from pathlib import Path
from types import SimpleNamespace
from typing import (
    Any,
    Dict,
    List,
    Tuple,
)
import urllib.parse as up

# --- Imports: Third-Party --- #
import aiohttp
from aiohttp import payload
import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm
from termcolor import cprint

# --- Imports: Local Application --- #
from .. import utils
from ..metadata import __title__, __url__, __version__
from ..utils import md5, LoggerConfig, dirs
from .parsers import parse_response, extract_query_field, response_to_df, format_df, process_content, APIResponseError

# --- Logger Setup --- #
logger = LoggerConfig.get_logger()

# --- Halo Spinner Import --- #
if dirs.is_notebook:
    from halo import HaloNotebook as Halo
else:
    from halo import Halo

# --- API URL Dictionaries --- #
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


# --- Default API Parameters --- #

DEFAULT_HTTP_RETRIES = 3
DEFAULT_HTTP_TIMEOUT: Tuple[float, float] = (5, 30)
DEFAULT_HTTP_BACKOFF = 1.0
DEFAULT_SOCKET_TIMEOUT = 2

# --- Functions --- #


def _request_with_retry(
    url: str,
    *,
    headers: Dict[str, str],
    params: Dict[str, Any] | None = None,
    retries: int = DEFAULT_HTTP_RETRIES,
    timeout: Tuple[float, float] = DEFAULT_HTTP_TIMEOUT,
    backoff: float = DEFAULT_HTTP_BACKOFF,
) -> requests.Response:
    """
    GET wrapper with exponential-backoff retries for transient API failures.
    Tries to make a GET request to the specified URL with the provided headers and query parameters.
    If the request fails due to a transient error (e.g., network issues, server errors), it will retry the request up to a specified number of retries with an exponential backoff strategy.

    Args:
        url (str): The URL to send the GET request to.
        headers (Dict[str, str]): A dictionary of HTTP headers to include in the request.
        params (Dict[str, Any], optional): A dictionary of query parameters to include in the request. Defaults to None.
        retries (int, optional): The number of retry attempts for transient failures. Defaults to DEFAULT_HTTP_RETRIES.
        timeout (Tuple[float, float], optional): A tuple specifying the connection and read timeouts for the request. Defaults to DEFAULT_HTTP_TIMEOUT.
        backoff (float, optional): The base backoff time in seconds for retries. Defaults to DEFAULT_HTTP_BACKOFF.

    Returns:
        requests.Response: The response object returned by the successful API request.

    Raises:
        ValueError: If the retries parameter is less than 1.
        requests.exceptions.RequestException: If an error occurs while making the API request after exhausting all retries.
        RuntimeError: If the request fails and no exception was captured.

    """
    if retries < 1:
        raise ValueError("retries must be at least 1")

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url=url, headers=headers, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as err:
            if attempt == retries:
                raise
            logger.debug("Attempt %s/%s failed: %s", attempt, retries, err)
            time.sleep(backoff * (2 ** (attempt - 1)))

    raise RuntimeError("Request failed and no exception was captured.")


# --- YGOPRODECK --- #


def fetch_ygoprodeck(misc=True) -> List[Dict[str, Any]]:
    """
    Fetch the card data from ygoprodeck.com.

    Returns:
        (List[Dict[str, Any]]): List of card data.

    Raises:
        requests.exceptions.RequestException: If an error occurs while making the API request.
        APIResponseError: If the API response contains an error or is invalid.
    """
    ydk_url = URLS.ygoprodeck
    if misc:
        ydk_url += "?misc=yes"
    response = _request_with_retry(ydk_url, headers=URLS.headers)

    try:
        payload = response.json()
    except ValueError as err:
        logger.error("Unable to parse YGOPRODeck JSON from URL: %s", response.url)
        raise APIResponseError(f"Invalid JSON response from {response.url}") from err

    try:
        return payload["data"]
    except KeyError as err:
        logger.error("Missing data in YGOPRODeck response from %s", response.url)
        raise APIResponseError(f"Missing data in response from {response.url}") from err


# --- API Status --- #


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
        response = _request_with_retry(URLS.base, params=params, headers=URLS.headers)
        cprint(text=f"{URLS.base} is up and running {response.json()['query']['general']['generator']}\n", color="green")
        logger.info("%s is up and running %s", URLS.base, response.json()["query"]["general"]["generator"])
        return True
    except requests.exceptions.RequestException as err:
        logger.error("%s is not alive", URLS.base)
        logger.error("API status request failed: %s", err)
        domain = up.urlparse(URLS.base).netloc
        port = 443

        try:
            socket.create_connection((domain, port), timeout=DEFAULT_SOCKET_TIMEOUT)
            cprint(text=f"{domain} is reachable\n", color="yellow")
            logger.warning("%s is reachable", domain)
        except OSError as err:
            cprint(text=f"{domain} is not reachable\n", color="red")
            logger.error("%s is not reachable", domain)
            logger.error("Socket probe failed: %s", err)

        return False


# --- Category Members --- #


def fetch_categorymembers(
    category: str,
    namespace: int | None = None,
    step: int = 500,
    iterator: tqdm | None = None,
) -> pd.DataFrame:
    """
    Fetches members of a category from the API by making iterative requests with a specified step size until all members are retrieved.

    Args:
        category (str): The category to retrieve members for.
        namespace (int| None, optional): The namespace ID to filter the members by. Defaults to None (no namespace).
        step (int, optional): The number of members to retrieve in each request. Defaults to 500.
        iterator (tqdm.std.tqdm | None, optional): A tqdm iterator to display progress updates. Defaults to None.

    Returns:
        pandas.DataFrame: A DataFrame containing the members of the category.

    Raises:
        requests.exceptions.RequestException: If an error occurs while making the API request.
        APIResponseError: If the API response contains an error or is invalid.
        KeyboardInterrupt: If the operation is interrupted by the user.
        SystemExit: If the operation is interrupted by the system.
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
                if iterator is not None:
                    iterator.set_postfix(it=i + 1)

                spinner.text = f"Fetching category members... Iteration {i+1}"

                params = params.copy()
                params.update(lastContinue)
                response = _request_with_retry(
                    URLS.base + URLS.categorymembers_action + up.quote(category),
                    params=params,
                    headers=URLS.headers,
                )
                with logging_redirect_tqdm():
                    logger.debug("%s", response.url)

                payload = parse_response(response)

                if "warnings" in payload:
                    spinner.warn(payload["warnings"])

                members = extract_query_field(payload, "categorymembers")
                all_results += members

                if "continue" not in payload:
                    spinner.succeed(f'{i+1} iteration(s) completed for category "{category}"')
                    with logging_redirect_tqdm():
                        logger.debug('%s iterations completed for category "%s"', i + 1, category)
                    break
                lastContinue = payload["continue"]
                i += 1

            if "PM_IN_EXECUTION" not in os.environ:
                time.sleep(0.5)

        except (KeyboardInterrupt, SystemExit):
            spinner.fail("Execution interrupted.")
            with logging_redirect_tqdm():
                logger.warning('Category members fetch interrupted for "%s" on iteration %s', category, i + 1)
            if "PM_IN_EXECUTION" not in os.environ:
                time.sleep(0.5)
            raise
        except (requests.exceptions.RequestException, APIResponseError) as err:
            spinner.fail(f"Request failed: {err}")
            with logging_redirect_tqdm():
                logger.error('Category members fetch failed for "%s" on iteration %s: %s', category, i + 1, err)
            if "PM_IN_EXECUTION" not in os.environ:
                time.sleep(0.5)
            raise

    results_df = pd.DataFrame(all_results)
    return results_df


# --- Properties --- #


def fetch_properties(
    condition: str,
    query: str,
    step: int = 500,
    limit: int = 5000,
    iterator: tqdm | None = None,
    include_all: bool = False,
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

    Returns:
        pandas.DataFrame: A DataFrame containing the properties matching the query and condition.

    Raises:
        requests.exceptions.RequestException: If an error occurs while making the API request.
        APIResponseError: If the API response contains an error or is invalid.
        KeyboardInterrupt: If the operation is interrupted by the user.
        SystemExit: If the operation is interrupted by the system.
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
                if iterator is not None:
                    iterator.set_postfix(it=i + 1)

                spinner.text = f"Fetching properties... Iteration {i+1}"

                response = _request_with_retry(
                    url=URLS.base
                    + URLS.ask_action
                    + up.quote(condition)
                    + query
                    + f"|limit%3D{step}|offset={i*step}|order%3Dasc",
                    headers=URLS.headers,
                )
                with logging_redirect_tqdm():
                    logger.debug("%s", response.url)

                query_df = response_to_df(response)
                formatted_df = format_df(input_df=query_df, include_all=include_all)
                df = pd.concat([df, formatted_df], ignore_index=True, axis=0)

                with logging_redirect_tqdm():
                    logger.debug("Iteration %s: %s results", i + 1, len(formatted_df.index))

                if len(formatted_df.index) < step or (i + 1) * step >= limit:
                    spinner.succeed(f"{i+1} iteration(s) completed")
                    complete = True
                else:
                    i += 1

            if "PM_IN_EXECUTION" not in os.environ:
                time.sleep(0.5)

            with logging_redirect_tqdm():
                logger.debug(
                    '%s iterations completed for condition "%s": %s total results',
                    i + 1,
                    condition,
                    len(df.index),
                )

        except (KeyboardInterrupt, SystemExit):
            spinner.fail("Execution interrupted.")
            with logging_redirect_tqdm():
                logger.warning(
                    'Properties fetch interrupted for condition "%s" on iteration %s',
                    condition,
                    i + 1,
                )
            if "PM_IN_EXECUTION" not in os.environ:
                time.sleep(0.5)
            raise
        except (requests.exceptions.RequestException, APIResponseError) as err:
            spinner.fail(f"Request failed: {err}")
            with logging_redirect_tqdm():
                logger.error(
                    'Properties fetch failed for condition "%s" on iteration %s: %s',
                    condition,
                    i + 1,
                    err,
                )
            if "PM_IN_EXECUTION" not in os.environ:
                time.sleep(0.5)
            raise

    return df


# --- Redirects --- #


def fetch_redirects(*titles: str) -> Dict[str, str]:
    """
    Fetches redirects for a list of page titles.

    Args:
        titles (str): Multiple title strings.

    Returns:
        Dict[str, str]: A dictionary mapping source titles to their corresponding redirect targets.
    Raises:
        requests.exceptions.RequestException: If an error occurs while making the API request.
        APIResponseError: If the API response contains an error or is invalid.
    """
    results = {}
    iterator = trange(np.ceil(len(titles) / 50).astype(int), desc="Redirects", leave=False)
    for i in iterator:
        first = i * 50
        last = (i + 1) * 50
        target_titles = "|".join(titles[first:last])
        response = _request_with_retry(
            url=URLS.base + URLS.redirects_action + target_titles,
            headers=URLS.headers,
        )
        with logging_redirect_tqdm():
            logger.debug("%s", response.url)

        payload = parse_response(response)
        redirects = extract_query_field(payload=payload, field="redirects")
        for redirect in redirects:
            results[redirect.get("from", "")] = redirect.get("to", "")

    return results


# --- Backlinks --- #


def fetch_backlinks(*titles: str) -> Dict[str, str]:
    """
    Fetches backlinks for a list of page titles.

    Args:
        titles (str): Multiple title strings.

    Returns:
        Dict[str, str]: A dictionary mapping backlink titles to their corresponding target titles.

    Raises:
        requests.exceptions.RequestException: If an error occurs while making the API request.
        APIResponseError: If the API response contains an error or is invalid.
    """
    results = {}
    iterator = tqdm(titles, dynamic_ncols=(not utils.dirs.is_notebook), desc="Backlinks", leave=False)
    for target_title in iterator:
        iterator.set_postfix(title=target_title)
        response = _request_with_retry(
            url=URLS.base + URLS.backlinks_action + target_title,
            headers=URLS.headers,
        )
        with logging_redirect_tqdm():
            logger.debug("%s", response.url)

        payload = parse_response(response)
        backlinks = extract_query_field(payload=payload, field="backlinks")
        for backlink in backlinks:
            if re.match(pattern=r"^[a-zA-Z]+$", string=backlink["title"]) and backlink["title"] not in target_title.split(
                " "
            ):
                results[backlink["title"]] = target_title

    return results


# --- Wrapper for dictionaries --- #


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

    Raises:
        requests.exceptions.RequestException: If an error occurs while making the API request.
        APIResponseError: If the API response contains an error or is invalid.
    """
    if isinstance(codes, str):
        codes = [codes]
    if isinstance(names, str):
        names = [names]
    if category:
        category_kwargs = dict(kwargs)
        category_kwargs.setdefault("namespace", 0)
        names.extend(fetch_categorymembers(category=category, **category_kwargs)["title"])

    backlinks = fetch_backlinks(*names)
    redirects = fetch_redirects(*codes)
    return redirects | backlinks


# --- Set Information --- #


def fetch_set_info(*sets: str, extra_info: List[str] = [], step: int = 15) -> pd.DataFrame:
    """
    Fetches information for a list of sets.

    Args:
        sets (str | List[str]): Multiple set names to fetch information for.
        extra_info (List[str], optional): A list of additional information to fetch for each set. Defaults to an empty list.
        step (int, optional): The number of sets to fetch information for at once. Defaults to 15.

    Returns:
        pd.DataFrame: A DataFrame containing information for all sets in the list.

    Raises:
        requests.exceptions.RequestException: If an error occurs while making the API request.
        APIResponseError: If the API response contains an error or is invalid.
        KeyboardInterrupt: If the operation is interrupted by the user.
        SystemExit: If the operation is interrupted by the system.
    """
    tqdm.write(
        f"Downloading information for {len(sets)} sets",
    )
    with logging_redirect_tqdm():
        logger.info("Downloading information for %s sets", len(sets))

    regions_dict = utils.load_json(utils.dirs.get_asset("json", "regions.json"))
    # Info to ask
    info = extra_info + ["Series", "Set type", "Cover card"]
    # Release to ask
    release = [i + " release date" for i in set(regions_dict.values())]
    # Ask list
    ask = up.quote(string="|".join(np.append(info, release)))

    # Get set info
    set_info_df = pd.DataFrame()

    try:
        for i in trange(np.ceil(len(sets) / step).astype(int), leave=False):
            first = i * step
            last = (i + 1) * step
            batch_sets = sets[first:last]
            titles = up.quote(string="]]OR[[".join(batch_sets))
            response = _request_with_retry(
                url=URLS.base + URLS.askargs_action + titles + f"&printouts={ask}",
                headers=URLS.headers,
            )
            with logging_redirect_tqdm():
                logger.debug("Iteration %s:", i + 1)
                logger.debug("%s", response.url)

            formatted_response = response_to_df(response)
            formatted_response.drop(
                "Page name", axis=1, inplace=True
            )  # Page name not needed - no set errata, set name same as page name
            formatted_df = format_df(input_df=formatted_response, include_all=(True if extra_info else False))
            with logging_redirect_tqdm():
                logger.debug(
                    "Information for %s sets downloaded - %s missing",
                    len(formatted_df),
                    len(batch_sets) - len(formatted_df),
                )

            set_info_df = pd.concat([set_info_df, formatted_df.dropna(axis=1, how="all")])

    except (KeyboardInterrupt, SystemExit):
        with logging_redirect_tqdm():
            logger.warning("Set info fetch interrupted on iteration %s", i + 1)
        raise
    except requests.exceptions.RequestException as err:
        with logging_redirect_tqdm():
            logger.error(
                "Set info request failed on iteration %s for sets %s: %s",
                i + 1,
                list(batch_sets),
                err,
            )
        raise
    except APIResponseError as err:
        with logging_redirect_tqdm():
            logger.error(
                "Set info parsing failed on iteration %s for sets %s: %s",
                i + 1,
                list(batch_sets),
                err,
            )
        raise

    set_info_df = set_info_df.convert_dtypes()
    set_info_df.sort_index(inplace=True)
    with logging_redirect_tqdm():
        logger.info("Information received for %s sets - %s missing", len(set_info_df), len(sets) - len(set_info_df))
    tqdm.write(
        f"Information received for {len(set_info_df)} sets - {len(sets) - len(set_info_df)} missing\n",
    )

    return set_info_df


# --- Set Lists --- #


def fetch_set_lists(*titles: str) -> None | Tuple[pd.DataFrame, int, int]:
    """
    Fetches card set lists from a list of page titles.

    Args:
        titles (str): Multiple page titles from which to fetch set lists.

    Returns:
        Tuple[pd.DataFrame, int, int]: A DataFrame containing the parsed card set lists, the number of successful requests, and the number of failed requests.

    Raises:
        requests.exceptions.RequestException: If an error occurs while making the API request.
        APIResponseError: If the API response contains an error or is invalid.
    """
    logger.debug("%s sets requested", len(titles))

    titles_str = up.quote(string="|".join(titles))
    rarities = utils.load_json(utils.dirs.get_asset("json", "rarities.json"))

    columns = [
        "Set",
        "Card number",
        "Name",
        "Rarity",
        "Print",
        "Quantity",
        "Region",
        "Page name",
    ]
    result = pd.DataFrame(columns=columns)
    total_success = 0
    total_error = 0

    response = _request_with_retry(
        url=URLS.base + URLS.revisions_action + titles_str,
        headers=URLS.headers,
    )
    logger.debug("%s", response.url)
    payload = parse_response(response)
    contents = extract_query_field(payload=payload, field="pages").values()

    for content in contents:
        if "revisions" in content.keys():
            page_df, success, error = process_content(content, rarities, columns)

            if page_df is not None:
                result = pd.concat([result, page_df], ignore_index=True).infer_objects().fillna(np.nan)

            total_success += success
            total_error += error

        else:
            total_error += 1
            logger.warning('No content for "%s"', content.get("title", "Unknown"))

    logger.debug("%s set lists received - %s missing", total_success, total_error)

    return result, total_success, total_error


# --- Media --- #


def fetch_page_images(
    *titles: str,
    featured: bool = False,
    batch_size: int = 50,
    imlimit: int = 500,
) -> Dict[str, List[str] | str]:
    """
    Fetch images from the MediaWiki API for the provided page titles.

    Args:
        titles (str): Page titles to fetch images for.
        featured (bool, optional): If True, fetch only the featured image of each page.
            If False, fetch the page image list. Defaults to False.
        batch_size (int, optional): Number of titles per API request. Defaults to 50.
        imlimit (int, optional): Maximum number of images per page when featured=False.
            Defaults to 500.

    Returns:
        Dict[str, List[str] | str]: Mapping of page title to image names.

    Raises:
        requests.exceptions.RequestException: If an error occurs while making the API request.
        APIResponseError: If the API response contains an error or is invalid.
    """
    results: Dict[str, List[str] | str] = {}

    for i in range(0, len(titles), batch_size):
        batch = titles[i : i + batch_size]
        titles_str = up.quote("|".join(batch))

        if featured:
            action = URLS.pageimages_action
        else:
            action = URLS.images_action.replace("&titles=", f"&imlimit={imlimit}&titles=")

        response = _request_with_retry(
            url=URLS.base + action + titles_str,
            headers=URLS.headers,
        )

        payload = parse_response(response)
        pages = extract_query_field(payload=payload, field="pages")
        for page in pages.values():
            if featured:
                original = page.get("original")
                if original and "source" in original:
                    results[page["title"]] = original["source"].split("/")[-1]
            else:
                images = page.get("images", [])
                results[page["title"]] = [img["title"].removeprefix("File:") for img in images]

    return results


async def download_media(
    *file_names: str,
    output_path: str | Path = "media",
    max_tasks: int = 10,
) -> List[Dict[str, str | bool]]:
    """
    Download media files from yugipedia media storage.

    Args:
        file_names (str): Media file names to download.
        output_path (str | Path, optional): Destination directory. Defaults to "media".
        max_tasks (int, optional): Maximum concurrent downloads. Defaults to 10.

    Returns:
        List[Dict[str, str | bool]]: Download status entries per file.
    """
    file_names_series = pd.Series(file_names)
    file_names_md5 = file_names_series.apply(md5)
    urls = file_names_md5.apply(lambda x: f"/{x[0]}/{x[0]}{x[1]}/") + file_names_series
    download_results = []

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
                        dynamic_ncols=(not utils.dirs.is_notebook),
                        disable=("PM_IN_EXECUTION" in os.environ),
                    )

                    if save_file.is_file():
                        save_file.unlink()

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
                if save_file.is_file():
                    save_file.unlink()
                download_results.append({"file_name": save_name, "url": URLS.media + url, "success": False})
                with logging_redirect_tqdm():
                    logger.warning("Failed to download %s: %s", save_name, e)
            finally:
                pbar.update()

    semaphore = asyncio.Semaphore(max_tasks)
    async with aiohttp.ClientSession(base_url=URLS.media, headers=URLS.headers) as session:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        with tqdm(
            total=len(urls),
            unit="file",
            dynamic_ncols=(not utils.dirs.is_notebook),
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
            await asyncio.gather(*tasks, return_exceptions=True)

    return download_results
