# yugiquery/api/wrappers.py

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import json
import os
import time
import urllib.parse as up
from enum import Enum
from typing import Dict

# --- Imports: Third-Party --- #
import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm

# --- Imports: Local Application --- #
from . import client
from ..utils import dirs, load_json, LoggerConfig, ensure_tuple_columns

# --- Logger Setup --- #
logger = LoggerConfig.get_logger()


# --- Card Game Enum & Properties --- #
class CG(Enum):
    """
    Enum representing the card game formats.

    Attributes:
        CG (str): Both TCG and OCG.
        ALL (CG): Alias for CG, representing all card games.
        BOTH (CG): Alias for CG, representing both card games.
        TCG (str): The 'trading card game' type.
        OCG (str): The 'official card game' type.
    """

    CG = "CG"
    ALL = CG
    BOTH = CG
    TCG = "TCG"
    OCG = "OCG"


#: A dictionary mapping card types to their corresponding properties to query.
_card_properties = {
    "monster": [
        "password",
        "card_type",
        "primary",
        "secondary",
        "attribute",
        "monster_type",
        "level_rank_link",
        "atk",
        "def",
        "scale",
        "arrows",
        "effect_type",
        "archseries",
        "alternate_artwork",
        "edited_artwork",
        "tcg_status",
        "ocg_status",
        "tcg_debut",
        "ocg_debut",
        "modified_date",
    ],
    "st": [
        "password",
        "card_type",
        "property",
        "effect_type",
        "archseries",
        "alternate_artwork",
        "edited_artwork",
        "tcg_status",
        "ocg_status",
        "tcg_debut",
        "ocg_debut",
        "modified_date",
    ],
    "tc": [
        "atk",
        "def",
        "attribute",
        "level_rank_link",
        "monster_type",
        "primary",
        "card_type",
        "effect_type",
        "archseries",
        "alternate_artwork",
        "edited_artwork",
        "tcg_status",
        "ocg_status",
        "tcg_debut",
        "ocg_debut",
        "modified_date",
    ],
    "skill": [
        "card_type",
        "property",
        "archseries",
        "tcg_status",
        "speed_status",
        "character",
        "speed_debut",
        "modified_date",
    ],
    "speed": [
        "password",
        "card_type",
        "property",
        "primary",
        "secondary",
        "attribute",
        "monster_type",
        "level_rank_link",
        "atk",
        "def",
        "effect_type",
        "archseries",
        "alternate_artwork",
        "edited_artwork",
        "tcg_status",
        "ocg_status",
        "speed_status",
        "speed_debut",
        "modified_date",
    ],
    "rush": [
        "card_type",
        "property",
        "primary",
        "attribute",
        "monster_type",
        "level",
        "atk",
        "def",
        "effect_type",
        "archseries",
        "modified_date",
        "rush_alt_artwork",
        "rush_edited_artwork",
        "maximum_atk",
        "misc",
        "debut",
    ],
    "bandai": [
        "card_type",
        "raw_level",
        "atk",
        "def",
        "number",
        "monster_type",
        "rule",
        "sets",
        "rarity",
        "ability",
        "modified_date",
        "debut",
    ],
}


# --- Asset Management Functions --- #
def update_rarities(save: bool = True) -> Dict[str, str]:
    """
    Update rarities dictionary with redirects/backlinks fetched from the API.

    Args:
        save (bool, optional): Whether to persist updates to rarities.json.

    Returns:
        Dict[str, str]: Updated rarities dictionary.
    """
    rarities_file = dirs.get_asset("json", "rarities.json")
    rarity_dict = load_json(rarities_file)

    codes = list(rarity_dict.keys())
    names = list(rarity_dict.values())
    new_rarity_dict = client.fetch_redirect_dict(codes=codes, names=names, category="Rarities", namespace=0)

    rarity_dict = rarity_dict | new_rarity_dict
    rarity_dict = {code: rarity_dict[code] for code in sorted(rarity_dict.keys())}

    if save:
        with open(rarities_file, "w+") as file:
            json.dump(rarity_dict, file, indent=4)

    return rarity_dict


def update_regions(save: bool = True) -> Dict[str, str]:
    """
    Update regions dictionary with redirects/backlinks fetched from the API.

    Args:
        save (bool, optional): Whether to persist updates to regions.json.

    Returns:
        Dict[str, str]: Updated regions dictionary.
    """
    regions_file = dirs.get_asset("json", "regions.json")
    regions_dict = load_json(regions_file)

    names = list(regions_dict.values())
    new_regions_dict = client.fetch_redirect_dict(names=names, category="Terminology", namespace=0)

    regions_dict = regions_dict | new_regions_dict
    regions_dict = {code: regions_dict[code] for code in sorted(regions_dict.keys())}

    if save:
        with open(regions_file, "w+") as file:
            json.dump(regions_dict, file, indent=4)

    return regions_dict


def get_ygoprodeck() -> pd.DataFrame:
    """
    Fetch YGOProDeck data from the API and persist a local cache.

    Returns:
        pd.DataFrame: YGOProDeck card data indexed by card id.

    Raises:
        Exception: Exceptions raised by client.fetch_ygoprodeck when no cache exists.
    """
    ygoprodeck_file = dirs.DATA / "ygoprodeck.json"
    try:
        result = client.fetch_ygoprodeck()
        ygoprodeck_file.parent.mkdir(parents=True, exist_ok=True)
        with open(ygoprodeck_file, "w+") as file:
            json.dump(result, file, indent=4)
        return pd.DataFrame(result).set_index("id")
    except (OSError, ValueError, RuntimeError) as e:
        logger.warning("Unable to fetch ygoprodeck data due to %s: %s. Attempting to use local cache.", type(e).__name__, e)
        if ygoprodeck_file.is_file():
            try:
                with open(ygoprodeck_file, "r") as file:
                    cached_result = json.load(file)
                return pd.DataFrame(cached_result).set_index("id")
            except (OSError, ValueError) as cache_err:
                logger.error("Failed to load ygoprodeck cache due to %s: %s", type(cache_err).__name__, cache_err)
                raise RuntimeError("Unable to fetch ygoprodeck data and failed to load cache.") from cache_err
        else:
            logger.error("Unable to obtain ygoprodeck data and no cache file exists.")
            raise RuntimeError("Unable to fetch ygoprodeck data and no cache file exists.") from e


# --- Query Builder & Property Mapping --- #
def card_query(*args, **kwargs) -> str:
    """
    Build the query string used for yugipedia card searches.

    Args:
        default (bool, optional): Include the default core card properties when True.
        *args: Extra properties to include.
        **kwargs: Property toggles where truthy adds and falsy removes.

    Returns:
        str: Query string for the API.
    """
    default_properties = [
        "password",
        "card_type",
        "property",
        "primary",
        "secondary",
        "attribute",
        "monster_type",
        "level_rank_link",
        "atk",
        "def",
        "scale",
        "arrows",
        "effect_type",
        "archseries",
        "alternate_artwork",
        "edited_artwork",
        "tcg_status",
        "ocg_status",
        "tcg_debut",
        "ocg_debut",
        "modified_date",
    ]

    property_dict = {
        "password": "Password",
        "card_type": "Card type",
        "property": "Property",
        "primary": "Primary type",
        "secondary": "Secondary type",
        "attribute": "Attribute",
        "monster_type": "Type=Monster type",
        "level_rank_link": "Level/Rank/Link string=Level/Rank/Link",
        "level_rank": "Level/Rank string=Level/Rank",
        "level": "Level string=Level",
        "rank": "Rank string=Rank",
        "link": "Link Rating string=Link",
        "atk": "ATK string=ATK",
        "def": "DEF string=DEF",
        "scale": "Pendulum Scale string=Pendulum Scale",
        "arrows": "Link Arrows",
        "effect_type": "Effect type",
        "archseries": "Archseries",
        "alternate_artwork": "Category:OCG/TCG cards with alternate artworks",
        "edited_artwork": "Category:OCG/TCG cards with edited artworks",
        "tcg_status": "TCG status",
        "ocg_status": "OCG status",
        "modified_date": "Modification date",
        "image_URL": "Card image",
        "misc": "Misc",
        "summoning": "Summoning",
        "debut": "Debut date=Debut",
        "tcg_debut": "TCG debut date=TCG debut",
        "ocg_debut": "OCG debut date=OCG debut",
        "speed_status": "TCG Speed Duel status",
        "character": "Character",
        "speed_debut": "TCG Speed Duel debut date=Speed debut",
        "rush_alt_artwork": "Category:Rush Duel cards with alternate artworks",
        "rush_edited_artwork": "Category:Rush Duel cards with edited artworks",
        "maximum_atk": "MAXIMUM ATK",
        "number": "Bandai number=Card number",
        "rule": "Bandai rule=Rule",
        "sets": "Sets=Set",
        "rarity": "Rarity",
        "ability": "Ability",
        "category": "category",
        # Bandai specific
        "raw_level": "Level",
    }

    search_string = "|?English name=Name"
    default = kwargs.pop("default", False)
    props = set(default_properties) if default else set()
    props.update(args)

    for key, value in kwargs.items():
        if value and key not in props:
            props.add(key)
        elif not value and key in props:
            props.discard(key)

    for prop in props:
        search_string += f"|?{up.quote(property_dict.get(prop, prop))}"

    return search_string


# --- Card Fetching Functions --- #
def fetch_bandai(bandai_query: str | None = None, limit: int = 200, **kwargs) -> pd.DataFrame:
    """
    Fetch Bandai cards from the API.

    Args:
        bandai_query (str | None, optional): Custom query string for Bandai cards.
        limit (int, optional): Maximum number of cards to fetch.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pd.DataFrame: DataFrame containing Bandai card data.
    """
    concept = "[[Medium::Bandai]]"
    if bandai_query is None:
        bandai_query = card_query(*_card_properties["bandai"])

    tqdm.write("Downloading bandai cards")
    with logging_redirect_tqdm():
        logger.info("Downloading bandai cards")

    bandai_df = client.fetch_properties(concept, bandai_query, step=limit, limit=limit, **kwargs)
    if "Monster type" in bandai_df:
        bandai_df["Monster type"] = bandai_df["Monster type"].dropna().apply(lambda x: x.split("(")[0])

    with logging_redirect_tqdm():
        logger.info("%s results", len(bandai_df.index))
    tqdm.write(f"{len(bandai_df.index)} results\n")

    time.sleep(0.5)

    return bandai_df


def fetch_st(
    st_query: str | None = None,
    st: str = "both",
    cg: CG = CG.ALL,
    step: int = 500,
    limit: int = 5000,
    **kwargs,
) -> pd.DataFrame:
    """
    Fetch Spell and Trap cards from the API.

    Args:
        st_query (str | None, optional): Custom query string for Spell and Trap cards.
        st (str, optional): The type of card to fetch. Default is "both".
        cg (CG, optional): The card game to fetch cards from. Default is CG.ALL.
        step (int, optional): The number of cards to fetch in each request. Default is 500.
        limit (int, optional): The maximum number of cards to fetch. Default is 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pd.DataFrame: DataFrame containing Spell and Trap card data.
    """
    st = st.capitalize()
    valid_st = {"Spell", "Trap", "Both", "All"}
    valid_cg = cg.value
    concept = f"[[Concept:CG non-monster cards]]"
    if st not in valid_st:
        raise ValueError("results: st must be one of %r." % valid_st)
    elif st == "Both" or st == "All":
        st = "Spells and Trap"
    else:
        concept += f"[[Card type::{st} card]]"
    if valid_cg != "CG":
        concept += f"[[Medium::{valid_cg}]]"

    if st_query is None:
        st_query = card_query(*_card_properties["st"])

    tqdm.write(f"Downloading {st}s")
    with logging_redirect_tqdm():
        logger.info("Downloading %ss", st)

    st_df = client.fetch_properties(concept, st_query, step=step, limit=limit, **kwargs)

    with logging_redirect_tqdm():
        logger.debug("- Total")
        logger.info("%s results", len(st_df.index))
    tqdm.write(f"{len(st_df.index)} results\n")

    return st_df


def fetch_spell(*query: str, cg=CG.ALL, step: int = 500, limit: int = 5000, **kwargs) -> pd.DataFrame:
    """
    Fetch Spell cards from the API.

    Args:
        *query (str): Variable length argument list of query strings for Spell cards.
        cg (CG, optional): The card game to fetch cards from. Default is CG.ALL.
        step (int, optional): The number of cards to fetch in each request. Default is 500.
        limit (int, optional): The maximum number of cards to fetch. Default is 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pd.DataFrame: DataFrame containing Spell card data.
    """
    return fetch_st(*query, st="Spell", cg=cg, step=step, limit=limit, **kwargs)


def fetch_trap(*query: str, cg=CG.ALL, step: int = 500, limit: int = 5000, **kwargs) -> pd.DataFrame:
    """
    Fetch Trap cards from the API.

    Args:
        *query (str): Variable length argument list of query strings for Trap cards.
        cg (CG, optional): The card game to fetch cards from. Default is CG.ALL.
        step (int, optional): The number of cards to fetch in each request. Default is 500.
        limit (int, optional): The maximum number of cards to fetch. Default is 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pd.DataFrame: DataFrame containing Trap card data.
    """
    return fetch_st(*query, st="Trap", cg=cg, step=step, limit=limit, **kwargs)


def fetch_monster(
    *query: str,
    cg: CG = CG.ALL,
    step: int = 500,
    limit: int = 5000,
    **kwargs,
) -> pd.DataFrame:
    """
    Fetch Monster cards from the API.

    Args:
        *query (str): Variable length argument list of query strings for Monster cards.
        cg (CG, optional): The card game to fetch cards from. Default is CG.ALL.
        step (int, optional): The number of cards to fetch in each request. Default is 500.
        limit (int, optional): The maximum number of cards to fetch. Default is 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pd.DataFrame: DataFrame containing Monster card data.
    """
    valid_cg = cg.value
    attributes = ["DIVINE", "LIGHT", "DARK", "WATER", "EARTH", "FIRE", "WIND", "?", "???"]
    if query:
        query_str = "|?".join(query)
    else:
        query_str = card_query(*_card_properties["monster"])

    logger.info("Downloading Monsters")
    monster_df = pd.DataFrame()
    iterator = tqdm(
        attributes,
        leave=False,
        unit="attribute",
        position=1,
        dynamic_ncols=(not dirs.is_notebook),
        disable=("PM_IN_EXECUTION" in os.environ),
    )
    for att in iterator:
        iterator.set_description(att)
        with logging_redirect_tqdm():
            logger.debug("- %s", att)

        concept = f"[[Concept:CG monsters]][[Attribute::{att}]]"

        if valid_cg != "CG":
            concept += f"[[Medium::{valid_cg}]]"

        tqdm.write(f'Downloading Monsters with attribute "{att}"')
        with logging_redirect_tqdm():
            logger.info('Downloading Monsters with attribute "%s"', att)

        temp_df = client.fetch_properties(concept, query_str, step=step, limit=limit, iterator=iterator, **kwargs)
        monster_df = pd.concat([monster_df, temp_df.dropna(how="all", axis=1)], ignore_index=True, axis=0)

    monster_df = ensure_tuple_columns(monster_df)
    with logging_redirect_tqdm():
        logger.debug("- Total")
        logger.info("%s results", len(monster_df.index))

    tqdm.write(f"\r{len(monster_df.index)} results\n")

    return monster_df


def fetch_tc(
    tc_query: str | None = None,
    tc: str = "both",
    cg: CG = CG.ALL,
    step: int = 500,
    limit: int = 5000,
    **kwargs,
) -> pd.DataFrame:
    """
    Fetch Token and Counter cards from the API.

    Args:
        tc_query (str | None, optional): Custom query string for Token and Counter cards.
        tc (str, optional): The type of card to fetch. Default is "both".
        cg (CG, optional): The card game to fetch cards from. Default is CG.ALL.
        step (int, optional): The number of cards to fetch in each request. Default is 500.
        limit (int, optional): The maximum number of cards to fetch. Default is 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pd.DataFrame: DataFrame containing Token and Counter card data.
    """
    tc = tc.capitalize()
    valid_tc = {"Token", "Counter", "Both", "All"}
    valid_cg = cg.value
    concept = f"[[Page type::Card page]]"
    if tc not in valid_tc:
        raise ValueError("results: tc must be one of %r." % valid_tc)
    elif tc == "Both" or tc == "All":
        tc = "Tokens and Counters"
        concept += "[[Category:Tokens||Counters]]"
    else:
        concept += f"[[Category:{tc}s]]"

    if valid_cg != "CG":
        concept += f"[[Medium::{tc}]]"
    else:
        concept += "[[Category:TCG cards||OCG cards]]"

    if tc_query is None:
        tc_query = card_query(*_card_properties["tc"])

    tqdm.write(f"Downloading {tc}s")
    with logging_redirect_tqdm():
        logger.info("Downloading %ss", tc)

    tc_df = client.fetch_properties(concept, tc_query, step=step, limit=limit, **kwargs)

    with logging_redirect_tqdm():
        logger.debug("- Total")
        logger.info("%s results", len(tc_df.index))
    tqdm.write(f"{len(tc_df.index)} results\n")

    return tc_df


def fetch_token(*query: str, cg=CG.ALL, step: int = 500, limit: int = 5000, **kwargs) -> pd.DataFrame:
    """
    Fetch Token cards from the API.

    Args:
        *query (str): Variable length argument list of query strings for Token cards.
        cg (CG, optional): The card game to fetch cards from. Default is CG.ALL.
        step (int, optional): The number of cards to fetch in each request. Default is 500.
        limit (int, optional): The maximum number of cards to fetch. Default is 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pd.DataFrame: DataFrame containing Token card data.
    """
    return fetch_tc(*query, tc="Token", cg=cg, step=step, limit=limit, **kwargs)


def fetch_counter(*query: str, cg=CG.ALL, step: int = 500, limit: int = 5000, **kwargs) -> pd.DataFrame:
    """
    Fetch Counter cards from the API.

    Args:
        *query (str): Variable length argument list of query strings for Counter cards.
        cg (CG, optional): The card game to fetch cards from. Default is CG.ALL.
        step (int, optional): The number of cards to fetch in each request. Default is 500.
        limit (int, optional): The maximum number of cards to fetch. Default is 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pd.DataFrame: DataFrame containing Counter card data.
    """
    return fetch_tc(*query, tc="Counter", cg=cg, step=step, limit=limit, **kwargs)


def fetch_speed(*query: str, step: int = 500, limit: int = 5000, **kwargs) -> pd.DataFrame:
    """
    Fetch Speed Duel cards from the API.

    Args:
        *query (str): Variable length argument list of query strings for Speed Duel cards.
        cg (CG, optional): The card game to fetch cards from. Default is CG.ALL.
        step (int, optional): The number of cards to fetch in each request. Default is 500.
        limit (int, optional): The maximum number of cards to fetch. Default is 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pd.DataFrame: DataFrame containing Speed Duel card data.
    """
    concept = "[[Category:TCG Speed Duel cards]]"
    if query:
        query_str = "|?".join(query)
    else:
        query_str = card_query(*_card_properties["speed"])

    tqdm.write("Downloading Speed duel cards")
    with logging_redirect_tqdm():
        logger.info("Downloading Speed duel cards")

    speed_df = client.fetch_properties(
        concept,
        query_str,
        step=step,
        limit=limit,
        **kwargs,
    )

    logger.debug("- Total")

    with logging_redirect_tqdm():
        logger.info("%s results", len(speed_df.index))
    tqdm.write(f"{len(speed_df.index)} results\n")

    return speed_df


def fetch_skill(*query: str, step: int = 500, limit: int = 5000, **kwargs) -> pd.DataFrame:
    """
    Fetch Skill cards for Speed Duel from the API.
    For Rush Duel Skill cards, use fetch_rush.

    Args:
        *query (str): Variable length argument list of query strings for Skill cards.
        cg (CG, optional): The card game to fetch cards from. Default is CG.ALL.
        step (int, optional): The number of cards to fetch in each request. Default is 500.
        limit (int, optional): The maximum number of cards to fetch. Default is 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pd.DataFrame: DataFrame containing Skill card data.
    """
    concept = "[[Category:Skill Cards]][[Card type::Skill Card]]"
    if query:
        query_str = "|?".join(query)
    else:
        query_str = card_query(*_card_properties["skill"])

    tqdm.write("Downloading skill cards")
    with logging_redirect_tqdm():
        logger.info("Downloading skill cards")

    skill_df = client.fetch_properties(concept, query_str, step=step, limit=limit, **kwargs)

    with logging_redirect_tqdm():
        logger.info("%s results", len(skill_df.index))
    tqdm.write(f"{len(skill_df.index)} results\n")

    return skill_df


def fetch_rush(*query: str, step: int = 500, limit: int = 5000, **kwargs) -> pd.DataFrame:
    """
    Fetch Rush Duel cards from the API.

    Args:
        *query (str): Variable length argument list of query strings for Rush Duel cards.
        cg (CG, optional): The card game to fetch cards from. Default is CG.ALL.
        step (int, optional): The number of cards to fetch in each request. Default is 500.
        limit (int, optional): The maximum number of cards to fetch. Default is 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pd.DataFrame: DataFrame containing Rush Duel card data.
    """
    concept = f"[[Category:Rush Duel cards]][[Medium::Rush Duel]]"
    if query:
        query_str = "|?".join(query)
    else:
        query_str = card_query(*_card_properties["rush"])

    tqdm.write("Downloading Rush Duel cards")
    with logging_redirect_tqdm():
        logger.info("Downloading Rush Duel cards")

    rush_df = client.fetch_properties(concept, query_str, step=step, limit=limit, **kwargs)

    with logging_redirect_tqdm():
        logger.info("%s results", len(rush_df.index))
    tqdm.write(f"{len(rush_df.index)} results\n")

    return rush_df


def fetch_unusable(
    *query: str,
    cg: CG = CG.ALL,
    filter=True,
    step: int = 500,
    limit: int = 5000,
    **kwargs,
) -> pd.DataFrame:
    """
    Fetch Unusable cards from the API.

    Args:
        *query (str): Variable length argument list of query strings for Unusable cards.
        cg (CG, optional): The card game to fetch cards from. Default is CG.ALL.
        filter (bool, optional): Whether to filter for only Character, Non-game, and Ticket cards. Default is True.
        step (int, optional): The number of cards to fetch in each request. Default is 500.
        limit (int, optional): The maximum number of cards to fetch. Default is 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pd.DataFrame: DataFrame containing Unusable card data.
    """
    concept = "[[Category:Unusable cards]]"

    if filter:
        concept += "[[Card type::Character Card||Non-game card||Ticket Card]]"

    valid_cg = cg.value
    if valid_cg == "CG":
        concept = "OR".join([concept + f"[[{s} status::+]]" for s in ["TCG", "OCG"]])
    else:
        concept += f"[[{valid_cg} status::+]]"

    if query:
        query_str = "|?".join(query)
    else:
        query_str = card_query(default=True)

    tqdm.write("Downloading unusable cards")
    with logging_redirect_tqdm():
        logger.info("Downloading unusable cards")

    unusable_df = client.fetch_properties(concept, query_str, step=step, limit=limit, **kwargs)

    unusable_df.dropna(how="all", axis=1, inplace=True)

    with logging_redirect_tqdm():
        logger.debug("- Total")
        logger.info("%s results", len(unusable_df.index))
    tqdm.write(f"{len(unusable_df.index)} results\n")

    return unusable_df


def fetch_errata(errata: str = "all", step: int = 500, **kwargs) -> pd.DataFrame:
    """
    Fetch cards with errata from the API.

    Args:
        errata (str, optional): The type of errata to fetch. Default is "all". Valid options are "name", "type", and "all".
        step (int, optional): The number of cards to fetch in each request. Default is 500.
        **kwargs: Additional keyword arguments. Not implemented.

    Returns:
        pd.DataFrame: DataFrame containing card errata information.
    """
    errata = errata.lower()
    valid = {"name", "type", "all"}
    categories = {
        "all": "Card Errata",
        "type": "Cards with card type errata",
        "name": "Cards with name errata",
    }
    if errata not in valid:
        raise ValueError("results: errata must be one of %r." % valid)
    elif errata == "all":
        errata = "all"
        categories = list(categories.values())
    else:
        categories = list(categories["errata"])

    tqdm.write(f"Downloading {errata} errata")
    with logging_redirect_tqdm():
        logger.info("Downloading %s errata", errata)

    errata_df = pd.DataFrame(dtype=bool)
    iterator = tqdm(
        categories,
        leave=False,
        unit="initial",
        dynamic_ncols=(not dirs.is_notebook),
        disable=("PM_IN_EXECUTION" in os.environ),
        position=1,
    )
    for cat in iterator:
        desc = cat.split("Category:")[-1]
        iterator.set_description(desc)

        with logging_redirect_tqdm():
            logger.debug("- %s", cat)

        temp = client.fetch_categorymembers(cat, namespace=3010, step=step, iterator=iterator)
        errata_data = temp["title"].apply(lambda x: x.split("Card Errata:")[-1])
        errata_series = pd.Series(data=True, index=errata_data, name=desc)
        errata_df = pd.concat([errata_df, errata_series], axis=1).astype("boolean").fillna(False).sort_index()

    with logging_redirect_tqdm():
        logger.info("%s results", len(errata_df.index))

    tqdm.write(f"\r{len(errata_df.index)} results\n")

    return errata_df


# --- Set List Fetching Functions --- #
def fetch_set_list_pages(cg: CG = CG.ALL, step: int = 500, limit=5000, **kwargs) -> pd.DataFrame:
    """
    Fetch the list of 'Set Card Lists' pages from the API.

    Args:
        cg (CG, optional): The card game to fetch set lists for. Default is CG.ALL.
        step (int, optional): The number of pages to fetch in each request. Default is 500.
        limit (int, optional): The maximum number of pages to fetch. Default is 5000.
        **kwargs: Additional keyword arguments to pass to fetch_properties.

    Returns:
        pd.DataFrame: DataFrame containing the list of 'Set Card Lists' pages.
    """
    valid_cg = cg.value
    if valid_cg == "CG":
        category = ["TCG Set Card Lists", "OCG Set Card Lists"]
    else:
        category = [f"{valid_cg} Set Card Lists"]

    tqdm.write("Downloading list of 'Set Card Lists' pages")
    logger.info("Downloading list of 'Set Card Lists' pages")
    set_list_pages = pd.DataFrame()
    iterator = tqdm(
        category,
        leave=False,
        unit="category",
        dynamic_ncols=(not dirs.is_notebook),
        disable=("PM_IN_EXECUTION" in os.environ),
        position=1,
    )
    for cat in iterator:
        with logging_redirect_tqdm():
            logger.info(cat)

        iterator.set_description(cat.split("Category:")[-1])
        temp = client.fetch_categorymembers(cat, namespace=None, step=step, iterator=iterator)
        sub_categories = pd.DataFrame(temp)["title"]
        sub_iterator = tqdm(
            sub_categories,
            leave=False,
            unit="subcategory",
            dynamic_ncols=(not dirs.is_notebook),
            disable=("PM_IN_EXECUTION" in os.environ),
            position=2,
        )
        for sub_cat in sub_iterator:
            with logging_redirect_tqdm():
                logger.debug("- %s", sub_cat)

            tqdm.write(f"Downloading properties for {sub_cat}")

            sub_iterator.set_description(sub_cat.split("Category:")[-1])
            temp = client.fetch_properties(
                f"[[{sub_cat}]]",
                query="|?Modification date",
                step=step,
                limit=limit,
                iterator=sub_iterator,
                **kwargs,
            )
            set_list_pages = pd.concat([set_list_pages, pd.DataFrame(temp)])

    set_list_pages = ensure_tuple_columns(set_list_pages)
    return set_list_pages


def fetch_all_set_lists(cg: CG = CG.ALL, step: int = 40, **kwargs) -> pd.DataFrame:
    """
    Fetch all set lists from the API.

    Args:
        cg (CG, optional): The card game to fetch set lists for. Default is CG.ALL.
        step (int, optional): The number of set lists to fetch in each request. Default is 40.
        **kwargs: Additional keyword arguments to pass to fetch_set_list_pages and fetch_set_lists.

    Returns:
        pd.DataFrame: DataFrame containing all set list data.
    """
    sets = fetch_set_list_pages(cg, **kwargs)
    keys = sets["Page name"]

    all_set_lists_df = pd.DataFrame(columns=["Set", "Card number", "Name", "Rarity", "Print", "Quantity", "Region"])
    total_success = 0
    total_error = 0

    tqdm.write(f" Downloading set lists for {len(keys)} sets")
    logger.info("Downloading set lists for %s sets", len(keys))
    for i in trange(np.ceil(len(keys) / step).astype(int), leave=False):
        success = 0
        error = 0

        first = i * step
        last = (i + 1) * step

        with logging_redirect_tqdm():
            logger.debug("Iteration %s:", i)
            result = client.fetch_set_lists(*keys[first:last])

        if result is None:
            continue
        set_lists_df, success, error = result
        set_lists_df = set_lists_df.merge(sets, on="Page name", how="left").drop("Page name", axis=1)
        all_set_lists_df = pd.concat([all_set_lists_df, set_lists_df], ignore_index=True)
        total_success += success
        total_error += error

    all_set_lists_df = all_set_lists_df.convert_dtypes()
    all_set_lists_df.sort_values(by=["Set", "Region", "Card number"]).reset_index(inplace=True)

    tqdm.write(f"{total_success} set lists received - {total_error} missing")
    with logging_redirect_tqdm():
        logger.info("%s set lists received - %s missing\n", total_success, total_error)

    return all_set_lists_df
