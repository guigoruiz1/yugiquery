# yugiquery/api/__init__.py

# -*- coding: utf-8 -*-

# --- Imports: Client Submodule --- #
from .client import (
    check_status,
    download_media,
    fetch_backlinks,
    fetch_categorymembers,
    fetch_properties,
    fetch_redirect_dict,
    fetch_redirects,
    fetch_page_images,
    fetch_set_info,
    fetch_set_lists,
    fetch_ygoprodeck,
    requests,
    URLS,
)

# --- Imports: Wrappers Submodule --- #
from .wrappers import (
    CG,
    _card_properties,
    card_query,
    fetch_all_set_lists,
    fetch_bandai,
    fetch_counter,
    fetch_errata,
    fetch_monster,
    fetch_rush,
    fetch_set_list_pages,
    fetch_skill,
    fetch_speed,
    fetch_st,
    fetch_token,
    fetch_tc,
    get_ygoprodeck,
    fetch_unusable,
    update_rarities,
    update_regions,
)
