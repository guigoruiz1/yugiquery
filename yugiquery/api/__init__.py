# yugiquery/api/__init__.py

# -*- coding: utf-8 -*-

from .client import (
    check_status,
    fetch_backlinks,
    fetch_categorymembers,
    fetch_properties,
    fetch_redirect_dict,
    fetch_redirects,
    fetch_set_info,
    fetch_set_lists,
    fetch_ygoprodeck,
    requests,
    URLS,
)
from .wrappers import (
    CG,
    card_properties,
    card_query,
    download_media,
    fetch_all_set_lists,
    fetch_bandai,
    fetch_counter,
    fetch_errata,
    fetch_monster,
    fetch_page_images,
    fetch_rush,
    fetch_set_list_pages,
    fetch_skill,
    fetch_speed,
    fetch_st,
    fetch_token,
    fetch_unusable,
    update_rarities,
    update_regions,
)
