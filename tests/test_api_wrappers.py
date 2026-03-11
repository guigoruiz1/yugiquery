import json

import pandas as pd
import pytest

from yugiquery.api import client, wrappers


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def test_card_query_default_and_overrides():
    query = wrappers.card_query(default=True, level=False, image_URL=True)
    assert "|?English%20name=Name" in query
    assert "|?Password" in query
    assert "|?Card%20image" in query
    # Explicitly disabled property should be absent when default=True.
    assert "|?Level%20string=Level" not in query


def test_update_rarities_merges_and_saves(tmp_path, monkeypatch):
    rarities_file = tmp_path / "rarities.json"
    rarities_file.write_text("{}")

    monkeypatch.setattr(wrappers.dirs, "get_asset", lambda *args: rarities_file)
    monkeypatch.setattr(wrappers, "load_json", lambda _: {"SR": "Super Rare"})
    monkeypatch.setattr(
        wrappers.client,
        "fetch_redirect_dict",
        lambda **kwargs: {"Super": "Super Rare"},
    )

    result = wrappers.update_rarities(save=True)
    assert result["SR"] == "Super Rare"
    assert result["Super"] == "Super Rare"

    saved = json.loads(rarities_file.read_text())
    assert saved == result


def test_update_regions_merges_without_save(monkeypatch):
    monkeypatch.setattr(wrappers.dirs, "get_asset", lambda *args: "unused.json")
    monkeypatch.setattr(wrappers, "load_json", lambda _: {"EU": "Europe"})
    monkeypatch.setattr(
        wrappers.client,
        "fetch_redirect_dict",
        lambda **kwargs: {"Europe region": "Europe"},
    )

    result = wrappers.update_regions(save=False)
    assert result == {"EU": "Europe", "Europe region": "Europe"}


def test_fetch_st_builds_query_and_filters(monkeypatch):
    calls = []

    def _fake_fetch_properties(concept, query, **kwargs):
        calls.append((concept, query, kwargs))
        return pd.DataFrame({"Name": ["Card"]})

    monkeypatch.setattr(wrappers.client, "fetch_properties", _fake_fetch_properties)

    result = wrappers.fetch_st(st_query="|?X", st="spell", cg=wrappers.CG.TCG, step=7, limit=9)

    assert len(result.index) == 1
    concept, query, kwargs = calls[0]
    assert "[[Concept:CG non-monster cards]]" in concept
    assert "[[Card type::Spell card]]" in concept
    assert "[[Medium::TCG]]" in concept
    assert query == "|?X"
    assert kwargs["step"] == 7
    assert kwargs["limit"] == 9


def test_fetch_st_rejects_invalid_st():
    with pytest.raises(ValueError):
        wrappers.fetch_st(st="invalid")


def test_fetch_monster_excludes_tokens(monkeypatch):
    monkeypatch.setattr(
        wrappers.client,
        "fetch_properties",
        lambda *args, **kwargs: pd.DataFrame({"Primary type": ["Monster Token", "Effect Monster"]}),
    )

    result = wrappers.fetch_monster(exclude_token=True)
    assert not (result["Primary type"] == "Monster Token").any()
    assert (result["Primary type"] == "Effect Monster").all()


def test_fetch_set_list_pages_collects_subcategories(monkeypatch):
    fetch_properties_calls = []

    def _fake_categorymembers(*args, **kwargs):
        return [{"title": "Category:Demo subcategory"}]

    def _fake_fetch_properties(concept, query, **kwargs):
        fetch_properties_calls.append((concept, query))
        return pd.DataFrame({"Page name": ["List A"], "Modification date": ["2024-01-01"]})

    monkeypatch.setattr(wrappers.client, "fetch_categorymembers", _fake_categorymembers)
    monkeypatch.setattr(wrappers.client, "fetch_properties", _fake_fetch_properties)

    result = wrappers.fetch_set_list_pages(cg=wrappers.CG.TCG)
    assert not result.empty
    assert fetch_properties_calls[0][0] == "[[Category:Demo subcategory]]"


def test_fetch_all_set_lists_batches_and_merges(monkeypatch):
    set_pages = pd.DataFrame(
        {
            "Page name": ["P1", "P2", "P3"],
            "Modification date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    )
    monkeypatch.setattr(wrappers, "fetch_set_list_pages", lambda *args, **kwargs: set_pages)

    batch_sizes = []

    def _fake_fetch_set_lists(*keys, **kwargs):
        batch_sizes.append(len(keys))
        df = pd.DataFrame(
            {
                "Page name": list(keys),
                "Set": ["S"] * len(keys),
                "Card number": [f"CN{i}" for i in range(len(keys))],
                "Name": ["Card"] * len(keys),
                "Rarity": ["Common"] * len(keys),
                "Print": ["1st"] * len(keys),
                "Quantity": [1] * len(keys),
                "Region": ["EN"] * len(keys),
            }
        )
        return df, len(keys), 0

    monkeypatch.setattr(wrappers.client, "fetch_set_lists", _fake_fetch_set_lists)

    result = wrappers.fetch_all_set_lists(step=2)
    assert len(result.index) == 3
    assert batch_sizes == [2, 1]


def test_fetch_page_images_featured_and_gallery(monkeypatch):
    payload = {
        "query": {
            "pages": {
                "1": {
                    "title": "Blue-Eyes White Dragon",
                    "original": {"source": "https://img.test/BlueEyes.png"},
                    "images": [{"title": "File:BlueEyes.png"}],
                }
            }
        }
    }
    monkeypatch.setattr(client.requests, "get", lambda **kwargs: _Resp(payload))

    featured = client.fetch_page_images("Blue-Eyes White Dragon", featured=True)
    gallery = client.fetch_page_images("Blue-Eyes White Dragon", featured=False)

    assert featured["Blue-Eyes White Dragon"] == "BlueEyes.png"
    assert gallery["Blue-Eyes White Dragon"] == ["BlueEyes.png"]
