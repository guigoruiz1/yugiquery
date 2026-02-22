from pathlib import Path

import pytest

from yugiquery.utils.dirs import dirs


def test_assets_user_and_get_asset(tmp_path, monkeypatch):
    d = dirs
    # point WORK to tmp_path and create an assets folder
    monkeypatch.setattr(d, "WORK", tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    filep = assets / "foo.txt"
    filep.write_text("hello")

    # ASSETS.user should point to tmp_path/assets
    assert d.ASSETS.user == assets

    # get_asset should return the user asset path
    got = d.get_asset("foo.txt")
    assert got == filep


def test_get_notebook_case_insensitive_and_find(tmp_path, monkeypatch):
    d = dirs
    monkeypatch.setattr(d, "WORK", tmp_path)
    notebooks = tmp_path / "notebooks"
    notebooks.mkdir()
    nb = notebooks / "MyNote.ipynb"
    nb.write_text("{}")

    # case-insensitive lookup
    got = d.get_notebook("mynote")
    assert got.name.lower() == "mynote.ipynb"

    # find_notebooks user
    found = d.find_notebooks("user")
    assert any(p.name.lower() == "mynote.ipynb" for p in found)
