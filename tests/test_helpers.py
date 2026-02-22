import hashlib
import json
import os

import arrow
import pytest

from yugiquery.utils import helpers


def test_md5_basic():
    assert helpers.md5("abc") == hashlib.md5(b"abc").hexdigest()


def test_escape_chars_default():
    s = "a_b.c-+@#=d"
    out = helpers.escape_chars(s)
    for ch in ["_", ".", "-", "+", "#", "@", "="]:
        assert f"\\{ch}" in out


def test_auto_or_bool_variants():
    assert helpers.auto_or_bool(None) is True
    assert helpers.auto_or_bool("auto") == "auto"
    # Note: non-empty strings are truthy
    assert helpers.auto_or_bool("False") is True


def test_get_ts_granularity_examples():
    assert helpers.get_ts_granularity(0) == ["second"]
    assert helpers.get_ts_granularity(3600) == ["hour"]
    assert helpers.get_ts_granularity(3661) == ["hour", "minute", "second"]


def test_make_filename_with_and_without_previous():
    ts = arrow.get("2020-01-01T00:00:00+00:00")
    prev = arrow.get("2019-12-31T23:00:00+00:00")
    f1 = helpers.make_filename("Report", ts)
    assert f1.startswith("report_data_") and f1.endswith(".bz2")
    f2 = helpers.make_filename("Report", ts, previous_timestamp=prev)
    assert "changelog" in f2 and f2.endswith(".bz2")


def test_load_json_and_missing(tmp_path, capsys):
    p = tmp_path / "test.json"
    data = {"x": 1}
    p.write_text(json.dumps(data))
    assert helpers.load_json(str(p)) == data

    # missing file returns empty dict and prints warning
    missing = helpers.load_json(str(p.parent / "nope.json"))
    assert missing == {}


def test_load_secrets_env_and_file(tmp_path, monkeypatch):
    # env overrides .env file
    monkeypatch.setenv("A", "envA")
    env_res = helpers.load_secrets(["A"])
    assert env_res["A"] == "envA"

    # secrets file
    env_file = tmp_path / ".env"
    env_file.write_text("B=fromfile\nC=three")
    # ensure requested subset is returned
    res = helpers.load_secrets(["B", "C"], secrets_file=str(env_file))
    assert res["B"] == "fromfile"

    # required flag raises KeyError when missing
    with pytest.raises(KeyError):
        helpers.load_secrets(["MISSING"], required=True)
