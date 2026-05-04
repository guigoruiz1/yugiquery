import hashlib
import json
import os
from typing import cast

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
    assert helpers.auto_or_bool(cast(str, None)) is True
    assert helpers.auto_or_bool("auto") == "auto"
    # Note: non-empty strings are truthy
    assert helpers.auto_or_bool("False") is True


def test_md5_unicode_and_consistency():
    value = "café"
    assert len(helpers.md5(value)) == 32
    assert helpers.md5(value) == helpers.md5(value)


def test_md5_long_string():
    assert len(helpers.md5("x" * 10000)) == 32


def test_escape_chars_without_specials():
    assert helpers.escape_chars("abcdef123") == "abcdef123"


def test_escape_chars_all_specials():
    special = "_.-+@#="
    escaped = helpers.escape_chars(special)
    for ch in special:
        assert f"\\{ch}" in escaped


def test_auto_or_bool_yes_string():
    assert helpers.auto_or_bool("yes") is True


def test_get_ts_granularity_examples():
    assert helpers.get_ts_granularity(0) == ["second"]
    assert helpers.get_ts_granularity(60) == ["minute"]
    assert helpers.get_ts_granularity(3600) == ["hour"]
    assert helpers.get_ts_granularity(86400) == ["day"]
    assert helpers.get_ts_granularity(3661) == ["hour", "minute", "second"]


def test_get_ts_granularity_mixed_units():
    result = helpers.get_ts_granularity(5445)
    assert isinstance(result, list)
    assert len(result) >= 1


def test_make_filename_with_and_without_previous():
    ts = arrow.get("2020-01-01T00:00:00+00:00")
    prev = arrow.get("2019-12-31T23:00:00+00:00")
    f1 = helpers.make_filename("Report", ts)
    assert f1.startswith("report_data_") and f1.endswith(".bz2")
    f2 = helpers.make_filename("Report", ts, previous_timestamp=prev)
    assert "changelog" in f2 and f2.endswith(".bz2")


def test_make_filename_lowercase_and_timestamp_content():
    ts = arrow.get("2024-01-15T10:30:00")
    filename = helpers.make_filename("MyReport", ts)
    assert filename.startswith("myreport_data_")
    assert "2024" in filename or "20240115" in filename


def test_load_json_and_missing(tmp_path, capsys):
    p = tmp_path / "test.json"
    data = {"x": 1}
    p.write_text(json.dumps(data))
    assert helpers.load_json(str(p)) == data

    # missing file returns empty dict and prints warning
    missing = helpers.load_json(str(p.parent / "nope.json"))
    assert missing == {}


def test_load_json_edge_cases(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text("")
    assert helpers.load_json(str(empty)) == {}

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{invalid json}")
    assert isinstance(helpers.load_json(str(invalid)), dict)

    nested = tmp_path / "nested.json"
    nested.write_text(json.dumps({"outer": {"inner": {"deep": "value"}}}))
    assert helpers.load_json(str(nested))["outer"]["inner"]["deep"] == "value"

    arrays = tmp_path / "arrays.json"
    arrays.write_text(json.dumps({"items": [1, 2, 3, 4, 5]}))
    assert len(helpers.load_json(str(arrays))["items"]) == 5


def test_load_json_with_special_path(tmp_path):
    special_path = tmp_path / "dir with spaces" / "file-2024.json"
    special_path.parent.mkdir(parents=True)
    special_path.write_text(json.dumps({"test": "data"}))

    result = helpers.load_json(str(special_path))
    assert result["test"] == "data"


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


def test_load_secrets_additional_cases(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_VAR", "env_value")
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_VAR=file_value\nVAR1=value1\nVAR2=value2\nVAR3=value3")

    result = helpers.load_secrets(["TEST_VAR"], secrets_file=str(env_file))
    assert result["TEST_VAR"] == "env_value"

    subset = helpers.load_secrets(["VAR1", "VAR2"], secrets_file=str(env_file))
    assert len(subset) == 2
    assert subset["VAR1"] == "value1"

    monkeypatch.setenv("EXISTING", "value")
    optional = helpers.load_secrets(["EXISTING", "NONEXISTENT"], required=False)
    assert "EXISTING" in optional
