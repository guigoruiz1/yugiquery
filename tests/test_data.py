from pathlib import Path

import pandas as pd

from yugiquery.core import data as core_data


def _write_cards_file(path: Path, secondary_type: str, debut: str):
    frame = pd.DataFrame(
        {
            "Name": ["Card"],
            "Secondary type": [secondary_type],
            "TCG debut": [debut],
        }
    )
    frame.to_csv(path, index=False, compression="bz2")


def test_load_latest_data_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(core_data.dirs, "WORK", tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)

    result = core_data.load_latest("cards")
    assert result == (None, None)


def test_load_latest_data_uses_latest_file_and_parses(monkeypatch, tmp_path):
    monkeypatch.setattr(core_data.dirs, "WORK", tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    old_file = data_dir / "cards_data_20240101.bz2"
    new_file = data_dir / "cards_data_20240201.bz2"
    _write_cards_file(old_file, "('Old',)", "2023-01-01")
    _write_cards_file(new_file, "('Flip', 'Effect')", "2024-02-20")

    ctimes = {str(old_file): 1, str(new_file): 2}
    monkeypatch.setattr(core_data.os.path, "getctime", lambda p: ctimes[str(p)])

    frame, ts = core_data.load_latest("cards", tuple_cols=["Secondary type"], return_ts=True)

    assert frame is not None
    assert frame.loc[0, "Secondary type"] == ("Flip", "Effect")
    assert str(frame.loc[0, "TCG debut"]).startswith("2024-02-20")
    assert ts is not None
    assert ts.format("YYYYMMDD") == "20240201"


def test_load_latest_data_skips_invalid_tuple_literal(monkeypatch, tmp_path):
    monkeypatch.setattr(core_data.dirs, "WORK", tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    file_path = data_dir / "cards_data_20240301.bz2"
    _write_cards_file(file_path, "not_a_tuple_literal", "2024-03-10")
    monkeypatch.setattr(core_data.os.path, "getctime", lambda p: 1)

    frame = core_data.load_latest("cards", tuple_cols=["Secondary type"])
    assert frame is not None
    assert frame.loc[0, "Secondary type"] == "not_a_tuple_literal"
