"""Unit tests for deck-related core helpers."""

import pytest
import pandas as pd
from pathlib import Path

from yugiquery.core import decks


# ============================================================================
# Tests for core/decks.py
# ============================================================================


class TestReadDecklist:
    """Tests for read_decklist function."""

    def test_read_decklist_basic(self, tmp_path):
        """Test reading a basic decklist file."""
        decklist_file = tmp_path / "test_deck.txt"
        decklist_content = """Main:
3x Blue-Eyes White Dragon
2x Dark Magician

Extra:
1x Blue-Eyes Ultimate Dragon
"""
        decklist_file.write_text(decklist_content)

        result = decks.read_decklist(decklist_file)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert result[result["Name"] == "Blue-Eyes White Dragon"].iloc[0]["Count"] == 3

    def test_read_decklist_with_path_object(self, tmp_path):
        """Test read_decklist accepts Path objects."""
        decklist_file = tmp_path / "deck.txt"
        decklist_file.write_text("Main:\n1x Blue-Eyes White Dragon\n")

        result = decks.read_decklist(decklist_file)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_read_decklist_with_string_path(self, tmp_path):
        """Test read_decklist accepts string paths."""
        decklist_file = tmp_path / "deck.txt"
        decklist_file.write_text("Main:\n2x Exodia\n")

        result = decks.read_decklist(str(decklist_file))
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["Count"] == 2

    def test_read_decklist_section_parsing(self, tmp_path):
        """Test that sections are correctly identified."""
        decklist_file = tmp_path / "deck.txt"
        content = """Main:
3x Monster
2x Spell

Extra:
1x Synchro
"""
        decklist_file.write_text(content)

        result = decks.read_decklist(decklist_file)
        # Should have 3 rows: 2 in Main, 1 in Extra
        assert len(result) == 3

    def test_read_decklist_deck_name_from_filename(self, tmp_path):
        """Test that deck name is extracted from filename."""
        decklist_file = tmp_path / "my_test_deck.txt"
        decklist_file.write_text("Main:\n1x Card\n")

        result = decks.read_decklist(decklist_file)
        assert result.iloc[0]["Deck"] == "My test deck"

    def test_read_decklist_returns_dataframe(self, tmp_path):
        """Test that read_decklist always returns a DataFrame."""
        decklist_file = tmp_path / "deck.txt"
        decklist_file.write_text("Main:\n1x Card A\n2x Card B\n")

        result = decks.read_decklist(decklist_file)
        # Result should have proper structure
        assert isinstance(result, pd.DataFrame)
        assert "Name" in result.columns
        assert "Count" in result.columns
        assert "Section" in result.columns
        assert "Deck" in result.columns


class TestGetDecklists:
    """Tests for get_decklists function."""

    def test_get_decklists_with_specific_files(self, tmp_path, monkeypatch):
        """Test get_decklists with specific files provided."""
        monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)  # Silence prints

        deck1 = tmp_path / "deck1.txt"
        deck2 = tmp_path / "deck2.txt"
        deck1.write_text("Main:\n2x Card1\n")
        deck2.write_text("Main:\n3x Card2\n")

        result = decks.get_decklists(deck1, deck2)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_get_decklists_section_normalization(self, tmp_path, monkeypatch):
        """Test that decklist sections are normalized (Monster/Spell/Trap -> Main)."""
        monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

        decklist_file = tmp_path / "deck.txt"
        content = """Monster:
1x Monster Card

Spell:
1x Spell Card

Trap:
1x Trap Card
"""
        decklist_file.write_text(content)

        result = decks.get_decklists(decklist_file)
        # After replacement, all should be "Main"
        assert all(result["Section"] == "Main")

    def test_get_decklists_returns_dataframe(self, tmp_path, monkeypatch):
        """Test that get_decklists returns a DataFrame."""
        monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

        deck_file = tmp_path / "test.txt"
        deck_file.write_text("Main:\n1x Card\n")

        result = decks.get_decklists(deck_file)
        assert isinstance(result, pd.DataFrame)


class TestAssignDeck:
    """Tests for assign_deck function."""

    def test_assign_deck_returns_dataframe(self):
        """Test assign_deck returns a DataFrame."""
        collection_df = pd.DataFrame(
            {
                "Name": ["Card1", "Card2"],
                "Count": [1, 2],
            }
        )

        deck_df = pd.DataFrame(
            {
                "Name": ["Card1"],
                "Count": [1],
                "Section": ["Main"],
                "Deck": ["TestDeck"],
            }
        )

        result = decks.assign_deck(collection_df, deck_df)
        assert isinstance(result, pd.DataFrame)
        assert "missing" in result.columns

    def test_assign_deck_with_return_collection_false(self):
        """Test assign_deck with return_collection=False."""
        collection_df = pd.DataFrame(
            {
                "Name": ["Card1"],
                "Count": [1],
            }
        )

        deck_df = pd.DataFrame(
            {
                "Name": ["Card1"],
                "Count": [1],
                "Section": ["Main"],
                "Deck": ["Deck1"],
            }
        )

        result = decks.assign_deck(collection_df, deck_df, return_collection=False)
        assert isinstance(result, pd.DataFrame)

    def test_assign_deck_with_return_collection_true(self):
        """Test assign_deck with return_collection=True."""
        collection_df = pd.DataFrame(
            {
                "Name": ["Card1", "Card2"],
                "Count": [1, 1],
            }
        )

        deck_df = pd.DataFrame(
            {
                "Name": ["Card1"],
                "Count": [1],
                "Section": ["Main"],
                "Deck": ["Deck1"],
            }
        )

        result = decks.assign_deck(collection_df, deck_df, return_collection=True)
        assert isinstance(result, pd.DataFrame)


class TestCoreModuleExports:
    """Tests for core module exports."""

    def test_core_imports_decks(self):
        """Test that core module has access to decks functions."""
        from yugiquery import core

        # Either imported as module or individual functions
        assert hasattr(core, "decks") or hasattr(core, "read_decklist") or hasattr(core, "get_decklists")

    def test_can_import_decks_directly(self):
        """Test direct import of decks module."""
        from yugiquery.core import decks

        assert hasattr(decks, "read_decklist")
        assert hasattr(decks, "get_decklists")
        assert hasattr(decks, "assign_deck")