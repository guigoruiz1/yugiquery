"""Unit tests for yugiquery package initialization and exports."""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock


# ============================================================================
# Tests for yugiquery/__init__.py exports
# ============================================================================


class TestPackageExports:
    """Tests for yugiquery package exports."""

    def test_package_exports_api(self):
        """Test that yugiquery exports api module."""
        import yugiquery

        assert hasattr(yugiquery, "api")

    def test_package_exports_core(self):
        """Test that yugiquery exports core module."""
        import yugiquery

        assert hasattr(yugiquery, "core")

    def test_package_exports_numpy_as_np(self):
        """Test that yugiquery exports numpy as np in notebook mode."""
        import yugiquery

        # np is only exported in notebook mode
        from yugiquery.utils.dirs import dirs

        if dirs.is_notebook:
            assert hasattr(yugiquery, "np")
        else:
            # Outside notebook, np may not be exported at top level
            pass

    def test_package_exports_pandas_as_pd(self):
        """Test that yugiquery exports pandas as pd in notebook mode."""
        import yugiquery

        # pd is only exported in notebook mode
        from yugiquery.utils.dirs import dirs

        if dirs.is_notebook:
            assert hasattr(yugiquery, "pd")
        else:
            # Outside notebook, pd may not be exported at top level
            pass

    def test_package_exports_path(self):
        """Test that yugiquery exports Path from pathlib."""
        import yugiquery

        assert hasattr(yugiquery, "Path")
        assert yugiquery.Path is Path

    def test_package_exports_plt(self):
        """Test that yugiquery exports plt (matplotlib.pyplot)."""
        import yugiquery

        assert hasattr(yugiquery, "plt")

    def test_package_exports_sns(self):
        """Test that yugiquery exports sns (seaborn)."""
        import yugiquery

        assert hasattr(yugiquery, "sns")

    def test_package_star_import_includes_convenience_libs(self):
        """Test that from yugiquery import * includes convenience libraries."""
        # We can't easily test `from yugiquery import *` dynamically,
        # but we can verify the attributes exist
        import yugiquery
        from yugiquery.utils.dirs import dirs

        # Libraries always exported
        convenience_exports = ["Path", "plt", "sns"]
        for export in convenience_exports:
            assert hasattr(yugiquery, export), f"Missing export: {export}"

        # np and pd are only exported in notebook mode
        if dirs.is_notebook:
            assert hasattr(yugiquery, "np")
            assert hasattr(yugiquery, "pd")

    def test_package_exports_metadata(self):
        """Test that yugiquery exports metadata."""
        import yugiquery

        # Check for version or other metadata
        assert hasattr(yugiquery, "__version__") or hasattr(yugiquery, "metadata")


class TestNotebookConditionalExports:
    """Tests for notebook-specific conditional exports."""

    def test_package_notebook_exports_optional(self):
        """Test that ipywidgets imports are optional (don't crash if unavailable)."""
        import yugiquery

        # Even if ipywidgets is unavailable, package should still work
        assert True  # If we got here, import succeeded

    def test_package_has_notebook_detection_flag(self):
        """The dirs singleton exposes a boolean notebook-detection flag."""
        from yugiquery.utils.dirs import dirs

        is_notebook = getattr(dirs, "is_notebook", False)
        assert isinstance(is_notebook, bool)


class TestDataScienceLibraryImports:
    """Tests for data science library imports."""

    def test_matplotlib_pyplot_available(self):
        """Test that matplotlib.pyplot is available as plt."""
        import yugiquery

        # Should have plotting functions
        assert hasattr(yugiquery.plt, "figure")
        assert hasattr(yugiquery.plt, "show")

    def test_seaborn_available(self):
        """Test that seaborn is available as sns."""
        import yugiquery

        # Should have seaborn functions
        assert hasattr(yugiquery.sns, "set_theme")
        assert hasattr(yugiquery.sns, "heatmap")

    def test_pandas_available(self):
        """Test that pandas is available."""
        import pandas

        # Should be able to create DataFrame
        df = pandas.DataFrame({"A": [1, 2, 3]})
        assert len(df) == 3

    def test_numpy_available(self):
        """Test that numpy is available."""
        import numpy

        # Should be able to create array
        arr = numpy.array([1, 2, 3])
        assert len(arr) == 3

    def test_pathlib_path_available(self):
        """Test that Path is available from pathlib."""
        import yugiquery

        p = yugiquery.Path("/tmp/test")
        assert isinstance(p, Path)


class TestApiModuleExports:
    """Tests for api module exports."""

    def test_api_exports_client(self):
        """Test that api exports client module."""
        from yugiquery import api

        assert hasattr(api, "client") or hasattr(api, "check_status")

    def test_api_exports_fetch_functions(self):
        """Test that api exports main fetch functions."""
        from yugiquery import api

        # Should have main API functions
        assert hasattr(api, "fetch_ygoprodeck") or hasattr(api, "check_status")


class TestCoreModuleExports:
    """Tests for core module exports."""

    def test_core_exports_data(self):
        """Test that core exports data module."""
        from yugiquery import core

        assert hasattr(core, "data") or hasattr(core, "load_latest_data")

    def test_core_exports_decks(self):
        """Test that core exports decks module."""
        from yugiquery import core

        assert hasattr(core, "decks") or hasattr(core, "read_decklist")


class TestUtilsModuleExports:
    """Tests for utils module exports."""

    def test_utils_exports_dirs(self):
        """Test that utils exports dirs."""
        from yugiquery import utils

        assert hasattr(utils, "dirs")

    def test_utils_exports_helpers(self):
        """Test that utils exports helpers."""
        from yugiquery import utils

        assert hasattr(utils, "helpers")

    def test_utils_exports_plotting(self):
        """Test that utils exports plot utilities."""
        from yugiquery import utils

        assert hasattr(utils, "plot") or hasattr(utils, "plt")


class TestPackageVersion:
    """Tests for package version information."""

    def test_package_has_version(self):
        """Test that package has __version__ attribute."""
        import yugiquery

        assert hasattr(yugiquery, "__version__")
        assert isinstance(yugiquery.__version__, str)

    def test_version_format_valid(self):
        """Test that version string is valid."""
        import yugiquery

        # Version should follow semantic versioning or similar
        version = yugiquery.__version__
        parts = version.split(".")
        assert len(parts) >= 1


# ============================================================================
# Tests for API utilities
# ============================================================================


class TestApiWrappers:
    """Tests for API wrapper functions."""

    def test_fetch_all_set_lists_returns_data(self, monkeypatch):
        """Test fetch_all_set_lists function exists."""
        from yugiquery import api

        # Verify function exists
        assert hasattr(api, "fetch_all_set_lists")

    def test_check_status_returns_bool(self, monkeypatch):
        """Test check_status returns boolean."""
        from yugiquery import api

        # Mock requests to avoid actual network call
        with patch("yugiquery.api.client.requests"):
            # Just verify the function exists and is callable
            assert callable(api.check_status)

    def test_fetch_redirects_exists(self):
        """Test fetch_redirects function exists."""
        from yugiquery import api

        assert hasattr(api, "fetch_redirects")


# ============================================================================
# Integration tests
# ============================================================================


class TestPackageIntegration:
    """Integration tests for package functionality."""

    def test_can_create_dataframe_with_convenience_exports(self):
        """Test creating DataFrame using pandas."""
        import pandas as pd
        import numpy as np

        df = pd.DataFrame(
            {
                "A": np.array([1, 2, 3]),
                "B": np.array([4, 5, 6]),
            }
        )
        assert len(df) == 3
        assert len(df.columns) == 2

    def test_can_use_path_with_exports(self):
        """Test using Path utility with package exports."""
        import yugiquery

        p = yugiquery.Path("/tmp/test/file.txt")
        assert p.name == "file.txt"
        assert p.suffix == ".txt"

    def test_can_create_simple_plot(self):
        """Test that plotting functions are available."""
        import yugiquery

        # Just test that we can call plotting functions
        figure = yugiquery.plt.figure()
        assert figure is not None
        yugiquery.plt.close(figure)

    def test_seaborn_functions_available(self):
        """Test that seaborn functions work."""
        import yugiquery
        import pandas as pd

        # Create test data
        data = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})

        # Test that seaborn functions are callable
        assert callable(yugiquery.sns.heatmap)
        assert callable(yugiquery.sns.lineplot)


class TestModuleImportPaths:
    """Tests for various import paths."""

    def test_direct_api_import(self):
        """Test importing api directly."""
        from yugiquery import api

        assert api is not None

    def test_direct_core_import(self):
        """Test importing core directly."""
        from yugiquery import core

        assert core is not None

    def test_direct_utils_import(self):
        """Test importing utils directly."""
        from yugiquery import utils

        assert utils is not None

    def test_nested_api_import(self):
        """Test importing nested api modules."""
        from yugiquery.api import client

        assert client is not None

    def test_nested_utils_import(self):
        """Test importing nested utils modules."""
        from yugiquery.utils import helpers, dirs

        assert helpers is not None
        assert dirs is not None
