# yugiquery/utils/dirs.py

# -*- coding: utf-8 -*-

# =========================== #
# Directory Management module #
# =========================== #

# ======= #
# Imports #
# ======= #

# Standard library imports
import os
import sysconfig
from pathlib import Path

# Third-party imports
from IPython import get_ipython
from platformdirs import user_data_dir, site_data_dir

# ======= #
# Classes #
# ======= #


class Dirs:
    """
    Singleton class to manage directory paths for the application.


    :ivar APP: The path to the application (YugiQuery) directory.
    :ivar ASSETS: The path to the assets directory.
    :ivar DATA: The path to the data directory.
    :ivar NOTEBOOKS: The path to the notebooks directory.
    :ivar REPORTS: The path to the reports directory.
    :ivar SHARE: The path to the share directory.
    :ivar UTILS: The path to the utils subpackage directory.
    :ivar WORK: The path to the working directory.

    :vartype APP: Path
    :vartype ASSETS: Path
    :vartype DATA: Path
    :vartype NOTEBOOKS: Path
    :vartype REPORTS: Path
    :vartype SHARE: Path
    :vartype UTILS: Path:
    :vartype WORK: Path
    """

    _instance = None
    APP: Path = None
    ASSETS: Path = None
    DATA: Path = None
    NOTEBOOKS: Path = None
    REPORTS: Path = None
    SHARE: Path = None
    UTILS: Path = None
    WORK: Path = None

    def __new__(cls, *args, **kwargs):
        """
        Create a new instance of the Dirs class if one does not already exist.
        """
        if cls._instance is None:
            cls._instance = super(Dirs, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize_dirs()
        return cls._instance

    def _initialize_dirs(self):
        """
        Initialize directory paths for various application components.
        """
        # Initialize directory paths
        self.UTILS = Path(__file__).resolve().parent
        self.APP = self.UTILS.parent
        self.WORK = Path.cwd()

        # Determine the ASSETS path
        self.ASSETS = Path(os.getenv("VIRTUAL_ENV", "")) / "share" / "yugiquery"
        if not self.ASSETS.is_dir():
            self.ASSETS = Path.home() / ".local" / "share" / "yugiquery"
            if not self.ASSETS.is_dir():
                self.ASSETS = Path(sysconfig.get_path("data")) / "share" / "yugiquery"
                if not self.ASSETS.is_dir():
                    self.ASSETS = Path(user_data_dir("yugiquery"))
                    if not self.ASSETS.is_dir():
                        self.ASSETS = Path(site_data_dir("yugiquery"))

        # Determine the SHARE parh from the ASSETS path
        self.SHARE = self.ASSETS.parent

        # Determine the REPORTS path
        if self.WORK.parent.joinpath("reports").is_dir():
            self.REPORTS = self.WORK.parent / "reports"
        else:
            self.REPORTS = self.WORK / "reports"

        # Determine the NOTEBOOKS path based on the environment and hierarchy
        if self.WORK.joinpath("notebooks").is_dir():
            self.NOTEBOOKS = self.WORK / "notebooks"
        elif self.WORK.parent.joinpath("notebooks").is_dir():
            self.NOTEBOOKS = self.WORK.parent / "notebooks"
        else:
            self.NOTEBOOKS = self.SHARE / "notebooks"

        # Define the DATA based on the WORK path
        if self.WORK.parent.joinpath("data").is_dir():
            self.DATA = self.WORK.parent / "data"
        else:
            self.DATA = self.WORK / "data"

        # Redefine the ASSETS based on the hierarchy
        if self.WORK.joinpath("assets").is_dir():
            self.ASSETS = self.WORK / "assets"
        elif self.WORK.parent.joinpath("assets").is_dir():
            self.ASSETS = self.WORK.parent / "assets"

    def print(self):
        """
        Print the directory paths managed by this class.
        """
        print(f"APP: {self.APP}")
        print(f"ASSETS: {self.ASSETS}")
        print(f"DATA: {self.DATA}")
        print(f"NOTEBOOKS: {self.NOTEBOOKS}")
        print(f"REPORTS: {self.REPORTS}")
        print(f"SHARE: {self.SHARE}")
        print(f"UTILS: {self.UTILS}")
        print(f"WORK: {self.WORK}")

    def make(self):
        """
        Ensure that the DATA and REPORTS directories exist.
        """
        os.makedirs(self.DATA, exist_ok=True)
        os.makedirs(self.REPORTS, exist_ok=True)

    @property
    def is_notebook(self):
        """
        Check if the current environment is a Jupyter notebook.
        """
        return get_ipython() is not None


# Global instance of Dirs
dirs = Dirs()