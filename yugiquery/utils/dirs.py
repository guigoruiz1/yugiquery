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
from types import SimpleNamespace

# Third-party imports
from IPython import get_ipython
from jupyter_core.paths import jupyter_path
from platformdirs import user_data_dir, site_data_dir
from termcolor import cprint

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
        self._PKG_ASSETS = Path(os.getenv("VIRTUAL_ENV", "")) / "share" / "yugiquery"
        if not self._PKG_ASSETS.is_dir():
            self._PKG_ASSETS = Path.home() / ".local" / "share" / "yugiquery"
            if not self._PKG_ASSETS.is_dir():
                self._PKG_ASSETS = Path(sysconfig.get_path("data")) / "share" / "yugiquery"
                if not self._PKG_ASSETS.is_dir():
                    self._PKG_ASSETS = Path(user_data_dir("yugiquery"))
                    if not self._PKG_ASSETS.is_dir():
                        self._PKG_ASSETS = Path(site_data_dir("yugiquery"))
                        if not self._PKG_ASSETS.is_dir():
                            self._PKG_ASSETS = self.APP.parent / "assets"
                            if not self._PKG_ASSETS.is_dir():
                                self._PKG_ASSETS = self.APP / "assets"
                                if not self._PKG_ASSETS.is_dir():
                                    self._PKG_ASSETS = None

        # Determine the Jupyter path
        self.NBCONVERT = next(Path(path) for path in jupyter_path("nbconvert", "templates") if Path(path).exists())

        # Determine the package notebooks path from the ASSETS path
        self._PKG_NOTEBOOKS = None
        if self._PKG_ASSETS is not None:
            self._PKG_NOTEBOOKS = self._PKG_ASSETS / "notebooks"

    @property
    def ASSETS(self) -> SimpleNamespace:
        """
        Returns a SimpleNamespace object containing the paths to the package and user assets directories.

        Returns:
            SimpleNamespace: A SimpleNamespace object containing the paths to the package and user assets directories.
        """
        assets = SimpleNamespace()

        if self._PKG_ASSETS is not None:
            assets.pkg = self._PKG_ASSETS
        else:
            assets.pkg = None

        if self.WORK.parent.joinpath("assets").is_dir():
            assets.user = self.WORK.parent / "assets"
        elif self.WORK.joinpath("assets").is_dir():
            assets.user = self.WORK / "assets"
        else:
            assets.user = None

        return assets

    @property
    def NOTEBOOKS(self) -> SimpleNamespace:
        """
        Return the path to the notebooks directory.

        Returns:
            Path: The path to the notebooks directory.
        """
        notebooks = SimpleNamespace()
        if self._PKG_NOTEBOOKS is not None:
            notebooks.pkg = self._PKG_NOTEBOOKS
        else:
            notebooks.pkg = None
        if self.WORK.parent.joinpath("notebooks").is_dir():
            notebooks.user = self.WORK.parent / "notebooks"
        else:
            notebooks.user = self.WORK / "notebooks"

        return notebooks

    @property
    def DATA(self) -> Path:
        """
        Return the path to the data directory.

        Returns:
            Path: The path to the data directory.
        """
        # Get the DATA path based on the WORK path
        if self.WORK.parent.joinpath("data").is_dir():
            return self.WORK.parent / "data"
        else:
            return self.WORK / "data"

    @property
    def REPORTS(self) -> Path:
        """
        Return the path to the reports directory.

        Returns:
            Path: The path to the reports directory.
        """
        # Get the REPORTS path based on the WORK path
        if self.WORK.parent.joinpath("reports").is_dir():
            return self.WORK.parent / "reports"
        else:
            return self.WORK / "reports"

    @property
    def is_notebook(self) -> bool:
        """
        Check if the current environment is a Jupyter notebook.

        Returns:
            bool: True if the current environment is a Jupyter notebook, False otherwise.
        """
        return get_ipython() is not None

    @property
    def secrets_file(self) -> Path:
        """
        Return the path to the secrets file following the hierarchy: first ASSETS, then WORK. Returns none if the file is not found.

        Returns:
            Path: The path to the secrets file.

        """
        try:
            secrets_file = self.get_asset("secrets.env")
        except FileNotFoundError:
            secrets_file = self.WORK / "secrets.env"
            if not secrets_file.is_file():
                secrets_file = None

        return secrets_file

    def print(self) -> None:
        """
        Prints the directory paths managed by this class.
        """

        def exists(path: Path) -> str:
            if path.exists():
                cprint("exists", color="green")
            else:
                cprint("missing", color="red")

        print(f"App: {self.APP} - ", end="", flush=True)
        exists(self.APP)
        assets = self.ASSETS
        if assets.pkg is not None or assets.user is not None:
            print(f"Assets:")
        if assets.pkg is not None:
            print(f"  Package: {assets.pkg} - ", end="", flush=True)
            exists(assets.pkg)
        if assets.user is not None:
            print(f"  User: {assets.user} - ", end="", flush=True)
            exists(assets.user)
        print(f"Data: {self.DATA} - ", end="", flush=True)
        exists(self.DATA)
        print(f"NbConvert: {self.NBCONVERT} - ", end="", flush=True)
        exists(self.NBCONVERT)
        notebooks = self.NOTEBOOKS
        if notebooks.pkg is not None or notebooks.user is not None:
            print(f"Notebooks:")
        if notebooks.pkg is not None:
            print(f"  Package: {notebooks.pkg} - ", end="", flush=True)
            exists(notebooks.pkg)
        if notebooks.user is not None:
            print(f"  User: {notebooks.user} - ", end="", flush=True)
            exists(notebooks.user)
        print(f"Reports: {self.REPORTS} - ", end="", flush=True)
        exists(self.REPORTS)
        print(f"Utils: {self.UTILS} - ", end="", flush=True)
        exists(self.UTILS)
        print(f"Work: {self.WORK} - ", end="", flush=True)
        exists(self.WORK)

    def make(self) -> None:
        """
        Ensure that the DATA and REPORTS directories exist.
        """
        os.makedirs(self.DATA, exist_ok=True)
        os.makedirs(self.REPORTS, exist_ok=True)
        os.makedirs(self.NOTEBOOKS.user, exist_ok=True)

    def get_asset(self, *parts: str) -> Path:
        """
        Return the path to an asset file.

        Args:
            *parts (str): The path parts to the asset file or directory.

        Returns:
            Path: The path to the asset file or directory.

        Raises:
            FileNotFoundError: If the asset file is not found
        """
        user_asset_path = self.ASSETS.user.joinpath(*parts) if self.ASSETS.user else None
        pkg_asset_path = self.ASSETS.pkg.joinpath(*parts) if self.ASSETS.pkg else None

        if user_asset_path and user_asset_path.exists():
            return user_asset_path
        elif pkg_asset_path and pkg_asset_path.exists():
            return pkg_asset_path

        print(self.ASSETS)

        raise FileNotFoundError(f'Asset "{Path(*parts)}" not found!')

    def get_notebook(self, *parts: str) -> Path:
        """
        Return the path to a notebook file.

        Args:
            *parts (str): The path parts to the notebook file.

        Returns:
            Path: The path to the notebook file.

        Raises:
            FileNotFoundError: If the notebook file is not found.
        """
        user_notebook_path = self.NOTEBOOKS.user.joinpath(*parts).with_suffix(".ipynb") if self.NOTEBOOKS.user else None
        pkg_notebook_path = self.NOTEBOOKS.pkg.joinpath(*parts).with_suffix(".ipynb") if self.NOTEBOOKS.pkg else None

        if user_notebook_path and user_notebook_path.exists():
            return user_notebook_path
        elif pkg_notebook_path and pkg_notebook_path.exists():
            return pkg_notebook_path

        raise FileNotFoundError(f"Notebook not found!")


# Global instance of Dirs
dirs = Dirs()
