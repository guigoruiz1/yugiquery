from IPython import get_ipython
import os
import sysconfig
from pathlib import Path
from platformdirs import user_data_dir, site_data_dir


class Dirs:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Dirs, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize_dirs()
        return cls._instance

    def _initialize_dirs(self):
        # Initialize directory paths
        self.UTILS = Path(__file__).resolve().parent
        self.SCRIPT = self.UTILS.parent
        self.WORK = Path.cwd()
        self.REPORTS = Path.cwd()

        # Determine the SHARE path
        self.SHARE = (
            Path(os.getenv("VIRTUAL_ENV", ""))
            / "share"
            / "yugiquery"
        )
        if not self.SHARE.is_dir():
            self.SHARE = (
                Path.home() / ".local" / "share" / "yugiquery"
            )
            if not self.SHARE.is_dir():
                self.SHARE = (
                    Path(sysconfig.get_path("data"))
                    / "share"
                    / "yugiquery"
                )
                if not self.SHARE.is_dir():
                    self.SHARE = Path(user_data_dir("yugiquery"))
                    if not self.SHARE.is_dir():
                        self.SHARE = Path(site_data_dir("yugiquery"))

        # Determine NOTEBOOKS_DIR based on the environment and hierarchy
        if self.REPORTS.name == "notebooks":
            self.NOTEBOOKS = self.REPORTS
        else:
            self.NOTEBOOKS = self.REPORTS / "notebooks"
            if not self.NOTEBOOKS.is_dir():
                self.NOTEBOOKS = self.SHARE / "notebooks"

        # Define DATA_DIR based on NOTEBOOKS_DIR
        if self.REPORTS == self.NOTEBOOKS:
            self.DATA = self.REPORTS.parent / "data"
        else:
            self.DATA = self.REPORTS / "data"

        # Define ASSETS_DIR based on the hierarchy
        if self.REPORTS.joinpath("assets").is_dir():
            self.ASSETS = self.REPORTS / "assets"
        else:
            self.ASSETS = self.SHARE / "assets"

        # Ensure directories exist
        os.makedirs(self.DATA, exist_ok=True)
        os.makedirs(self.NOTEBOOKS, exist_ok=True)

    def print(self):
        print(f"UTILS: {self.UTILS}")
        print(f"SCRIPT: {self.SCRIPT}")
        print(f"WORK: {self.WORK}")
        print(f"REPORTS: {self.REPORTS}")
        print(f"SHARE: {self.SHARE}")
        print(f"NOTEBOOKS: {self.NOTEBOOKS}")
        print(f"DATA: {self.DATA}")
        print(f"ASSETS: {self.ASSETS}")

    @property
    def is_notebook(self):
        return (get_ipython() is not None)


# Global instance of Dirs
dirs = Dirs()