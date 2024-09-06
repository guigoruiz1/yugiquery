from IPython import get_ipython
import os
import sys
import sysconfig
from pathlib import Path


class Dirs:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Dirs, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize_dirs()
        return cls._instance

    def _initialize_dirs(self):
        if get_ipython() is not None:
            self.is_notebook = True
        else:
            self.is_notebook = False

        # Initialize directory paths
        self.UTILS = Path(__file__).resolve().parent
        self.SCRIPT = self.UTILS.parent
        self.WORK = Path.cwd()

        # Determine REPORTS_DIR based on the environment and hierarchy
        if self.is_notebook:
            self.REPORTS = self.WORK
        else:
            self.REPORTS = self.WORK / "reports"
            if not self.REPORTS.is_dir():
                self.REPORTS = (
                    Path(os.getenv("VIRTUAL_ENV", ""))
                    / "share"
                    / "yugiquery"
                    / "reports"
                )
                if not self.REPORTS.is_dir():
                    self.REPORTS = (
                        Path.home() / ".local" / "share" / "yugiquery" / "reports"
                    )
                    if not self.REPORTS.is_dir():
                        self.REPORTS = (
                            Path(sysconfig.get_path("data"))
                            / "share"
                            / "yugiquery"
                            / "reports"
                        )

        # Define DATA_DIR based on REPORTS_DIR
        if self.REPORTS.name == "reports":
            self.DATA = self.REPORTS.parent / "data"
        else:
            self.DATA = self.WORK / "data"

        # Define ASSETS_DIR based on the hierarchy
        if self.SCRIPT.joinpath("assets").is_dir():
            self.ASSETS = self.SCRIPT / "assets"
        elif os.getenv("VIRTUAL_ENV"):
            self.ASSETS = (
                Path(os.getenv("VIRTUAL_ENV")) / "share" / "yugiquery" / "assets"
            )
        elif Path.home().joinpath(".local", "share", "yugiquery", "assets").is_dir():
            self.ASSETS = Path.home() / ".local" / "share" / "yugiquery" / "assets"
        else:
            self.ASSETS = (
                Path(sysconfig.get_path("data")) / "share" / "yugiquery" / "assets"
            )

        # Ensure directories exist
        os.makedirs(self.DATA, exist_ok=True)
        os.makedirs(self.REPORTS, exist_ok=True)

    def print(self):
        print(f"UTILS: {self.UTILS}")
        print(f"SCRIPT: {self.SCRIPT}")
        print(f"WORK: {self.WORK}")
        print(f"REPORTS: {self.REPORTS}")
        print(f"DATA: {self.DATA}")
        print(f"ASSETS: {self.ASSETS}")


# Global instance of Dirs
dirs = Dirs()
