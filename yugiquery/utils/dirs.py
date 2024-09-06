from IPython import get_ipython
import os
import sysconfig
from pathlib import Path
from platformdirs import user_data_dir, site_data_dir


class Dirs:
    """
    Singleton class to manage directory paths for the application.
    """

    _instance = None

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

        # Determine the SHARE path
        self.SHARE = Path(os.getenv("VIRTUAL_ENV", "")) / "share"
        if not self.SHARE.is_dir():
            self.SHARE = Path.home() / ".local" / "share"
            if not self.SHARE.is_dir():
                self.SHARE = Path(sysconfig.get_path("data")) / "share"
                if not self.SHARE.is_dir():
                    self.SHARE = Path(user_data_dir())
                    if not self.SHARE.is_dir():
                        self.SHARE = Path(site_data_dir())

        # Determine the REPORTS path
        if self.WORK.parent.joinpath("reports").is_dir():
            self.REPORTS = self.WORK.parent / "reports"
        else:
            self.REPORTS = self.WORK / "reports"

        # Determine NOTEBOOKS_DIR based on the environment and hierarchy
        if self.WORK.joinpath("notebooks").is_dir():
            self.NOTEBOOKS = self.WORK / "notebooks"
        elif self.WORK.parent.joinpath("notebooks").is_dir():
            self.NOTEBOOKS = self.WORK.parent / "notebooks"
        else:
            self.NOTEBOOKS = self.SHARE / "notebooks"

        # Define dirs.DATA based on NOTEBOOKS_DIR
        if self.WORK.parent.joinpath("data").is_dir():
            self.DATA = self.WORK.parent / "data"
        else:
            self.DATA = self.WORK / "data"

        # Define ASSETS_DIR based on the hierarchy
        if self.WORK.joinpath("assets").is_dir():
            self.ASSETS = self.WORK / "assets"
        elif self.WORK.parent.joinpath("assets").is_dir():
            self.ASSETS = self.WORK.parent / "assets"
        else:
            self.ASSETS = self.SHARE / "yugiquery"

    def print(self):
        """
        Print the directory paths managed by this class.
        """
        print(f"SCRIPT: {self.APP}")
        print(f"UTILS: {self.UTILS}")
        print(f"SHARE: {self.SHARE}")
        print(f"ASSETS: {self.ASSETS}")
        print(f"WORK: {self.WORK}")
        print(f"REPORTS: {self.REPORTS}")
        print(f"NOTEBOOKS: {self.NOTEBOOKS}")
        print(f"DATA: {self.DATA}")

    def make(self):
        """
        Ensure that the necessary directories exist.
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
