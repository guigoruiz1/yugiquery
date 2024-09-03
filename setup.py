from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import os
from pathlib import Path  # Import Path from pathlib


class CustomInstallCommand(install):
    """Customized setuptools install command - runs custom installation script."""

    def run(self):
        # Run the standard install process
        install.run(self)

        # Define the base directory and path to the shell script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(
            base_dir, "yugiquery", "assets", "bash", "install.sh"
        )

        # Run the shell script
        subprocess.check_call(["bash", script_path])


# Import the metadata from __version__.py
about = {}
with open(Path(__file__).parent / "yugiquery" / "version.py") as f:
    exec(f.read(), about)

# Read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name=about["__title__"],
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__author_email"],
    description=about["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=about["__url__"],
    packages=find_packages(),
    cmdclass={"install": CustomInstallCommand},
    entry_points={
        "console_scripts": [
            "yugiquery=yugiquery.__main__:main",  # This handles both yugiquery and yugiquery bot
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        f"Development Status :: {about['__status__']}",
    ],
    python_requires=">=3.6",
    install_requires=[
        "arrow>=1.3.0",
        "discord.py>=2.3.2",
        "GitPython>=3.1.42",
        "ipykernel>=6.29.3",
        "ipylab>=1.0.0",
        "ipython>=8.12.3",
        "itables>=1.7.1",
        "jupyter_client>=8.6.1",
        "matplotlib>=3.8.3",
        "matplotlib_venn>=0.11.10",
        "nbconvert>=7.11.0",
        "nbformat>=5.10.3",
        "nbstripout>=0.6.1",
        "numpy>=1.26.4",
        "pandas>=2.2.1",
        "papermill>=2.5.0",
        "python-dotenv>=1.0.1",
        "python-telegram-bot>=21.0.1",
        "Requests>=2.31.0",
        "seaborn>=0.13.2",
        "wikitextparser>=0.55.8",
        "tqdm @ git+https://github.com/guigoruiz1/tqdm.git",
        "halo @ git+https://github.com/guigoruiz1/halo.git",
    ],
    include_package_data=True,  # Ensures files listed in MANIFEST.in are included
)
