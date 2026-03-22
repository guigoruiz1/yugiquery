# yugiquery/scripts/optionals.py

#!/usr/bin/env python3

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import argparse
import os
import subprocess
import sys
import shutil
from termcolor import cprint

# --- Imports: Local Application --- #
from ..utils import LoggerConfig


# --- Logger Setup --- #
logger = LoggerConfig.get_logger()


def install_templates() -> None:
    """
    Copy the notebook templates from the package's `ASSETS` directory to the user's `NOTEBOOKS` directory
    and .xlsx files to the `DATA` directory.
    """
    from yugiquery.utils.dirs import dirs

    src_dir = dirs.ASSETS.pkg / "templates"
    notebooks_dst_dir = dirs.NOTEBOOKS.user
    data_dst_dir = dirs.DATA
    dirs.make()

    try:
        # Copy .ipynb files to dirs.NOTEBOOKS.user
        for ipynb_file in src_dir.glob("*.ipynb"):
            shutil.copy(ipynb_file, notebooks_dst_dir)

        # Copy .xlsx files to dirs.DATA
        for xlsx_file in src_dir.glob("*.xlsx"):
            shutil.copy(xlsx_file, data_dst_dir)

        cprint(text=f"Templates copied to {notebooks_dst_dir} and {data_dst_dir}.", color="green")
        logger.info("Templates copied to %s and %s.", notebooks_dst_dir, data_dst_dir)
    except Exception as e:
        logger.error("Failed to copy templates. %s", e)


def install_kernel(venv: bool = False) -> None:
    """
    Create a virtual environment, install YugiQuery inside it, and install it as a Jupyter kernel.

    Args:
        venv (bool, optional): Whether to create a virtual environment. Default is True.
    """
    from yugiquery import __title__
    from IPython.core.profileapp import ProfileCreate

    name = __title__.lower()
    if venv:
        from yugiquery.utils.dirs import dirs

        venv_name = "venv"
        venv_path = dirs.WORK / venv_name
        python_path = (
            os.path.join(venv_path, "bin", "python3") if os.name != "nt" else os.path.join(venv_path, "Scripts", "python3")
        )

        # Create a virtual environment.
        if not os.path.exists(venv_path):
            result = subprocess.run([sys.executable, "-m", "venv", venv_path], text=True)
            if result.returncode != 0:
                logger.error("Failed to create virtual environment '%s'.", venv_name)
                return
            else:
                cprint(text=f"{__title__} virtual environment created at {venv_path}.", color="green")
                logger.info("%s virtual environment created at %s.", __title__, venv_path)

        # Install YugiQuery inside the virtual environment.
        cache_dir = subprocess.run(
            args=f"{sys.executable} -m pip freeze | grep {name}",
            capture_output=True,
            text=True,
            shell=True,
        ).stdout.strip()
        result = subprocess.run([python_path, "-m", "pip", "install", "--force-reinstall", cache_dir], text=True)

        # If cache not found, install from GitHub with the same version
        if result.returncode != 0:
            commit_hash = None
            try:
                from yugiquery import __version__, __version_tuple__, __url__

                # If __version_tuple__ exists, extract parts from the version tuple
                if len(__version_tuple__) > 3:
                    commit_hash = __version_tuple__[-1]
                    commit_hash = commit_hash.split("g")[-1].split(".")[0] if isinstance(commit_hash, str) else None
            except ImportError:
                # Fallback to __version__ if __version_tuple__ is not available
                from yugiquery import __version__, __url__

            # Check if there's a commit hash in the version string
            if commit_hash:
                git_ref = commit_hash
            else:
                git_ref = f"V{__version__}"

            github_url = f"{__url__}.git@{git_ref}"

            result = subprocess.run(
                args=[
                    python_path,
                    "-m",
                    "pip",
                    "install",
                    "--force-reinstall",
                    github_url,
                ],
                text=True,
            )

        if result.returncode != 0:
            logger.error("Error installing %s in %s", __title__, venv_name)
            return
        else:
            cprint(text=f"{__title__} installed in {venv_name}.", color="green")
            logger.info("%s installed in %s.", __title__, venv_name)
    else:
        python_path = sys.executable

    # Create an IPython profile for YugiQuery.
    try:
        # Step 1: Initialize the profile creation process
        profile_creator = ProfileCreate(profile=name)

        # Step 2: Create the profile directory and default config files
        profile_creator.init_config_files()

        # Step 3: Manually write the config to ipython_config.py
        profile_dir = profile_creator.profile_dir.location
        config_file = os.path.join(profile_dir, "ipython_config.py")

        # Step 4: Write the configuration manually
        with open(config_file, "w") as f:
            f.write("c = get_config()\n")
            f.write("c.InteractiveShellApp.exec_lines = ['from yugiquery import *']\n")
    except:
        logger.error("Failed to create IPython profile for YugiQuery!")
        return

    cprint(text="IPython profile created for YugiQuery.", color="green")
    logger.info("IPython profile created for YugiQuery.")

    # Install the Jupyter kernel using ipykernel.
    display_name = f"Python3 ({__title__})"

    result = subprocess.run(
        [
            python_path,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            name,
            "--display-name",
            display_name,
            "--profile",
            name,
        ],
        text=True,
    )

    if result.returncode != 0:
        logger.error("Failed to install Jupyter kernel '%s'!", name)
        return
    else:
        cprint(text=f"Jupyter kernel '{name}' installed.", color="green")
        logger.info("Jupyter kernel '%s' installed.", name)


def install_nbconvert() -> None:
    """
    Patch the nbconvert "Lab" template to include a dynamic light and dark theme, and preprocessor to remove cells tagged with "exclude".
    """
    from . import generate_auto_theme

    try:
        generate_auto_theme.main()
        cprint(text="nbconvert templates installed.", color="green")
        logger.info("nbconvert templates installed.")
    except Exception as e:
        logger.error("Failed to install nbconvert templates. %s", e)


def install_filters() -> None:
    """
    Install Git filters to automatically clean notebooks before committing them and redacting secrets from "secrets" files.
    Will initializes a new Git repository if one does not exist.
    """
    from yugiquery.utils.dirs import dirs
    from yugiquery.utils.git import ensure_repo

    try:
        repo_root = ensure_repo().working_dir  # Still unsure about this
        if os.name == "nt":
            args = [dirs.get_asset("scripts", "git_filters.bat")]
        else:
            args = ["sh", dirs.get_asset("scripts", "git_filters.sh")]

        result = subprocess.run(
            args=args,
            text=True,
            cwd=repo_root,
        )
        if result.returncode == 0:
            cprint(text="Git filters have been installed in the current repository.", color="green")
            logger.info("Git filters have been installed in the current repository.")
            return
        else:
            logger.error("Failed to install Git filters!")
    except Exception as e:
        logger.error("Failed to install Git filters! %s", e)


def set_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--templates", action="store_true", help="install template notebooks")
    parser.add_argument("--nbconvert", action="store_true", help="install nbconvert templates patch")
    parser.add_argument("--filters", action="store_true", help="install git filters")
    parser.add_argument("--kernel", action="store_true", help="install Jupyter kernel")
    parser.add_argument(
        "--venv",
        action="store_true",
        help="whether to create a virtual environment to install Jupyter Kernel. Has no effect if --kernel is not passed",
    )
    debug_group = parser.add_argument_group("Debugging")
    debug_group.add_argument(
        "--log-level",
        type=str,
        required=False,
        default=None,
        help="set log verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    debug_group.add_argument(
        "--log-file",
        type=str,
        required=False,
        default=None,
        help="write log output to a file in addition to stderr",
    )


def main(args):
    LoggerConfig.setup(level=args.log_level, log_file=args.log_file)
    no_flags = not (args.templates or args.kernel or args.nbconvert or args.filters)
    if args.venv and not args.kernel:
        logger.warning("The --venv flag has no effect if --kernel is not passed.")
        if no_flags:
            return

    # If no flags are passed, install everything.
    if no_flags:
        args.templates = args.kernel = args.nbconvert = args.filters = True

    if args.templates:
        install_templates()
    if args.kernel:
        install_kernel(venv=args.venv)
    if args.nbconvert:
        install_nbconvert()
    if args.filters:
        install_filters()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Install various optional components. If no flags are passed, all components will be installed"
    )
    set_parser(parser)
    args = parser.parse_args()
    main(args)
