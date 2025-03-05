#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys
import shutil
from termcolor import cprint


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

        cprint(text=f"\nTemplates copied to {notebooks_dst_dir} and {data_dst_dir}.", color="green")
    except Exception as e:
        cprint(text=f"\nFailed to copy templates.", color="red")
        print(e)


def install_kernel(venv: bool = False) -> None:
    """
    Create a virtual environment, install YugiQuery inside it, and install it as a Jupyter kernel.

    Args:
        venv (bool, optional): Whether to create a virtual environment. Default is True.
    """
    from yugiquery import __title__
    from IPython.core.profileapp import ProfileCreate

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
                cprint(text=f"\nFailed to create virtual environment '{venv_name}'.", color="red")
                return
            else:
                cprint(text=f"\n{__title__} virtual environment created at {venv_path}.", color="green")

        # Install YugiQuery inside the virtual environment.
        cache_dir = subprocess.run(
            args=f"{sys.executable} -m pip freeze | grep {__title__.lower()}",
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
                    commit_hash = __version_tuple__[-1].split("g")[-1].split(".")[0]
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
            cprint(text=f"Error installing {__title__} in {venv_name}", color="red")
            return
        else:
            cprint(text=f"\n{__title__} installed in {venv_name}.", color="green")
    else:
        python_path = sys.executable

    # Create an IPython profile for YugiQuery.
    try:
        # Step 1: Initialize the profile creation process
        profile_creator = ProfileCreate(profile="yugiquery")

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
        cprint(text=f"\nFailed to create IPython profile for YugiQuery!", color="red")
        return

    cprint(text=f"\nIPython profile created for YugiQuery.", color="green")

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
            __title__.lower(),
            "--display-name",
            display_name,
            "--profile",
            __title__.lower(),
        ],
        text=True,
    )

    if result.returncode != 0:
        cprint(text=f"\nFailed to install Jupyter kernel '{__title__.lower()}'!", color="red")
        return
    else:
        cprint(text=f"\nJupyter kernel '{__title__.lower()}' installed.", color="green")


def install_nbconvert() -> None:
    """
    Copy the nbconvert templates from YugiQuery to the appropriate Jupyter NbConvert directory.
    """
    from yugiquery.utils.dirs import dirs

    src_dir = dirs.ASSETS.pkg / "nbconvert"
    dst_dir = dirs.NBCONVERT
    try:
        shutil.copytree(src=src_dir, dst=dst_dir, dirs_exist_ok=True)
        cprint(text="\nnbconvert templates installed.", color="green")
    except Exception as e:
        cprint(text=f"\nFailed to install nbconvert templates", color="red")
        print(e)


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
            cprint(text="\nGit filters have been installed in the current repository.", color="green")
            return
        else:
            cprint(text=f"\nFailed to install Git filters!", color="red")
    except Exception as e:
        cprint(text=f"\nFailed to install Git filters!", color="red")
        print(e)


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


def main(args):
    if args.venv and not args.kernel:
        cprint(text="The --venv flag has no effect if --kernel is not passed.", color="yellow")

    # If no flags are passed, install everything.
    if not (args.templates or args.kernel or args.nbconvert or args.filters):
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
        description="Install various additional components. If no flags are passed, all components will be installed"
    )
    set_parser(parser)
    args = parser.parse_args()

    main(args)
