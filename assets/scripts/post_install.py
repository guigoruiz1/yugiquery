#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import shutil
from termcolor import cprint


def install_kernel() -> None:
    from yugiquery.utils.git import get_repo
    from yugiquery.utils.dirs import dirs
    from yugiquery import __url__

    venv_name = "yqvenv"
    venv_path = dirs.WORK / venv_name

    # Create a virtual environment.
    result = subprocess.run([sys.executable, "-m", "venv", venv_path], text=True)
    if result.returncode != 0:
        cprint(text=f"\nFailed to create virtual environment '{venv_name}'.", color="red")
        return
    else:
        cprint(text=f"\n{venv_name} virtual environment created.", color="green")

    # Install YugiQuery inside the virtual environment.
    pip_path = os.path.join(venv_path, "bin", "pip") if os.name != "nt" else os.path.join(venv_path, "Scripts", "pip")
    try:
        remote_url = get_repo().remote().url
    except:
        remote_url = f"{__url__}.git"

    result = subprocess.run(
        [pip_path, "install", f"git+{remote_url}"],
        text=True,
    )

    if result.returncode != 0:
        cprint(text=f"\nFailed to install YugiQuery in {venv_name}!", color="red")
        return
    else:
        cprint(text=f"\nYugiQuery installed in the virtual environment {venv_name}.", color="green")

    # Install the Jupyter kernel using ipykernel.
    python_path = (
        os.path.join(venv_path, "bin", "python3") if os.name != "nt" else os.path.join(venv_path, "Scripts", "python3")
    )
    display_name = "Python3 (yugiquery)"

    result = subprocess.run(
        [
            python_path,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            venv_name,
            "--display-name",
            display_name,
            "--matplotlib",
            "svg",
            "--IPKernelApp.exec_lines=['from yugiquery import *']",
        ],
        text=True,
    )

    if result.returncode != 0:
        cprint(text=f"\nFailed to install Jupyter kernel 'yugiquery'!", color="red")
    else:
        cprint(text="\nJupyter kernel 'yugiquery' installed.", color="green")


def install_tqdm() -> None:
    result = subprocess.run(
        ["pip", "install", "--no-deps", "tqdm[notebook] @ git+https://github.com/guigoruiz1/tqdm.git"],
        text=True,
    )

    if result.returncode == 0:
        cprint(text="\nTQDM fork for Discord bot installed.", color="green")
    else:
        cprint(text=f"\nFailed to install TQDM fork for Discord bot!", color="red")


def install_nbconvert() -> None:
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
    from yugiquery.utils.dirs import dirs
    from yugiquery.utils.git import get_repo

    try:
        repo_root = get_repo().working_dir
        script_path = dirs.get_asset("scripts", "git_filters.sh")
        result = subprocess.run(
            ["bash", script_path],
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


def main(args):
    if not (args.tqdm or args.kernel or args.nbconvert or args.filters):
        args.tqdm = args.kernel = args.nbconvert = args.filters = True

    if args.tqdm:
        install_tqdm()
    if args.kernel:
        install_kernel()
    if args.nbconvert:
        install_nbconvert()
    if args.filters:
        install_filters()


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(description="Install various additional components. If no flags are passed, all components will be installed.")
    argparser.add_argument("--tqdm", action="store_true", help="Install TQDM fork for Discord bot.")
    argparser.add_argument("--kernel", action="store_true", help="Install Jupyter kernel.")
    argparser.add_argument("--nbconvert", action="store_true", help="Install nbconvert templates.")
    argparser.add_argument("--filters", action="store_true", help="Install git filters.")
    args = argparser.parse_args()

    main(args)
