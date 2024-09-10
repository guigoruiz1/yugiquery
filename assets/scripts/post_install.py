#!/usr/bin/env python3

import os
import subprocess
import sys
import shutil
from termcolor import cprint


def install_kernel() -> None:
    from yugiquery.utils import git
    from yugiquery import __url__

    venv_name = "yqvenv"
    # Create a virtual environment.
    result = subprocess.run([sys.executable, "-m", "venv", venv_name], text=True)
    if result.returncode != 0:
        cprint(text=f"Failed to create virtual environment '{venv_name}'.", color="red")
        return
    else:
        print(f"{venv_name} virtual environment created.")

    # Install YugiQuery inside the virtual environment.
    pip_path = os.path.join(venv_name, "bin", "pip") if os.name != "nt" else os.path.join(venv_name, "Scripts", "pip")
    try:
        remote_url = git.get_repo().remote().url
    except:
        remote_url = f"{__url__}.git"

    result = subprocess.run(
        [pip_path, "install", f"git+{remote_url}"],
        text=True,
    )

    if result.returncode != 0:
        cprint(text=f"Failed to install YugiQuery in {venv_name}!", color="red")
        return
    else:
        print(f"YugiQuery installed in the virtual environment {venv_name}.")

    # Install the Jupyter kernel using ipykernel.
    python_path = (
        os.path.join(venv_name, "bin", "python3") if os.name != "nt" else os.path.join(venv_name, "Scripts", "python3")
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
        cprint(text=f"Failed to install Jupyter kernel 'yugiquery'!", color="red")
    else:
        print("Jupyter kernel 'yugiquery' installed.")


def install_tqdm() -> None:
    result = subprocess.run(
        ["pip", "install", "--no-deps", "git+https://github.com/guigoruiz1/tqdm.git"],
        text=True,
    )

    if result.returncode == 0:
        print("tqdm fork for Discord bot installed.")
    else:
        cprint(f"Failed to install tqdm fork for Discord bot!", color="red")
        print(result.stderr)


def install_nbconvert() -> None:
    from yugiquery.utils.dirs import dirs

    src_dir = dirs.ASSETS / "nbconvert"
    dst_dir = dirs.SHARE / "jupyter" / "nbconvert" / "templates"
    try:
        shutil.copytree(src=src_dir, dst=dst_dir, dirs_exist_ok=True)
        print("nbconvert templates installed.")
    except Exception as e:
        cprint(f"Failed to install nbconvert templates", color="red")
        print(e)


def install_filters() -> None:
    from yugiquery.utils.dirs import dirs
    from yugiquery.utils import git

    try:
        repo_root = git.get_repo().rev_parse("--show-toplevel")
        script_path = dirs.ASSETS / "scripts" / "git_filters.sh"
        result = subprocess.run(
            ["cd", repo_root, "&&", "bash", script_path],
            text=True,
        )
        if result.returncode == 0:
            print("Git filters have been installed in the current repository.")
        else:
            raise RuntimeError(result.stderr)
    except Exception as e:
        cprint(f"Failed to install tqdm fork for Discord bot!", color="red")
        print(e)


def main(args):
    if args.tqdm or args.all:
        install_tqdm()
    if args.kernel or args.all:
        install_kernel()
    if args.nbconvert or args.all:
        install_nbconvert()
    if args.filters or args.all:
        install_filters()


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--tqdm", action="store_true", help="Install tqdm fork for Discord bot.")
    argparser.add_argument("--kernel", action="store_true", help="Install Jupyter kernel.")
    argparser.add_argument("--nbconvert", action="store_true", help="Install nbconvert templates.")
    argparser.add_argument("--filters", action="store_true", help="Install git filters.")
    argparser.add_argument("--all", action="store_true", help="Install all.")
    args = argparser.parse_args()

    main(args)
