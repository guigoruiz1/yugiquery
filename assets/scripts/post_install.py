#!/usr/bin/env python3

import os
import subprocess
import sys
import shutil


def install_kernel() -> None:
    from yugiquery.utils import git
    from yugiquery import __url__

    venv_path = "yqvenv"
    # Create a virtual environment.
    result = subprocess.run([sys.executable, "-m", "venv", venv_path], text=True)
    if result.returncode != 0:
        print(f"Failed to create virtual environment '{venv_path}'.")
        return
    else:
        print(f"{venv_path} virtual environment created.")

    # Install YugiQuery inside the virtual environment.
    pip_path = os.path.join(venv_path, "bin", "pip") if os.name != "nt" else os.path.join(venv_path, "Scripts", "pip")
    try:
        remote_url = git.get_repo().remote().url
    except:
        remote_url = f"{__url__}.git"

    result = subprocess.run(
        [pip_path, "install", f"git+{remote_url}"],
        text=True,
    )

    if result.returncode != 0:
        print(f"Failed to install YugiQuery in {venv_path}!")
    else:
        print(f"YugiQuery installed in the virtual environment {venv_path}.")

    # Install the Jupyter kernel using ipykernel.
    python_path = (
        os.path.join(venv_path, "bin", "python") if os.name != "nt" else os.path.join(venv_path, "Scripts", "python")
    )
    kernel_name = "yugiquery"
    display_name = "Python (yugiquery)"

    result = subprocess.run(
        [
            python_path,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            kernel_name,
            "--display-name",
            display_name,
        ],
        text=True,
    )

    if result.returncode != 0:
        print(f"Failed to install Jupyter kernel 'yugiquery'.")
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
        print(f"Failed to install tqdm fork for Discord bot. Error: {result.stderr}")


def install_nbconvert() -> None:
    from yugiquery.utils.dirs import dirs

    src_dir = dirs.ASSETS / "nbconvert"
    dst_dir = dirs.SHARE / "jupyter" / "nbconvert" / "templates"
    shutil.copytree(src=src_dir, dst=dst_dir, dirs_exist_ok=True)
    print("nbconvert templates installed.")


def main(args):
    if args.tqdm or args.all:
        install_tqdm()
    if args.kernel or args.all:
        install_kernel()
    if args.nbconvert or args.all:
        install_nbconvert()


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--tqdm", action="store_true", help="Install tqdm fork for Discord bot.")
    argparser.add_argument("--kernel", action="store_true", help="Install Jupyter kernel.")
    argparser.add_argument("--nbconvert", action="store_true", help="Install nbconvert templates.")
    argparser.add_argument("--all", action="store_true", help="Install all.")
    args = argparser.parse_args()

    main(args)
