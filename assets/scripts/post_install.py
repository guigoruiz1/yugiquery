#!/usr/bin/env python3

import subprocess
import sys
import shutil
import importlib


def install_kernel():
    # Run the jupyter command to install the kernel
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name=yugiquery",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("Jupyter kernel 'yugiquery' installed.")
    else:
        print(f"Failed to install Jupyter kernel 'yugiquery'. Error: {result.stderr}")


def install_tqdm():
    result = subprocess.run(
        ["pip", "install", "--no-deps", "git+https://github.com/guigoruiz1/tqdm.git"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("tqdm fork for Discord bot installed.")
    else:
        print(f"Failed to install tqdm fork for Discord bot. Error: {result.stderr}")


def install_nbconvert():
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
    argparser.add_argument(
        "--tqdm", action="store_true", help="Install tqdm fork for Discord bot."
    )
    argparser.add_argument(
        "--kernel", action="store_true", help="Install Jupyter kernel."
    )
    argparser.add_argument(
        "--nbconvert", action="store_true", help="Install nbconvert templates."
    )
    argparser.add_argument("--all", action="store_true", help="Install all.")
    args = argparser.parse_args()

    main(args)
