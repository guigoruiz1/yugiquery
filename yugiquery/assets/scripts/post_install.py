#!/usr/bin/env python3

import subprocess
import sys


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


if __name__ == "__main__":
    install_tqdm()
    install_kernel()
