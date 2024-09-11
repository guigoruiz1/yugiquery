# yugiquery/utils/git.py

# -*- coding: utf-8 -*-

# ===================== #
# Git Management module #
# ===================== #

# ======= #
# Imports #
# ======= #

# Standard library imports
import subprocess
from pathlib import Path
from typing import (
    List,
    Union,
)

# Third-party imports
import git
from termcolor import cprint

# Local application imports
from .helpers import *
from .dirs import dirs

# ========= #
# Functions #
# ========= #


def assure_repo() -> git.Repo:
    """
    Assures the script is inside a git repository. Initializes a repository if one is not found.

    Raises:
        git.InvalidGitRepositoryError: If the dirs.SCRIPT is not in a git repository.
        Exception: For any other unexpected errors.

    Returns:
        git.Repo: The git repository object.
    """
    try:
        # Try to create a Repo object
        repo = git.Repo(dirs.WORK, search_parent_directories=True)
        dirs.WORK = Path(repo.working_dir)

    except git.InvalidGitRepositoryError:
        # Handle the case when the path is not a valid Git repository
        repo = git.Repo.init(dirs.WORK)
        cprint(text=f"Git repository initialized in {dirs.WORK}\n", color="yellow")

    except Exception as e:
        # Handle any exceptions (e.g., invalid path)
        raise RuntimeError(f"Unable to init Git repository: {e}\n")
    finally:
        # Ensure the data and reports directories exist
        dirs.make()

    return repo


def get_repo() -> git.Repo:
    """
    Gets the current git repository if there is one.

    Args:
        None

    Raises:
        git.InvalidGitRepositoryError: If the dirs.SCRIPT is not in a git repository.
        git.GitCommandError: If an error occurs while committing the changes.
        Exception: For any other unexpected errors.

    Returns:
        git.Repo: The git repository object.
    """
    try:
        return git.Repo(dirs.WORK, search_parent_directories=True)

    except git.InvalidGitRepositoryError as e:
        raise RuntimeError(f"Unable to find a git repository: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


def unlock(passphrase: str = "") -> str:
    """
    Unlock the git credential store.

    Args:
        passphrase (str, optional): The passphrase to unlock your Git credential store. Defaults to empty.

    Returns:
        str: The result of the unlock operation.
    """
    # TODO: Better error handling
    script = dirs.get_asset("scripts", "unlock_git.sh")

    result = subprocess.run(
        [
            "sh",
            script,
            passphrase,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result


def commit(files: Union[str, List[str]], commit_message: str = "", repo: git.Repo = None) -> str:
    """
    Commits the specified files to the git repository after staging them.

    Args:
        files (Union[str, List[str]]): A list of file paths to be committed.
        commit_message (str, optional): The commit message. If not provided, a default message will be used.
        repo (git.Repo, optional): The git repository object. If not provided, the current repository will be used.

    Raises:
        git.GitCommandError: If an error occurs while committing the changes.
        Exception: For any other unexpected errors.

    Returns:
        str: The commit result.
    """
    if commit_message is None:
        commit_message = f"Commit - {arrow.utcnow().isoformat()}"
    if isinstance(files, str):
        files = [files]
    if repo is None:
        repo = get_repo()
    with repo:
        # Stage the files before committing
        try:
            repo.git.add(*files)
            return repo.git.commit(message=commit_message)
        except git.GitCommandError as e:
            raise RuntimeError(f"Failed to commit changes: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred: {e}")


def restore(files: Union[str, List[str]], repo: git.Repo = None) -> str:
    """
    Restores the specified files on the git repository.

    Args:
        files (Union[str, List[str]]): A list of file paths to be restored.
        repo (git.Repo, optional): The git repository object. If not provided, the current repository will be used.

    Raises:
        git.GitCommandError: If an error occurs while committing the changes.
        Exception: For any other unexpected errors.

    Returns:
        str: The restore result.
    """
    if isinstance(files, str):
        files = [files]
    if repo is None:
        repo = get_repo()
    with repo:
        # Stage the files before committing
        try:
            return repo.git.restore(*files)
        except git.GitCommandError as e:
            raise RuntimeError(f"Failed to restore files: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred: {e}")


def pull(passphrase: str = "", repo: git.Repo = None) -> str:
    """
    Pulls changes from the remote git repository.

    Args:
        passphrase (str, optional): The passphrase to unlock your Git credential store. Defaults to empty.
        repo (git.Repo, optional): The git repository object. If not provided, the current repository will be used.

    Raises:
        git.GitCommandError: If an error occurs while committing the changes.
        Exception: For any other unexpected errors.

    Returns:
        None
    """
    if repo is None:
        repo = get_repo()
    with repo:
        try:
            result = unlock(passphrase)
            if result.returncode != 0:
                return result.stdout.decode("utf-8")
        except Exception as e:
            cprint(text="Failed to unlock Git credential store.", color="yellow")
            print(e)
        try:
            return repo.git.pull()
        except git.GitCommandError as e:
            raise RuntimeError(f"Failed to pull changes: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred: {e}")


def push(passphrase: str = "", repo: git.Repo = None) -> str:
    """
    Pushes commits to the remote git repository.

    Args:
        passphrase (str, optional): The passphrase to unlock your Git credential store. Defaults to empty.
        repo (git.Repo, optional): The git repository object. If not provided, the current repository will be used.

    Raises:
        git.GitCommandError: If an error occurs while committing the changes.
        Exception: For any other unexpected errors.

    Returns:
        None
    """
    if repo is None:
        repo = get_repo()
    with repo:
        try:
            result = unlock()
            if result.returncode != 0:
                return result.stdout.decode("utf-8")
        except Exception as e:
            cprint(text="Failed to unlock Git credential store.", color="yellow")
            print(e)

        try:
            return repo.git.push()
        except git.GitCommandError as e:
            raise RuntimeError(f"Failed to push changes: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred: {e}")
