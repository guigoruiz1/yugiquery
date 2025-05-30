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
from typing import List


# Third-party imports
import git
from termcolor import cprint

# Local application imports
from .helpers import *
from .dirs import dirs

# ========= #
# Functions #
# ========= #


def ensure_repo() -> git.Repo:
    """
    Ensures the execution happens inside a git repository. Initializes a repository if one is not found.

    Raises:
        git.InvalidGitRepositoryError: If the dirs.SCRIPT is not in a git repository.
        Exception: For any other unexpected errors.

    Returns:
        git.Repo: The git repository object.

    Raises:
        RuntimeError: If a Git repository cannot be initialized.
    """
    try:
        # Try to create a Repo object
        repo = git.Repo(dirs.WORK, search_parent_directories=True)
        dirs.WORK = Path(repo.working_dir)

    except git.InvalidGitRepositoryError:
        # Handle the case when the path is not a valid Git repository
        # Check if the work directory is a child of the root repository.
        if dirs.DATA.parent == dirs.WORK.parent:
            repo_root = dirs.WORK.parent
        else:
            repo_root = dirs.WORK
        repo = git.Repo.init(repo_root)
        cprint(text=f"\nGit repository initialized in {dirs.WORK}\n", color="yellow")

    except Exception as e:
        # Handle any exceptions (e.g., invalid path)
        raise RuntimeError(f"\nUnable to init Git repository: {e}\n")
    finally:
        # Ensure the data and reports directories exist
        dirs.make()

    return repo


def get_repo() -> git.Repo:
    """
    Gets the current git repository if there is one.

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
    if os.name == "nt":
        script = dirs.get_asset("scripts", "unlock_git.bat")
        args = [script, passphrase]
    else:
        script = dirs.get_asset("scripts", "unlock_git.sh")
        args = ["sh", script, passphrase]

    result = subprocess.run(
        args=args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result


def commit(files: str | List[str], message: str = "", repo: git.Repo | None = None) -> str:
    """
    Commits the specified files to the git repository after staging them.

    Args:
        files (str | List[str]): A list of file paths to be committed.
        message (str, optional): The commit message. If not provided, a default message will be used.
        repo (git.Repo | None, optional): The git repository object. If none provided, the current repository will be used.

    Raises:
        git.GitCommandError: If an error occurs while committing the changes.
        Exception: For any other unexpected errors.

    Returns:
        str: The commit result.
    """
    if message is None:
        message = f"Commit - {arrow.utcnow().isoformat()}"
    if isinstance(files, str):
        files = [files]
    if repo is None:
        repo = get_repo()
    with repo:
        # Stage the files before committing
        try:
            repo.git.add(*files)
            diff = repo.index.diff("HEAD")
            if diff:
                return repo.git.commit(message=message)
            else:
                return "No changes to commit."
        except git.GitCommandError as e:
            raise RuntimeError(f"Failed to commit changes: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred: {e}")


def restore(files: str | List[str], repo: git.Repo | None = None) -> str:
    """
    Restores the specified files on the git repository.

    Args:
        files (str | List[str]): A list of file paths to be restored.
        repo (git.Repo | None, optional): The git repository object. If none provided, the current repository will be used.

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
        try:
            return repo.git.restore(*files)
        except git.GitCommandError as e:
            raise RuntimeError(f"Failed to restore files: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred: {e}")


def pull(passphrase: str = "", repo: git.Repo | None = None) -> str:
    """
    Pulls changes from the remote git repository.

    Args:
        passphrase (str, optional): The passphrase to unlock your Git credential store. Defaults to empty.
        repo (git.Repo | None, optional): The git repository object. If none provided, the current repository will be used.

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


def push(passphrase: str = "", repo: git.Repo | None = None) -> str:
    """
    Pushes commits to the remote git repository.

    Args:
        passphrase (str, optional): The passphrase to unlock your Git credential store. Defaults to empty.
        repo (git.Repo | None, optional): The git repository object. If none provided, the current repository will be used.

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
            return repo.git.push()
        except git.GitCommandError as e:
            raise RuntimeError(f"Failed to push changes: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred: {e}")


def squash_commits(start_commit: git.Commit, repo: git.Repo | None = None, message: str = "") -> str:
    """
    Squashes all commits from the start_commit to HEAD into a single commit.

    Args:
        start_commit (git.Commit): The commit object to start squashing from.
        repo (git.Repo | None, optional): The git repository object. If none provided, the current repository will be used.
        message (str, optional): The commit message. If not provided, a default message will generated from commits.
    """
    if repo is None:
        repo = get_repo()
    with repo:
        try:
            # Get all commits from start_commit to HEAD
            commits = list(repo.iter_commits(f"{start_commit.hexsha}..HEAD"))
            if not message:
                # Collect commit messages from the range
                commit_messages = [commit.message.strip() for commit in commits]
                # Form a single commit message by joining the individual messages
                message = "\n\n".join(commit_messages)
            # Reset the branch to the start_commit (soft reset)
            repo.git.reset("--soft", start_commit.hexsha)
            # Create a new commit with the combined commit message
            return repo.git.commit(message=message)

        except git.GitCommandError as e:
            raise RuntimeError(f"Failed to commit changes: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred: {e}")
