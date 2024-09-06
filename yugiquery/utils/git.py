# yugiquery/utils/git.py

# -*- coding: utf-8 -*-

# ===================== #
# Git Management module #
# ===================== #

import git
import subprocess
from .helpers import *
from .dirs import dirs


def assure_repo():
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
        print(f"Git repository initialized in {dirs.WORK}")

    except Exception as e:
        # Handle any exceptions (e.g., invalid path)
        raise RuntimeError(f"Unable to init Git repository: {e}")
    finally:
        dirs.DATA = dirs.WORK / "data"
        dirs.REPORTS = dirs.WORK / "reports"

        # Ensure the data and reports directories exist
        os.makedirs(dirs.DATA, exist_ok=True)
        os.makedirs(dirs.REPORTS, exist_ok=True)

    return repo


def commit(files: Union[str, List[str]], commit_message: str = None):
    """
    Commits the specified files to the git repository after staging them.

    Args:
        files (Union[str, List[str]]): A list of file paths to be committed.
        commit_message (str, optional): The commit message. If not provided, a default message will be used.

    Raises:
        git.InvalidGitRepositoryError: If the dirs.SCRIPT is not in a git repository.
        git.GitCommandError: If an error occurs while committing the changes.
        Exception: For any other unexpected errors.

    Returns:
        None
    """
    if commit_message is None:
        commit_message = f"Commit - {arrow.utcnow().isoformat()}"
    if isinstance(files, str):
        files = [files]
    try:
        with git.Repo(dirs.APP, search_parent_directories=True) as repo:
            # Stage the files before committing
            repo.git.add(*files)
            return repo.git.commit(message=commit_message)

    except git.InvalidGitRepositoryError as e:
        raise RuntimeError(f"Unable to find a git repository: {e}")
    except git.GitCommandError as e:
        raise RuntimeError(f"Failed to commit changes: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


def pull(passphrase: str = None):
    """
    Pulls changes from the remote git repository.

    Args:
        passphrase (str, optional): The passphrase to unlock your Git credential store.

    Raises:
        git.InvalidGitRepositoryError: If dirs.SCRIPT is not in a git repository.
        git.GitCommandError: If an error occurs while committing the changes.
        Exception: For any other unexpected errors.

    Returns:
        None
    """
    result = subprocess.run(
        [
            "sh",
            dirs.ASSETS / "scripts" / "unlock_git.sh",
            passphrase if passphrase else "",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return result.stdout.decode("utf-8")
    else:
        try:
            with git.Repo(dirs.APP, search_parent_directories=True) as repo:
                return repo.git.pull()

        except git.InvalidGitRepositoryError as e:
            raise RuntimeError(f"Unable to find a git repository: {e}")
        except git.GitCommandError as e:
            raise RuntimeError(f"Failed to push changes: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred: {e}")


def push(passphrase: str = None):
    """
    Pushes commits to the remote git repository.

    Args:
        passphrase (str, optional): The passphrase to unlock your Git credential store.

    Raises:
        git.InvalidGitRepositoryError: If dirs.SCRIPT is not in a git repository.
        git.GitCommandError: If an error occurs while committing the changes.
        Exception: For any other unexpected errors.

    Returns:
        None
    """
    result = subprocess.run(
        [
            "sh",
            dirs.ASSETS / "scripts" / "unlock_git.sh",
            passphrase if passphrase else "",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return result.stdout.decode("utf-8")
    else:
        try:
            with git.Repo(dirs.APP, search_parent_directories=True) as repo:
                return repo.git.push()

        except git.InvalidGitRepositoryError as e:
            raise RuntimeError(f"Unable to find a git repository: {e}")
        except git.GitCommandError as e:
            raise RuntimeError(f"Failed to push changes: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred: {e}")
