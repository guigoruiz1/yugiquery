import git
import subprocess
from utils.helpers import *

# ======================== #
# Git Management functions #
# ======================== #


def assure_repo():
    """
    Assures the script is inside a git repository. Initializes a repository if one is not found.

    Raises:
        Exception: For any unexpected errors.

    Returns:
        None
    """
    try:
        # Try to create a Repo object
        repo = git.Repo(PARENT_DIR)

    except git.InvalidGitRepositoryError:
        # Handle the case when the path is not a valid Git repository
        git.Repo.init(PARENT_DIR)
        print(f"Git repository initialized in {PARENT_DIR}")

    except Exception as e:
        # Handle any exceptions (e.g., invalid path)
        raise RuntimeError(f"Unable to init Git repository: {e}")


def commit(files: Union[str, List[str]], commit_message: str = None):
    """
    Commits the specified files to the git repository after staging them.

    Args:
        files (Union[str, List[str]]): A list of file paths to be committed.
        commit_message (str, optional): The commit message. If not provided, a default message will be used.

    Raises:
        git.InvalidGitRepositoryError: If the PARENT_DIR is not a git repository.
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
        with git.Repo(SCRIPT_DIR, search_parent_directories=True) as repo:
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
        git.InvalidGitRepositoryError: If the PARENT_DIR is not a git repository.
        git.GitCommandError: If an error occurs while committing the changes.
        Exception: For any other unexpected errors.

    Returns:
        None
    """
    result = subprocess.run(
        [
            "sh",
            os.path.join(SCRIPT_DIR, "assets/bash/unlock_git.sh"),
            passphrase if passphrase else "",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return result.stdout.decode("utf-8")
    else:
        try:
            with git.Repo(SCRIPT_DIR, search_parent_directories=True) as repo:
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
        git.InvalidGitRepositoryError: If the PARENT_DIR is not a git repository.
        git.GitCommandError: If an error occurs while committing the changes.
        Exception: For any other unexpected errors.

    Returns:
        None
    """
    result = subprocess.run(
        [
            "sh",
            os.path.join(SCRIPT_DIR, "assets/bash/unlock_git.sh"),
            passphrase if passphrase else "",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return result.stdout.decode("utf-8")
    else:
        try:
            with git.Repo(SCRIPT_DIR, search_parent_directories=True) as repo:
                return repo.git.push()

        except git.InvalidGitRepositoryError as e:
            raise RuntimeError(f"Unable to find a git repository: {e}")
        except git.GitCommandError as e:
            raise RuntimeError(f"Failed to push changes: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred: {e}")
