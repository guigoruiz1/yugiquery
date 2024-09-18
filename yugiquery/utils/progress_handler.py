# yugiquery/utils/progress_handler.py

# -*- coding: utf-8 -*-

# ======================= #
# Progress handler Module #
# ======================= #

# ======= #
# Imports #
# ======= #

# Standard library packages
import io
import multiprocessing as mp
from typing import Any, Dict

# Third-party imports
from tqdm.auto import tqdm


class ProgressHandler:
    """
    Progress handler class.

    Args:
        queue (multiprocessing.Queue): The multiprocessing queue to communicate progress status.
        progress_bar (tqdm, optional): The tqdm progress bar implementation. Defaults to None.
        pbar_kwargs (Dict[str, Any], optional): Keyword arguments to customize the progress bar. Defaults to None.

    Attributes:
        queue (multiprocessing.Queue): The multiprocessing queue to communicate progress status.
        progress_bar (tqdm, optional): The tqdm progress bar implementation.
        pbar_kwargs (Dict[str, Any], optional): Keyword arguments to customize the progress bar.
    """

    def __init__(
        self,
        queue: mp.Queue,
        progress_bar: tqdm | None = None,
        pbar_kwargs: Dict[str, Any] = {},
    ):
        """
        Initializes the ProgressHandler class.

        Args:
            queue (multiprocessing.Queue): The multiprocessing queue to communicate progress status.
            progress_bar (tqdm | None, optional): The tqdm progress bar implementation. Defaults to None.
            pbar_kwargs (Dict[str, Any], optional): Keyword arguments to customize the progress bar. Defaults to None.
        """
        self.queue = queue
        self.progress_bar = progress_bar
        self.pbar_kwargs = pbar_kwargs

    def pbar(self, iterable, **kwargs) -> None | tqdm:
        """
        Initializes and returns a progress bar instance if progress_bar is not None.

        Args:
            iterable (iterable): The iterable to track progress.
            **kwargs: Additional keyword arguments for the progress bar.

        Returns:
            Progress bar instance or None: The initialized progress bar instance or None if progress_bar is None.
        """
        if self.progress_bar is None:
            return None
        else:
            return self.progress_bar(iterable, file=io.StringIO(), **self.pbar_kwargs, **kwargs)

    def exit(self, API_status: bool = True) -> None:
        """
        Puts the API status in the queue.

        Args:
            API_status (bool, optional): The status to put in the queue. Defaults to True.

        Returns:
            None
        """
        self.queue.put(API_status)
