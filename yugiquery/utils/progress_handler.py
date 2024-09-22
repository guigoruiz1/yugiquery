# yugiquery/utils/progress_handler.py

# -*- coding: utf-8 -*-

# ======================= #
# Progress handler Module #
# ======================= #

# ======= #
# Imports #
# ======= #

# Standard library packages
# from __future__ import annotations
import asyncio
import multiprocessing as mp
from typing import Any, Dict, Callable, Optional

# Third-party imports
from tqdm.auto import tqdm


class ProgressHandler:
    """
    A progress handler class to comunicate progress of a process execution.

    Args:
        queue (multiprocessing.Queue | None, optional): The multiprocessing queue to communicate progress status. If None, a new queue is created. Defaults to None.
        progress_bar (tqdm | None, optional): The tqdm progress bar implementation. Defaults to None.
        pbar_kwargs (Dict[str, Any], optional): Keyword arguments to customize the progress bar. Defaults to None.

    Attributes:
        queue (multiprocessing.Queue | None, optional): The multiprocessing queue to communicate progress status. If None, a new queue is created. Defaults to None.
        progress_bar (tqdm | None, optional): The tqdm progress bar implementation. Defaults to None.
        pbar_kwargs (Dict[str, Any], optional): Keyword arguments to customize the progress bar. Defaults to None.
    """

    def __init__(
        self,
        queue: Optional[mp.Queue] | None = None,
        progress_bar: tqdm | None = None,
        pbar_kwargs: Dict[str, Any] = {},
    ):
        """
        Initializes the ProgressHandler class.

        Args:
            queue (multiprocessing.Queue | None, optional): The multiprocessing queue to communicate progress status. If None, a new queue is created. Defaults to None.
            progress_bar (tqdm | None, optional): The tqdm progress bar implementation. Defaults to None.
            pbar_kwargs (Dict[str, Any], optional): Keyword arguments to customize the progress bar. Defaults to None.
        """
        self.queue = queue if queue is not None else mp.Queue()
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
            return self.progress_bar(iterable, **self.pbar_kwargs, **kwargs)

    def send(self, **kwargs) -> None:
        """
        Puts the keyword arguments into the queue

        Args:
            **kwargs: Keyword arguments to put into the queue.

        Returns:
            None
        """
        self.queue.put(kwargs)

    async def await_result(self, process) -> tuple[int, bool, list]:
        while process.is_alive():
            await asyncio.sleep(1)

        API_status = None
        errors = []
        while not self.queue.empty():
            message = self.queue.get()
            if "API_status" in message:
                API_status = message.get("API_status")
            if "error" in message:
                errors.append(message.get("error"))

        self.queue.close()
        return API_status, errors
