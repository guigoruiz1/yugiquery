# yugiquery/utils/progress_handler.py

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import asyncio
import multiprocessing as mp
from typing import Any, Dict, Iterable, Optional

# --- Imports: Third-Party --- #
from tqdm.auto import tqdm

# --- Classes --- #


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
        progress_bar: type[tqdm] | None = None,
        pbar_kwargs: Dict[str, Any] | None = {},
    ):
        """
        Initializes the ProgressHandler class.

        Args:
            queue (multiprocessing.Queue | None, optional): The multiprocessing queue to communicate progress status. If None, a new queue is created. Defaults to None.
            progress_bar (Type[tqdm] | None, optional): The tqdm progress bar class. Defaults to None.
            pbar_kwargs (Dict[str, Any], optional): Keyword arguments to customize the progress bar. Defaults to None.
        """
        self.queue = queue if queue is not None else mp.Queue()
        self.progress_bar = progress_bar
        self.pbar_kwargs = pbar_kwargs or {}

    def pbar(self, iterable: Iterable, **kwargs) -> tqdm | None:
        """
        Creates and returns a progress bar instance with merged kwargs.

        Args:
            iterable: The iterable to track progress.
            **kwargs: Additional keyword arguments for the progress bar (merged with stored pbar_kwargs).

        Returns:
            tqdm | None: The initialized progress bar instance or None if progress_bar is None.
        """
        if self.progress_bar is None:
            return None
        # Merge stored pbar_kwargs with runtime kwargs (runtime takes precedence)
        merged_kwargs = {**self.pbar_kwargs, **kwargs}
        return self.progress_bar(iterable, **merged_kwargs)

    def send(self, **kwargs) -> None:
        """
        Puts the keyword arguments into the queue

        Args:
            **kwargs: Keyword arguments to put into the queue.

        Returns:
            None
        """
        self.queue.put(kwargs)

    async def await_result(self, process: mp.Process) -> tuple[int | None, list]:  # TODO: Docstrings
        """
        Waits asynchronously for a multiprocessing process to finish and retrieves results from a queue.

        Args:
            process (multiprocessing.Process): The multiprocessing process to await.

        Returns:
            tuple[int | None, list]: A tuple containing the API status (int or None) and a list of errors.

        Notes:
            - The method periodically checks if the process is alive and waits until it finishes.
            - After completion, it extracts 'API_status' and 'error' messages from the queue.
            - The queue is closed after processing.
        """
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
