"""
Progress callback utilities for command line and web interfaces.
"""

from typing import Optional, Callable
from tqdm import tqdm


def create_tqdm_callback(
    description: str = "Processing", unit: str = "item", verbose: bool = True
) -> Callable:
    """
    Create a progress callback that uses tqdm for command line progress display.

    Parameters
    ----------
    description : str
        Description to show in the progress bar
    unit : str
        Unit name for the progress bar (e.g., 'pathway', 'sample')
    verbose : bool
        Whether to show the progress bar

    Returns
    -------
    callable
        Progress callback function that accepts (current, total, message) parameters
    """
    pbar = None

    def callback(current: int, total: int, message: str = ""):
        nonlocal pbar

        if pbar is None:
            pbar = tqdm(
                total=total,
                desc=description,
                unit=unit,
                disable=not verbose,
            )

        pbar.update(current - pbar.n)  # Update to current position

        if message:
            pbar.set_postfix_str(message)

        if current >= total:
            pbar.close()
            pbar = None

    return callback


def get_progress_callback(
    progress_callback: Optional[Callable] = None,
    description: str = "Processing",
    unit: str = "item",
    verbose: bool = True,
) -> Callable:
    """
    Get appropriate progress callback - either provided one or create tqdm fallback.

    Parameters
    ----------
    progress_callback : callable, optional
        Optional progress callback function
    description : str
        Description for fallback tqdm progress bar
    unit : str
        Unit for fallback tqdm progress bar
    verbose : bool
        Whether to show fallback progress bar

    Returns
    -------
    callable
        Progress callback function
    """
    if progress_callback is not None:
        return progress_callback
    else:
        return create_tqdm_callback(description, unit, verbose)
