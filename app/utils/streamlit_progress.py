"""
Streamlit progress callback utilities for web interface.
"""

import streamlit as st
from typing import Callable


def create_streamlit_progress_callback() -> Callable:
    """
    Create a progress callback that uses Streamlit progress bar and text.

    Returns
    -------
    callable
        Progress callback function that accepts (current, total, message) parameters
    """
    progress_bar = st.progress(0)
    progress_text = st.empty()

    def callback(current: int, total: int, message: str = ""):
        progress = int(100 * current / total)
        progress_bar.progress(progress)

        if message:
            progress_text.text(f"Processing: {message} ({current}/{total})")
        else:
            progress_text.text(f"Progress: {current}/{total}")

    return callback
