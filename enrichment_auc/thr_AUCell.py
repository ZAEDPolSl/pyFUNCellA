import numpy as np
import pandas as pd
from typing import Optional, Callable, Union, Dict, Any
from enrichment_auc.utils.progress_callbacks import get_progress_callback
from .utils.r_executor import execute_r_code, check_r_available, RProcessError


def _check_aucell_installed() -> bool:
    """Check if AUCell package is installed in R."""
    if not check_r_available():
        return False
    try:
        result = execute_r_code(
            """
        tryCatch({
            library(AUCell)
            success <- TRUE
        }, error = function(e) {
            success <- FALSE
        })
        success
        """
        )
        return result.get("success", False)
    except Exception:
        return False


def thr_AUCell(
    df_path,
    pathway_names=None,
    sample_names=None,
    progress_callback: Optional[Callable] = None,
):
    """
    Calculate activity thresholds for pathways using AUCell's thresholding algorithm.

    Parameters
    ----------
    df_path : pd.DataFrame or np.ndarray
        Pathway activity scores (rows: pathways, columns: samples).
    pathway_names : list, optional
        Names for pathways (rows). Used if df_path is numpy array.
    sample_names : list, optional
        Names for samples (columns). Used if df_path is numpy array.
    progress_callback : callable, optional
        Optional callback function for progress reporting.
        Should accept (current, total, message) parameters.

    Returns
    -------
    dict
        Dictionary mapping pathway names to threshold activity score.
    """
    if not check_r_available():
        raise RuntimeError("R is not available")

    if not _check_aucell_installed():
        raise RuntimeError("AUCell package is not installed in R")

    # Convert input to DataFrame if needed
    if isinstance(df_path, np.ndarray):
        n_pathways, n_samples = df_path.shape
        if pathway_names is None:
            pathway_names = [f"pathway_{i}" for i in range(n_pathways)]
        if sample_names is None:
            sample_names = [f"sample_{i}" for i in range(n_samples)]
        sample_names = [f"sample_{i}" for i in range(n_samples)]
        df = pd.DataFrame(df_path, index=pathway_names, columns=sample_names)
    else:
        df = df_path.copy()
        if df.index is None or not hasattr(df.index, "size") or df.index.size == 0:
            df = df.copy()
            df.index = pd.Index([f"pathway_{i}" for i in range(df.shape[0])])

    # Get the path to the R script and read it
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    r_script_path = os.path.join(script_dir, "aucell_threshold.R")

    # Read the R script content
    with open(r_script_path, "r") as f:
        r_code = f.read()

    try:
        if progress_callback:
            progress_callback(0, len(df), "Starting AUCell threshold calculation")

        thresholds = {}

        # Process each pathway individually for proper progress reporting
        for i, (pathway_name, pathway_data) in enumerate(df.iterrows()):
            if progress_callback:
                progress_callback(i, len(df), f"Processing {pathway_name}")

            # Prepare single pathway data for R
            pathway_df = pd.DataFrame([pathway_data], index=[pathway_name])
            data_inputs = {"pathway_scores": pathway_df}

            # Execute R code for this pathway
            result = execute_r_code(r_code, data_inputs)

            if result.get("success", False):
                threshold_value = result.get("aucell_results")
                if threshold_value is not None:
                    thresholds[pathway_name] = threshold_value
            else:
                print(f"Warning: Failed to calculate threshold for {pathway_name}")

        if progress_callback:
            progress_callback(
                len(df), len(df), "AUCell threshold calculation completed"
            )

        return thresholds

    except Exception as e:
        raise RProcessError(f"AUCell threshold calculation failed: {str(e)}")
