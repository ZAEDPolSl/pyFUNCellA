import numpy as np
import pandas as pd
from typing import Optional, Callable, Union, Dict, Any
from pyfuncella.utils.progress_callbacks import get_progress_callback
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

    Automatically normalizes data to [0,1] range if values are outside this range,
    calculates thresholds on normalized data, then rescales thresholds back to
    original data range.

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
        Thresholds are in the original data scale.
    """
    if not check_r_available():
        raise RuntimeError(
            "R is not available. AUCell thresholding requires R with the AUCell package.\n"
            "Please ensure:\n"
            "1. R is installed and accessible from your PATH\n"
            "2. Install R dependencies by running: Rscript setup_docker_compatible_renv.R\n"
            "3. Or manually install AUCell: BiocManager::install('AUCell')"
        )

    if not _check_aucell_installed():
        raise RuntimeError(
            "AUCell package is not installed in R.\n"
            "Please install it by running: Rscript setup_docker_compatible_renv.R\n"
            "Or manually: BiocManager::install('AUCell')"
        )

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

    # Check if normalization is needed and store per-pathway scaling info
    pathway_scaling_info = {}
    df_normalized = df.copy()

    for pathway_name in df.index:
        pathway_data = df.loc[pathway_name]
        data_min = np.min(pathway_data)
        data_max = np.max(pathway_data)

        # Check if this pathway needs normalization
        if data_min < 0 or data_max > 1:
            # Store scaling info for this pathway
            original_range = data_max - data_min
            if original_range == 0:
                # Handle constant values - no normalization needed
                pathway_scaling_info[pathway_name] = None
            else:
                pathway_scaling_info[pathway_name] = {
                    "min": data_min,
                    "max": data_max,
                    "range": original_range,
                }
                # Normalize this pathway to [0, 1]
                df_normalized.loc[pathway_name] = (
                    pathway_data - data_min
                ) / original_range
        else:
            # No normalization needed for this pathway
            pathway_scaling_info[pathway_name] = None

    # Count how many pathways needed normalization
    normalized_count = sum(
        1 for info in pathway_scaling_info.values() if info is not None
    )
    total_pathways = len(df.index)

    if progress_callback:
        if normalized_count > 0:
            progress_callback(
                0,
                len(df),
                f"Normalized {normalized_count}/{total_pathways} pathways to [0, 1] range",
            )
        else:
            progress_callback(0, len(df), "All pathways already in [0, 1] range")

    # Get the path to the R script and read it
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    r_script_path = os.path.join(script_dir, "aucell_threshold.R")

    # Read the R script content
    with open(r_script_path, "r") as f:
        r_code = f.read()

    try:
        if progress_callback:
            progress_callback(
                0, len(df_normalized), "Starting AUCell threshold calculation"
            )

        thresholds = {}

        # Process each pathway individually for proper progress reporting
        for i, (pathway_name, pathway_data) in enumerate(df_normalized.iterrows()):
            if progress_callback:
                progress_callback(i, len(df_normalized), f"Processing {pathway_name}")

            # Prepare single pathway data for R
            pathway_df = pd.DataFrame([pathway_data], index=[pathway_name])
            data_inputs = {"pathway_scores": pathway_df}

            # Execute R code for this pathway
            result = execute_r_code(r_code, data_inputs)

            if result.get("success", False):
                threshold_value = result.get("aucell_results")
                if threshold_value is not None:
                    # Rescale threshold back to original data range if normalization was applied
                    scaling_info = pathway_scaling_info[pathway_name]
                    if scaling_info is not None:
                        # Rescale: threshold_original = threshold_normalized * range + min
                        threshold_value = (
                            threshold_value * scaling_info["range"]
                            + scaling_info["min"]
                        )
                    thresholds[pathway_name] = threshold_value
            else:
                print(f"Warning: Failed to calculate threshold for {pathway_name}")

        if progress_callback:
            if normalized_count > 0:
                progress_callback(
                    len(df),
                    len(df),
                    f"AUCell completed ({normalized_count} pathways rescaled to original ranges)",
                )
            else:
                progress_callback(
                    len(df), len(df), "AUCell threshold calculation completed"
                )

        return thresholds

    except Exception as e:
        raise RProcessError(f"AUCell threshold calculation failed: {str(e)}")
