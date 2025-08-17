"""
GMM Decomposition using R's dpGMM package through process execution.

This module provides a Python wrapper for the GMMdecomp function
originally implemented in R using the dpGMM package.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, Callable

from .utils.r_executor import execute_r_code, check_r_available, RProcessError


def _check_r_available() -> bool:
    """Check if R is available (backward compatibility for tests)."""
    return check_r_available()


def _check_dpgmm_installed() -> bool:
    """Check if dpGMM package is installed in R."""
    if not check_r_available():
        return False
    try:
        result = execute_r_code(
            """
        tryCatch({
            library(dpGMM)
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


def _install_dpgmm() -> bool:
    """Attempt to install dpGMM package in R."""
    if not check_r_available():
        return False

    try:
        result = execute_r_code(
            """
        # Try to install dpGMM from GitHub
        if (!require("devtools", quietly = TRUE)) {
            install.packages("devtools", repos='https://cloud.r-project.org/')
        }
        
        devtools::install_github("ZAEDPolSl/dpGMM")
        
        # Verify installation
        library(dpGMM)
        installation_success <- TRUE
        """
        )
        return result.get("success", False) and result.get(
            "installation_success", False
        )
    except Exception:
        return False


def _create_simple_result(pathway_data, multiply):
    """Create a simple result for constant or near-constant data."""
    mean_value = np.mean(pathway_data)
    std_value = np.std(pathway_data)

    # Scale back if multiply was used
    if multiply:
        scaling_factor = 10
        mean_value = mean_value / scaling_factor
        std_value = std_value / scaling_factor

    return {
        "model": {
            "alpha": np.array([1.0]),
            "mu": np.array([mean_value]),
            "sigma": np.array([std_value]),
        },
        "thresholds": np.array([]),  # No thresholds for single component
        "IC_value": 0.0 if std_value == 0 else None,
        "converged": True,
    }


def _is_constant_data(pathway_data):
    """Check if pathway data is constant or near-constant."""
    return len(np.unique(pathway_data)) == 1 or np.std(pathway_data) < 1e-10


def _validate_inputs(X, K, multiply, IC, parallel, verbose):
    """Validate input parameters for GMMdecomp."""
    # Input validation
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise ValueError("X must be a pandas DataFrame or numpy array")

    if not isinstance(K, int) or K <= 0:
        raise ValueError("K must be a positive integer")

    if not isinstance(multiply, bool):
        raise ValueError("multiply must be a boolean value")

    IC_options = ["AIC", "AICc", "BIC", "ICL-BIC", "LR"]
    if IC not in IC_options:
        raise ValueError(f"IC must be one of {IC_options}")

    if not isinstance(parallel, bool):
        raise ValueError("parallel must be a boolean value")

    if not isinstance(verbose, bool):
        raise ValueError("verbose must be a boolean value")


def _prepare_data(X, multiply, verbose):
    """Prepare and validate data for GMM decomposition."""
    # Convert input to DataFrame if needed
    if isinstance(X, np.ndarray):
        if X.size == 0:
            raise ValueError("Input data X cannot be empty")
        X = pd.DataFrame(
            X,
            index=[f"pathway_{i}" for i in range(X.shape[0])],
            columns=[f"sample_{i}" for i in range(X.shape[1])],
        )
    elif isinstance(X, pd.DataFrame):
        if X.empty:
            raise ValueError("Input data X cannot be empty")

    # Convert to numeric and handle missing values
    X_numeric = X.select_dtypes(include=[np.number]).copy()
    if X_numeric.empty:
        raise ValueError("No numeric columns found in input data")

    # Apply scaling if requested
    if multiply:
        X_for_r = X_numeric * 10
    else:
        X_for_r = X_numeric.copy()

    return X_for_r


def _process_constant_pathways(X_for_r, multiply, verbose):
    """Process pathways with constant data and identify non-constant ones."""
    results = {}
    non_constant_pathways = []

    if verbose:
        print("Preprocessing pathways for constant data...")

    for pathway_name in X_for_r.index:
        pathway_data = X_for_r.loc[pathway_name].values.astype(float)

        if _is_constant_data(pathway_data):
            results[pathway_name] = _create_simple_result(pathway_data, multiply)
            if verbose:
                print(
                    f"Pathway {pathway_name}: Constant data detected, using simple result"
                )
        else:
            non_constant_pathways.append(pathway_name)

    return results, non_constant_pathways


def _process_single_pathway_gmm(pathway_name, X_for_r, K, IC, multiply, verbose):
    """Process a single pathway through GMM decomposition."""
    # Get single pathway data
    single_pathway_data = X_for_r.loc[[pathway_name]]

    # Prepare data for R (single pathway)
    data_inputs = {
        "X_data": single_pathway_data,
        "K_param": K,
        "IC_param": IC,
        "verbose_param": False,  # Reduce R verbosity for individual pathways
    }

    try:
        # Execute R code for single pathway
        result = execute_r_code(_get_gmm_r_code(), data_inputs)

        if not result.get("success", False):
            if verbose:
                print(f"Warning: GMM failed for pathway {pathway_name}")
            return _create_failed_result()

        # Extract and process results for this pathway
        gmm_results = result.get("gmm_results", {})

        if pathway_name in gmm_results:
            pathway_result = gmm_results[pathway_name]
            if isinstance(pathway_result, dict) and pathway_result.get(
                "success", False
            ):
                # Process successful result
                return _process_single_pathway_result(pathway_result, multiply)
            else:
                # Handle failed pathway
                return _create_failed_result()
        else:
            return _create_failed_result()

    except Exception as e:
        if verbose:
            print(f"Error processing pathway {pathway_name}: {str(e)}")
        return _create_failed_result()


def GMMdecomp(
    X: Union[pd.DataFrame, np.ndarray],
    K: int = 10,
    multiply: bool = True,
    IC: str = "BIC",
    parallel: bool = False,
    verbose: bool = True,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Perform Gaussian Mixture Model (GMM) decomposition on pathway enrichment scores.

    This function reduces pathway-level data to probabilistic components, which can
    help detect multimodal activity patterns in gene sets.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Data enrichment scores (rows: pathways, columns: samples).
        If numpy array, row names will be generated automatically.
    K : int, default=10
        Maximum number of components for GMM decomposition.
    multiply : bool, default=True
        Whether to scale values by 10 before fitting GMM.
    IC : str, default="BIC"
        Information criterion to use for model selection.
        Options: "AIC", "AICc", "BIC", "ICL-BIC", "LR".
    parallel : bool, default=False
        Whether to perform decomposition in parallel.
        Note: Parallel execution is not yet implemented.
    verbose : bool, default=True
        Whether to show progress messages.
    progress_callback : callable, optional
        Optional callback function for progress reporting.
        Should accept (current, total, message) parameters.

    Returns
    -------
    dict
        Dictionary of GMM decomposition results for each pathway.
        Keys are pathway names, values are dictionaries containing:
        - 'model': Fitted GMM model parameters (mixture weights, means, and standard deviations)
        - 'thresholds': Array of decision thresholds calculated by the model
        - Other diagnostics from dpGMM package

    Raises
    ------
    RuntimeError
        If R or dpGMM package is not available.
    ValueError
        If input parameters are invalid.
    """

    # Check R availability first with informative error message
    if not check_r_available():
        raise RuntimeError(
            "R is not available. GMM decomposition requires R with the dpGMM package.\n"
            "Please ensure:\n"
            "1. R is installed and accessible from your PATH\n"
            "2. Install R dependencies by running: Rscript setup_docker_compatible_renv.R\n"
            "3. Or manually install required packages: BiocManager::install('GSVA'); "
            "remotes::install_github('ZAEDPolSl/dpGMM')"
        )

    # Validate inputs
    _validate_inputs(X, K, multiply, IC, parallel, verbose)

    # Check dpGMM availability
    if not _check_dpgmm_installed():
        if verbose:
            print("dpGMM package not found. Attempting to install...")
        if not _install_dpgmm():
            raise RuntimeError(
                "Failed to install dpGMM package. "
                "Please install it manually by running: Rscript setup_docker_compatible_renv.R\n"
                "Or use: remotes::install_github('ZAEDPolSl/dpGMM')"
            )

    # Prepare data
    X_for_r = _prepare_data(X, multiply, verbose)

    # Process constant pathways and identify non-constant ones
    results, non_constant_pathways = _process_constant_pathways(
        X_for_r, multiply, verbose
    )

    # If all pathways were constant, return results
    if not non_constant_pathways:
        if verbose:
            print("All pathways contain constant data. Returning simple results.")
        return results

    if verbose:
        print(
            f"Processing {len(non_constant_pathways)} non-constant pathways through R..."
        )

    # Process pathways individually to avoid timeout and provide progress updates
    if progress_callback:
        progress_callback(
            0, len(non_constant_pathways), "Starting GMM decomposition..."
        )

    for i, pathway_name in enumerate(non_constant_pathways):
        if progress_callback:
            progress_callback(
                i, len(non_constant_pathways), f"Processing {pathway_name}"
            )

        if verbose:
            print(
                f"Processing pathway {i+1}/{len(non_constant_pathways)}: {pathway_name}"
            )

        # Process this pathway
        results[pathway_name] = _process_single_pathway_gmm(
            pathway_name, X_for_r, K, IC, multiply, verbose
        )

    if progress_callback:
        progress_callback(
            len(non_constant_pathways),
            len(non_constant_pathways),
            "GMM decomposition completed",
        )

    if verbose:
        successful_pathways = sum(1 for r in results.values() if r.get("K", 0) > 0)
        print(
            f"GMM decomposition completed: {successful_pathways}/{len(results)} pathways successful"
        )

    return results


def _get_gmm_r_code():
    """Get the R code for GMM decomposition from separate file."""
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    r_script_path = os.path.join(script_dir, "gmm_decomposition.R")

    # Read the R script content
    with open(r_script_path, "r") as f:
        return f.read()


def _process_single_pathway_result(pathway_result, multiply):
    """Process a single pathway result from R."""

    # Helper function to handle NA values from R
    def _clean_r_value(value):
        """Convert R values, handling NA strings."""
        if isinstance(value, str) and value.upper() in ["NA", "NAN", "NULL"]:
            return np.nan
        elif isinstance(value, (list, tuple)):
            return [_clean_r_value(v) for v in value]
        else:
            return value

    # Convert the flat structure from R to nested structure expected by tests
    alpha = _clean_r_value(pathway_result.get("alpha", []))
    mu = _clean_r_value(pathway_result.get("mu", []))
    sigma = _clean_r_value(pathway_result.get("sigma", []))
    threshold = _clean_r_value(pathway_result.get("threshold", []))

    # Convert scalars to arrays for consistency
    if np.isscalar(alpha):
        alpha = [alpha]
    if np.isscalar(mu):
        mu = [mu]
    if np.isscalar(sigma):
        sigma = [sigma]
    if np.isscalar(threshold):
        threshold = [threshold]

    # Filter out NaN values and handle empty results
    alpha = (
        [a for a in alpha if not (isinstance(a, float) and np.isnan(a))]
        if isinstance(alpha, (list, tuple))
        else ([alpha] if not (isinstance(alpha, float) and np.isnan(alpha)) else [])
    )
    mu = (
        [m for m in mu if not (isinstance(m, float) and np.isnan(m))]
        if isinstance(mu, (list, tuple))
        else ([mu] if not (isinstance(mu, float) and np.isnan(mu)) else [])
    )
    sigma = (
        [s for s in sigma if not (isinstance(s, float) and np.isnan(s))]
        if isinstance(sigma, (list, tuple))
        else ([sigma] if not (isinstance(sigma, float) and np.isnan(sigma)) else [])
    )
    threshold = (
        [t for t in threshold if not (isinstance(t, float) and np.isnan(t))]
        if isinstance(threshold, (list, tuple))
        else (
            [threshold]
            if not (isinstance(threshold, float) and np.isnan(threshold))
            else []
        )
    )

    # If we have no valid values, return a failed result
    if not alpha or not mu or not sigma or not threshold:
        return _create_failed_result()

    # Unscale values if multiply=True was used
    if multiply:
        # Use exact arithmetic to avoid floating point precision issues
        scaling_factor = 10
        mu = [
            m / scaling_factor if isinstance(m, (int, float)) and not np.isnan(m) else m
            for m in mu
        ]
        sigma = [
            s / scaling_factor if isinstance(s, (int, float)) and not np.isnan(s) else s
            for s in sigma
        ]
        threshold = [
            t / scaling_factor if isinstance(t, (int, float)) and not np.isnan(t) else t
            for t in threshold
        ]

    return {
        "model": {
            "alpha": np.array(alpha),
            "mu": np.array(mu),
            "sigma": np.array(sigma),
        },
        "thresholds": np.array(threshold),
        "K": pathway_result.get("K", 1),
        "IC": pathway_result.get("IC", 0.0),
        "loglik": pathway_result.get("loglik", 0.0),
        "cluster": np.array(pathway_result.get("cluster", [])),
    }


def _create_failed_result():
    """Create a failed result structure."""
    return {
        "model": {
            "alpha": np.array([]),
            "mu": np.array([]),
            "sigma": np.array([]),
        },
        "thresholds": np.array([]),
        "K": 0,
        "IC": float("inf"),
        "loglik": float("-inf"),
        "cluster": np.array([]),
    }


# Legacy function for backwards compatibility
def gmm_decomposition_parallel(*args, **kwargs):
    """
    Legacy function name for backwards compatibility.
    Calls GMMdecomp with parallel=True by default.
    """
    kwargs.setdefault("parallel", True)
    return GMMdecomp(*args, **kwargs)
