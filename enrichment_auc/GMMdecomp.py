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

    # Check R availability
    if not check_r_available():
        raise RuntimeError(
            "R is not available. Please ensure R is installed and accessible."
        )

    # Check dpGMM availability
    if not _check_dpgmm_installed():
        if verbose:
            print("dpGMM package not found. Attempting to install...")
        if not _install_dpgmm():
            raise RuntimeError(
                "Failed to install dpGMM package. "
                "Please install it manually: devtools::install_github('ZAEDPolSl/dpGMM')"
            )

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

    # Check for constant data and handle separately
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
                # Create failed result structure
                results[pathway_name] = {
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
                continue

            # Extract and process results for this pathway
            gmm_results = result.get("gmm_results", {})

            if pathway_name in gmm_results:
                pathway_result = gmm_results[pathway_name]
                if isinstance(pathway_result, dict) and pathway_result.get(
                    "success", False
                ):
                    # Process successful result
                    results[pathway_name] = _process_single_pathway_result(
                        pathway_result, multiply
                    )
                else:
                    # Handle failed pathway
                    results[pathway_name] = _create_failed_result()
            else:
                results[pathway_name] = _create_failed_result()

        except Exception as e:
            if verbose:
                print(f"Error processing pathway {pathway_name}: {str(e)}")
            results[pathway_name] = _create_failed_result()

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
    """Get the R code for GMM decomposition as a separate function."""
    return """
    # Load required libraries
    library(dpGMM)
    library(jsonlite)
    
    # Get parameters
    K <- K_param
    IC <- IC_param
    X <- X_data
    verbose_flag <- verbose_param
    
    # GMM options setup
    opt <- dpGMM::GMM_1D_opts
    opt$max_iter <- 1000
    opt$KS <- K
    opt$plot <- FALSE
    opt$quick_stop <- FALSE
    opt$SW <- 0.05
    opt$sigmas.dev <- 0
    opt$IC <- IC
    
    # Helper function for row calculation that extracts serializable components
    row_multiple <- function(row) {
        tmp <- as.numeric(row)
        result <- dpGMM::runGMM(tmp, opts = opt)
        
        # Extract key components that can be serialized based on runGMM structure
        serializable_result <- list(
            K = result$KS,  # Number of components
            IC = result[[IC]],  # Information criterion value
            loglik = result$logLik,  # Log-likelihood
            threshold = result$threshold,  # The thresholds we need!
            cluster = as.vector(result$cluster),  # Cluster assignments
            mu = result$model$mu,  # Component means
            sigma = result$model$sigma,  # Component standard deviations
            alpha = result$model$alpha,  # Component weights (not lambda)
            success = TRUE
        )
        
        # Handle potential NULL values
        if (is.null(serializable_result$mu)) serializable_result$mu <- numeric(0)
        if (is.null(serializable_result$sigma)) serializable_result$sigma <- numeric(0)
        if (is.null(serializable_result$alpha)) serializable_result$alpha <- numeric(0)
        if (is.null(serializable_result$threshold)) serializable_result$threshold <- numeric(0)
        if (is.null(serializable_result$cluster)) serializable_result$cluster <- integer(0)
        
        return(serializable_result)
    }
    
    # GMM Calculation for each row (should be just one row now)
    results_list <- list()
    total_rows <- nrow(X)
    
    for (i in 1:total_rows) {
        tryCatch({
            row_result <- row_multiple(X[i, ])
            results_list[[rownames(X)[i]]] <- row_result
        }, error = function(e) {
            if (verbose_flag) {
                cat("Warning: Failed to process pathway", rownames(X)[i], ":", e$message, "\\n")
            }
            results_list[[rownames(X)[i]]] <- list(error = e$message, success = FALSE)
        })
    }
    
    # Use jsonlite to properly serialize the results
    tryCatch({
        # Prepare final results using the expected variable name
        gmm_results <- results_list
    }, error = function(e) {
        cat("Error serializing results:", e$message, "\\n")
        gmm_results <- list(error = "Serialization failed", details = e$message)
    })
    """


def _process_single_pathway_result(pathway_result, multiply):
    """Process a single pathway result from R."""
    # Convert the flat structure from R to nested structure expected by tests
    alpha = pathway_result.get("alpha", [])
    mu = pathway_result.get("mu", [])
    sigma = pathway_result.get("sigma", [])
    threshold = pathway_result.get("threshold", [])

    # Convert scalars to arrays for consistency
    if np.isscalar(alpha):
        alpha = [alpha]
    if np.isscalar(mu):
        mu = [mu]
    if np.isscalar(sigma):
        sigma = [sigma]
    if np.isscalar(threshold):
        threshold = [threshold]

    # Unscale values if multiply=True was used
    if multiply:
        # Use exact arithmetic to avoid floating point precision issues
        scaling_factor = 10
        mu = [m / scaling_factor if isinstance(m, (int, float)) else m for m in mu]
        sigma = [
            s / scaling_factor if isinstance(s, (int, float)) else s for s in sigma
        ]
        threshold = [
            t / scaling_factor if isinstance(t, (int, float)) else t for t in threshold
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


def validate_gmm_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean GMM decomposition results.

    Parameters
    ----------
    results : dict
        Results from GMMdecomp function

    Returns
    -------
    dict
        Validated and cleaned results
    """
    validated = {}

    for pathway, result in results.items():
        if isinstance(result, dict) and "error" not in result:
            validated[pathway] = result
        else:
            print(f"Warning: Excluding invalid result for pathway {pathway}")

    return validated


def extract_thresholds(results: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Extract thresholds from GMM decomposition results.

    Parameters
    ----------
    results : dict
        Results from GMMdecomp function

    Returns
    -------
    dict
        Dictionary mapping pathway names to threshold arrays
    """
    thresholds = {}

    for pathway, result in results.items():
        if isinstance(result, dict) and "thresholds" in result:
            thresholds[pathway] = np.array(result["thresholds"])
        elif isinstance(result, dict) and "model" in result:
            # Try to extract thresholds from model if available
            model = result["model"]
            if isinstance(model, dict) and "thresholds" in model:
                thresholds[pathway] = np.array(model["thresholds"])

    return thresholds


def summarize_gmm_results(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a summary DataFrame of GMM decomposition results.

    Parameters
    ----------
    results : dict
        Results from GMMdecomp function

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with pathway statistics
    """
    summary_data = []

    for pathway, result in results.items():
        if isinstance(result, dict) and "error" not in result:
            row = {"pathway": pathway}

            # Extract basic information
            if "model" in result:
                model = result["model"]
                if isinstance(model, dict):
                    row["n_components"] = str(len(model.get("weights", [])))
                    row["converged"] = model.get("converged", False)

            if "thresholds" in result:
                thresholds = result["thresholds"]
                if isinstance(thresholds, (list, np.ndarray)):
                    row["n_thresholds"] = str(len(thresholds))

            summary_data.append(row)

    return pd.DataFrame(summary_data)
