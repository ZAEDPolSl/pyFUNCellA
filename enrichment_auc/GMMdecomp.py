"""
GMM Decomposition using R's dpGMM package through rpy2.

This module provides a Python wrapper for the GMMdecomp function
originally implemented in R using the dpGMM package.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union

try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter

    RPY2_AVAILABLE = True
except ImportError:
    RPY2_AVAILABLE = False
    robjects = None  # type: ignore
    pandas2ri = None  # type: ignore
    importr = None  # type: ignore
    localconverter = None  # type: ignore


def _check_r_available() -> bool:
    """Check if R is available through rpy2."""
    if not RPY2_AVAILABLE or robjects is None:
        return False
    try:
        robjects.r("R.version")
        return True
    except Exception:
        return False


def _check_dpgmm_installed() -> bool:
    """Check if dpGMM package is installed in R."""
    if not _check_r_available() or robjects is None:
        return False
    try:
        r_code = """
        tryCatch({
            library(dpGMM)
            TRUE
        }, error = function(e) {
            FALSE
        })
        """
        result = robjects.r(r_code)
        return bool(result[0])  # type: ignore
    except Exception:
        return False


def _install_dpgmm():
    """Install dpGMM package in R if not already installed."""
    if not _check_r_available() or robjects is None:
        raise RuntimeError("R is not available through rpy2")

    r_code = """
    if (!require("devtools", quietly = TRUE))
        install.packages("devtools", repos = "http://cran.us.r-project.org")
    if (!require("dpGMM", quietly = TRUE))
        devtools::install_github("ZAEDPolSl/dpGMM")
    library(dpGMM)
    """
    robjects.r(r_code)


def GMMdecomp(
    X: Union[pd.DataFrame, np.ndarray],
    K: int = 10,
    multiply: bool = True,
    IC: str = "BIC",
    parallel: bool = False,
    verbose: bool = True,
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

    Returns
    -------
    dict
        Dictionary of GMM decomposition results for each pathway.
        Keys are pathway names, values are dictionaries containing:
        - 'model': Fitted GMM model parameters (mixture weights, means, and standard deviations)
        - 'threshold': Decision threshold calculated by the model
        - Other diagnostics from dpGMM package

    Raises
    ------
    RuntimeError
        If R or dpGMM package is not available.
    ValueError
        If input parameters are invalid.
    """

    # Check if R and dpGMM are available
    if not _check_r_available() or robjects is None:
        raise RuntimeError(
            "R is not available through rpy2. Please install R and rpy2 package."
        )

    if not _check_dpgmm_installed():
        try:
            _install_dpgmm()
        except Exception as e:
            raise RuntimeError(
                f"Failed to install dpGMM package: {e}. "
                "Please install it manually: devtools::install_github('ZAEDPolSl/dpGMM')"
            )

    # Parameter validation
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

    # Apply scaling if requested
    if multiply:
        X_scaled = X * 10
    else:
        X_scaled = X.copy()

    # Define R function
    r_code = """
    GMMdecomp <- function(X, K=10, IC="BIC") {
        # Load required libraries
        library(dpGMM)
        
        # GMM options setup
        opt <- dpGMM::GMM_1D_opts
        opt$max_iter <- 1000
        opt$KS <- K
        opt$plot <- FALSE
        opt$quick_stop <- FALSE
        opt$SW <- 0.05
        opt$sigmas.dev <- 0
        opt$IC <- IC
        
    # Helper function for row calculation
    row_multiple <- function(row) {
        tmp <- as.numeric(row)
        result <- runGMM(tmp, opts = opt)
        return(result)
    }
        
        # GMM Calculation
        results_list <- lapply(1:nrow(X), function(i) {
            row_multiple(X[i, ])
        })
        
        names(results_list) <- rownames(X)
        return(results_list)
    }
    """

    # Execute R code to define the function
    if robjects is None:
        raise RuntimeError("robjects is None")
    robjects.r(r_code)

    # Get the R function
    r_gmmdecomp = robjects.globalenv["GMMdecomp"]

    # Convert pandas DataFrame to R data.frame
    if localconverter is None or pandas2ri is None:
        raise RuntimeError("rpy2 converters not available")

    def _create_simple_result(pathway_data, multiply):
        """Create a simple result for constant or near-constant data."""
        mean_value = np.mean(pathway_data)
        std_value = np.std(pathway_data)

        # Scale back if multiply was used
        if multiply:
            mean_value = mean_value / 10
            std_value = std_value / 10

        return {
            "model": {
                "alpha": np.array([1.0]),
                "mu": np.array([mean_value]),
                "sigma": np.array([std_value]),
            },
            "threshold": mean_value,
            "IC_value": 0.0 if std_value == 0 else None,
            "converged": True,
        }

    def _is_constant_data(pathway_data):
        """Check if pathway data is constant or near-constant."""
        return len(np.unique(pathway_data)) == 1 or np.std(pathway_data) < 1e-10

    def _extract_r_result(pathway_result, multiply):
        """Extract and process R result for a pathway."""

        def safe_extract(obj, key, default=None, converter=None):
            try:
                value = obj[key]
                if value is None:
                    return default
                return converter(value) if converter else value
            except (KeyError, TypeError, IndexError):
                return default

        # Extract model components
        model = safe_extract(pathway_result, "model", {})
        model_alpha = safe_extract(model, "alpha", np.array([]), np.array)
        model_mu = safe_extract(model, "mu", np.array([]), np.array)
        model_sigma = safe_extract(model, "sigma", np.array([]), np.array)

        # Extract threshold
        threshold = safe_extract(
            pathway_result,
            "threshold",
            0.0,
            lambda x: float(x[0]) if hasattr(x, "__getitem__") else float(x),
        )

        # Scale back if multiply was used
        if multiply:
            if model_mu is not None and len(model_mu) > 0:
                model_mu = model_mu / 10
            if model_sigma is not None and len(model_sigma) > 0:
                model_sigma = model_sigma / 10
            if threshold is not None:
                threshold = threshold / 10

        return {
            "model": {
                "alpha": model_alpha,
                "mu": model_mu,
                "sigma": model_sigma,
            },
            "threshold": threshold,
            "IC_value": safe_extract(pathway_result, "IC", None, lambda x: float(x[0])),
            "converged": safe_extract(
                pathway_result, "converged", None, lambda x: bool(x[0])
            ),
        }

    # Process pathways and separate constant from non-constant
    results = {}
    non_constant_pathways = []

    for pathway_name in X_scaled.index:
        pathway_data = X_scaled.loc[pathway_name].values.astype(float)

        if _is_constant_data(pathway_data):
            results[pathway_name] = _create_simple_result(pathway_data, multiply)
        else:
            non_constant_pathways.append(pathway_name)

    # If all pathways were constant, return results
    if not non_constant_pathways:
        return results

    # Process non-constant pathways through R
    X_for_r = X_scaled.loc[non_constant_pathways]

    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_X = robjects.conversion.py2rpy(X_for_r)
        r_results = r_gmmdecomp(r_X, K=K, IC=IC)

    # Process R results
    for pathway_name in non_constant_pathways:
        try:
            pathway_result = r_results[pathway_name]
            results[pathway_name] = _extract_r_result(pathway_result, multiply)
        except Exception:
            # Fallback to simple result if R processing fails
            pathway_data = X_scaled.loc[pathway_name].values.astype(float)
            results[pathway_name] = _create_simple_result(pathway_data, multiply)
            results[pathway_name]["converged"] = False

    return results
