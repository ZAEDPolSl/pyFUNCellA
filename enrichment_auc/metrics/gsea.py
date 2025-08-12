import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union
from ..utils.r_executor import execute_r_code, check_r_available, RProcessError


def _check_gsva_installed() -> bool:
    """Check if GSVA package is installed in R."""
    if not check_r_available():
        return False
    try:
        result = execute_r_code(
            """
        tryCatch({
            library(GSVA)
            success <- TRUE
        }, error = function(e) {
            success <- FALSE
        })
        success
        """
        )
        return result.get("success", False)
    except Exception as e:
        return False


def _run_analysis(genesets, data, genes, method, progress_callback=None):
    """
    Run GSVA/SSGSEA analysis using R executor pattern.

    Parameters
    ----------
    genesets : dict
        Dictionary mapping pathway names to lists of gene names
    data : array-like
        Gene expression data (genes x samples)
    genes : list
        List of gene names corresponding to data rows
    method : str
        Method to use ('gsva' or 'ssgsea')

    Returns
    -------
    numpy.ndarray
        Pathway enrichment scores (pathways x samples)
    """

    # Note: GSVA availability is verified during container build
    # Skip the runtime check to avoid false negatives in different execution contexts

    # Convert data to DataFrame
    df = pd.DataFrame(data, index=genes)

    # Prepare data for R
    data_inputs = {"gene_expr": df, "genesets": genesets, "method_param": method}

    # Get the path to the R script and read it
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    r_script_path = os.path.join(script_dir, "gsva_analysis.R")

    # Read the R script content
    with open(r_script_path, "r") as f:
        r_code = f.read()

    try:
        result = execute_r_code(r_code, data_inputs)

        if not result.get("success", False):
            raise RProcessError(f"{method.upper()} analysis failed in R")

        # Extract results - try different keys where results might be stored
        gsva_results = result.get("gsva_results")

        # If gsva_results is None, try other possible keys
        if gsva_results is None:
            for possible_key in ["scores", "samples", "pathways", "data", "results"]:
                if possible_key in result:
                    gsva_results = {possible_key: result[possible_key]}
                    break

        if gsva_results is None or not isinstance(gsva_results, dict):
            raise RProcessError(f"No results returned from {method.upper()} analysis")

        # Check for errors
        if "error" in gsva_results:
            raise RProcessError(
                f"{method.upper()} analysis failed: {gsva_results['error']}"
            )

        # Extract structured results
        scores = gsva_results.get("scores", {})
        samples = gsva_results.get("samples", [])
        pathways = gsva_results.get("pathways", [])

        # Reconstruct DataFrame
        try:
            # Create DataFrame with pathways as rows and samples as columns
            data_matrix = []
            pathway_names = []

            for pathway in pathways:
                if pathway in scores:
                    data_matrix.append(scores[pathway])
                    pathway_names.append(pathway)

            if not data_matrix:
                raise RProcessError(
                    f"No pathway scores found in {method.upper()} results"
                )

            results_df = pd.DataFrame(data_matrix, index=pathway_names, columns=samples)

            return results_df.values  # Return numpy array instead of DataFrame

        except Exception as e:
            # If all else fails, let's see what we actually got
            raise RProcessError(
                f"Cannot convert {method.upper()} results to DataFrame: {str(e)}"
            )

    except Exception as e:
        raise RProcessError(f"{method.upper()} analysis failed: {str(e)}")


def GSVA(genesets, data, genes):
    """
    Run GSVA analysis using R executor pattern.

    Parameters
    ----------
    genesets : dict
        Dictionary mapping pathway names to lists of gene names
    data : array-like
        Gene expression data (genes x samples)
    genes : list
        List of gene names corresponding to data rows

    Returns
    -------
    numpy.ndarray
        GSVA enrichment scores (pathways x samples)
    """
    return _run_analysis(genesets, data, genes, "gsva", None)


def SSGSEA(genesets, data, genes):
    """
    Run ssGSEA analysis using R executor pattern.

    Parameters
    ----------
    genesets : dict
        Dictionary mapping pathway names to lists of gene names
    data : array-like
        Gene expression data (genes x samples)
    genes : list
        List of gene names corresponding to data rows

    Returns
    -------
    numpy.ndarray
        ssGSEA enrichment scores (pathways x samples)
    """
    return _run_analysis(genesets, data, genes, "ssgsea", None)
