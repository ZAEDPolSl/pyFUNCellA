"""
Transform Gene-level Data to Pathway-level Scores Using Single-Sample Methods

This module provides a comprehensive function similar to FUNCellA's gene2path.R
for reducing gene-level data to pathway-level activity scores using various
single-sample pathway enrichment methods with filtering capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Literal
import warnings

# Import your existing metrics
from pyfuncella.metrics.mean import MEAN
from pyfuncella.metrics.bina import BINA
from pyfuncella.metrics.cerno import AUC
from pyfuncella.metrics.aucell import AUCELL
from pyfuncella.metrics.jasmine import JASMINE
from pyfuncella.metrics.z import Z as ZSCORE
from pyfuncella.metrics.gsea import SSGSEA

# Import preprocessing functions
from pyfuncella.preprocess.filter import (
    filter as variance_filter,
    filter_coverage,
    filter_size,
)


def _validate_inputs(data, genesets, method):
    """Validate input parameters for gene2path."""
    if data is None or len(data) == 0:
        raise ValueError("No data provided")

    if genesets is None or len(genesets) == 0:
        raise ValueError("No pathway list provided")

    valid_methods = [
        "CERNO",
        "MEAN",
        "BINA",
        "AUCELL",
        "JASMINE",
        "ZSCORE",
        "SSGSEA",
    ]
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")


def _prepare_data_and_genes(data, genes):
    """Prepare data array and gene names from input."""
    if isinstance(data, pd.DataFrame):
        if genes is None:
            genes = list(data.index)
        data_array = data.values
        sample_names = list(data.columns)
    else:
        data_array = np.array(data)
        if genes is None:
            genes = [f"Gene_{i}" for i in range(data_array.shape[0])]
        sample_names = [f"Sample_{i}" for i in range(data_array.shape[1])]

    if len(genes) != data_array.shape[0]:
        raise ValueError("Number of genes must match number of rows in data")

    return data_array, genes, sample_names


def _apply_variance_filtering(data, data_array, genes, variance_filter_threshold):
    """Apply variance filtering to the data if requested."""
    if variance_filter_threshold is None:
        return data_array, genes

    print(f"Applying variance filtering (keep top {variance_filter_threshold:.2%})")

    if isinstance(data, pd.DataFrame):
        try:
            filtered_data = variance_filter(data, leave_best=variance_filter_threshold)
            if isinstance(filtered_data, pd.DataFrame):
                data_array = filtered_data.values
                genes = list(filtered_data.index)
            else:
                raise ValueError("Unexpected return type from variance filter")
        except Exception as e:
            print(f"Warning: Variance filtering failed: {e}")
            print("Continuing without variance filtering...")
    else:
        # Use enhanced filter function for numpy arrays
        try:
            filtered_result = variance_filter(
                data_array, leave_best=variance_filter_threshold, genes=genes
            )
            if isinstance(filtered_result, tuple) and len(filtered_result) == 2:
                data_array, genes = filtered_result
            else:
                raise ValueError("Unexpected return type from variance filter")
        except Exception as e:
            print(f"Warning: Variance filtering failed: {e}")
            print("Continuing without variance filtering...")

    if genes is not None:
        print(f"After variance filtering: {len(genes)} genes")

    return data_array, genes


def _apply_pathway_filtering(genesets, genes, filt_min, filt_max, filt_cov):
    """Apply pathway filtering based on size and coverage."""
    # Filter pathways by size (filt_min, filt_max)
    if filt_min > 0 or filt_max < float("inf"):
        genesets = filter_size(genesets, min_size=filt_min, max_size=filt_max)

    # Filter pathways by coverage (filt_cov)
    if filt_cov > 0:
        genesets = filter_coverage(genesets, genes, min_coverage=filt_cov)

    return genesets


def _calculate_pathway_scores(
    method, genesets, data_array, genes, aucell_threshold, type
):
    """Calculate pathway scores using the specified method."""
    print(f"Calculating {method} scores...")

    if method == "CERNO":
        scores = AUC(genesets, data_array, genes)  # Use existing AUC function
    elif method == "MEAN":
        scores = MEAN(genesets, data_array, genes)
    elif method == "BINA":
        scores = BINA(genesets, data_array, genes)
    elif method == "AUCELL":
        scores = AUCELL(genesets, data_array, genes, aucell_threshold)
    elif method == "JASMINE":
        scores = JASMINE(genesets, data_array, genes, effect_size=type)
    elif method == "ZSCORE":
        pval, qval, z = ZSCORE(genesets, data_array, genes)
        scores = z  # Use z-scores as the primary result
    elif method == "SSGSEA":
        scores = SSGSEA(genesets, data_array, genes)
    else:
        raise ValueError(f"Method {method} not implemented")

    return scores


def gene2path(
    data: Union[np.ndarray, pd.DataFrame],
    genesets: Dict[str, List[str]],
    genes: Optional[List[str]] = None,
    method: Literal[
        "CERNO", "MEAN", "BINA", "AUCELL", "JASMINE", "ZSCORE", "SSGSEA"
    ] = "CERNO",
    filt_cov: float = 0,
    filt_min: int = 15,
    filt_max: int = 500,
    aucell_threshold: float = 0.05,
    variance_filter_threshold: Optional[float] = None,
    type: Literal["oddsratio", "likelihood"] = "oddsratio",
) -> pd.DataFrame:
    """
    Transform gene-level data to pathway-level scores using single-sample methods.

    This function reduces gene-level data to pathway-level activity scores using a variety of
    single-sample pathway enrichment methods. It supports filtering pathways based on coverage
    and size, similar to FUNCellA's gene2path function.

    Args:
        data: Gene expression matrix with genes as rows and samples as columns
        genesets: Dictionary mapping pathway names to lists of gene identifiers
        genes: List of gene names (if None, uses data index if DataFrame or creates range)
        method: Single-sample enrichment method to use
        filt_cov: Minimum fraction of pathway genes that must be present in data (0-1)
        filt_min: Minimum number of genes a pathway must have
        filt_max: Maximum number of genes a pathway can have
        aucell_threshold: Threshold parameter for AUCELL method (fraction of top genes to consider)
        variance_filter_threshold: If provided, filter genes by variance (keep top fraction)
        type: Type of effect size adjustment for JASMINE method ("oddsratio" or "likelihood")

    Returns:
        DataFrame with pathways as rows and samples as columns containing pathway activity scores

    Methods:
        - CERNO: Non-parametric method based on gene expression ranks using Mann-Whitney U statistic
        - MEAN: Simple mean expression of pathway genes per sample
        - BINA: Binary scoring based on proportion of expressed genes with logit transformation
        - AUCELL: Area Under the Curve method for gene set enrichment
        - JASMINE: Dropout-aware method for single-cell data with effect size adjustment
                   Uses 'type' parameter to specify effect size method (oddsratio or likelihood)
        - ZSCORE: Z-score based method using Stouffer integration
        - SSGSEA: Single-sample Gene Set Enrichment Analysis (requires R)

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> # Create example data
        >>> data = pd.DataFrame(np.random.randn(1000, 50))  # 1000 genes, 50 samples
        >>> data.index = [f"Gene_{i}" for i in range(1000)]
        >>> genesets = {
        ...     "Pathway1": [f"Gene_{i}" for i in range(0, 20)],
        ...     "Pathway2": [f"Gene_{i}" for i in range(10, 30)]
        ... }
        >>> scores = gene2path(data, genesets, method="CERNO")
    """

    # Input validation
    _validate_inputs(data, genesets, method)

    # Convert to numpy array if needed and get gene names
    data_array, genes, sample_names = _prepare_data_and_genes(data, genes)

    print(f"Starting gene2path transformation using {method} method")
    print(f"Input data: {data_array.shape[0]} genes x {data_array.shape[1]} samples")
    print(f"Input pathways: {len(genesets)}")

    # Apply variance filtering if requested
    data_array, genes = _apply_variance_filtering(
        data, data_array, genes, variance_filter_threshold
    )

    # Ensure genes is not None for the rest of the function
    if genes is None:
        raise ValueError("Gene names are required for filtering operations")

    # Filter pathways
    genesets = _apply_pathway_filtering(genesets, genes, filt_min, filt_max, filt_cov)

    print(f"Final pathways for analysis: {len(genesets)}")

    if len(genesets) == 0:
        warnings.warn("No pathways remain after filtering")
        return pd.DataFrame()

    # Calculate pathway scores based on method
    scores = _calculate_pathway_scores(
        method, genesets, data_array, genes, aucell_threshold, type
    )

    # Create result DataFrame
    pathway_names = list(genesets.keys())
    result_df = pd.DataFrame(scores, index=pathway_names, columns=sample_names)

    print(f"{method} scores calculated successfully")
    print(f"Output: {result_df.shape[0]} pathways x {result_df.shape[1]} samples")

    return result_df
