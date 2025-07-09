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
from enrichment_auc.metrics.mean import MEAN
from enrichment_auc.metrics.bina import BINA
from enrichment_auc.metrics.cerno import AUC
from enrichment_auc.metrics.aucell import AUCELL
from enrichment_auc.metrics.jasmine import JASMINE
from enrichment_auc.metrics.z import Z as ZSCORE
from enrichment_auc.metrics.gsea import SSGSEA

# Import preprocessing functions
from enrichment_auc.preprocess.filter import (
    filter as variance_filter,
    filter_coverage,
    filter_size,
)


def gene2path(
    data: Union[np.ndarray, pd.DataFrame],
    genesets: Dict[str, List[str]],
    genes: Optional[List[str]] = None,
    method: Literal[
        "CERNO", "MEAN", "BINA", "AUCELL", "JASMINE", "ZSCORE", "SSGSEA", "AUC"
    ] = "CERNO",
    filt_cov: float = 0,
    filt_min: int = 15,
    filt_max: int = 500,
    aucell_threshold: float = 0.05,
    variance_filter_threshold: Optional[float] = None,
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

    Returns:
        DataFrame with pathways as rows and samples as columns containing pathway activity scores

    Methods:
        - CERNO: Non-parametric method based on gene expression ranks using Mann-Whitney U statistic
        - MEAN: Simple mean expression of pathway genes per sample
        - BINA: Binary scoring based on proportion of expressed genes with logit transformation
        - AUCELL: Area Under the Curve method for gene set enrichment
        - JASMINE: Dropout-aware method for single-cell data with effect size adjustment
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
    if data is None or len(data) == 0:
        raise ValueError("No data provided")

    if genesets is None or len(genesets) == 0:
        raise ValueError("No pathway list provided")

    # Convert to numpy array if needed and get gene names
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

    # Validate method
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

    print(f"Starting gene2path transformation using {method} method")
    print(f"Input data: {data_array.shape[0]} genes x {data_array.shape[1]} samples")
    print(f"Input pathways: {len(genesets)}")

    # Apply variance filtering if requested
    if variance_filter_threshold is not None:
        print(f"Applying variance filtering (keep top {variance_filter_threshold:.2%})")
        if isinstance(data, pd.DataFrame):
            filtered_data = variance_filter(data, leave_best=variance_filter_threshold)
            data_array = filtered_data.values
            genes = list(filtered_data.index)
        else:
            # Use enhanced filter function for numpy arrays
            filtered_result = variance_filter(
                data_array, leave_best=variance_filter_threshold, genes=genes
            )
            if isinstance(filtered_result, tuple):
                data_array, genes = filtered_result
            else:
                raise ValueError("Unexpected return type from variance filter")
        print(f"After variance filtering: {len(genes)} genes")

    # Ensure genes is not None for the rest of the function
    if genes is None:
        raise ValueError("Gene names are required for filtering operations")

    # Filter pathways by size (filt_min, filt_max)
    if filt_min > 0 or filt_max < float("inf"):
        genesets = filter_size(genesets, min_size=filt_min, max_size=filt_max)

    # Filter pathways by coverage (filt_cov)
    if filt_cov > 0:
        genesets = filter_coverage(genesets, genes, min_coverage=filt_cov)

    print(f"Final pathways for analysis: {len(genesets)}")

    if len(genesets) == 0:
        warnings.warn("No pathways remain after filtering")
        return pd.DataFrame()

    # Calculate pathway scores based on method
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
        scores = JASMINE(genesets, data_array, genes)
    elif method == "ZSCORE":
        scores = ZSCORE(genesets, data_array, genes)
    elif method == "SSGSEA":
        scores = SSGSEA(genesets, data_array, genes)
    else:
        raise ValueError(f"Method {method} not implemented")

    # Create result DataFrame
    pathway_names = list(genesets.keys())
    result_df = pd.DataFrame(scores, index=pathway_names, columns=sample_names)

    print(f"{method} scores calculated successfully")
    print(f"Output: {result_df.shape[0]} pathways x {result_df.shape[1]} samples")

    return result_df


# Example usage and utility functions
def create_example_data(
    n_genes: int = 1000, n_samples: int = 50, n_pathways: int = 10
) -> tuple:
    """
    Create example data for testing gene2path function.

    Args:
        n_genes: Number of genes
        n_samples: Number of samples
        n_pathways: Number of pathways to create

    Returns:
        Tuple of (data, genesets, genes) for testing
    """
    # Create gene expression data
    data = np.random.randn(n_genes, n_samples)

    # Create gene names
    genes = [f"Gene_{i}" for i in range(n_genes)]

    # Create pathways
    genesets = {}
    pathway_size_range = (15, 50)

    for i in range(n_pathways):
        pathway_size = np.random.randint(pathway_size_range[0], pathway_size_range[1])
        pathway_genes = np.random.choice(genes, size=pathway_size, replace=False)
        genesets[f"Pathway_{i}"] = list(pathway_genes)

    return data, genesets, genes


def run_example():
    """
    Run an example of the gene2path function with various methods.
    """
    print("=== Gene2Path Example ===")

    # Create example data
    data, genesets, genes = create_example_data(n_genes=500, n_samples=20, n_pathways=5)
    print(
        f"Created example data: {len(genes)} genes, {len(genesets)} pathways, {data.shape[1]} samples"
    )

    # Test different methods
    methods = ["CERNO", "MEAN", "BINA", "AUCELL", "JASMINE", "ZSCORE", "SSGSEA", "AUC"]

    for method in methods:
        try:
            print(f"\nTesting {method} method...")
            scores = gene2path(
                data=data,
                genesets=genesets,
                genes=genes,
                method=method,  # type: ignore
            )
            print(
                f"✓ {method}: Generated {scores.shape[0]} pathway scores for {scores.shape[1]} samples"
            )

        except Exception as e:
            print(f"✗ {method}: Failed with error: {str(e)}")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    # Run example when script is executed directly
    run_example()
