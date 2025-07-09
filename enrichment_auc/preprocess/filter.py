import sys
from typing import Union, List, Optional

import numpy as np
import pandas as pd


def filter(
    cells: Union[pd.DataFrame, np.ndarray],
    leave_best: float = 0.25,
    genes: Optional[List[str]] = None,
) -> Union[pd.DataFrame, tuple]:
    """
    Filter genes by variance, keeping the most variable ones.

    Args:
        cells: Gene expression data (DataFrame or numpy array)
        leave_best: Fraction of genes to keep (0-1)
        genes: Gene names (required if cells is numpy array)

    Returns:
        If input is DataFrame: filtered DataFrame
        If input is numpy array: tuple of (filtered_array, filtered_gene_names)
    """
    leave_best = max(0, min(leave_best, 1))

    if isinstance(cells, pd.DataFrame):
        # Original DataFrame logic
        cells.dropna(inplace=True)
        vars = np.var(cells, axis=1)
        vars = vars[vars != 0]
        vars = vars.sort_values()
        cells = cells.loc[vars[int((1 - leave_best) * vars.shape[0]) :].index]
        return cells

    elif isinstance(cells, np.ndarray):
        # Numpy array logic
        if genes is None:
            raise ValueError("Gene names must be provided when filtering numpy arrays")

        if len(genes) != cells.shape[0]:
            raise ValueError("Number of genes must match number of rows in data")

        # Calculate variances
        variances = np.var(cells, axis=1)

        # Remove genes with zero variance
        non_zero_mask = variances != 0
        cells_filtered = cells[non_zero_mask, :]
        genes_filtered = [genes[i] for i in range(len(genes)) if non_zero_mask[i]]
        variances_filtered = variances[non_zero_mask]

        # Sort by variance and keep top fraction
        n_keep = int(leave_best * len(variances_filtered))
        if n_keep == 0:
            n_keep = 1  # Keep at least one gene

        # Get indices of top variable genes
        top_indices = np.argsort(variances_filtered)[-n_keep:]

        # Filter data and genes
        final_data = cells_filtered[top_indices, :]
        final_genes = [genes_filtered[i] for i in top_indices]

        return final_data, final_genes

    else:
        raise ValueError("Input must be pandas DataFrame or numpy array")


def filter_coverage(
    genesets: dict,
    genes: List[str],
    min_coverage: float = 0.0,
) -> dict:
    """
    Filter pathways by gene coverage in expression data.

    Args:
        genesets: Dictionary of pathway name -> list of genes
        genes: List of gene names available in the dataset
        min_coverage: Minimum fraction of pathway genes that must be present in data (0-1)

    Returns:
        Filtered genesets dictionary
    """
    if min_coverage <= 0:
        return genesets

    print(f"PATHWAY COVERAGE FILTRATION (min_coverage={min_coverage})")

    filtered_genesets = {}
    removed_count = 0

    for pathway_name, pathway_genes in genesets.items():
        # Check how many pathway genes are present in the data
        genes_present = [gene for gene in pathway_genes if gene in genes]
        coverage = (
            len(genes_present) / len(pathway_genes) if len(pathway_genes) > 0 else 0
        )

        if coverage >= min_coverage:
            filtered_genesets[pathway_name] = pathway_genes
        else:
            removed_count += 1

    print(f"In total removed: {removed_count} pathways")
    return filtered_genesets


def filter_size(
    genesets: dict,
    min_size: int = 15,
    max_size: int = 500,
) -> dict:
    """
    Filter pathways by size (number of genes).

    Args:
        genesets: Dictionary of pathway name -> list of genes
        min_size: Minimum number of genes a pathway must have
        max_size: Maximum number of genes a pathway can have

    Returns:
        Filtered genesets dictionary
    """
    print(f"PATHWAY SIZE FILTRATION (min_size={min_size}, max_size={max_size})")

    filtered_genesets = {}
    removed_count = 0

    for pathway_name, pathway_genes in genesets.items():
        pathway_size = len(pathway_genes)

        if min_size <= pathway_size <= max_size:
            filtered_genesets[pathway_name] = pathway_genes
        else:
            removed_count += 1

    print(f"In total removed: {removed_count} pathways")
    return filtered_genesets


if __name__ == "__main__":
    folder = sys.argv[1]
    datatype = sys.argv[2]
    path = sys.argv[3]
    infile = path + folder + "/" + datatype + "_data.csv"
    outfile = path + folder + "/" + datatype + "_filtered_data.csv"
    cells = pd.read_csv(infile, index_col=0)
    filtered_cells = filter(cells)
    # Since we're reading a CSV, the result will always be a DataFrame
    if isinstance(filtered_cells, pd.DataFrame):
        filtered_cells.to_csv(outfile)
    else:
        raise ValueError("Unexpected return type from filter function")
