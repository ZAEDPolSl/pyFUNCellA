import numpy as np
from tqdm import tqdm
from scipy.stats.mstats import rankdata


def dropout(data):
    ma_data = np.ma.masked_equal(data, 0)
    return ma_data


def rank_genes(masked_data):
    ranks = np.zeros(masked_data.data.shape)
    N = np.count_nonzero(masked_data.mask == 0, axis=0)
    good_idx = np.where(N != 0)
    ranks[:, good_idx] = rankdata(masked_data[:, good_idx], axis=0, use_missing=False)
    ranks = np.ma.masked_array(ranks, masked_data.mask)
    return ranks


def calc_odds_ratio(data, geneset_genes, genes):
    """
    Calculate odds ratio for each sample, assessing the enrichment
    of a pathway by comparing the presence (non-zero expression) of genes
    in the pathway versus those not in the pathway.

    Parameters:
    -----------
    data : np.ndarray
        Gene expression matrix with genes as rows and samples as columns
    geneset_genes : list
        List of gene names in the geneset
    genes : list
        List of all gene names

    Returns:
    --------
    np.ndarray
        Odds ratios for each sample
    """
    # Get indices of genes in and not in the geneset
    sig_gene_indices = [i for i, gene in enumerate(genes) if gene in geneset_genes]
    non_sig_gene_indices = [
        i for i, gene in enumerate(genes) if gene not in geneset_genes
    ]

    if len(sig_gene_indices) == 0:
        return np.zeros(data.shape[1])

    # Count expressed genes (non-zero) for signature and non-signature genes
    sig_genes_exp = np.sum(data[sig_gene_indices, :] != 0, axis=0)
    non_sig_genes_exp = np.sum(data[non_sig_gene_indices, :] != 0, axis=0)

    # Count non-expressed genes with pseudocount to prevent division by zero
    sig_genes_ne = np.maximum(len(sig_gene_indices) - sig_genes_exp, 1)
    non_sig_genes_ne = np.maximum(len(non_sig_gene_indices) - non_sig_genes_exp, 1)

    # Calculate odds ratio with protection against division by zero
    denominator = sig_genes_ne * non_sig_genes_exp
    denominator = np.maximum(denominator, 1e-10)  # Avoid division by zero
    or_values = (sig_genes_exp * non_sig_genes_ne) / denominator

    return or_values


def calc_likelihood(data, geneset_genes, genes):
    """
    Calculate likelihood ratio for each sample, comparing the expression
    of genes in the provided gene set against non-pathway genes.

    Parameters:
    -----------
    data : np.ndarray
        Gene expression matrix with genes as rows and samples as columns
    geneset_genes : list
        List of gene names in the geneset
    genes : list
        List of all gene names

    Returns:
    --------
    np.ndarray
        Likelihood ratios for each sample
    """
    # Get indices of genes in and not in the geneset
    sig_gene_indices = [i for i, gene in enumerate(genes) if gene in geneset_genes]
    non_sig_gene_indices = [
        i for i, gene in enumerate(genes) if gene not in geneset_genes
    ]

    if len(sig_gene_indices) == 0:
        return np.zeros(data.shape[1])

    # Count expressed genes (non-zero) for signature and non-signature genes
    sig_genes_exp = np.sum(data[sig_gene_indices, :] != 0, axis=0)
    non_sig_genes_exp = np.sum(data[non_sig_gene_indices, :] != 0, axis=0)

    # Count non-expressed genes with pseudocount to prevent division by zero
    sig_genes_ne = np.maximum(len(sig_gene_indices) - sig_genes_exp, 1)
    non_sig_genes_ne = np.maximum(len(non_sig_gene_indices) - non_sig_genes_exp, 1)

    # Calculate likelihood ratio
    lr1 = sig_genes_exp * (non_sig_genes_exp + non_sig_genes_ne)
    lr2 = non_sig_genes_exp * (sig_genes_exp + sig_genes_ne)

    # Avoid division by zero
    lr2 = np.maximum(lr2, 1e-10)
    lr_values = lr1 / lr2

    return lr_values


def scale_minmax(x):
    """
    Apply min-max normalization to scale values to range [0, 1].

    Parameters:
    -----------
    x : np.ndarray
        Input array to normalize

    Returns:
    --------
    np.ndarray
        Min-max normalized array
    """
    x = np.asarray(x)
    if np.isnan(x).all():
        return np.zeros_like(x)

    x_range = np.max(x) - np.min(x)
    if x_range == 0 or np.isnan(x_range):
        return np.zeros_like(x)

    x_scaled = (x - np.min(x)) / x_range
    return x_scaled


def _jasmine(geneset, ranks, genes, data=None, effect_size="oddsratio", gs_name=""):
    genes_in_ds = [gene in geneset for gene in genes]
    # find the number of nonzero genes for each cell
    N = np.count_nonzero(ranks.mask == 0, axis=0)

    # Calculate ranking component (RM) - dropout-aware ranking
    ranking_scores = ranks[genes_in_ds, :].mean(axis=0) / N
    ranking_scores = ranking_scores.filled(0)

    # If data is provided, calculate and combine with effect size component (ES)
    if data is not None:
        # Calculate effect size component
        if effect_size == "oddsratio":
            effect_scores = calc_odds_ratio(data, geneset, genes)
        elif effect_size == "likelihood":
            effect_scores = calc_likelihood(data, geneset, genes)
        else:
            effect_scores = np.zeros(data.shape[1])

        # Normalize both components with min-max scaling
        ranking_scores_norm = scale_minmax(ranking_scores)
        effect_scores_norm = scale_minmax(effect_scores)

        # Combine ranking and effect size (if effect size is valid)
        if np.isnan(effect_scores_norm).all():
            final_scores = ranking_scores_norm
        else:
            final_scores = (ranking_scores_norm + effect_scores_norm) / 2

        return final_scores
    else:
        # Return only ranking scores for backward compatibility (original method)
        return ranking_scores


def JASMINE(genesets, data, genes, use_effect_size=True, effect_size="oddsratio"):
    """
    Calculate JASMINE pathway enrichment scores.

    Parameters:
    -----------
    genesets : dict
        Dictionary mapping geneset names to lists of gene names
    data : np.ndarray
        Gene expression matrix with genes as rows and samples as columns
    genes : list
        List of all gene names
    use_effect_size : bool, default=True
        Whether to use effect size component (FUNCellA-style implementation)
    effect_size : str, default="oddsratio"
        Effect size method: "oddsratio" or "likelihood"

    Returns:
    --------
    np.ndarray
        JASMINE scores with shape (n_genesets, n_samples)
    """
    jasmine = np.empty((len(genesets), data.shape[1]))
    masked_data = dropout(data)
    ranks = rank_genes(masked_data)

    for i, (gs_name, geneset_genes) in tqdm(
        enumerate(genesets.items()), total=len(genesets)
    ):
        if use_effect_size:
            # Use enhanced method with effect size
            jasmine[i] = _jasmine(
                geneset_genes, ranks, genes, data, effect_size, gs_name
            )
        else:
            # Use original method
            jasmine[i] = _jasmine(geneset_genes, ranks, genes, gs_name=gs_name)

    # Apply final normalization only for original method
    if not use_effect_size:
        # standardize the results for each geneset using min-max scaling
        for i in range(jasmine.shape[0]):
            jasmine[i] = scale_minmax(jasmine[i])

    return jasmine
