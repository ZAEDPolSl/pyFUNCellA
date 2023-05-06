import math

import numpy as np
from scipy import stats
from tqdm import tqdm


def _calculate_gene_zscores(x, batch_size=30):
    if x.shape[0] < batch_size:
        print("Preparing for subtraction...")
        y = np.repeat(x, x.shape[1], axis=0)
        y = y.reshape((x.shape[0], x.shape[1], x.shape[1]))
        print("Subtracting..")
        y = -y + x[:, :, None]
        print("Calculating probabilities...")
        y = stats.norm.cdf(y)
        print("Calculating the means...")
        results = np.mean(y, axis=-1)
    else:
        results = np.zeros(x.shape)
        idx = 0
        num_sections = math.ceil(x.shape[0] / batch_size)
        for part in tqdm(np.array_split(x, num_sections)):
            y = np.repeat(part, part.shape[1], axis=0)
            y = y.reshape((part.shape[0], part.shape[1], part.shape[1]))
            y = -y + part[:, :, None]
            y = stats.norm.cdf(y)
            results[idx: idx + part.shape[0], :] = np.mean(y, axis=-1)
            idx += part.shape[0]
    return results


def _calculate_gene_poissons(x):
    x = x.astype(float)
    for i in range(x.shape[0]):
        x[i, :] = stats.poisson.cdf(
            x[i, :].reshape((x.shape[1], 1)), x[i, :] + 0.5
        ).mean(axis=1)
    return x


def _rank_expressions(phenotype_expressions):
    ranks = (-phenotype_expressions).argsort(axis=0).argsort(axis=0) + 1
    ranks = ranks - ranks.shape[0] / 2
    return np.abs(ranks)


def _get_miss_increment(genes_in_ds, genes):
    N = len(genes)
    n = len(genes_in_ds)
    increment = (-1) / (N - n)
    return increment


def get_ranks(data, genes, datatype="single_cell"):
    print("Transforming the geneset...\n")
    if datatype == "single_cell":
        transformed = _calculate_gene_poissons(data)
    else:
        bandwidths = (np.std(data, axis=1) / 4)[:, None]
        if np.where(bandwidths == 0)[0].shape[0] != 0:
            good_indices = np.where(bandwidths != 0)[0]
            data = data[good_indices]
            bandwidths = bandwidths[good_indices]
            genes = [genes[i] for i in good_indices.tolist()]
            print(
                "Some of the genes had 0 standard deviation. Removing them from the dataset."
            )
        # divide each row by its bandwidth
        data = data / bandwidths
        transformed = _calculate_gene_zscores(data)
    print("Getting the ranks...\n")
    ranks = _rank_expressions(transformed)
    return ranks, genes


def _gsva(geneset, ranks, genes, gs_name=""):
    genes_in_ds = [gene in geneset for gene in genes]
    not_in_gs = [not present for present in genes_in_ds]
    present_genes = [genes[i] for i in range(len(genes)) if genes_in_ds[i]]
    if len(genes) == len(present_genes):
        print("Incorrect geneset format:", gs_name)
        return np.zeros(ranks.shape[1])
    # increment the same for all phenotypes
    miss_increment = _get_miss_increment(present_genes, genes)
    # N_R for each patient
    N_R = np.sum(ranks[genes_in_ds], axis=0)
    ranks = ranks / N_R
    ranks[not_in_gs] = miss_increment  # already negative
    ES = np.cumsum(ranks, axis=0)
    ES_max = np.max(ES, axis=0)
    ES_min = np.abs(np.min(ES, axis=0))
    return ES_max - ES_min


def GSVA(genesets, ranks, genes):
    print("Calculating the score...\n")
    gsva = np.empty((len(genesets), ranks.shape[1]))
    for i, (gs_name, geneset_genes) in tqdm(
        enumerate(genesets.items()), total=len(genesets)
    ):
        gsva[i] = _gsva(geneset_genes, ranks, genes, gs_name)
    return gsva
