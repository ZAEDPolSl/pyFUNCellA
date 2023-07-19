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


def _jasmine(geneset, ranks, genes, gs_name=""):
    genes_in_ds = [gene in geneset for gene in genes]
    # find the number of nonzero genes for each cell
    N = np.count_nonzero(ranks.mask == 0, axis=0)
    jasmine = ranks[genes_in_ds, :].mean(axis=0) / N
    return jasmine.filled(0)


def JASMINE(genesets, data, genes):
    jasmine = np.empty((len(genesets), data.shape[1]))
    masked_data = dropout(data)
    ranks = rank_genes(masked_data)
    for i, (gs_name, geneset_genes) in tqdm(
        enumerate(genesets.items()), total=len(genesets)
    ):
        jasmine[i] = _jasmine(geneset_genes, ranks, genes, gs_name)
    # standardize the results for each geneset
    jasmine = (jasmine - jasmine.min(axis=1)[:, None]) / (
        jasmine.max(axis=1) - jasmine.min(axis=1)
    )[:, None]
    return jasmine
