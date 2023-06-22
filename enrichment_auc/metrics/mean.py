import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


def _mean(geneset, data, genes, gs_name=""):
    genes_in_ds = [gene in geneset for gene in genes]
    N = len(genes)
    in_gs = data[genes_in_ds, :]
    N_gs = in_gs.shape[0]
    if N <= N_gs or N_gs == 0:
        print("Incorrect geneset format:", gs_name)
        return np.zeros(data.shape[1])
    means = np.mean(in_gs, axis=0)
    return means


def MEAN(genesets, data, genes):
    means = np.empty((len(genesets), data.shape[1]))
    for i, (gs_name, geneset_genes) in tqdm(
        enumerate(genesets.items()), total=len(genesets)
    ):
        means[i] = _mean(geneset_genes, data, genes, gs_name)
    return means
