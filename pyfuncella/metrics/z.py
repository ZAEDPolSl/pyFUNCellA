import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


def _z(geneset, data, genes, alpha=0.05, gs_name=""):
    # should not be run on its own because of the normalisation in function
    genes_in_ds = [gene in geneset for gene in genes]
    N = len(genes)
    in_gs = data[genes_in_ds, :]
    N_gs = in_gs.shape[0]
    if N <= N_gs or N_gs == 0:
        print("Incorrect geneset format:", gs_name)
        return np.ones(data.shape[1]), np.ones(data.shape[1]), np.zeros(data.shape[1])
    z = np.sum(in_gs, axis=0) / (N_gs ** (1 / 2))

    pvals = stats.norm.sf(z)
    _, qvals, _, _ = multipletests(pvals, alpha=alpha, method="fdr_tsbh")
    return pvals, qvals, z


def Z(genesets, data, genes, alpha=0.05):
    # gene expression for each patient, genesets
    m = data.mean(axis=1, keepdims=True)
    sd = data.std(axis=1, keepdims=True, ddof=1)
    data = (data - m) / sd
    pval = np.empty((len(genesets), data.shape[1]))
    qval = np.empty((len(genesets), data.shape[1]))
    z = np.empty((len(genesets), data.shape[1]))
    for i, (gs_name, geneset_genes) in tqdm(
        enumerate(genesets.items()), total=len(genesets)
    ):
        pval[i], qval[i], z[i] = _z(geneset_genes, data, genes, alpha, gs_name)
    return pval, qval, z
