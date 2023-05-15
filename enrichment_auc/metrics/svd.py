import numpy as np
from sklearn.decomposition import PCA, SparsePCA
from tqdm import tqdm


def _sparse_pca(geneset, data, genes, gs_name=""):
    genes_in_ds = [gene in geneset for gene in genes]
    in_gs = data[genes_in_ds, :]
    if (in_gs.T == in_gs.T[0]).all():
        print("Incorrect geneset format:", gs_name)
        return np.zeros(data.shape[1])
    gs_expression = SparsePCA(n_components=1).fit_transform(in_gs.T)
    return gs_expression.flatten()


def sparse_PCA(genesets, data, genes):
    res = np.empty((len(genesets), data.shape[1]))
    for i, (gs_name, geneset_genes) in tqdm(
        enumerate(genesets.items()), total=len(genesets)
    ):
        res[i] = _sparse_pca(geneset_genes, data, genes, gs_name)
    return res


def _svd(geneset, data, genes, gs_name=""):
    genes_in_ds = [gene in geneset for gene in genes]
    in_gs = data[genes_in_ds, :]
    if (in_gs.T == in_gs.T[0]).all():
        print("Incorrect geneset format:", gs_name)
        return np.zeros(data.shape[1])
    gs_expression = PCA(n_components=1).fit_transform(in_gs.T)
    return gs_expression.flatten()


def SVD(genesets, data, genes):
    res = np.empty((len(genesets), data.shape[1]))
    for i, (gs_name, geneset_genes) in tqdm(
        enumerate(genesets.items()), total=len(genesets)
    ):
        res[i] = _svd(geneset_genes, data, genes, gs_name)
    return res
