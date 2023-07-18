import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from enrichment_auc.metrics.rank import rank_genes


def _fisher(
    geneset, data, genes, alpha=0.05, gs_name=""
):  # ordered gene names for each patient, geneset
    genes_in_ds = [gene in geneset for gene in genes]
    in_gs = data[genes_in_ds, :]
    N_gs = in_gs.shape[0]  # number of genes in GS
    N_tot = len(genes)  # total number of genes
    if N_tot <= N_gs or N_gs == 0:
        print("Incorrect geneset format:", gs_name)
        return (
            np.zeros(data.shape[1]),
            np.ones(data.shape[1]),
            np.ones(data.shape[1]),
        )
    cerno = np.sum(np.log(in_gs / N_tot), axis=0) * (-2)
    pvals = 1 - stats.chi2.cdf(cerno, 2 * N_gs)
    _, qvals, _, _ = multipletests(pvals, alpha=alpha, method="fdr_tsbh")
    return cerno, pvals, qvals


def FISHER(genesets, data, genes, alpha=0.05):
    data = rank_genes(data, ordinal=True)
    cerno = np.empty((len(genesets), data.shape[1]))
    pval = np.empty((len(genesets), data.shape[1]))
    qval = np.empty((len(genesets), data.shape[1]))
    for i, (gs_name, geneset_genes) in tqdm(
        enumerate(genesets.items()), total=len(genesets)
    ):
        cerno[i], pval[i], qval[i] = _fisher(geneset_genes, data, genes, alpha, gs_name)
    return cerno, pval, qval


def _auc(
    geneset, data, genes, gs_name=""
):  # ordered gene names for each patient, geneset
    genes_in_ds = [gene in geneset for gene in genes]
    in_gs = data[genes_in_ds, :]
    N_gs = in_gs.shape[0]  # number of genes in GS
    N_tot = len(genes)  # total number of genes
    if N_tot <= N_gs or N_gs == 0:
        print("Incorrect geneset format:", gs_name)
        return np.zeros(data.shape[1])
    R = np.sum(in_gs, axis=0)
    auc = N_gs * (N_tot - N_gs) + (N_gs + 1) * N_gs / 2 - R
    auc /= N_gs * (N_tot - N_gs)
    return auc


def AUC(genesets, data, genes):
    data = rank_genes(data)
    auc = np.empty((len(genesets), data.shape[1]))
    for i, (gs_name, geneset_genes) in tqdm(
        enumerate(genesets.items()), total=len(genesets)
    ):
        auc[i] = _auc(geneset_genes, data, genes, gs_name)
    return auc
