import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multitest import multipletests

def rank_genes(data, descending=True):
    # for each patient return the gene ranking
    if descending:
        return (-data).argsort(axis=0).argsort(axis=0) + 1
    return np.argsort(data, axis=0) + 1

def _ratio(geneset, data, genes):
    genes_in_ds = [gene in geneset for gene in genes]
    in_gs = data[genes_in_ds, :]
    gs_expression = np.count_nonzero(in_gs, axis=0)/in_gs.shape[0]
    return gs_expression

def calculate_ratios(genesets, data, genes):
    res = np.empty((len(genesets), data.shape[1]))
    for i, (gs_name, geneset_genes) in enumerate(genesets.items()):
        res[i] = _ratio(geneset_genes, data, genes)
    return res

def _svd(geneset, data, genes):
    genes_in_ds = [gene in geneset for gene in genes]
    in_gs = data[genes_in_ds, :]
    if (in_gs.T == in_gs.T[0]).all():
        return np.zeros(data.shape[1])
    gs_expression = PCA(n_components=1).fit_transform(in_gs.T)
    return gs_expression.flatten()

def SVD(genesets, data, genes):
    res = np.empty((len(genesets), data.shape[1]))
    for i, (gs_name, geneset_genes) in enumerate(genesets.items()):
        res[i] = _svd(geneset_genes, data, genes)
    return res

def _cerno(geneset, data, genes, alpha=0.05, gs_name=""): # ordered gene names for each patient, geneset
    genes_in_ds = [gene in geneset for gene in genes]
    in_gs = data[genes_in_ds, :]
    N_gs = in_gs.shape[0]  # number of genes in GS
    N_tot = len(genes) # total number of genes 
    if N_tot <= N_gs or N_gs==0:
        print("Incorrect geneset format:", gs_name)
        return np.zeros(data.shape[1]), np.zeros(data.shape[1]), np.ones(data.shape[1])
    R = np.sum(in_gs, axis=0)
    AUC = (N_gs*(N_tot-N_gs)+(N_gs+1)*N_gs/2 - R)/(N_gs*(N_tot-N_gs))
    cerno = np.sum(np.log(in_gs/N_tot), axis=0)*(-2)
    pvals = 1 - stats.chi2.cdf(cerno, 2*N_gs)
    _, qvals, _, _ = multipletests(pvals, alpha=alpha, method='fdr_tsbh')
    return cerno, AUC, qvals

def CERNO(genesets, data, genes, alpha=0.05): # ordered gene names for each patient, genesets
    cerno = np.empty((len(genesets), data.shape[1]))
    auc = np.empty((len(genesets), data.shape[1]))
    pval = np.empty((len(genesets), data.shape[1]))
    for i, (gs_name, geneset_genes) in enumerate(genesets.items()):
        cerno[i], auc[i], pval[i] = _cerno(geneset_genes, data, genes, alpha, gs_name)
    return cerno, auc, pval
