import json
import sys

import pandas as pd

from enrichment_auc.metrics import (
    aucell,
    cerno,
    gsva,
    rank,
    ratio,
    svd,
    vision,
    z,
    mean,
)

if __name__ == "__main__":
    inpath = sys.argv[1]
    outpath = sys.argv[2]
    datatype = sys.argv[3]
    outpath = outpath + "/" + datatype
    # load genesets
    with open(inpath + "/genesets_genes.json") as file:
        genesets = json.load(file)
    gs_names = list(genesets.keys())
    # load gene expressions
    gene_expr = pd.read_csv(inpath + "/" + datatype + "_filtered_data.csv", index_col=0)
    patients_names = gene_expr.columns.to_list()
    genes = gene_expr.index.tolist()
    gene_expr = gene_expr.to_numpy().astype(float)
    # get scores
    # ratio
    ratio_ = ratio.calculate_ratios(genesets, gene_expr, genes)
    df_ratio = pd.DataFrame(
        data=ratio_, index=list(genesets.keys()), columns=patients_names
    )
    df_ratio.to_csv(outpath + "/ratios.csv")
    # ranks
    ranks = rank.rank_genes(gene_expr)
    # aucell
    aucell_ = aucell.AUCELL(genesets, ranks, genes)
    df_aucell = pd.DataFrame(
        data=aucell_, index=list(genesets.keys()), columns=patients_names
    )
    df_aucell.to_csv(outpath + "/aucell.csv")
    # cerno
    cerno_, auc, pvals_005, qvals_005 = cerno.CERNO(genesets, ranks, genes, alpha=0.05)
    df_cerno = pd.DataFrame(
        data=cerno_, index=list(genesets.keys()), columns=patients_names
    )
    df_cerno.to_csv(outpath + "/cerno.csv")
    df_auc = pd.DataFrame(data=auc, index=list(genesets.keys()), columns=patients_names)
    df_auc.to_csv(outpath + "/auc.csv")
    df_pvals_005 = pd.DataFrame(
        data=pvals_005, index=list(genesets.keys()), columns=patients_names
    )
    df_pvals_005.to_csv(outpath + "/pvals_cerno005.csv")
    df_qvals_005 = pd.DataFrame(
        data=qvals_005, index=list(genesets.keys()), columns=patients_names
    )
    df_qvals_005.to_csv(outpath + "/qvals_cerno005.csv")
    # svd
    svd_ = svd.SVD(genesets, gene_expr, genes)
    df_svd = pd.DataFrame(
        data=svd_, index=list(genesets.keys()), columns=patients_names
    )
    df_svd.to_csv(outpath + "/svd.csv")
    # sparse pca
    svd_ = svd.sparse_PCA(genesets, gene_expr, genes)
    df_svd = pd.DataFrame(
        data=svd_, index=list(genesets.keys()), columns=patients_names
    )
    df_svd.to_csv(outpath + "/sparse_pca.csv")
    # vision
    vision_ = vision.VISION(genesets, gene_expr, genes)
    df_vision = pd.DataFrame(
        data=vision_, index=list(genesets.keys()), columns=patients_names
    )
    df_vision.to_csv(outpath + "/vision.csv")
    # mean
    mean_ = mean.MEAN(genesets, gene_expr, genes)
    df_mean = pd.DataFrame(
        data=mean_, index=list(genesets.keys()), columns=patients_names
    )
    df_mean.to_csv(outpath + "/mean.csv")
    # z
    pvals_005, qvals_005, z_ = z.Z(genesets, ranks, genes, alpha=0.05)
    df_z = pd.DataFrame(data=z_, index=list(genesets.keys()), columns=patients_names)
    df_z.to_csv(outpath + "/z.csv")
    df_pvals_005 = pd.DataFrame(
        data=pvals_005, index=list(genesets.keys()), columns=patients_names
    )
    df_pvals_005.to_csv(outpath + "/pvals_z005.csv")
    df_qvals_005 = pd.DataFrame(
        data=qvals_005, index=list(genesets.keys()), columns=patients_names
    )
    df_qvals_005.to_csv(outpath + "/qvals_z005.csv")
    # gsva
    del ratio_, cerno_, auc, pvals_005, qvals_005, svd_, vision_, z_
    del df_ratio, df_auc, df_qvals_005, df_cerno, df_vision, df_z
    ranks, _ = gsva.get_ranks(gene_expr, genes)
    df_gsva = pd.DataFrame(data=ranks, index=genes, columns=patients_names)
    df_gsva.to_csv(outpath + "/ranks_gsva.csv")
    gsva_ = gsva.GSVA(genesets, ranks, genes)
    df_gsva = pd.DataFrame(
        data=gsva_, index=list(genesets.keys()), columns=patients_names
    )
    df_gsva.to_csv(outpath + "/gsva.csv")
