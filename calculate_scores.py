import json
import sys

import numpy as np
import pandas as pd
from time import time

from enrichment_auc.metrics import (
    aucell,
    cerno,
    gsea,
    gsva,
    ssgsea,
    rank,
    ratio,
    svd,
    jasmine,
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
    with open(inpath + "/filtered_genesets_genes.json") as file:
        genesets = json.load(file)
    gs_names = list(genesets.keys())
    # load gene expressions
    gene_expr = pd.read_csv(inpath + "/" + datatype + "_filtered_data.csv", index_col=0)
    patients_names = gene_expr.columns.to_list()
    genes = gene_expr.index.tolist()
    gene_expr = gene_expr.to_numpy().astype(float)

    # get scores with no ranks and single output:
    # scores_functions = [
    #     ratio.RATIO,
    #     svd.SVD,
    #     svd.sparse_PCA,
    #     vision.VISION,
    #     mean.MEAN,
    #     jasmine.JASMINE,
    # ]
    scores_names = [
        # "ratios",
        # "svd",
        # "sparse_pca",
        # "vision",
        # "mean",
        # "jasmine",
        # "gsva",
        # "ssgsea",
        # "cerno",
        # "aucell",
        # "auc",
        "z",
    ]
    elapsed_times = np.zeros(len(scores_names))
    # for i in range(len(scores_functions)):
    #     t = time()
    #     score = scores_functions[i](genesets, gene_expr, genes)
    #     elapsed = time() - t
    #     elapsed_times[i] = elapsed
    #     df_score = pd.DataFrame(
    #         data=score, index=list(genesets.keys()), columns=patients_names
    #     )
    #     df_score.to_csv(outpath + "/" + scores_names[i] + ".csv")

    # z
    z_names = ["pvals_005", "qvals_005", "z"]
    t = time()
    output = z.Z(genesets, gene_expr, genes, alpha=0.05)
    elapsed = time() - t
    elapsed_times[scores_names.index("z")] = elapsed
    for i in range(len(z_names)):
        df = pd.DataFrame(
            data=output[i], index=list(genesets.keys()), columns=patients_names
        )
        df.to_csv(outpath + "/" + z_names[i] + ".csv")

    # get GSEA based scores:
    # gsea_based_names = ["gsva", "ssgsea"]
    # gsea_based_functions = [gsva.get_ranks, ssgsea.get_ranks]
    # for i in range(len(gsea_based_names)):
    #     del df_score, score
    #     t = time()
    #     ranks, _ = gsea_based_functions[i](gene_expr, genes)
    #     score = gsea.GSEA(genesets, ranks, genes)
    #     elapsed = time() - t
    #     elapsed_times[scores_names.index(gsea_based_names[i])] = elapsed
    #     df_ranks = pd.DataFrame(data=ranks, index=genes, columns=patients_names)
    #     df_ranks.to_csv(outpath + "/ranks_" + gsea_based_names[i] + ".csv")
    #     df_score = pd.DataFrame(
    #         data=score, index=list(genesets.keys()), columns=patients_names
    #     )
    #     df_score.to_csv(outpath + "/" + gsea_based_names[i] + ".csv")
    #     del ranks

    # get scores based on simple ranks:
    # ranks
    # elements = [
    #     scores_names.index("cerno"),
    #     scores_names.index("auc"),
    #     scores_names.index("aucell"),
    # ]
    # t = time()
    # ranks = rank.rank_genes(gene_expr)
    # elapsed = time() - t
    # elapsed_times[elements] = elapsed
    # # aucell
    # t = time()
    # aucell_ = aucell.AUCELL(genesets, ranks, genes)
    # elapsed = time() - t
    # elapsed_times[scores_names.index("aucell")] += elapsed
    # df_aucell = pd.DataFrame(
    #     data=aucell_, index=list(genesets.keys()), columns=patients_names
    # )
    # df_aucell.to_csv(outpath + "/aucell.csv")
    # # cerno
    # cerno_names = ["cerno", "auc", "pvals_cerno005", "qvals_cerno005"]
    # t = time()
    # output = cerno.CERNO(genesets, ranks, genes, alpha=0.05)
    # elapsed = time() - t
    # elapsed_times[scores_names.index("cerno")] += elapsed
    # elapsed_times[scores_names.index("auc")] += elapsed
    # for i in range(len(cerno_names)):
    #     df = pd.DataFrame(
    #         data=output[i], index=list(genesets.keys()), columns=patients_names
    #     )
    #     df.to_csv(outpath + "/" + cerno_names[i] + ".csv")

    # times_df = pd.DataFrame(data=elapsed_times, index=scores_names, columns=["time"])
    # times_df.to_csv(outpath + "/times.csv")
