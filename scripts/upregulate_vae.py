import os
import json
import numpy as np
import pandas as pd
import sys


import enrichment_auc.metrics.vae as vae

if __name__ == "__main__":
    tissue = sys.argv[1]
    norm = sys.argv[2]
    data_folder = os.path.join(sys.argv[3], tissue)
    res_folder = os.path.join(sys.argv[4], tissue, norm)

    gene_expr = pd.read_csv(
        os.path.join(data_folder, norm + "_filtered_data.csv"), index_col=0
    )
    genes = gene_expr.index.tolist()
    patients_names = gene_expr.columns.to_list()
    gene_expr = gene_expr.to_numpy().astype(float)

    with open(os.path.join(data_folder, "filtered_genesets_genes.json")) as file:
        genesets = json.load(file)
    gs_names = list(genesets.keys())
    pas = pd.read_csv(os.path.join(res_folder, "vae.csv"), index_col=0)
    genesets = dict((k, genesets[k]) for k in list(pas.index))
    gs_names = list(genesets.keys())
    pas_corr = vae.correct_order(
        gene_expr, genesets, genes, pas.to_numpy().astype(float)
    )
    df = pd.DataFrame(data=pas_corr, index=pas.index, columns=patients_names)
    df.to_csv(os.path.join(res_folder, "vae_corr.csv"))
