import json
import os
import shutil
import sys
import subprocess

import pandas as pd


def _run_analysis(genesets, data, genes, method):
    if not os.path.isdir("tmp"):
        os.makedirs("tmp")
    df = pd.DataFrame(data, index=genes)
    df.to_csv("tmp/data.csv")
    del df
    with open("tmp/genesets_genes.json", "w") as fp:
        json.dump(genesets, fp)
    rscript = "enrichment_auc/metrics/gsea.R"
    outpath = os.getcwd() + "/"
    cmd = ["Rscript", rscript] + [x for x in [method, outpath]]
    cmd = " ".join(cmd)
    subprocess.call(cmd, shell=True)
    res = pd.read_csv("tmp/res.csv", index_col=0, header=0)
    shutil.rmtree("tmp")
    return res.to_numpy()


def GSVA(genesets, data, genes):
    method = "gsva"
    gsea = _run_analysis(genesets, data, genes, method)
    return gsea


def SSGSEA(genesets, data, genes):
    method = "ssgsea"
    gsea = _run_analysis(genesets, data, genes, method)
    return gsea
