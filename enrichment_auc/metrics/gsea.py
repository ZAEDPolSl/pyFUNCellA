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
    rscript = os.path.join(os.path.dirname(__file__), "gsea.R")
    outpath = os.getcwd() + "/"
    cmd = ["Rscript", rscript, method, outpath]
    subprocess.call(cmd)
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
