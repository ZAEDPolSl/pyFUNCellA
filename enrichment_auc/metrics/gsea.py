import json
import os
import shutil
import sys
from subprocess import PIPE, Popen

import pandas as pd


def _run_analysis(genesets, data, genes, method):
    os.makedirs("tmp")
    df = pd.DataFrame(data, index=genes)
    df.to_csv("tmp/data.csv")
    with open("tmp/genesets_genes.json", "w") as fp:
        json.dump(genesets, fp)
    rscript = "enrichment_auc/metrics/gsea.R"
    outpath = os.getcwd() + "/"
    cmd = ["Rscript", rscript] + [x for x in [method, outpath]]
    destination = PIPE
    destination = sys.stderr
    sp = Popen(cmd, stdout=PIPE, stderr=destination)
    sp.communicate()
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
