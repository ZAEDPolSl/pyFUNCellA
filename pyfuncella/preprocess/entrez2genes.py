import sys
import warnings

import numpy as np
import pandas as pd


def remove_duplicates(genes, cells, gene_name):
    duplicates = pd.unique(genes[genes.duplicated(gene_name, keep=False)][gene_name])
    duplicated_rows = genes[genes.duplicated(gene_name, keep=False)]
    for name in duplicates:
        id = genes[genes[gene_name] == name].index
        to_stay = cells.loc[id].var(axis=1).idxmax()
        duplicated_rows = duplicated_rows.drop(index=to_stay)
    genes = genes.drop(index=duplicated_rows.index)
    return genes


def entrez2genes(genes, cells, gene_name="external_gene_name"):
    if gene_name == "" or gene_name is None:
        warnings.warn(
            "No column name for genes given. Using the first one by default",
            UserWarning,
        )
        gene_name = genes.columns[0]
    genes.replace("", np.nan, inplace=True)
    genes.dropna(inplace=True)
    genes = remove_duplicates(genes, cells, gene_name)
    new_cells = pd.concat([cells, genes[gene_name]], join="inner", axis=1)
    new_cells = new_cells.set_index(gene_name)
    new_cells.index.name = None
    return new_cells


if __name__ == "__main__":
    folder = sys.argv[1]
    datatype = sys.argv[2]
    path = sys.argv[3]
    gene_name = "external_gene_name"
    infile = path + folder + "/" + datatype + "_data.csv"
    outfile = path + folder + "/" + datatype + "_filtered_data.csv"
    genes = pd.read_csv(
        path + folder + "/genes.csv",
        usecols=["genes", gene_name],
        index_col="genes",
    )
    cells = pd.read_csv(infile, index_col=0)
    cells = entrez2genes(genes, cells, gene_name)
    cells.to_csv(outfile)
