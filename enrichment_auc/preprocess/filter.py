import sys

import numpy as np
import pandas as pd


def filter(cells, leave_best=0.25):
    # input: cells - pd df of gene expressions
    # leave_best - the percentage of genes to leave after filtering
    # output: cells - pd df of filtered gene expressions
    leave_best = max(0, min(leave_best, 1))
    cells.dropna(inplace=True)
    vars = np.var(cells, axis=1)
    vars = vars[vars != 0]
    vars = vars.sort_values()
    cells = cells.loc[vars[int((1 - leave_best) * vars.shape[0]) :].index]
    return cells


if __name__ == "__main__":
    folder = sys.argv[1]
    datatype = sys.argv[2]
    path = sys.argv[3]
    infile = path + folder + "/" + datatype + "_data.csv"
    outfile = path + folder + "/" + datatype + "_filtered_data.csv"
    cells = pd.read_csv(infile, index_col=0)
    cells = filter(cells)
    cells.to_csv(outfile)
