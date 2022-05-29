import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

def rank_genes(data, descending=True):
    # for each patient return the gene ranking
    if descending:
        return (-data).argsort(axis=0).argsort(axis=0) + 1
    return np.argsort(data, axis=0) + 1
