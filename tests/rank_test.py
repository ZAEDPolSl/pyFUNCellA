import pytest
import pandas as pd
import numpy as np
import source.metrics.rank as rank
from statsmodels.stats.multitest import multipletests

def test_rank_genes():
    x = np.array([[0, 2],[3,5]])
    res = rank.rank_genes(x)
    correct = np.array([[2,2],[1,1]])
    assert np.array_equal(correct, res)
