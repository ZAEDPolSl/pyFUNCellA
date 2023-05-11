import numpy as np
import pandas as pd

from enrichment_auc.preprocess import filter


def test_thresholds_on_1():
    d = {"p1": [1.0, 2.01, 5.0, 7.0], "p2": [1.001, 2.0, 6.0, 0.01]}
    cells = pd.DataFrame(data=d)
    filtered = filter.filter(cells, 1.0001)
    assert filtered.equals(cells)


def test_leaves_at_least_one():
    # tests if filtering leaves at least one gene if leave_best is small
    d = {"p1": [1.0, 2.01, 5.0, 0.0], "p2": [4.0, 2.0, 6.0, 0.01]}
    d1 = {"p1": [1.0], "p2": [4.0]}
    cells = pd.DataFrame(data=d)
    expected = pd.DataFrame(data=d1, index=[0])
    filtered = filter.filter(cells, 0.0001)
    assert filtered.equals(expected)


def test_removes_nans():
    d = {"p1": [1.0, 2.0, 3.0], "p2": [4.0, np.nan, 6.0]}
    d1 = {"p1": [1.0, 3.0], "p2": [4.0, 6.0]}
    cells = pd.DataFrame(data=d)
    expected = pd.DataFrame(data=d1, index=[0, 2])
    filtered = filter.filter(cells, 1)
    assert filtered.equals(expected)


def test_removes_zero_var():
    d = {"p1": [1.0, 2.0, 3.0, 0.0], "p2": [4.0, 2.0, 6.0, 0.0]}
    d1 = {"p1": [1.0, 3.0], "p2": [4.0, 6.0]}
    cells = pd.DataFrame(data=d)
    expected = pd.DataFrame(data=d1, index=[0, 2])
    filtered = filter.filter(cells, 1)
    assert filtered.equals(expected)


def test_filters_out():
    d = {"p1": [2.01, 1.0, 5.0, 7.0], "p2": [2.0, 1.001, 6.0, 0.01]}
    d1 = {"p1": [2.01, 5.0, 7.0], "p2": [2.0, 6.0, 0.01]}
    cells = pd.DataFrame(data=d)
    expected = pd.DataFrame(data=d1, index=[0, 2, 3])
    filtered = filter.filter(cells, 0.75)
    assert filtered.equals(expected)
