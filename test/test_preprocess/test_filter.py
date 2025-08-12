import numpy as np
import pandas as pd

from pyfuncella.preprocess import filter


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


def test_filter_size():
    # Test pathway size filtering
    genesets = {
        "small": ["gene1", "gene2"],  # size 2
        "medium": ["gene1", "gene2", "gene3", "gene4", "gene5"],  # size 5
        "large": [
            "gene1",
            "gene2",
            "gene3",
            "gene4",
            "gene5",
            "gene6",
            "gene7",
            "gene8",
        ],  # size 8
    }

    # Filter with min_size=3, max_size=6
    filtered = filter.filter_size(genesets, min_size=3, max_size=6)

    # Should only keep "medium" pathway
    assert len(filtered) == 1
    assert "medium" in filtered
    assert "small" not in filtered
    assert "large" not in filtered


def test_filter_coverage():
    # Test pathway coverage filtering
    genesets = {
        "high_coverage": ["gene1", "gene2", "gene3"],  # 3/3 = 100% coverage
        "medium_coverage": ["gene1", "gene2", "missing1"],  # 2/3 = 67% coverage
        "low_coverage": ["missing1", "missing2", "gene1"],  # 1/3 = 33% coverage
    }
    genes = ["gene1", "gene2", "gene3", "gene4"]  # Available genes

    # Filter with min_coverage=0.5 (50%)
    filtered = filter.filter_coverage(genesets, genes, min_coverage=0.5)

    # Should keep pathways with >= 50% coverage
    assert len(filtered) == 2
    assert "high_coverage" in filtered
    assert "medium_coverage" in filtered
    assert "low_coverage" not in filtered


def test_filter_coverage_zero_threshold():
    # Test that zero coverage threshold returns all pathways
    genesets = {
        "pathway1": ["gene1", "missing1"],
        "pathway2": ["missing1", "missing2"],
    }
    genes = ["gene1", "gene2"]

    filtered = filter.filter_coverage(genesets, genes, min_coverage=0.0)

    # Should return all pathways when threshold is 0
    assert len(filtered) == len(genesets)
    assert filtered == genesets


def test_filter_size_edge_cases():
    # Test edge cases for size filtering
    genesets = {
        "exactly_min": ["gene1", "gene2", "gene3"],  # size 3
        "exactly_max": ["gene1", "gene2", "gene3", "gene4", "gene5"],  # size 5
    }

    # Filter with min_size=3, max_size=5 (inclusive)
    filtered = filter.filter_size(genesets, min_size=3, max_size=5)

    # Should keep both pathways (inclusive boundaries)
    assert len(filtered) == 2
    assert "exactly_min" in filtered
    assert "exactly_max" in filtered
