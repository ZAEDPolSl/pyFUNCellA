import numpy as np
import pytest

from pyfuncella.metrics import jasmine


def test_ranks():
    data = np.array(
        [[0.1, 0.7, 0.9], [0.8, 0.6, 0.2], [0.0, 0.3, 0.5], [0.7, 0.0, 0.1]]
    )
    data = np.ma.masked_equal(data, 0)
    ranks = np.array([[1, 3, 4], [3, 2, 2], [0, 1, 3], [2, 0, 1]])
    ranks = np.ma.masked_equal(ranks, 0)
    ranks_jasmine = jasmine.rank_genes(data)
    assert np.ma.allequal(ranks_jasmine, ranks, fill_value=True)


def test_ranks_ties():
    data = np.array(
        [[0.1, 0.7, 0.9], [0.8, 0.6, 0.2], [0.0, 0.3, 0.5], [0.1, 0.0, 0.1]]
    )
    data = np.ma.masked_equal(data, 0)
    ranks = np.array([[1.5, 3, 4], [3, 2, 2], [0, 1, 3], [1.5, 0, 1]])
    ranks = np.ma.masked_equal(ranks, 0)
    ranks_jasmine = jasmine.rank_genes(data)
    assert np.ma.allequal(ranks_jasmine, ranks, fill_value=True)


def test_jasmine_is_array():
    genes = ["a", "b", "c", "d"]
    genesets = {"geneset": ["a", "b", "d"]}
    ranks = np.array([[3, 1, 1], [1, 2, 3], [4, 3, 2], [2, 4, 4]])
    ranks = np.ma.masked_equal(ranks, 0)
    jasmine_ = jasmine._jasmine(genesets["geneset"], ranks, genes)
    assert isinstance(jasmine_, np.ndarray)


def test_jasmine_on_ranks():
    genes = ["a", "b", "c", "d"]
    genesets = {"geneset": ["a", "b", "d"]}
    ranks = np.array([[3, 1, 1], [1, 2, 3], [0, 3, 2], [2, 0, 4]])
    ranks = np.ma.masked_equal(ranks, 0)
    res_expected = np.array([2.0 / 3, 0.5, 2.0 / 3])
    jasmine_ = jasmine._jasmine(genesets["geneset"], ranks, genes)
    assert np.array_equal(res_expected, jasmine_)


def test_full_jasmine():
    genes = ["a", "b", "c", "d"]
    genesets = {"geneset": ["a", "b", "d"], "geneset1": ["d"]}
    data = np.array(
        [[0.1, 0.7, 0.9], [0.8, 0.6, 0.2], [0.0, 0.3, 0.5], [0.7, 0.0, 0.1]]
    )
    res_expected = np.array([[1.0 / 3, 1, 0.0], [1, 0, 3.0 / 8]])
    jasmine_ = jasmine.JASMINE(genesets, data, genes, use_effect_size=False)
    assert np.isclose(res_expected, jasmine_, atol=10e-4).all()


def test_full_jasmine_ties():
    genes = ["a", "b", "c", "d"]
    genesets = {"geneset": ["a", "b", "d"], "geneset1": ["d"]}
    data = np.array(
        [[0.1, 0.7, 0.9], [0.8, 0.6, 0.2], [0.0, 0.3, 0.5], [0.1, 0.0, 0.1]]
    )
    res_expected = np.array([[1.0 / 3, 1, 0.0], [1, 0, 1.0 / 2]])
    jasmine_ = jasmine.JASMINE(genesets, data, genes, use_effect_size=False)
    assert np.isclose(res_expected, jasmine_, atol=10e-4).all()


def test_jasmine_is_0_robust():
    genes = ["a", "b", "c", "d"]
    genesets = {"geneset": ["a", "b", "d"], "geneset1": ["d"]}
    data = np.array(
        [[0.1, 0.7, 0.0], [0.8, 0.6, 0.0], [0.0, 0.3, 0.0], [0.7, 0.0, 0.0]]
    )
    res_expected = np.array([[0.8, 1.0, 0.0], [1.0, 0, 0.0]])
    jasmine_ = jasmine.JASMINE(genesets, data, genes, use_effect_size=False)
    assert np.isclose(res_expected, jasmine_, atol=10e-4).all()


def test_scale_minmax():
    """Test min-max normalization function"""
    x = np.array([1, 2, 3, 4, 5])
    expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    result = jasmine.scale_minmax(x)
    assert np.allclose(result, expected)

    # Test with constant values
    x_constant = np.array([2, 2, 2, 2])
    result_constant = jasmine.scale_minmax(x_constant)
    assert np.allclose(result_constant, np.zeros(4))


def test_calc_odds_ratio():
    """Test odds ratio calculation"""
    genes = ["a", "b", "c", "d"]
    geneset_genes = ["a", "b"]
    data = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0]])

    result = jasmine.calc_odds_ratio(data, geneset_genes, genes)
    assert len(result) == 3  # Should have one value per sample
    assert all(result >= 0)  # Odds ratios should be non-negative


def test_calc_likelihood():
    """Test likelihood ratio calculation"""
    genes = ["a", "b", "c", "d"]
    geneset_genes = ["a", "b"]
    data = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0]])

    result = jasmine.calc_likelihood(data, geneset_genes, genes)
    assert len(result) == 3  # Should have one value per sample
    assert all(result >= 0)  # Likelihood ratios should be non-negative


def test_jasmine_with_effect_size():
    """Test JASMINE method with effect size"""
    genes = ["a", "b", "c", "d"]
    genesets = {"geneset": ["a", "b", "d"], "geneset1": ["d"]}
    data = np.array(
        [[0.1, 0.7, 0.9], [0.8, 0.6, 0.2], [0.0, 0.3, 0.5], [0.7, 0.0, 0.1]]
    )

    # Test with odds ratio
    jasmine_odds = jasmine.JASMINE(
        genesets, data, genes, use_effect_size=True, effect_size="oddsratio"
    )
    assert jasmine_odds.shape == (2, 3)
    assert np.all(jasmine_odds >= 0)

    # Test with likelihood
    jasmine_likelihood = jasmine.JASMINE(
        genesets, data, genes, use_effect_size=True, effect_size="likelihood"
    )
    assert jasmine_likelihood.shape == (2, 3)
    assert np.all(jasmine_likelihood >= 0)


def test_jasmine_backward_compatibility():
    """Test that original method still works as before"""
    genes = ["a", "b", "c", "d"]
    genesets = {"geneset": ["a", "b", "d"], "geneset1": ["d"]}
    data = np.array(
        [[0.1, 0.7, 0.9], [0.8, 0.6, 0.2], [0.0, 0.3, 0.5], [0.7, 0.0, 0.1]]
    )

    # Test original method (explicitly set to False)
    jasmine_original = jasmine.JASMINE(genesets, data, genes, use_effect_size=False)

    # Test with effect size (default behavior)
    jasmine_enhanced = jasmine.JASMINE(genesets, data, genes)

    # They should have same shape but potentially different values
    assert jasmine_original.shape == jasmine_enhanced.shape


def test_jasmine_edge_cases():
    """Test edge cases for JASMINE function"""
    genes = ["a", "b", "c", "d"]
    genesets = {"empty_geneset": []}
    data = np.array(
        [[0.1, 0.7, 0.9], [0.8, 0.6, 0.2], [0.0, 0.3, 0.5], [0.7, 0.0, 0.1]]
    )

    # Test with empty geneset
    jasmine_empty = jasmine.JASMINE(genesets, data, genes, use_effect_size=True)
    assert jasmine_empty.shape == (1, 3)

    # Test with non-existent genes in geneset
    genesets_missing = {"geneset": ["x", "y", "z"]}
    jasmine_missing = jasmine.JASMINE(
        genesets_missing, data, genes, use_effect_size=True
    )
    assert jasmine_missing.shape == (1, 3)


def test_jasmine_parameter_validation():
    """Test that invalid parameters work as expected"""
    genes = ["a", "b", "c", "d"]
    genesets = {"geneset": ["a", "b", "d"]}
    data = np.array(
        [[0.1, 0.7, 0.9], [0.8, 0.6, 0.2], [0.0, 0.3, 0.5], [0.7, 0.0, 0.1]]
    )

    # Test invalid effect_size (should default to zeros)
    jasmine_invalid = jasmine.JASMINE(
        genesets, data, genes, use_effect_size=True, effect_size="invalid_effect"
    )
    assert jasmine_invalid.shape == (1, 3)
    # Should still work, just use zeros for effect size
