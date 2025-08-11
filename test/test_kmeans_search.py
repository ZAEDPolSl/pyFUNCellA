import numpy as np
import pytest

from enrichment_auc.utils.kmeans_search import last_consecutive_true, km_search


@pytest.mark.parametrize(
    "arr,expected",
    [
        ([False, True, True, False, True, True, True], [4, 5, 6]),
        ([False, False, False], None),
        ([False, False, True], [2]),
        ([True, True, False, False], [0, 1]),
        ([False, False, True, True], [2, 3]),
        ([True, True, True], [0, 1, 2]),
        ([False, True, False, True, False, True], [5]),
        ([True], [0]),
        ([False], None),
        ([True, False, True, True, False, True, True, True], [5, 6, 7]),
    ],
)
def test_last_consecutive_true_various(arr, expected):
    assert last_consecutive_true(arr) == expected


def test_km_search_edge_cases():
    # 1 component
    gmm_result = {
        "model": {
            "mu": np.array([1.0]),
            "sigma": np.array([0.1]),
            "alpha": np.array([1.0]),
        },
        "thresholds": np.array([]),
    }
    assert np.isnan(km_search(gmm_result))

    # 2 components
    gmm_result = {
        "model": {
            "mu": np.array([1.0, 2.0]),
            "sigma": np.array([0.1, 0.2]),
            "alpha": np.array([0.5, 0.5]),
        },
        "thresholds": np.array([1.5]),
    }
    assert km_search(gmm_result) == 1.5

    # No valid params
    gmm_result = {
        "model": {"mu": np.array([]), "sigma": np.array([]), "alpha": np.array([])},
        "thresholds": np.array([]),
    }
    assert np.isnan(km_search(gmm_result))

    # Final component not in target cluster (simulate by shifting means)
    gmm_result = {
        "model": {
            "mu": np.array([10.0, 20.0, 5.0]),
            "sigma": np.array([0.1, 0.2, 0.3]),
            "alpha": np.array([0.3, 0.3, 0.4]),
        },
        "thresholds": np.array([1.5, 2.5]),
    }
    result = km_search(gmm_result)
    assert result == 2.5

    # All components their own cluster (simulate by making all params unique)
    gmm_result = {
        "model": {
            "mu": np.array([1.0, 2.0, 3.0]),
            "sigma": np.array([0.1, 0.2, 0.3]),
            "alpha": np.array([0.3, 0.2, 0.5]),
        },
        "thresholds": np.array([1.5, 2.5]),
    }
    result = km_search(gmm_result)
    assert result == 2.5

    # Normal case: last run of TRUEs is not at start
    gmm_result = {
        "model": {
            "mu": np.array([1.0, 2.0, 3.0]),
            "sigma": np.array([0.1, 0.2, 0.3]),
            "alpha": np.array([0.3, 0.3, 0.4]),
        },
        "thresholds": np.array([1.5, 2.5]),
    }
    result = km_search(gmm_result)
    assert isinstance(result, float)


def test_km_search_randomized():
    # Randomized test: 5 components, thresholds
    rng = np.random.default_rng(42)
    mu = rng.normal(0, 1, 5)
    sigma = rng.uniform(0.1, 1.0, 5)
    alpha = rng.dirichlet(np.ones(5))
    thrs = np.sort(rng.uniform(-2, 2, 4))
    gmm_result = {
        "model": {"mu": mu, "sigma": sigma, "alpha": alpha},
        "thresholds": thrs,
    }
    result = km_search(gmm_result)
    assert isinstance(result, float)
    assert thrs[0] <= result <= thrs[-1]
