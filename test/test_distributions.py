import numpy as np

import enrichment_auc.distributions as dist


def test_find_dist_zero_var():
    scores = np.ones(100)
    dist_found = dist.find_distribution(scores, "")
    assert dist_found["weights"].shape[0] == 0
    assert dist_found["mu"].shape[0] == 0
    assert dist_found["sigma"].shape[0] == 0
    assert np.isnan(dist_found["TIC"])
    assert np.isnan(dist_found["l_lik"])


def test_merge_gmm_retains_dists():
    mu = np.array([0, 5, 10, 15])
    sigma = np.array([1, 2, 3, 4])
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    dists = {
        "sigma": sigma,
        "mu": mu,
        "weights": weights,
    }
    pred_dist = dist._merge_gmm(dists)
    np.testing.assert_array_equal(mu, pred_dist["mu"])
    np.testing.assert_array_equal(weights, pred_dist["weights"])
    np.testing.assert_array_equal(sigma, pred_dist["sigma"])


def test_merge_gmm_merges_dists_sigma():
    mu = np.array([0.0, 9.5, 10.0, 15])
    sigma = np.array([1.0, 2.0, 3, 4])
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    dists = {
        "sigma": sigma,
        "mu": mu,
        "weights": weights,
    }
    pred_dist = dist._merge_gmm(dists)
    np.testing.assert_array_almost_equal(np.array([0, 9.75, 15]), pred_dist["mu"])
    np.testing.assert_array_equal(np.array([0.25, 0.5, 0.25]), pred_dist["weights"])
    np.testing.assert_array_almost_equal(
        np.array([1.0, 2.56173769, 4]), pred_dist["sigma"]
    )


def test_merge_gmm_merges_dists_alpha():
    mu = np.array([-5.0, 0.0, 10.0, 15])
    sigma = np.array([2.0, 1.0, 3, 4])
    weights = np.array([0.4999, 0.0001, 0.25, 0.25])
    dists = {
        "sigma": sigma,
        "mu": mu,
        "weights": weights,
    }
    pred_dist = dist._merge_gmm(dists)
    np.testing.assert_array_almost_equal(np.array([-4.999, 10, 15]), pred_dist["mu"])
    np.testing.assert_array_equal(np.array([0.5, 0.25, 0.25]), pred_dist["weights"])
    np.testing.assert_array_almost_equal(
        np.array([2.0010994478036324, 3, 4]), pred_dist["sigma"]
    )
