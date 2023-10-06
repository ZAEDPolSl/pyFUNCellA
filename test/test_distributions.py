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


def test_categorize_by_thresholds_no_thr():
    # check if all data is non significant and in one group when no thr is found
    scores = np.random.normal(0, 1, 100)
    labels = np.zeros(100)
    thresholds = np.array([])
    groupings = dist.categorize_by_thresholds(scores, thresholds)
    np.testing.assert_array_equal(groupings, labels)


def test_filter_thresholds_retains_thresholds():
    localizer = np.array([0, 1, 2])
    mu = np.array([0, 1, 2])
    thresholds = np.array([0.5, 1.5])
    thr_found = dist._filter_thresholds(localizer, mu, thresholds)
    np.testing.assert_array_equal(thr_found, thresholds)


def test_filter_thresholds_retains_thresholds_mixed_up():
    localizer = np.array([0, 1, 0])
    mu = np.array([0, 1, 2])
    thresholds = np.array([0.5, 1.5])
    thr_found = dist._filter_thresholds(localizer, mu, thresholds)
    np.testing.assert_array_equal(thr_found, thresholds)


def test_filter_thresholds_removes_single_thresholds():
    localizer = np.array([1, 1, 0])
    mu = np.array([0, 1, 2])
    thresholds = np.array([0.5, 1.5])
    thr_found = dist._filter_thresholds(localizer, mu, thresholds)
    np.testing.assert_array_equal(thr_found, np.array([1.5]))


def test_filter_thresholds_removes_multiple_thresholds_one_label():
    localizer = np.array([0, 1, 1, 1])
    mu = np.array([0, 1, 2, 3])
    thresholds = np.array([0.5, 1.5, 2.5])
    thr_found = dist._filter_thresholds(localizer, mu, thresholds)
    np.testing.assert_array_equal(thr_found, np.array([0.5]))


def test_filter_thresholds_removes_multiple_thresholds_diff_label():
    localizer = np.array([0, 0, 1, 1])
    mu = np.array([0, 1, 2, 3])
    thresholds = np.array([0.5, 1.5, 2.5])
    thr_found = dist._filter_thresholds(localizer, mu, thresholds)
    np.testing.assert_array_equal(thr_found, np.array([1.5]))


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


# def remove_redundant_thresholds_start():
#     scores = np.linspace(0, 1, 100)
#     thresholds = np.array([0])
#     thr_found = dist._remove_redundant_thresholds(thresholds, scores)
#     assert thr_found.shape[0] == 0


# def remove_redundant_thresholds_before():
#     scores = np.linspace(0, 1, 100)
#     thresholds = np.array([-0.01])
#     thr_found = dist._remove_redundant_thresholds(thresholds, scores)
#     assert thr_found.shape[0] == 0


# def remove_redundant_thresholds_end():
#     scores = np.linspace(0, 1, 100)
#     thresholds = np.array([1])
#     thr_found = dist._remove_redundant_thresholds(thresholds, scores)
#     assert thr_found.shape[0] == 0


# def remove_redundant_thresholds_after():
#     scores = np.linspace(0, 1, 100)
#     thresholds = np.array([1.01])
#     thr_found = dist._remove_redundant_thresholds(thresholds, scores)
#     assert thr_found.shape[0] == 0


# def remove_redundant_thresholds_mixed_ends():
#     scores = np.linspace(0, 1, 100)
#     thresholds = np.array([0, 1.0])
#     thr_found = dist._remove_redundant_thresholds(thresholds, scores)
#     assert thr_found.shape[0] == 0


# def remove_redundant_thresholds_mix():
#     scores = np.linspace(0, 1, 100)
#     thresholds = np.array([0, 0.5, 0.75, 1.0])
#     thr_found = dist._remove_redundant_thresholds(thresholds, scores)
#     thr_expected = np.array([0.5, 0.75])
#     np.testing.assert_array_equal(thr_found, thr_expected)


def test_correct_via_kmeans_skips_smaller_than_2():
    distributions_0 = {
        "mu": np.array([]),
        "sigma": np.array([]),
        "weights": np.array([]),
    }
    distributions_1 = {
        "mu": np.array([0]),
        "sigma": np.array([1.0]),
        "weights": np.array([1.0]),
    }
    distributions_2 = {
        "mu": np.array([0, 1]),
        "sigma": np.array([1, 1]),
        "weights": np.array([0.5, 0.5]),
    }
    thresholds_0 = np.array([])
    thresholds_1 = np.array([])
    thresholds_2 = np.array([0.5])
    thr_found_0 = dist.correct_via_kmeans(distributions_0, thresholds_0)
    thr_found_1 = dist.correct_via_kmeans(distributions_1, thresholds_1)
    thr_found_2 = dist.correct_via_kmeans(distributions_2, thresholds_2)
    np.testing.assert_array_equal(thr_found_0, thresholds_0)
    np.testing.assert_array_equal(thr_found_1, thresholds_1)
    np.testing.assert_array_equal(thr_found_2, thresholds_2)


def test_correct_via_kmeans_skips_missing_thrs():
    distributions = {
        "mu": np.array([0, 1, 2]),
        "sigma": np.array([1, 1, 1]),
        "weights": np.array([0.25, 0.25, 0.5]),
    }
    thresholds = np.array([0.5])
    thr_found = dist.correct_via_kmeans(distributions, thresholds)
    np.testing.assert_array_equal(thr_found, thresholds)
