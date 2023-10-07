import numpy as np

import enrichment_auc.gmm.thresholds as thr_tools


def test_categorize_by_thresholds_no_thr():
    # check if all data is non significant and in one group when no thr is found
    scores = np.random.normal(0, 1, 100)
    labels = np.zeros(100)
    thresholds = np.array([])
    groupings = thr_tools.categorize_by_thresholds(scores, thresholds)
    np.testing.assert_array_equal(groupings, labels)


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
    thr_found_0 = thr_tools.correct_via_kmeans(distributions_0, thresholds_0)
    thr_found_1 = thr_tools.correct_via_kmeans(distributions_1, thresholds_1)
    thr_found_2 = thr_tools.correct_via_kmeans(distributions_2, thresholds_2)
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
    thr_found = thr_tools.correct_via_kmeans(distributions, thresholds)
    np.testing.assert_array_equal(thr_found, thresholds)


# def remove_redundant_thresholds_start():
#     scores = np.linspace(0, 1, 100)
#     thresholds = np.array([0])
#     thr_found = thr_tools._remove_redundant_thresholds(thresholds, scores)
#     assert thr_found.shape[0] == 0


# def remove_redundant_thresholds_before():
#     scores = np.linspace(0, 1, 100)
#     thresholds = np.array([-0.01])
#     thr_found = thr_tools._remove_redundant_thresholds(thresholds, scores)
#     assert thr_found.shape[0] == 0


# def remove_redundant_thresholds_end():
#     scores = np.linspace(0, 1, 100)
#     thresholds = np.array([1])
#     thr_found = thr_tools._remove_redundant_thresholds(thresholds, scores)
#     assert thr_found.shape[0] == 0


# def remove_redundant_thresholds_after():
#     scores = np.linspace(0, 1, 100)
#     thresholds = np.array([1.01])
#     thr_found = thr_tools._remove_redundant_thresholds(thresholds, scores)
#     assert thr_found.shape[0] == 0


# def remove_redundant_thresholds_mixed_ends():
#     scores = np.linspace(0, 1, 100)
#     thresholds = np.array([0, 1.0])
#     thr_found = thr_tools._remove_redundant_thresholds(thresholds, scores)
#     assert thr_found.shape[0] == 0


# def remove_redundant_thresholds_mix():
#     scores = np.linspace(0, 1, 100)
#     thresholds = np.array([0, 0.5, 0.75, 1.0])
#     thr_found = thr_tools._remove_redundant_thresholds(thresholds, scores)
#     thr_expected = np.array([0.5, 0.75])
#     np.testing.assert_array_equal(thr_found, thr_expected)


def test_filter_thresholds_retains_thresholds():
    localizer = np.array([0, 1, 2])
    mu = np.array([0, 1, 2])
    thresholds = np.array([0.5, 1.5])
    thr_found = thr_tools._filter_thresholds(localizer, mu, thresholds)
    np.testing.assert_array_equal(thr_found, thresholds)


def test_filter_thresholds_retains_thresholds_mixed_up():
    localizer = np.array([0, 1, 0])
    mu = np.array([0, 1, 2])
    thresholds = np.array([0.5, 1.5])
    thr_found = thr_tools._filter_thresholds(localizer, mu, thresholds)
    np.testing.assert_array_equal(thr_found, thresholds)


def test_filter_thresholds_removes_single_thresholds():
    localizer = np.array([1, 1, 0])
    mu = np.array([0, 1, 2])
    thresholds = np.array([0.5, 1.5])
    thr_found = thr_tools._filter_thresholds(localizer, mu, thresholds)
    np.testing.assert_array_equal(thr_found, np.array([1.5]))


def test_filter_thresholds_removes_multiple_thresholds_one_label():
    localizer = np.array([0, 1, 1, 1])
    mu = np.array([0, 1, 2, 3])
    thresholds = np.array([0.5, 1.5, 2.5])
    thr_found = thr_tools._filter_thresholds(localizer, mu, thresholds)
    np.testing.assert_array_equal(thr_found, np.array([0.5]))


def test_filter_thresholds_removes_multiple_thresholds_diff_label():
    localizer = np.array([0, 0, 1, 1])
    mu = np.array([0, 1, 2, 3])
    thresholds = np.array([0.5, 1.5, 2.5])
    thr_found = thr_tools._filter_thresholds(localizer, mu, thresholds)
    np.testing.assert_array_equal(thr_found, np.array([1.5]))
