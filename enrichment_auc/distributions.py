import numpy as np
import numpy_indexed as npi
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from enrichment_auc._matlab_legacy import find_gaussian_mixtures


def find_distribution(scores, gs_name=""):
    if np.var(scores) == 0:
        print("All scores were of the same value in {}.".format(gs_name))
        return {
            "weights": np.array([]),
            "mu": np.array([]),
            "sigma": np.array([]),
            "TIC": np.nan,
            "l_lik": np.nan,
        }
    # find proper Gaussian Mixture model approximating the scores' distribution
    counts = np.ones(scores.shape)
    if stats.shapiro(scores).pvalue > 0.05:  # check if distribution is normal
        cur_dist = find_gaussian_mixtures(scores, counts, 1)
    else:  # if not, approximate with Gaussian Mixture model
        min_BIC = np.Inf
        cur_dist = {}
        for components_no in range(1, 11):
            distribution = find_gaussian_mixtures(scores, counts, components_no)
            l_lik = distribution["l_lik"]
            BIC = -2 * l_lik + (3 * components_no - 1) * np.log(scores.size)

            if (
                min_BIC > BIC
                and not np.isnan(BIC)
                and not np.isnan(distribution["TIC"])
            ):
                cur_dist = distribution
                min_BIC = BIC

        if min_BIC == np.inf:
            cur_dist = {
                "weights": np.array([]),
                "mu": np.array([]),
                "sigma": np.array([]),
                "TIC": np.nan,
                "l_lik": np.nan,
            }
    return cur_dist


def find_pdfs(mu, sig, alpha, x_temp, localizer):
    x_temp = x_temp.reshape(
        x_temp.shape[0], -1
    )  # so that scipy can recast it together with distribution parameters
    pdfs = (
        stats.norm.pdf(x_temp, mu, sig) * alpha
    )  # calculate pdfs for all distributions
    pdfs = pdfs.transpose()
    unique_loc, pdfs = npi.group_by(localizer).sum(
        pdfs
    )  # sum them up based on their localizer value
    return pdfs, unique_loc


def find_crossing(pdf1, pdf2, mu1, mu2, x_temp):
    # find crossing between two given pdfs
    idxs = np.argwhere(np.diff(np.sign(pdf1 - pdf2))).flatten()
    if idxs.size == 0:
        return None
    thrs = x_temp[idxs]
    if thrs[-1] > mu1 and thrs[-1] < mu2:
        return thrs[-1]
    if thrs[0] > mu1 and thrs[0] < mu2:
        return thrs[0]
    # do the closest to means' mean
    means = np.mean(np.array([mu1, mu2]))
    if np.abs(thrs[-1] - means) < np.abs(thrs[0] - means):
        return thrs[-1]
    return thrs[0]


def find_dist_crossings(distributions, localizer, unique_loc, pdfs, x_temp, gs_name):
    if (
        unique_loc.size == 1
    ):  # all the distributions are grouped up together, no crossings detected
        print("All distributions are grouped together for {}.".format(gs_name))
        return np.array([])
    thresholds = []
    for i in range(unique_loc.size - 1):
        mu1 = distributions["mu"][np.where(localizer == unique_loc[i])][-1]
        mu2 = distributions["mu"][np.where(localizer == unique_loc[i + 1])][-1]
        thr = find_crossing(pdfs[i, :], pdfs[i + 1, :], mu1, mu2, x_temp)
        if thr is not None:  # and thr != x_temp[-1] or x_temp[0] (approx)
            thresholds.append(thr)
    if len(thresholds) == 0:
        print(
            "{}: No crossings found for {} distributions.".format(
                gs_name, unique_loc.size
            )
        )
    elif len(thresholds) != unique_loc.size - 1:
        print(
            "{}: {} thresholds found for {} distributions.".format(
                gs_name, len(thresholds), unique_loc.size
            )
        )
    thresholds = np.array(thresholds)
    return thresholds


def _group_by_gmm(distributions):
    # check distributions' grouping
    localizer = np.arange(distributions["mu"].size, dtype=np.int8)
    diff = distributions["mu"] + distributions["sigma"]
    diff = np.roll(diff, 1)
    # checking the first component for falling back just in case
    if distributions["mu"].size > 1:
        diff[0] = distributions["mu"][1] - distributions["sigma"][1]
    else:
        diff[0] = distributions["mu"][0]

    diff_merge = np.where(distributions["mu"] < diff)[0]
    weight_merge = np.where(distributions["weights"] < 0.001)[0]
    merge = np.unique(np.concatenate((diff_merge, weight_merge)))
    for i in merge:
        if i == 0:
            localizer[i] = localizer[i + 1]
        else:
            localizer[i] = localizer[i - 1]
    # clean them up to start from 0 and change by 1
    localizer = np.unique(localizer, return_inverse=True)[1]
    return localizer


def _group_by_kmeans(distributions):
    if distributions["mu"].size == 2:
        return np.array([0, 1])
    mu = np.array(distributions["mu"])
    sig = np.array(distributions["sigma"])
    features = np.stack([mu, mu - sig, mu + sig]).T
    # calculate Kmeans on the parameters of the distributions
    best_sil = -1.1
    localizer = np.zeros(mu.size)
    best_centers = np.zeros((1, features.shape[1]))
    for k in range(2, mu.size - 1):
        km = KMeans(k, n_init="auto")
        labels = km.fit_predict(features)
        sil = silhouette_score(features, labels)
        if sil > best_sil:
            best_sil = sil
            best_centers = km.cluster_centers_
            localizer = labels
    # make them go in correct order
    localizer = np.nonzero(localizer[:, None] == best_centers[:, 0].argsort())[1]
    return localizer


def group_distributions(distributions, method="gmm"):
    if distributions["mu"].size <= 1:
        return np.zeros(distributions["mu"].size)
    if method == "gmm":
        return _group_by_gmm(distributions)
    if method == "kmeans":
        return _group_by_kmeans(distributions)
    print("no such method existing.")
    return np.zeros(distributions["mu"].size)


def _remove_redundant_thresholds(thresholds, scores):
    if thresholds.shape[0] == 0:
        return thresholds
    if len(np.where(scores > thresholds[-1])[0]) <= 1:
        thresholds = thresholds[:-1]
        if thresholds.shape[0] == 0:
            return thresholds
    if len(np.where(scores <= thresholds[0])[0]) <= 1:
        thresholds = thresholds[1:]
    return thresholds


def find_grouped_dist_thresholds(distributions, localizer, scores, gs_name):
    # if there is only one distribution/all are grouped together
    if localizer.size <= 1 or np.unique(localizer).size == 1:
        return np.array([])
    # find pdfs
    x_temp = np.linspace(np.min(scores), np.max(scores), 10**6)
    # remove unique_loc possibly - cleaned that up
    pdfs, unique_loc = find_pdfs(
        distributions["mu"],
        distributions["sigma"],
        distributions["weights"],
        x_temp,
        localizer,
    )
    # find the crossings between the distributions
    thresholds = find_dist_crossings(
        distributions, localizer, unique_loc, pdfs, x_temp, gs_name
    )
    thresholds = _remove_redundant_thresholds(thresholds, scores)
    return thresholds


def categorize_by_thresholds(score, thresholds):
    group = np.zeros(score.shape[0])
    for i in range(score.shape[0]):
        group[i] = (thresholds <= score[i]).sum()
    return group
