import numpy as np
import numpy_indexed as npi
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def categorize_by_thresholds(score, thresholds):
    group = np.zeros(score.shape[0])
    for i in range(score.shape[0]):
        group[i] = (thresholds <= score[i]).sum()
    return group


def correct_via_kmeans(distributions, thresholds):
    if distributions["mu"].size <= 2 or thresholds.size != distributions["mu"].size - 1:
        return thresholds
    mu = distributions["mu"]
    sig = distributions["sigma"]
    alpha = distributions["weights"]
    features = np.stack([mu, sig, alpha]).T
    # scale the features
    features = features - features.min(axis=0)
    features = features / features.max(axis=0)
    best_sil = -1.1
    localizer = np.zeros(mu.size)
    for k in range(2, mu.size - 1):
        km = KMeans(k, n_init="auto")
        labels = km.fit_predict(features)
        sil = silhouette_score(features, labels)
        if sil > best_sil:
            best_sil = sil
            localizer = labels
    thresholds = _filter_thresholds(localizer, mu, thresholds)
    return thresholds


def _filter_thresholds(localizer, mu, thresholds):
    # sort the clusters to go in the correct order
    _, group_min = npi.group_by(localizer).min(mu)
    localizer = np.nonzero(localizer[:, None] == group_min.argsort())[1]
    # check if neighbouring distributions are together
    differences = np.diff(localizer)
    if np.all(differences >= 0):
        # perform the correction
        thresholds = thresholds[np.where(differences)]
    return thresholds


def _remove_redundant_thresholds(thresholds, scores, counter):
    if thresholds.shape[0] == 0:
        return thresholds, counter
    if len(np.where(scores > thresholds[-1])[0]) <= 1:
        # thresholds = thresholds[:-1]
        # if thresholds.shape[0] == 0:
        #     return thresholds
        counter += 1
    if len(np.where(scores <= thresholds[0])[0]) <= 1:
        # thresholds = thresholds[1:]
        counter += 1
    return thresholds, counter


def _find_thr_by_params(distributions, x_temp, pdfs, sigma_dev=2.5):
    tol = 1e-10
    thresholds = []
    thrs_from_dists = np.array([])
    for i in range(distributions["mu"].size - 1):
        A = 1 / (2 * distributions["sigma"][i] ** 2) - 1 / (
            2 * distributions["sigma"][i + 1] ** 2
        )
        B = distributions["mu"][i + 1] / (
            distributions["sigma"][i + 1] ** 2
        ) - distributions["mu"][i] / (distributions["sigma"][i] ** 2)
        C = (
            distributions["mu"][i] ** 2 / (2 * distributions["sigma"][i] ** 2)
            - distributions["mu"][i + 1] ** 2 / (2 * distributions["sigma"][i + 1] ** 2)
            - np.log(
                (distributions["weights"][i] * distributions["sigma"][i + 1])
                / (distributions["weights"][i + 1] * distributions["sigma"][i])
            )
        )
        if np.abs(A) < tol:
            if np.abs(B) < tol:
                print("the Gaussians are the same.")
            else:
                x1 = -C / B
                thresholds.append(x1)
        else:
            delta = B**2 - 4 * A * C
            if delta < 0:
                if thrs_from_dists.size == 0:
                    thrs_from_dists = _find_thr_by_dist(
                        distributions, x_temp, pdfs, sigma_dev
                    )
                if np.isfinite(thrs_from_dists[i]):
                    thresholds.append(thrs_from_dists[i])
                pass
            else:
                x1 = (-B - np.sqrt(delta)) / (2 * A)
                x2 = (-B + np.sqrt(delta)) / (2 * A)
                if x1 > distributions["mu"][i] and x1 < distributions["mu"][i + 1]:
                    thresholds.append(x1)
                elif x2 > distributions["mu"][i] and x2 < distributions["mu"][i + 1]:
                    thresholds.append(x2)
                else:
                    d1 = np.min(
                        np.array(
                            [
                                np.abs(x1 - distributions["mu"][i]),
                                np.abs(x1 - distributions["mu"][i + 1]),
                            ]
                        )
                    )
                    d2 = np.min(
                        np.array(
                            [
                                np.abs(x2 - distributions["mu"][i]),
                                np.abs(x2 - distributions["mu"][i + 1]),
                            ]
                        )
                    )
                    if d1 < d2:
                        thresholds.append(x1)
                    else:
                        thresholds.append(x2)
    return np.array(thresholds)


def _find_thr_by_dist(distributions, x_temp, pdfs, sigma_dev=2.5):
    ranges = np.array(
        [
            distributions["mu"][0] - sigma_dev * distributions["sigma"][0],
            distributions["mu"][-1] + sigma_dev * distributions["sigma"][-1],
        ]
    )
    x_temp, pdfs = _restrict_thr_ranges(ranges, x_temp, pdfs)
    # sum of all consecutive pdfs except for last one, summing from first
    f1 = np.cumsum(pdfs, axis=0)[:-1, :]
    # sum of all consecutive pdfs except for first one, but summing from last
    f2 = np.cumsum(pdfs[::-1, :], axis=0)[::-1, :][1:, :]
    to_discard = _detect_noncrossing(f1, f2)
    thrs = _find_closest_location(f1, f2, x_temp)
    thrs[to_discard] = np.nan
    return thrs


def _find_closest_location(f1, f2, x_temp):
    f_diff = np.abs(f1 - f2)
    idx = np.argmin(f_diff, axis=1)
    # x_temp from argmin
    thrs = x_temp[idx]
    return thrs


def _detect_noncrossing(f1, f2):
    checks = np.sum(f2 > f1, axis=1)
    idxs = np.where((checks == 0) | (checks == f1.shape[1]))
    return idxs


def _restrict_thr_ranges(ranges, x_temp, pdfs):
    to_keep = np.where((x_temp > ranges[0]) & (x_temp < ranges[1]))
    x_temp = x_temp[to_keep]
    x_temp = x_temp[1:-1]
    pdfs = pdfs[:, to_keep].squeeze()
    pdfs = pdfs[:, 1:-1]
    return x_temp, pdfs


def find_pdfs(mu, sig, alpha, x_temp):
    x_temp = x_temp.reshape(
        x_temp.shape[0], -1
    )  # so that scipy can recast it together with distribution parameters
    pdfs = (
        stats.norm.pdf(x_temp, mu, sig) * alpha
    )  # calculate pdfs for all distributions
    pdfs = pdfs.transpose()
    return pdfs


def find_thresholds(distributions, scores, gs_name, counter):
    if distributions["mu"].size == 0:
        return np.array([])
    x_temp = np.linspace(np.min(scores), np.max(scores), 10**6)
    pdfs = find_pdfs(
        distributions["mu"], distributions["sigma"], distributions["weights"], x_temp
    )
    thresholds = _find_thr_by_params(distributions, x_temp, pdfs)
    if thresholds.size != distributions["mu"].size - 1:
        print(
            "{}: {} thresholds fonud for {} distributions.".format(
                gs_name, thresholds.size, distributions["mu"].size
            )
        )
    thresholds, counter = _remove_redundant_thresholds(thresholds, scores, counter)
    return thresholds
