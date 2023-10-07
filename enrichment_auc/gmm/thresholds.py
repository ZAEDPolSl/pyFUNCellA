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


def _find_thr_by_params(distributions, scores, pdfs, sigma_dev=2.5):
    tol = 1e-10
    thresholds = []
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
                found = _find_thr_by_dist(distributions, scores, pdfs)[i]
                thresholds.append(found)
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


def _find_thr_by_dist(distributions, scores, pdfs, sigma_dev=2.5):
    return 0


def find_pdfs(mu, sig, alpha, x_temp):
    x_temp = x_temp.reshape(
        x_temp.shape[0], -1
    )  # so that scipy can recast it together with distribution parameters
    pdfs = (
        stats.norm.pdf(x_temp, mu, sig) * alpha
    )  # calculate pdfs for all distributions
    pdfs = pdfs.transpose()
    return pdfs


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


def find_thresholds(distributions, scores, gs_name, counter):
    # for one distribution only
    if distributions["mu"].size == 0:
        return np.array([])
    thresholds = []
    x_temp = np.linspace(np.min(scores), np.max(scores), 10**6)
    pdfs = find_pdfs(
        distributions["mu"], distributions["sigma"], distributions["weights"], x_temp
    )
    for i in range(distributions["mu"].size - 1):
        thr = find_crossing(
            pdfs[i, :],
            pdfs[i + 1, :],
            distributions["mu"][i],
            distributions["mu"][i + 1],
            x_temp,
        )
        if thr is not None:
            thresholds.append(thr)
    if len(thresholds) != distributions["mu"].size - 1:
        print(
            "{}: {} thresholds found for {} distributions.".format(
                gs_name, len(thresholds), distributions["mu"].size
            )
        )
    thresholds = np.array(thresholds)
    thresholds, counter = _remove_redundant_thresholds(thresholds, scores, counter)
    return thresholds, counter


def new_find_thresholds(distributions, scores, gs_name, counter):
    if distributions["mu"].size == 0:
        return np.array([])
    x_temp = np.linspace(np.min(scores), np.max(scores), 10**6)
    pdfs = find_pdfs(
        distributions["mu"], distributions["sigma"], distributions["weights"], x_temp
    )
    thresholds = _find_thr_by_params(distributions, scores, pdfs)
    if thresholds.size != distributions["mu"].size - 1:
        print(
            "{}: {} thresholds fonud for {} distributions.".format(
                gs_name, thresholds.size, distributions["mu"].size
            )
        )
    thresholds, counter = _remove_redundant_thresholds(thresholds, scores, counter)
    return thresholds
