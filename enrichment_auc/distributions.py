import numpy as np
import numpy_indexed as npi
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from enrichment_auc._matlab_legacy import find_gaussian_mixtures


def _merge_gmm(dist, sigma_dev=1.0, alpha_limit=0.001):
    KS = dist["mu"].size  # components number
    mu = np.array([dist["mu"][0]])
    sigma = np.array([dist["sigma"][0]])
    alpha = np.array([dist["weights"][0]])
    for i in range(1, KS):
        mu_diff = dist["mu"][i] - mu[-1]
        min_displacement = np.min(np.append(sigma, dist["sigma"][i])) * sigma_dev
        if mu_diff < min_displacement or dist["weights"][i] < alpha_limit:
            # merge
            pp_est = np.array([dist["weights"][i], alpha[-1]])
            mu_est = np.array([dist["mu"][i], mu[-1]])
            sigma_est = np.array([dist["sigma"][i], sigma[-1]])
            ww_temp = np.sum(pp_est)
            mu_temp = (
                dist["weights"][i] * dist["mu"][i] + alpha[-1] * mu[-1]
            ) / ww_temp
            sigma_temp = np.sqrt(
                np.sum(pp_est * (mu_est**2 + sigma_est**2)) / ww_temp - mu[-1] ** 2
            )
            mu[-1] = mu_temp
            sigma[-1] = sigma_temp
            alpha[-1] = ww_temp
        else:
            # add new distribution
            mu = np.append(mu, dist["mu"][i])
            sigma = np.append(sigma, dist["sigma"][i])
            alpha = np.append(alpha, dist["weights"][i])
    return {
        "weights": np.array(alpha),
        "mu": np.array(mu),
        "sigma": np.array(sigma),
    }


def find_distribution(scores, gs_name="", sigma_dev=1.0, alpha_limit=0.001):
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
    # merge the special cases
    dist = _merge_gmm(cur_dist, sigma_dev, alpha_limit)
    return dist


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


def _remove_redundant_thresholds(thresholds, scores, counter):
    if thresholds.shape[0] == 0:
        return thresholds
    if len(np.where(scores > thresholds[-1])[0]) <= 1:
        # thresholds = thresholds[:-1]
        # if thresholds.shape[0] == 0:
        #     return thresholds
        counter += 1
    if len(np.where(scores <= thresholds[0])[0]) <= 1:
        # thresholds = thresholds[1:]
        counter += 1
    return thresholds, counter


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
            distributions["sigma"],
            distributions["mu"][i + 1],
            x_temp,
        )
        if thr is not None:
            thresholds.append(thr)
    if len(thresholds) != distributions["mu"].size - 1:
        print(
            "{}: {} thresholds fonud for {} distributions.".format(
                gs_name, len(thresholds), distributions["mu"].size
            )
        )
    thresholds = np.array(thresholds)
    thresholds, counter = _remove_redundant_thresholds(thresholds, scores, counter)
    return thresholds, counter


def correct_via_kmeans(distributions, thresholds):
    if distributions["mu"].size == 2:
        return thresholds
    mu = distributions["mu"]
    sig = distributions["sigma"]
    alpha = distributions["weights"]
    features = np.stack([mu, sig, alpha]).T
    # scale the features
    features = features - features.min(axis=0)[:, None]
    features = features / features.max(axis=0)[:, None]
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


def categorize_by_thresholds(score, thresholds):
    group = np.zeros(score.shape[0])
    for i in range(score.shape[0]):
        group[i] = (thresholds <= score[i]).sum()
    return group
