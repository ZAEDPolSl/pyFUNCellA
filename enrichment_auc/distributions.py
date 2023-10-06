import numpy as np
from scipy import stats

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
            mu[-1] = (dist["weights"][i] * dist["mu"][i] + alpha[-1] * mu[-1]) / ww_temp
            sigma[-1] = np.sqrt(
                np.sum(pp_est * (mu_est**2 + sigma_est**2)) / ww_temp - mu[-1] ** 2
            )
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


def find_distribution(scores, gs_name="", sigma_dev=2.5, alpha_limit=0.001):
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
