import numpy as np
from scipy import stats

# from enrichment_auc._matlab_legacy import find_gaussian_mixtures
from enrichment_auc.gmm.gaussian_mixture_hist import gaussian_mixture_hist


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


def find_distribution(scores, gs_name="", sigma_dev=2.5, alpha_limit=0.001, SW=0.25):
    if np.var(scores) == 0:
        print("All scores were of the same value in {}.".format(gs_name))
        return {
            "weights": np.array([]),
            "mu": np.array([]),
            "sigma": np.array([]),
        }
    # find proper Gaussian Mixture model approximating the scores' distribution
    counts = np.ones(scores.shape)
    scores = np.sort(scores)
    if stats.shapiro(scores).pvalue > 0.05:  # check if distribution is normal
        pp, mu, sig = gaussian_mixture_hist(
            scores,
            counts,
            SW=SW,
            n_clusters=1,
        )
    else:  # if not, approximate with Gaussian Mixture model
        pp, mu, sig = gaussian_mixture_hist(
            scores,
            counts,
            KS=10,
            SW=SW,
            n_clusters=None,
        )
    cur_dist = {
        "weights": pp,
        "mu": mu,
        "sigma": sig,
    }
    # merge the special cases
    dist = _merge_gmm(cur_dist, sigma_dev, alpha_limit)
    return dist
