import numpy as np
from sklearn.cluster import KMeans
from gmm._matlab_legacy import find_gaussian_mixtures
from scipy import stats
import numpy_indexed as npi

def find_distribution(scores, gs_name):
    counts = np.ones(scores.shape)
    if stats.shapiro(scores).pvalue > 0.05:
        cur_dist =  find_gaussian_mixtures(scores, counts, 1)
    else:
        min_BIC = np.Inf
        cur_dist = {}
        for components_no in range(1, 11):
            distribution =  find_gaussian_mixtures(scores, counts, components_no)
            l_lik = distribution["l_lik"]
            BIC = -2*l_lik+(3*components_no-1)*np.log(scores.size) 
            
            if min_BIC > BIC and not np.isnan(BIC) and not np.isnan(distribution["TIC"]):
                cur_dist = distribution
                min_BIC = BIC

        if min_BIC == np.inf:
            cur_dist = {"weights": np.array([]), "mu": np.array([]), "sigma": np.array([]), "TIC": np.nan, "l_lik": np.nan}
    return cur_dist


def find_pdfs(mu, sig, alpha, x_temp, localizer):
    x_temp = x_temp.reshape(x_temp.shape[0], -1) # so that scipy can recast it together with distribution parameters
    pdfs = stats.norm.pdf(x_temp, mu, sig)*alpha # calculate pdfs for all distributions
    pdfs = pdfs.transpose()
    unique_loc, pdfs = npi.group_by(localizer).sum(pdfs) # sum them up based on their localizer value
    return pdfs, unique_loc


def find_crossing(pdf1, pdf2, mu1, mu2, x_temp, gs_name):
    idxs=np.argwhere(np.diff(np.sign(pdf1 - pdf2))).flatten()
    if idxs.size == 0:
        # print(gs_name)
        # print("No crossings found")
        return None
    thrs = x_temp[idxs]
    if thrs[-1]>mu1 and thrs[-1]<mu2:
        return thrs[-1]
    if thrs[0]>mu1 and thrs[0]<mu2:
        return thrs[0]
    # do the closest to means' mean
    means = np.mean(np.array([mu1, mu2]))
    if np.abs(thrs[-1]-means) < np.abs(thrs[0]-means):
        return thrs[-1]
    return thrs[0]

def find_thresholds(distributions, scores, gs_name):
    # return thresholds and generated pdfs; x_temp can be fast reconstructed in plotting functions
    x_temp = np.linspace(np.min(scores), np.max(scores), 10**6)
    if distributions["mu"].size == 0:
        return np.array([np.max(scores)]), np.zeros(distributions["mu"].size)
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
            localizer[i] = localizer[i+1]
        else:
            localizer[i] = localizer[i-1]
    
    
    # find pdfs, even if it is a single distribution
    # use means -> (0.5*u1+0.5*u2)/(0.5+0.5)
    pdfs, unique_loc = find_pdfs(distributions["mu"], distributions["sigma"], distributions["weights"], x_temp, localizer)
    if unique_loc.size == 1: # all the distributions are grouped up together, no crossings detected
        return np.array([np.max(scores)]), localizer
    thresholds = np.zeros(unique_loc.size-1)
    thresholds = []
    # finding thresholds
    for i in range(unique_loc.size-1):
        
        mu1 = distributions["mu"][np.where(localizer==unique_loc[i])][-1]
        mu2= distributions["mu"][np.where(localizer==unique_loc[i+1])][-1]

        thr = find_crossing(pdfs[i, :], pdfs[i+1, :], mu1, mu2, x_temp, gs_name)
        if thr is not None:
            thresholds.append(thr)
    if len(thresholds) == 0:
        thresholds.append(np.max(scores))
    thresholds = np.array(thresholds)
    return thresholds, localizer

def take_top_one(thresholds):
    return thresholds[-1] 

def binarize_thresholds(distributions, thresholds): # to be changed
    # k-means them independently of the merges for finding crossings
    if thresholds.shape[0] == 1:
        return thresholds[0]
    mu_ = distributions["mu"].reshape(-1, 1)
    sig_ = distributions["sigma"].reshape(-1, 1)
    features = np.append(mu_, mu_ + sig_, axis=1)
    features = np.append(features, mu_ - sig_, axis=1)
    if np.isnan(np.sum(features)):
        return take_top_one(thresholds)
    comp_group = KMeans(2).fit_predict(features)
    if comp_group[-1] == 1:
        comp_group = 1 - comp_group
    return thresholds[sum(comp_group)-1]

def categorize_by_thresholds(score, thresholds):
    group = np.zeros(score.shape[0])
    for i in range(score.shape[0]):
        group[i] = (thresholds <= score[i]).sum()
    return group