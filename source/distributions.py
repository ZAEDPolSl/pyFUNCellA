import numpy as np
from sklearn.cluster import KMeans
from divik._matlab_legacy import find_thresholds, find_gaussian_mixtures
from scipy import stats

def find_distribution(scores):
    counts = np.ones(scores.shape)
    if stats.shapiro(scores).pvalue > 0.05:
        distribution =  find_gaussian_mixtures(scores, counts, 1)
        loc = distribution["mu"][0]
        scale = distribution["sigma"][0]
        thresholds = np.array([stats.norm.ppf(.95, loc=loc, scale=scale)])
    else:
        thresholds = find_thresholds(scores)
        distribution = find_gaussian_mixtures(scores, counts, thresholds.shape[0]+1)
        if (thresholds.shape[0]==0): # should not happen
            loc = distribution["mu"][0]
            scale = distribution["sigma"][0]
            thresholds = np.array([stats.norm.ppf(.95, loc=loc, scale=scale)])
    return distribution, thresholds

def take_top_one(thresholds):
    return thresholds[-1] 

def binarize_thresholds(distributions, thresholds):
    if thresholds.shape[0] == 1:
        return thresholds[0]
    mu_ = distributions["mu"].reshape(-1, 1)
    sig_ = distributions["sigma"].reshape(-1, 1)
    features = np.append(mu_, mu_ + sig_, axis=1)
    features = np.append(features, mu_ - sig_, axis=1)
    comp_group = KMeans(2).fit_predict(features)
    if comp_group[-1] == 1:
        comp_group = 1 - comp_group
    return thresholds[sum(comp_group)-1]

def categorize_by_thresholds(score, thresholds):
    group = np.zeros(score.shape[0])
    for i in range(score.shape[0]):
        group[i] = (thresholds <= score[i]).sum()
    return group