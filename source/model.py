import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy import stats
import warnings

def _find_the_gmm(data, components=(2,10)):
    n = data.shape[0]
    if components[0] < 1:
        raise ValueError("The minimal components' number is 1")
    if components[1] < components[1]:
        raise ValueError("The minimal components' number cannot be bigger than the maximal one")
    best_model = GaussianMixture(components[0]).fit(data)
    bic = best_model.bic(data)

    for i in range(components[0]+1, components[1]+1):
        model = GaussianMixture(i).fit(data)
        cur_bic = model.bic(data)
        if cur_bic < bic:
            best_model = model            
            if bic - cur_bic > 3:
                bic = cur_bic
                break
            bic = cur_bic
    return best_model

def choose_distribution(data):
    data = data.reshape(-1, 1)
    if stats.shapiro(data).pvalue > 0.05:
        return GaussianMixture(1).fit(data)
    return _find_the_gmm(data, components=(2,10))

def cluster_gmms(model):
    if model.n_components == 1:
        return np.array([0])
    else:
        features = np.append(model.weights_.reshape(-1, 1), 
                             model.means_.reshape(-1, 1), axis=1)
        features = np.append(features,
                             model.covariances_.reshape(-1, 1), axis=1)
        comp_group = KMeans(2).fit_predict(features)
    if comp_group[np.argmax(model.means_)] == 0:
        comp_group = 1 - comp_group
    return comp_group

def get_predictions(model, comp_group, dist):
    if len(comp_group) == 1:
        thr = stats.norm.ppf(0.95, loc=model.means_[0,0], scale=model.covariances_[0, 0])
        binary_labels = [int(score > thr) for score in dist.tolist() ]
        labels = None
    else:         
        labels = model.predict(dist.reshape(-1,1))
        binary_labels = [comp_group[i] for i in labels.tolist()]
        if len(comp_group) == 2:
            labels = None
    return labels, binary_labels