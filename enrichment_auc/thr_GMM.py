import numpy as np
from enrichment_auc.utils.kmeans_search import km_search

def thr_GMM(gmms):
    """
    Calculate thresholds for pathway activity based on GMMs.

    Parameters
    ----------
    gmms : dict
        Dictionary of GMM results, one per pathway. Each value must have:
        - 'thresholds': numeric vector of GMM thresholds
        - 'model': dict with 'mu', 'sigma', 'alpha' (component params)

    Returns
    -------
    dict
        Dictionary with keys as pathway names, values as dicts:
        - 'Kmeans_thr': threshold chosen via k-means (or fallback)
        - 'Top1_thr': maximum original threshold
        - 'All_thr': all thresholds (after NA removal)
    """
    results = {}
    for name, gmm in gmms.items():
        thrs = np.asarray(gmm.get('thresholds', []), dtype=float)
        model = gmm.get('model', {})
        # Remove NaNs from thresholds and corresponding params
        valid_mask = ~np.isnan(thrs)
        thrs_clean = thrs[valid_mask]
        # Top1 threshold: max of cleaned thresholds, fallback to nan if empty
        Top1_thr = np.nanmax(thrs_clean) if thrs_clean.size > 0 else float('nan')
        All_thr = thrs_clean.tolist()

        # Prepare params for k-means threshold
        mu = np.asarray(model.get('mu', []))
        sigma = np.asarray(model.get('sigma', []))
        alpha = np.asarray(model.get('alpha', []))
        # Drop params corresponding to NaN thresholds (thresholds are between components)
        # For n components, there are n-1 thresholds
        n_components = mu.size
        # Kmeans threshold logic
        if thrs_clean.size > 0:
            if thrs_clean.size == 1 and n_components == 2:
                Kmeans_thr = thrs_clean[0]
            elif thrs_clean.size != n_components - 1:
                Kmeans_thr = Top1_thr
            else:
                # Build a GMM result dict for km_search
                gmm_for_kmeans = {
                    'model': {'mu': mu, 'sigma': sigma, 'alpha': alpha},
                    'thresholds': thrs_clean
                }
                Kmeans_thr = km_search(gmm_for_kmeans)
        else:
            Kmeans_thr = float('-inf')

        results[name] = {
            'Kmeans_thr': Kmeans_thr,
            'Top1_thr': Top1_thr,
            'All_thr': All_thr
        }
    return results
