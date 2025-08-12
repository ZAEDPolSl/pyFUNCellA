import numpy as np
from sklearn.cluster import KMeans
from .optimize_clusters import find_optimal_clusters


def last_consecutive_true(row):
    """
    Finds the indices of the last run of consecutive True values in a boolean array.
    Returns None if no True values exist.
    """
    row = np.asarray(row, dtype=bool)
    if not np.any(row):
        return None
    # Find transitions from False to True and True to False
    diff = np.diff(np.concatenate(([0], row.astype(int), [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if starts.size == 0:
        return None
    # Last run is from starts[-1] to ends[-1]-1
    return list(range(starts[-1], ends[-1]))


def km_search(gmm_result):
    """
    Chooses the most appropriate threshold from a GMMdecomp output dict by k-means clustering on the GMM components' parameters.

    Parameters
    ----------
    gmm_result : dict
        Output dictionary from GMMdecomp for a single pathway, must contain keys:
        - 'model': dict with 'mu', 'alpha', 'sigma' (all np.ndarray)
        - 'thresholds': np.ndarray (thresholds between components, len = len(mu) - 1)

    Returns
    -------
    float
        Selected threshold.
    """
    model = gmm_result.get("model", {})
    mu = np.asarray(model.get("mu", []))
    sigma = np.asarray(model.get("sigma", []))
    alpha = np.asarray(model.get("alpha", []))
    thrs = np.asarray(gmm_result.get("thresholds", []))
    if mu.size == 0 or sigma.size == 0 or alpha.size == 0 or thrs.size == 0:
        return float("nan")
    params = np.column_stack([mu, sigma, alpha])
    n_components = params.shape[0]
    if n_components <= 2 or thrs.size == 0:
        return np.nanmax(thrs) if thrs.size > 0 else float("nan")

    # Standardize columns only if needed (vectorized)
    params_std = params.copy()
    col_means = np.mean(params, axis=0)
    col_stds = np.std(params, axis=0) + 1e-10
    for col in range(params.shape[1]):
        if np.unique(params[:, col]).size > 1:
            params_std[:, col] = (params[:, col] - col_means[col]) / col_stds[col]

    # Find optimal clusters using silhouette analysis
    max_k = min(5, n_components)
    optimal_k = find_optimal_clusters(
        params_std, max_k, KMeans(), n_clusters_param="n_clusters"
    )
    kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(params_std)
    # Use mean column to find strongest cluster
    mean_col = params_std[:, 0]
    cluster_means = np.array(
        [mean_col[cluster_labels == k].mean() for k in range(optimal_k)]
    )
    best_cluster = np.argmax(cluster_means)
    ord = cluster_labels == best_cluster

    # Edge case: every component is its own cluster
    if optimal_k == n_components or not np.any(ord):
        return np.nanmax(thrs)

    # If final component is not in target cluster, fallback
    if not ord[-1]:
        return np.nanmax(thrs)

    # Use last_consecutive_true to find last run of TRUEs
    last_run = last_consecutive_true(ord)
    if not last_run or last_run[0] == 0:
        return np.nanmax(thrs)
    idx = last_run[0] - 1
    if idx < 0 or idx >= thrs.size:
        return np.nanmax(thrs)
    return np.nanmin(thrs[: idx + 1])
