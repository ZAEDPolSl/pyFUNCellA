from typing import Any

import numpy as np
from sklearn.metrics import silhouette_score


def find_optimal_clusters(
    values: np.ndarray, max_k: int, model: Any, n_clusters_param: str = "n_clusters"
) -> int:
    """
    Find the optimal number of clusters using silhouette analysis.

    Parameters
    ----------
    values : numpy.ndarray
        1D array of values to cluster.
    max_k : int
        Maximum number of clusters to consider.
    model : Any
        Pre-configured clustering model instance. The model should support
        fit_predict() and have a parameter for number of clusters.
        Most sklearn clustering models inherit from BaseEstimator and ClusterMixin.
    n_clusters_param : str, default='n_clusters'
        The name of the parameter that controls the number of clusters in the model.
        Common values: 'n_clusters' (KMeans, AgglomerativeClustering),
        'n_components' (GaussianMixture).

    Returns
    -------
    int
        Optimal number of clusters.
    """
    if len(np.unique(values)) == 1:
        return 1

    if max_k == 1:
        return 1

    # Limit max_k to the number of unique values
    n_unique = len(np.unique(values))
    max_k = min(max_k, n_unique)

    if max_k == 1:
        return 1

    sample_matrix = values.reshape(-1, 1)
    silhouette_scores = []

    # Try different numbers of clusters from 2 to max_k
    k_range = range(2, max_k + 1)

    for k in k_range:
        try:
            # Create a copy of the model with the specific number of clusters
            # Most sklearn clustering models inherit from BaseEstimator and have get_params()
            if hasattr(model, "get_params") and callable(getattr(model, "get_params")):
                model_params = model.get_params()
                model_params[n_clusters_param] = k
                # Create new instance of the same class
                clustering_model = model.__class__(**model_params)
            else:
                # Fallback: create a new instance and try to set the parameter
                clustering_model = model.__class__()
                if hasattr(clustering_model, n_clusters_param):
                    setattr(clustering_model, n_clusters_param, k)
                else:
                    # If the parameter doesn't exist, skip this iteration
                    silhouette_scores.append(-1)
                    continue

            cluster_labels = clustering_model.fit_predict(sample_matrix)

            # Calculate silhouette score
            if len(np.unique(cluster_labels)) > 1:  # Need at least 2 clusters
                score = silhouette_score(sample_matrix, cluster_labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1)  # Invalid score

        except Exception:
            # If clustering fails for this k, assign a low score
            silhouette_scores.append(-1)

    if not silhouette_scores or all(score <= 0 for score in silhouette_scores):
        # If no valid silhouette scores, default to 2 clusters
        return 2

    # Find k with the highest silhouette score
    best_idx = np.argmax(silhouette_scores)
    optimal_k = k_range[best_idx]

    return optimal_k
