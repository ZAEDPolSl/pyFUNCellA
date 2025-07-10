"""
Thresholding pathway activity scores using k-means clustering.
Based on the FUNCellA R package thr_KM.R implementation.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

from enrichment_auc.utils.optimize_clusters import find_optimal_clusters


def thr_KM(
    df_path: Union[np.ndarray, pd.DataFrame],
    K: int = 10,
    random_state: Optional[int] = 42,
    verbose: bool = True,
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Threshold pathway activity scores using k-means clustering.

    This function applies k-means clustering to threshold pathway activity scores
    for each pathway (row) individually. The optimal number of clusters is
    estimated using the silhouette method. Samples belonging to the cluster
    with the highest mean activity are marked as active (1); all others are
    marked inactive (0).

    Parameters
    ----------
    df_path : numpy.ndarray or pandas.DataFrame
        A numeric matrix or DataFrame of pathway activity scores,
        where rows correspond to pathways and columns to samples.
    K : int, default=10
        Maximum number of clusters to consider when estimating the optimal
        number of clusters using the silhouette method.
    random_state : int, optional, default=42
        Random state for reproducible k-means clustering.
    verbose : bool, default=True
        Whether to print progress information.
    Returns
    -------
    numpy.ndarray or pandas.DataFrame
        A binary matrix of the same dimensions as df_path, where
        values are 1 for samples in the most active cluster per pathway,
        and 0 otherwise. Returns the same type as the input.
    """
    # Input validation
    if not isinstance(df_path, (np.ndarray, pd.DataFrame)):
        raise TypeError("df_path must be a numpy array or pandas DataFrame")

    if K < 2:
        raise ValueError("K must be at least 2")

    # Convert to numpy array for processing
    is_dataframe = isinstance(df_path, pd.DataFrame)
    if is_dataframe:
        values = df_path.values
        index = df_path.index
        columns = df_path.columns
    else:
        values = df_path

    n_pathways, n_samples = values.shape

    if n_samples < 2:
        raise ValueError("Need at least 2 samples for clustering")

    # Initialize result matrix
    result = np.zeros_like(values, dtype=int)

    # Process each pathway (row) with progress bar
    pathway_iterator = tqdm(
        range(n_pathways),
        desc="Processing pathways",
        unit="pathway",
        disable=not verbose,
    )

    for i in pathway_iterator:

        # Get pathway activity scores for current pathway
        sample_values = values[i, :]

        if len(np.unique(sample_values)) == 1:
            result[i, :] = 1
            continue

        if n_samples < K:
            max_k = n_samples
        else:
            max_k = K

        # Find optimal number of clusters using silhouette analysis
        kmeans_model = KMeans(random_state=random_state, n_init=10)
        param_name = "n_clusters"
        best_k = find_optimal_clusters(sample_values, max_k, kmeans_model, param_name)

        # Perform k-means clustering with optimal k
        if best_k == 1:
            # Only one cluster, mark all as active
            result[i, :] = 1
        else:
            kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
            sample_matrix = sample_values.reshape(-1, 1)
            cluster_labels = kmeans.fit_predict(sample_matrix)

            # Find the cluster with the highest mean activity
            cluster_centers = kmeans.cluster_centers_.flatten()
            most_active_cluster = np.argmax(cluster_centers)

            # Mark samples in the most active cluster as 1
            result[i, cluster_labels == most_active_cluster] = 1

    if is_dataframe:
        return pd.DataFrame(result, index=index, columns=columns)
    else:
        return result
