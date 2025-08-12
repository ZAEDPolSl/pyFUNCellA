"""
Thresholding pathway activity scores using k-means clustering.
Based on the FUNCellA R package thr_KM.R implementation.
"""

from typing import Optional, Union, Callable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from pyfuncella.utils.optimize_clusters import find_optimal_clusters
from pyfuncella.utils.progress_callbacks import get_progress_callback


def thr_KM(
    df_path: Union[np.ndarray, pd.DataFrame],
    K: int = 10,
    random_state: Optional[int] = 42,
    verbose: bool = True,
    progress_callback: Optional[Callable] = None,
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

    # Initialize result matrices
    result = np.zeros_like(values, dtype=int)
    cluster_assignments = np.zeros_like(values, dtype=int)

    # Process each pathway (row) with progress reporting
    progress_cb = get_progress_callback(
        progress_callback,
        description="Processing pathways",
        unit="pathway",
        verbose=verbose,
    )

    for i in range(n_pathways):
        progress_cb(i + 1, n_pathways, f"pathway {i + 1}")

        # Get pathway activity scores for current pathway
        sample_values = values[i, :]

        if len(np.unique(sample_values)) == 1:
            result[i, :] = 1
            cluster_assignments[i, :] = 0
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
            cluster_assignments[i, :] = 0
        else:
            kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
            sample_matrix = sample_values.reshape(-1, 1)
            cluster_labels = kmeans.fit_predict(sample_matrix)

            # Sort clusters by mean activity and relabel
            cluster_centers = kmeans.cluster_centers_.flatten()
            sorted_idx = np.argsort(cluster_centers)
            relabel_map = {old: new for new, old in enumerate(sorted_idx)}
            relabeled_clusters = np.vectorize(lambda x: relabel_map[x])(cluster_labels)
            cluster_assignments[i, :] = relabeled_clusters

            # Find the cluster with the highest mean activity (now the last label)
            most_active_cluster = best_k - 1
            result[i, relabeled_clusters == most_active_cluster] = 1

    if is_dataframe:
        # Create DataFrame with binary and cluster columns
        binary_df = pd.DataFrame(result, index=index, columns=columns)
        cluster_df = pd.DataFrame(cluster_assignments, index=index, columns=columns)
        # Suffix columns for clarity
        binary_df.columns = [f"{col}_binary" for col in binary_df.columns]
        cluster_df.columns = [f"{col}_cluster" for col in cluster_df.columns]
        combined_df = pd.concat([binary_df, cluster_df], axis=1)
        return combined_df
    else:
        # For ndarray, concatenate binary and cluster assignments along columns
        combined = np.concatenate([result, cluster_assignments], axis=1)
        return combined
