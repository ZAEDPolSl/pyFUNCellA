"""
Visualization utilities for pathway analysis.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from enrichment_auc.plot.plot_distributed_data import plot_pas_distribution
from enrichment_auc.plot.plot_scatter_flow import scatterplot_subplots


def calculate_pca_tsne_coordinates():
    """
    Calculate PCA (40 components) + t-SNE (2D) coordinates from original data.
    This is computed once for all pathways and stored in session state.
    """
    data = st.session_state.get("data")
    if data is None:
        return None

    # Check if coordinates are already calculated
    if "pca_tsne_coords" in st.session_state:
        return st.session_state["pca_tsne_coords"]

    try:
        # Transpose data so samples are rows and features are columns
        data_T = data.T

        # Apply PCA to reduce to 40 components
        pca = PCA(n_components=min(40, data_T.shape[1]))
        pca_result = pca.fit_transform(data_T)

        # Apply t-SNE to reduce to 2D
        tsne = TSNE(
            n_components=2, random_state=42, perplexity=min(30, data_T.shape[0] - 1)
        )
        tsne_result = tsne.fit_transform(pca_result)

        # Store coordinates in session state
        coords = {"x": tsne_result[:, 0], "y": tsne_result[:, 1]}
        st.session_state["pca_tsne_coords"] = coords
        return coords
    except Exception as e:
        st.error(f"Error calculating PCA + t-SNE coordinates: {str(e)}")
        return None


def visualize_scatter(
    result, thresholded, method, selected_pathway, ranked_labels=None
):
    """
    Visualize PAS scatterplots for AUCell, GMM, and K-Means thresholding.
    Uses PCA + t-SNE coordinates for x and y positions.
    """
    # Get or calculate PCA + t-SNE coordinates
    coords = calculate_pca_tsne_coordinates()
    if coords is None:
        st.warning(
            "Could not calculate PCA + t-SNE coordinates. Falling back to sample indices."
        )
        pas_scores = result.loc[selected_pathway].values
        x = np.arange(len(pas_scores))
        y = pas_scores
    else:
        pas_scores = result.loc[selected_pathway].values
        x = coords["x"]
        y = coords["y"]

    binary_labels = (
        thresholded.loc[selected_pathway].values if thresholded is not None else None
    )
    continuous_labels = pas_scores
    title = f"{selected_pathway} ({method})"
    pas_method = st.session_state.get("pas_method", "PAS")

    if method == "AUCell":
        fig = scatterplot_subplots(
            x=x,
            y=y,
            continuous_labels=continuous_labels,
            binary_labels=binary_labels,
            ranked_labels=ranked_labels,
            title=title,
            pas_method=pas_method,
        )
    elif method == "GMM":
        gmm_thresholds = st.session_state.get("gmm_thresholds")
        thr = None
        all_thr = None
        if gmm_thresholds:
            thr = gmm_thresholds.get(selected_pathway, {}).get(
                "Kmeans_thr", float("nan")
            )
            all_thr = gmm_thresholds.get(selected_pathway, {}).get("All_thr", [])
        if thr is None:
            thr = float("nan")
        binary_labels = (pas_scores >= thr).astype(int)
        # Calculate group index for each PAS score based on All_thr
        if all_thr and len(all_thr) > 0:
            # Sort thresholds and assign group index by np.digitize
            sorted_thr = np.sort(np.array(all_thr))
            gmm_groups = np.digitize(pas_scores, sorted_thr)
        else:
            gmm_groups = binary_labels

        # Don't show ranked labels if there are only 2 or fewer groups
        if gmm_groups is not None and len(np.unique(gmm_groups)) <= 2:
            gmm_groups = None

        fig = scatterplot_subplots(
            x=x,
            y=y,
            continuous_labels=continuous_labels,
            binary_labels=binary_labels,
            ranked_labels=gmm_groups,
            title=title,
            pas_method=pas_method,
        )
    elif method == "K-Means":
        if ranked_labels is not None and len(np.unique(ranked_labels)) <= 2:
            ranked_labels = None

        fig = scatterplot_subplots(
            x=x,
            y=y,
            continuous_labels=continuous_labels,
            binary_labels=binary_labels,
            ranked_labels=ranked_labels,
            title=title,
            pas_method=pas_method,
        )
    st.plotly_chart(fig, use_container_width=True)


def visualize_dist(result, method, selected_pathway):
    """
    Visualize PAS score distribution for AUCell and GMM thresholding.
    """
    pas_scores = result.loc[selected_pathway].values
    pathway_name = selected_pathway
    pas_method = st.session_state.get("pas_method", "PAS")
    fig = None

    if method == "AUCell":
        aucell_thresholds = st.session_state.get("aucell_thresholds")
        thr = aucell_thresholds.get(pathway_name) if aucell_thresholds else None
        if thr is None:
            st.warning(
                f"AUCell threshold not found for pathway {pathway_name}. Please re-run thresholding."
            )
            return
        fig = plot_pas_distribution(pas_scores, pas_method, pathway_name, thr)

    elif method == "GMM":
        gmm_params = st.session_state.get("gmm_params", {})
        gmm_model = (
            gmm_params.get(pathway_name, {}).get("model") if gmm_params else None
        )
        gmm_thresholds = st.session_state.get("gmm_thresholds")
        thr = None
        if gmm_thresholds:
            thr = gmm_thresholds.get(pathway_name, {}).get("Kmeans_thr", float("nan"))
        if thr is None:
            thr = float("nan")
        fig = plot_pas_distribution(
            pas_scores, pas_method, pathway_name, thr, gmm=gmm_model
        )

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
