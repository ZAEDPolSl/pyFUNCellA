import numpy as np
import pandas as pd
import streamlit as st

from app.utils.streamlit_progress import create_streamlit_progress_callback
from app.utils.visualize import visualize_scatter, visualize_dist


def _run_kmeans_thresholding(result, progress_callback):
    """Run K-Means thresholding on PAS results."""
    from pyfuncella.thr_KM import thr_KM

    thresholded_df = thr_KM(result, verbose=False, progress_callback=progress_callback)

    # thr_KM returns a DataFrame with binary and cluster columns when input is DataFrame
    assert isinstance(thresholded_df, pd.DataFrame), "Expected DataFrame from thr_KM"

    binary_cols = [col for col in thresholded_df.columns if col.endswith("_binary")]
    cluster_cols = [col for col in thresholded_df.columns if col.endswith("_cluster")]
    return thresholded_df[binary_cols], thresholded_df[cluster_cols]


def _run_gmm_thresholding(result, progress_callback, status):
    """Run GMM thresholding on PAS results."""
    from pyfuncella.GMMdecomp import GMMdecomp
    from pyfuncella.thr_GMM import thr_GMM

    # Step 1: GMM Decomposition
    status.update(label="Running GMM decomposition...", state="running")
    gmms = GMMdecomp(result, verbose=False, progress_callback=progress_callback)
    st.session_state["gmm_params"] = gmms

    # Step 2: Threshold calculation
    status.update(label="Calculating GMM thresholds...", state="running")
    gmm_thresholds = thr_GMM(gmms, progress_callback=progress_callback)
    st.session_state["gmm_thresholds"] = gmm_thresholds

    thresholded = pd.DataFrame(0, index=result.index, columns=result.columns)
    for pathway in result.index:
        thr = gmm_thresholds.get(pathway, {}).get("Kmeans_thr", float("nan"))
        thresholded.loc[pathway] = (result.loc[pathway] >= thr).astype(int)

    return thresholded, None


def _run_aucell_thresholding(result, progress_callback):
    """Run AUCell thresholding on PAS results."""
    from pyfuncella.thr_AUCell import thr_AUCell

    aucell_thresholds = thr_AUCell(result, progress_callback=progress_callback)
    st.session_state["aucell_thresholds"] = aucell_thresholds

    thresholded = pd.DataFrame(0, index=result.index, columns=result.columns)
    for pathway in result.index:
        thr = aucell_thresholds.get(pathway, float("nan"))
        thresholded.loc[pathway] = (result.loc[pathway] >= thr).astype(int)

    return thresholded, None


def _handle_r_not_available_error(method, status, e):
    """Handle R not available errors."""
    status.update(
        label=f"{method} thresholding failed - R not available",
        state="error",
    )
    st.error("**R Not Available**")
    st.error(f"Error: {str(e)}")
    st.info("**Try:**")
    st.info("• Using K-Means thresholding instead (doesn't require R)")
    st.info("• Checking that R is properly installed and accessible")


def _handle_package_missing_error(method, status, e, package_name):
    """Handle missing R package errors."""
    status.update(
        label=f"{method} thresholding failed - package missing",
        state="error",
    )
    st.error(f"**{package_name} Package Missing**")
    st.error(f"Error: {str(e)}")
    st.info("**Try:**")
    st.info("• Using K-Means thresholding instead")
    if package_name == "AUCell":
        st.info("• Installing AUCell: BiocManager::install('AUCell')")


def _handle_missing_values_error(method, status, e):
    """Handle missing values in data errors."""
    status.update(label=f"{method} thresholding failed", state="error")
    st.error(
        f"{method} failed due to missing values in the data. This may be caused by:"
    )
    st.error("• Pathways with insufficient data points")
    st.error("• NaN/infinite values in PAS scores")
    st.error("• Data distribution issues preventing histogram calculation")
    st.info(
        "Try using K-Means thresholding instead, which is more robust to data quality issues."
    )


def _handle_general_error(method, status, e):
    """Handle general thresholding errors."""
    status.update(label=f"{method} thresholding failed", state="error")
    st.error(f"{method} thresholding failed: {str(e)}")
    st.info("**Try:**")
    st.info("• Using K-Means thresholding instead")
    st.info("• Checking that R and required packages are properly installed")


def threshold_tab():
    """Main threshold tab function with refactored helper functions."""
    st.header("Step 3: Thresholding and Post-processing")
    result = st.session_state.get("pas_result")
    if result is not None:
        st.write("Choose thresholding method and apply to PAS results.")
        method = st.selectbox(
            "Thresholding method",
            ["K-Means", "GMM", "AUCell"],
            index=0,
            key="threshold_method",
        )

        # Use session_state to persist thresholded results and cluster assignments
        if (
            "thresholded" not in st.session_state
            or "cluster_assignments" not in st.session_state
            or st.session_state.get("last_threshold_method") != method
        ):
            st.session_state["thresholded"] = None
            st.session_state["cluster_assignments"] = None

        run_threshold = st.button("Run thresholding", key="run_threshold")
        if run_threshold:
            with st.status(f"Running {method}...", expanded=True) as status:
                progress_callback = create_streamlit_progress_callback()

                try:
                    if method == "K-Means":
                        thresholded, cluster_assignments = _run_kmeans_thresholding(
                            result, progress_callback
                        )
                        st.session_state["thresholded"] = thresholded
                        st.session_state["cluster_assignments"] = cluster_assignments
                        status.update(
                            label="K-Means thresholding completed!", state="complete"
                        )

                    elif method == "GMM":
                        thresholded, cluster_assignments = _run_gmm_thresholding(
                            result, progress_callback, status
                        )
                        st.session_state["thresholded"] = thresholded
                        st.session_state["cluster_assignments"] = cluster_assignments
                        status.update(
                            label="GMM thresholding completed!", state="complete"
                        )

                    elif method == "AUCell":
                        thresholded, cluster_assignments = _run_aucell_thresholding(
                            result, progress_callback
                        )
                        st.session_state["thresholded"] = thresholded
                        st.session_state["cluster_assignments"] = cluster_assignments
                        status.update(
                            label="AUCell thresholding completed!", state="complete"
                        )

                    st.session_state["last_threshold_method"] = method

                except RuntimeError as e:
                    if "R is not available" in str(e):
                        _handle_r_not_available_error(method, status, e)
                    elif "AUCell package" in str(e):
                        _handle_package_missing_error(method, status, e, "AUCell")
                    else:
                        _handle_general_error(method, status, e)
                except Exception as e:
                    if "missing value where TRUE/FALSE needed" in str(e):
                        _handle_missing_values_error(method, status, e)
                    else:
                        _handle_general_error(method, status, e)
        thresholded = st.session_state.get("thresholded")
        cluster_assignments = st.session_state.get("cluster_assignments")
        if thresholded is not None:
            st.dataframe(thresholded)
            st.download_button(
                "Download thresholded results as CSV",
                thresholded.to_csv(),
                file_name=f"{method}_thresholded_results.csv",
                on_click="ignore",
            )
            selected_pathway = st.selectbox(
                "Select a single pathway to visualize",
                list(result.index),
                key="visualize_pathway",
            )
            st.markdown(f"**Selected pathway:** `{selected_pathway}`")
            # Only show distribution plots for methods that use thresholds
            if method in ["AUCell", "GMM"]:
                visualize_dist(result, method, selected_pathway)
            if method == "K-Means" and cluster_assignments is not None:
                ranked_labels = cluster_assignments.loc[selected_pathway].values
                visualize_scatter(
                    result,
                    thresholded,
                    method,
                    selected_pathway,
                    ranked_labels=ranked_labels,
                )
            else:
                visualize_scatter(result, thresholded, method, selected_pathway)
            st.markdown("---")
    else:
        st.info("Run PAS calculation first to see results here.")
