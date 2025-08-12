import numpy as np
import pandas as pd
import streamlit as st

from app.utils.streamlit_progress import create_streamlit_progress_callback
from app.utils.visualize import visualize_scatter, visualize_dist


def threshold_tab():
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

                if method == "K-Means":
                    from pyfuncella.thr_KM import thr_KM

                    thresholded_df = thr_KM(
                        result, verbose=False, progress_callback=progress_callback
                    )

                    binary_cols = [
                        col for col in thresholded_df.columns if col.endswith("_binary")
                    ]
                    cluster_cols = [
                        col
                        for col in thresholded_df.columns
                        if col.endswith("_cluster")
                    ]
                    st.session_state["thresholded"] = thresholded_df[binary_cols]
                    st.session_state["cluster_assignments"] = thresholded_df[
                        cluster_cols
                    ]
                    st.session_state["last_threshold_method"] = method
                    status.update(
                        label="K-Means thresholding completed!", state="complete"
                    )

                elif method == "GMM":
                    try:
                        from pyfuncella.GMMdecomp import GMMdecomp
                        from pyfuncella.thr_GMM import thr_GMM

                        # Step 1: GMM Decomposition
                        status.update(
                            label="Running GMM decomposition...", state="running"
                        )
                        gmms = GMMdecomp(
                            result, verbose=False, progress_callback=progress_callback
                        )
                        st.session_state["gmm_params"] = (
                            gmms  # Step 2: Threshold calculation
                        )
                        status.update(
                            label="Calculating GMM thresholds...", state="running"
                        )
                        gmm_thresholds = thr_GMM(
                            gmms, progress_callback=progress_callback
                        )
                        st.session_state["gmm_thresholds"] = gmm_thresholds

                        thresholded = pd.DataFrame(
                            0, index=result.index, columns=result.columns
                        )
                        for pathway in result.index:
                            thr = gmm_thresholds.get(pathway, {}).get(
                                "Kmeans_thr", float("nan")
                            )
                            thresholded.loc[pathway] = (
                                result.loc[pathway] >= thr
                            ).astype(int)

                        st.session_state["thresholded"] = thresholded
                        st.session_state["cluster_assignments"] = None
                        st.session_state["last_threshold_method"] = method
                        status.update(
                            label="GMM thresholding completed!", state="complete"
                        )
                    except RuntimeError as e:
                        if "R is not available" in str(e):
                            status.update(
                                label="GMM thresholding failed - R not available",
                                state="error",
                            )
                            st.error(f"R is not available: {str(e)}")
                            st.info("**Try:**")
                            st.info(
                                "• Using K-Means thresholding instead (doesn't require R)"
                            )
                            st.info(
                                "• Checking that R is properly installed and accessible"
                            )
                        else:
                            status.update(
                                label="GMM thresholding failed", state="error"
                            )
                            st.error(f"GMM thresholding failed: {str(e)}")
                    except Exception as e:
                        status.update(label="GMM thresholding failed", state="error")
                        st.error(f"Unexpected error during GMM thresholding: {str(e)}")
                        st.info("**Try:**")
                        st.info("• Using K-Means thresholding instead")
                        st.info("• Checking the data for missing or invalid values")

                elif method == "AUCell":
                    try:
                        from pyfuncella.thr_AUCell import thr_AUCell

                        # Check for negative values which can cause AUCell to fail
                        min_value = result.min().min()
                        pas_method = st.session_state.get("pas_method", "PAS")
                        if min_value < 0:
                            st.warning(
                                f"⚠️ Detected negative values in {pas_method} data (minimum: {min_value:.4f})"
                            )
                            st.info(
                                "AUCell expects non-negative enrichment scores. Some PAS methods can produce negative values."
                            )
                            st.info("Consider:")
                            st.info(
                                "• Using K-Means or GMM thresholding (more robust to negative values)"
                            )
                            st.info(
                                "• Using a different PAS method that produces non-negative scores"
                            )
                            status.update(
                                label="AUCell thresholding skipped - negative values detected",
                                state="error",
                            )
                            return

                        aucell_thresholds = thr_AUCell(
                            result, progress_callback=progress_callback
                        )
                        st.session_state["aucell_thresholds"] = aucell_thresholds

                        thresholded = pd.DataFrame(
                            0, index=result.index, columns=result.columns
                        )
                        for pathway in result.index:
                            thr = aucell_thresholds.get(pathway, float("nan"))
                            thresholded.loc[pathway] = (
                                result.loc[pathway] >= thr
                            ).astype(int)

                        st.session_state["thresholded"] = thresholded
                        st.session_state["cluster_assignments"] = None
                        st.session_state["last_threshold_method"] = method
                        status.update(
                            label="AUCell thresholding completed!", state="complete"
                        )
                    except RuntimeError as e:
                        if "R is not available" in str(e):
                            status.update(
                                label="AUCell thresholding failed - R not available",
                                state="error",
                            )
                            st.error("**R Not Available**")
                            st.error(f"Error: {str(e)}")
                            st.info("**Try:**")
                            st.info(
                                "• Using K-Means thresholding instead (doesn't require R)"
                            )
                            st.info(
                                "• Checking that R is properly installed and accessible"
                            )
                        elif "AUCell package" in str(e):
                            status.update(
                                label="AUCell thresholding failed - package missing",
                                state="error",
                            )
                            st.error("**AUCell Package Missing**")
                            st.error(f"Error: {str(e)}")
                            st.info("**Try:**")
                            st.info("• Using K-Means thresholding instead")
                            st.info(
                                "• Installing AUCell: BiocManager::install('AUCell')"
                            )
                        else:
                            status.update(
                                label="AUCell thresholding failed", state="error"
                            )
                            st.error(f"AUCell thresholding failed: {str(e)}")
                    except Exception as e:
                        status.update(label="AUCell thresholding failed", state="error")
                        if "missing value where TRUE/FALSE needed" in str(e):
                            st.error(
                                "AUCell failed due to missing values in the data. This may be caused by:"
                            )
                            st.error("• Pathways with insufficient data points")
                            st.error("• NaN/infinite values in PAS scores")
                            st.error(
                                "• Data distribution issues preventing histogram calculation"
                            )
                            st.info(
                                "Try using K-Means thresholding instead, which is more robust to data quality issues."
                            )
                        else:
                            st.error(f"AUCell thresholding failed: {str(e)}")
                            st.info("**Try:**")
                            st.info("• Using K-Means thresholding instead")
                            st.info(
                                "• Checking that R and AUCell package are properly installed"
                            )
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
