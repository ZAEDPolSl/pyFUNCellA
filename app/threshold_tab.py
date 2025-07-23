import numpy as np
import pandas as pd
import streamlit as st


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
        thresholded = None

        if method == "K-Means":
            from enrichment_auc.thr_KM import thr_KM

            thresholded = thr_KM(result, verbose=False)
            st.info("K-Means thresholding applied.")
        elif method == "GMM":
            from enrichment_auc.GMMdecomp import GMMdecomp
            from enrichment_auc.thr_GMM import thr_GMM

            # Run GMM decomposition
            gmms = GMMdecomp(result, verbose=False)
            # Store GMM parameters in session_state for later use/visualization
            st.session_state["gmm_params"] = gmms
            # Get thresholds using thr_GMM
            gmm_thresholds = thr_GMM(gmms)
            # Apply threshold per pathway
            thresholded = pd.DataFrame(0, index=result.index, columns=result.columns)
            for i, pathway in enumerate(result.index):
                thr = gmm_thresholds.get(pathway, {}).get("Kmeans_thr", np.nan)
                thresholded.loc[pathway] = (result.loc[pathway] >= thr).astype(int)
            st.info(
                "GMM thresholding applied using GMMdecomp and thr_GMM. GMM parameters stored in session_state['gmm_params']."
            )
        elif method == "AUCell":
            from enrichment_auc.thr_AUCell import thr_AUCell

            # Compute AUCell thresholds for each pathway
            aucell_thresholds = thr_AUCell(result)
            thresholded = pd.DataFrame(0, index=result.index, columns=result.columns)
            for pathway in result.index:
                thr = aucell_thresholds.get(pathway, np.nan)
                thresholded.loc[pathway] = (result.loc[pathway] >= thr).astype(int)
            st.info("AUCell thresholding applied using thr_AUCell.")
        if thresholded is not None:
            st.dataframe(thresholded)
            st.download_button(
                "Download thresholded results as CSV",
                thresholded.to_csv(),
                file_name=f"{method}_thresholded_results.csv",
                on_click="ignore",
            )
            # Pathway/geneset selection for visualization (single selection)
            selected_pathway = st.selectbox(
                "Select a single pathway/geneset to visualize",
                list(result.index),
                key="visualize_pathway",
            )
            st.markdown(f"**Selected pathway:** `{selected_pathway}`")
        st.markdown("---")
        st.write("Visualizations will appear here (coming soon).")
    else:
        st.info("Run PAS calculation first to see results here.")
