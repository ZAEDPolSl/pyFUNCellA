import streamlit as st
import pandas as pd
import numpy as np
import io
from enrichment_auc.gene2path import gene2path

st.title("FUNCellA (Functional Cell Analysis)")
st.write(
    "Upload your data and genesets, choose method and filtering options, calculate PAS and thresholding."
)

tabs = st.tabs(["Upload Files", "PAS Calculation", "Thresholding"])

# --- Tab 1: Upload Files ---
with tabs[0]:
    st.header("Step 1: Upload Data and Genesets")
    data_file = st.file_uploader(
        "Upload gene expression data (CSV)", type=["csv"], key="data_file"
    )
    geneset_file = st.file_uploader(
        "Upload genesets file (JSON)", type=["json"], key="geneset_file"
    )
    geneset_source = st.radio(
        "Genesets source",
        ["Upload genesets file", "Use default genesets"],
        key="geneset_source",
    )
    default_genesets = {}  # TODO: Fill with your default genesets

    if st.button("Upload", key="save_files"):
        if data_file is not None:
            try:
                st.session_state["data"] = pd.read_csv(data_file, index_col=0)
                st.session_state["genes"] = list(st.session_state["data"].index)
                st.success("Data file loaded.")
            except Exception as e:
                st.error(f"Error loading data file: {e}")
        else:
            st.session_state["data"] = None
            st.session_state["genes"] = None
        if geneset_source == "Upload geneset file" and geneset_file is not None:
            try:
                import json

                genesets = json.load(io.StringIO(geneset_file.getvalue().decode()))
                st.session_state["genesets"] = genesets
                st.success("Genesets file loaded.")
            except Exception as e:
                st.error(f"Error loading geneset file: {e}")
        elif geneset_source == "Use default genesets":
            st.session_state["genesets"] = default_genesets
            st.success("Default genesets selected.")
        else:
            st.session_state["genesets"] = None

# --- Tab 2: PAS Calculation ---
with tabs[1]:
    st.header("Step 2: PAS Calculation")
    # Method selection
    method = st.selectbox(
        "Select calculation method",
        ["CERNO", "MEAN", "BINA", "AUCELL", "JASMINE", "ZSCORE", "SSGSEA"],
        index=0,
        key="method_select",
    )
    # Filtering options
    filt_cov = st.slider(
        "Minimum pathway coverage (filt_cov)", 0.0, 1.0, 0.0, 0.01, key="filt_cov"
    )
    filt_min = st.number_input(
        "Minimum pathway size (filt_min)", min_value=0, value=15, key="filt_min"
    )
    filt_max = st.number_input(
        "Maximum pathway size (filt_max)", min_value=1, value=500, key="filt_max"
    )
    variance_filter_threshold = st.slider(
        "Variance filter threshold (fraction of genes to keep)",
        0.0,
        1.0,
        0.0,
        0.01,
        key="variance_filter",
    )
    aucell_threshold = None
    if method == "AUCELL":
        aucell_threshold = st.slider(
            "AUCELL threshold (fraction of top genes)",
            0.0,
            1.0,
            0.05,
            0.01,
            key="aucell_thr",
        )
    type_option = None
    if method == "JASMINE":
        type_option = st.selectbox(
            "JASMINE effect size type",
            ["oddsratio", "likelihood"],
            index=0,
            key="jasmine_type",
        )

    if st.button("Run PAS calculation", key="run_pas"):
        data = st.session_state.get("data")
        genes = st.session_state.get("genes")
        genesets = st.session_state.get("genesets")
        if data is None or genesets is None or len(genesets) == 0:
            st.error("Both data and genesets must be loaded and non-empty.")
            st.stop()
        kwargs = dict(
            data=data,
            genesets=genesets,
            genes=genes,
            method=method,
            filt_cov=filt_cov,
            filt_min=filt_min,
            filt_max=filt_max,
            variance_filter_threshold=(
                variance_filter_threshold if variance_filter_threshold > 0 else None
            ),
        )
        if method == "AUCELL":
            kwargs["aucell_threshold"] = (
                aucell_threshold if aucell_threshold is not None else 0.05
            )
        if method == "JASMINE":
            kwargs["type"] = type_option
        try:
            result = gene2path(**kwargs)
            st.session_state["pas_result"] = result
            st.success(f"PAS calculation completed. Result shape: {result.shape}")
            st.dataframe(result)
            st.download_button(
                "Download results as CSV",
                result.to_csv(),
                file_name=f"{method}_results.csv",
            )
        except Exception as e:
            st.error(f"Error running {method}: {e}")

# --- Tab 3: Thresholding ---
with tabs[2]:
    st.header("Step 3: Thresholding and Post-processing")
    result = st.session_state.get("pas_result")
    if result is not None:
        st.write("You can add thresholding, visualization, or post-processing here.")
        # Example: threshold slider
        threshold = st.slider(
            "Set threshold for PAS",
            float(result.min().min()),
            float(result.max().max()),
            float(result.mean().mean()),
            0.01,
            key="pas_threshold",
        )
        thresholded = result.applymap(lambda x: 1 if x >= threshold else 0)
        st.dataframe(thresholded)
        st.download_button(
            "Download thresholded results as CSV",
            thresholded.to_csv(),
            file_name="thresholded_results.csv",
        )
    else:
        st.info("Run PAS calculation first to see results here.")
