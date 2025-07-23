import streamlit as st
from enrichment_auc.gene2path import gene2path


def pas_tab():
    st.header("Step 2: PAS Calculation")
    method = st.selectbox(
        "Select calculation method",
        ["CERNO", "MEAN", "BINA", "AUCELL", "JASMINE", "ZSCORE", "SSGSEA"],
        index=0,
        key="method_select",
    )
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
                on_click="ignore",
            )
        except Exception as e:
            st.error(f"Error running {method}: {e}")
