import streamlit as st
import pandas as pd
import io


def upload_tab(default_genesets):
    st.header("Step 1: Upload Data and Genesets")
    data_file = st.file_uploader(
        "Upload gene expression data (CSV)", type=["csv"], key="data_file"
    )
    geneset_file = st.file_uploader(
        "Upload genesets file (JSON or CSV)", type=["json", "csv"], key="geneset_file"
    )
    geneset_source = st.radio(
        "Genesets source",
        ["Upload genesets file", "Use default genesets"],
        key="geneset_source",
    )

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

                if geneset_file.name.endswith(".json"):
                    genesets = json.load(io.StringIO(geneset_file.getvalue().decode()))
                elif geneset_file.name.endswith(".csv"):
                    df = pd.read_csv(geneset_file)
                    if df.shape[1] < 2:
                        raise ValueError(
                            "Geneset CSV must have at least two columns (pathway, gene)"
                        )
                    genesets = (
                        df.groupby(df.columns[0])[df.columns[1]].apply(list).to_dict()
                    )
                else:
                    raise ValueError(
                        "Unsupported geneset file type. Please upload JSON or CSV."
                    )
                st.session_state["genesets"] = genesets
                st.success("Genesets file loaded.")
            except Exception as e:
                st.error(f"Error loading geneset file: {e}")
        elif geneset_source == "Use default genesets":
            st.session_state["genesets"] = default_genesets
            st.success("Default genesets selected.")
        else:
            st.session_state["genesets"] = None
