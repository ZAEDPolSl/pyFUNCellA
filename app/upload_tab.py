import streamlit as st
import pandas as pd
import io


def load_csv(file):
    content = file.getvalue().decode()
    first_line = content.splitlines()[0]
    if "\t" in first_line:
        sep = "\t"
    elif ";" in first_line:
        sep = ";"
    else:
        sep = ","
    lines = content.splitlines()
    genesets = {}
    for line in lines:
        parts = line.strip().split(sep)
        if len(parts) < 2:
            continue
        pathway = parts[0]
        genes = [g for g in parts[1:] if g]
        genesets[pathway] = genes
    return genesets


def upload_tab():
    st.header("Step 1: Upload Data and Genesets")
    data_file = st.file_uploader(
        "Upload gene expression data (CSV)",
        type=["csv"],
        key="data_file",
        help="The genes should be in rows, the patients in columns. The application also expects gene names and patient identifiers.",
    )
    geneset_file = st.file_uploader(
        "Upload genesets file (JSON or CSV)",
        type=["json", "csv"],
        key="geneset_file",
        help="For csv, the genes should be in rows. Each row should start with the pathway name.",
    )
    geneset_source = st.radio(
        "Genesets source",
        ["Upload genesets file", "Use default genesets"],
        key="geneset_source",
    )

    if st.button("Upload", key="save_files"):
        missing = []
        # Data file check
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
            missing.append("gene expression data file")
        # Genesets check
        genesets_loaded = False
        if geneset_source == "Upload genesets file" and geneset_file is not None:
            try:
                import json

                if geneset_file.name.endswith(".json"):
                    genesets = json.load(io.StringIO(geneset_file.getvalue().decode()))
                elif geneset_file.name.endswith(".csv"):
                    genesets = load_csv(geneset_file)
                else:
                    raise ValueError(
                        "Unsupported geneset file type. Please upload JSON or CSV."
                    )
                st.session_state["genesets"] = genesets
                st.success("Genesets file loaded.")
                genesets_loaded = True
            except Exception as e:
                st.error(f"Error loading geneset file: {e}")
        elif geneset_source == "Use default genesets":
            from pyfuncella import load_pathways

            st.session_state["genesets"] = load_pathways("data")
            st.success("Default genesets selected.")
            genesets_loaded = True
        else:
            st.session_state["genesets"] = None
            missing.append("genesets (upload or select default)")
        # Show info if anything is missing
        if missing:
            st.info(f"Please provide: {', '.join(missing)} before proceeding.")
