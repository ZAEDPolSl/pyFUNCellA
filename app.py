import streamlit as st
from app.upload_tab import upload_tab
from app.pas_tab import pas_tab
from app.threshold_tab import threshold_tab

default_genesets = {}  # TODO: Fill with your default genesets

st.title("FUNCellA (Functional Cell Analysis)")
st.write(
    "Upload your data and genesets, choose method and filtering options, calculate PAS and thresholding."
)

tabs = st.tabs(["Upload Files", "PAS Calculation", "Thresholding"])

with tabs[0]:
    upload_tab(default_genesets)
with tabs[1]:
    pas_tab()
with tabs[2]:
    threshold_tab()
