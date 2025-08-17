import streamlit as st


st.set_page_config(page_title="FUNCellA", page_icon="🧬")

from app.pas_tab import pas_tab
from app.threshold_tab import threshold_tab
from app.upload_tab import upload_tab

st.title("FUNCellA (Functional Cell Analysis)")
st.write(
    "Upload your data and genesets, choose method and filtering options, calculate PAS and thresholding."
)

tabs = st.tabs(["Upload Files", "PAS Calculation", "Thresholding"])

with tabs[0]:
    upload_tab()
with tabs[1]:
    pas_tab()

with tabs[2]:
    threshold_tab()


footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #154c8e;
    color: white;
    padding: 10px;
    z-index: 100;
    display: flex;
    align-items: center;
}
.footer-logos {
    display: flex;
    align-items: center;
}
.footer img {
    margin: 0px 10px;
    vertical-align: middle;
    height: 40px;
    transition: transform 0.2s;
}
.footer img:hover {
    transform: scale(1.2);
}
.footer-text {
    margin-left: auto;
    margin-right: 20px;
}
</style>
<div class="footer">
    <div class="footer-logos">
        <img src="https://raw.githubusercontent.com/ZAEDPolSl/enrichment-auc/main/app/static/politechnika_sl_logo_poziom_inwersja_en_rgb.png" alt="Politechnika Śląska Logo">
        <img src="https://raw.githubusercontent.com/ZAEDPolSl/enrichment-auc/main/app/static/EN-logo-poziome-rgb.png" alt="EN Logo">
    </div>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)
