import streamlit as st
from pathlib import Path

st.set_page_config(page_title="css-test")
css_file = Path(__file__).parent.parent / "assets" / "style.css"
if css_file.exists():
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown('<div class="f1-hero"><div class="f1-hero-title">TEST HERO</div></div>')
st.write("hello")
