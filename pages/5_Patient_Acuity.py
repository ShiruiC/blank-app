import streamlit as st
from utils import init_state, render_sidebar, header
init_state(); render_sidebar(current_file=__file__)
header("Patient Acuity")
st.write("Acuity assessment (placeholder).")
