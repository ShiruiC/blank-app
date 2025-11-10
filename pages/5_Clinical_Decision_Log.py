import streamlit as st
from utils import init_state, render_sidebar, header
init_state(); render_sidebar(current_file=__file__)
header("Clinical Decision Log")
st.write("Review clinician decisions versus AI recommendations for each patient case (placeholder).")