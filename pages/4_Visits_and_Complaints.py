import streamlit as st
from utils import init_state, render_sidebar, header
init_state(); render_sidebar(current_file=__file__)
header("Visits & Complaints")
st.write("Chief complaint & visit history (placeholder).")
