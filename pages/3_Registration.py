import streamlit as st
from utils import init_state, render_sidebar, header
init_state(); render_sidebar(current_file=__file__)
header("Registration")
st.write("Record newly arrived patients and their basic admission details before clinical assessment (placeholder).")
