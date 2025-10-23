# streamlit_app.py
import streamlit as st
st.set_page_config(page_title="EHR SYSTEM", page_icon="🏥", layout="wide")

from utils import init_state, render_sidebar, header, reset_on_landing
init_state()
render_sidebar(current_file=__file__)
reset_on_landing()


header("EHR System", "Welcome to the ED decision-support workspace.")

colA, colB = st.columns(2)
with colA:
    st.subheader("Quick Access")
    st.page_link("pages/1_ED_Track_Board.py", label="📋 Open ED Track Board")
    st.page_link("pages/2_Patient_Chart.py", label="🩺 Open Patient Chart")
with colB:
    st.subheader("How to use")
    st.markdown(
        "- Use the **blue left sidebar** to switch pages. Click the top button to **collapse/expand**.\n"
        "- **ED Track Board**: view all patients, then click a row to open **Patient Chart**.\n"
        "- Other pages are templates for fields & workflows."
    )

st.divider()
st.subheader("Today’s Tip ")
st.info("Chest pain triage: ECG + first hs-Tn. If suggested **T2**, consider starting the ACS pathway.")
