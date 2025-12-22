# streamlit_app.py
import streamlit as st
st.set_page_config(page_title="EHR SYSTEM", page_icon="🏥", layout="wide")

from utils import init_state, render_sidebar, header, reset_on_landing
init_state()
render_sidebar(current_file=__file__)
reset_on_landing()

header("EHR System", "Welcome to the Emergency Department decision-support workspace.")

# Make the upper section visually dominant via wider columns
colA, colB = st.columns([3, 2])
with colA:
    st.subheader("Quick Access")
    st.page_link("pages/1_ED_Track_Board.py", label="📋 Open ED Track Board")
    st.page_link("pages/2_Patient_Chart.py", label="🩺 Open Patient Chart")
with colB:
    st.subheader("How to use")
    st.markdown(
        "- Use the **blue left sidebar** to switch pages. Click the top button to **collapse/expand**.\n"
        "- **ED Track Board**: view all patients, then click a row to open Patient Chart.\n"
        "- **Patient Chart**: view patient details, AI recommendations with confidence illustration, and the clinician’s final decision with next-step actions.\n"
        "- Other pages are templates for fields & workflows."
    )

st.divider()

# ---- Attention needed (AI–Clinician disagreement) ----
st.subheader("Attention needed — AI–Clinician Decision Mismatch")
# Pull from session state if already populate it elsewhere; otherwise use a harmless placeholder.
mismatches = st.session_state.get(
    "decision_mismatches",
    [
        {"MRN": "CP-1004", "Patient": "Müller, Jonas", "AI": "Consult", "Clinician": "Confirm/Admit", "Next action": "Cardiology consult; continuous monitor"},
        {"MRN": "CP-1008", "Patient": "Mustermann, Max", "AI": "Discharge", "Clinician": "Observe", "Next action": "Repeat labs in 2h; reassess"}
    ]
)
if mismatches:
    st.dataframe(mismatches, use_container_width=True, hide_index=True)
else:
    st.info("No current mismatches requiring action. 🎉")

# Keep Today’s Tip at the very bottom
st.subheader("Today’s Tip")
st.info("Chest pain triage: ECG + first hs-Tn. If suggested **T2**, consider starting the ACS pathway.")