# pages/2_Patient_Chart.py
import streamlit as st
from utils import init_state, render_sidebar, header, TRACKBOARD_DF, PATIENT_DB

init_state()
render_sidebar(current_file=__file__)
header("Patient Chart")

mrn = st.session_state.get("selected_patient_id")
if not mrn:
    st.warning("Please select a patient in the ED Track Board first.")
    st.page_link("pages/1_ED_Track_Board.py", label="📋 Go to ED Track Board")
    st.stop()

row = TRACKBOARD_DF[TRACKBOARD_DF["MRN"] == mrn].iloc[0]
pdata = PATIENT_DB.get(mrn)

st.caption(f"MRN: {mrn} | {row['Patient']} | {row['AgeSex']} | Complaint: {row['Complaint']} | Acuity: {row['Acuity']}")

if pdata:
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("HR", pdata["vitals"]["HR"]);  st.metric("RR", pdata["vitals"]["RR"])
    with col2: st.metric("BP", pdata["vitals"]["BP"]);  st.metric("SpO₂", pdata["vitals"]["SpO2"])
    with col3: st.metric("Temp", pdata["vitals"]["Temp"])

    st.divider()
    left, right = st.columns([0.55, 0.45])
    with left:
        st.markdown("#### Chief Complaint")
        st.write(pdata["chief_complaint"])
        st.markdown("#### History")
        st.write(" • " + "\n • ".join(pdata["history"]))
        st.markdown("#### Orders")
        st.write(" • " + "\n • ".join(pdata["orders"]))
    with right:
        st.markdown("#### Decision Support (demo)")
        st.success(pdata["acuity_suggestion"])
        st.markdown("**Uncertainty snapshot**")
        st.json(pdata["uncertainty"])
        st.markdown("**Notes**")
        st.write(pdata["notes"])
        st.selectbox("Next step suggestion", ["—", "Repeat troponin at 1h", "CTA chest", "Cardiology consult"])
        st.button("Save to Triage Notes", use_container_width=True)
else:
    st.info("No demo detail for this MRN. Showing header only.")
