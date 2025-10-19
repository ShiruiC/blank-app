# pages/1_ED_Track_Board.py
import streamlit as st
from utils import init_state, render_sidebar, header, TRACKBOARD_DF
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

init_state()
render_sidebar(current_file=__file__)
header("ED Track Board", "Click a patient row to open the chart.")

# Build AgGrid with single-row selection
df = TRACKBOARD_DF.copy()
display_cols = ["Room","Patient","AgeSex","Complaint","Acuity","Indicators","RN","MD","Lab","Rad","Dispo","Comments"]

gob = GridOptionsBuilder.from_dataframe(df[display_cols])
gob.configure_selection(selection_mode="single", use_checkbox=False)
gob.configure_grid_options(rowHeight=28, headerHeight=32)
gob.configure_column("Indicators", width=90)
gob.configure_column("Room", width=70)
gob.configure_column("Acuity", width=90)
go = gob.build()

grid = AgGrid(
    df[display_cols],
    gridOptions=go,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    height=400,
    theme="balham",
    fit_columns_on_grid_load=True
)

selected = grid["selected_rows"]
if selected:
    # find MRN behind the selected patient
    patient_name = selected[0]["Patient"]
    mrn = df.loc[df["Patient"] == patient_name, "MRN"].iloc[0]
    st.session_state["selected_patient_id"] = mrn
    st.success(f"Selected: {patient_name}  •  MRN {mrn}")
    st.page_link("pages/2_Patient_Chart.py", label="➡️ Open Patient Chart", help="Go to patient details")

st.caption("Legend: **●** Unacknowledged item  |  **▪** New  |  **✓** Completed")
