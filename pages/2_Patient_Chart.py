# pages/2_Patient_Chart.py
import re
import streamlit as st
from datetime import datetime

# Safe utils import (optional in your project)
def _no_op(*a, **k): return None
try:
    from utils import init_state as _init_state, enter_page as _enter_page, show_back_top_right as _show_back, render_sidebar as _render_sidebar
except Exception:
    _init_state = _no_op; _enter_page = _no_op; _show_back = _no_op; _render_sidebar = _no_op

# Component toolkit (risk model + panel renderers)
from components.patient_view import make_patient_from_row, compute_summary, render_patient_panel, render_clinician_panel

# ───────── Demo fallback profiles (only used if no trackboard_df available) ─────────
FIXED_PROFILES = {
    "CP-1000": {"MRN":"CP-1000","Patient":"Weber, Charlotte","Age":27,"Sex":"Female","ESI":2,
                "HR":78,"SBP":164,"SpO₂":96,"Temp":"98.6°F","ECG":"Normal","hs-cTn (ng/L)":None,
                "OnsetMin":90,"CC":"Chest pain at rest, mild SOB.","Arrival":"10:39", "DOB":"—",
                "point_risk":33,"ci_low":29,"ci_high":37,"drivers":"All vitals & ECG reassuring"},
    "CP-1001": {"MRN":"CP-1001","Patient":"Green, Gary","Age":59,"Sex":"Male","ESI":3,
                "HR":102,"SBP":138,"SpO₂":95,"Temp":"99.1°F","ECG":"Borderline","hs-cTn (ng/L)":0.03,
                "OnsetMin":60,"CC":"Intermittent chest tightness with exertion.","Arrival":"11:07","DOB":"—",
                "point_risk":21,"ci_low":16,"ci_high":27,"drivers":"ECG slightly abnormal; vitals fair"},
}

# ───────── Small helpers ─────────
def risk_band(point_risk: float) -> str:
    try: r = float(point_risk)
    except: return "—"
    if r < 10: return "Low"
    if r < 30: return "Moderate"
    return "High"

TRACK_BOARD_PALETTE = {"High":"#DC2626","Moderate":"#F59E0B","Low":"#16A34A"}  # tweak to match your board

def risk_badge(point_risk: float, palette=None):
    band = risk_band(point_risk)
    colors = {"High":"#DC2626","Moderate":"#F59E0B","Low":"#16A34A"}
    if palette: colors.update(palette)
    color = colors.get(band, "#9CA3AF")
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:.5rem;margin-top:.25rem;">
          <span style="font-weight:600;">Risk band:</span>
          <span style="padding:.15rem .55rem;border-radius:999px;background:{color};color:white;font-weight:700;">
            {band}
          </span>
        </div>
        """, unsafe_allow_html=True
    )

def temp_to_celsius(value):
    """Accepts 98.6, '98.6°F', '37 C', returns string like '37.0'."""
    if value is None: return ""
    s = str(value).strip().replace(" ", "")
    m = re.match(r"^(-?\d+(\.\d+)?)(°?[CFcf])?$", s)
    if not m:
        try:
            return f"{round(float(s),1)}"
        except:
            return ""
    num = float(m.group(1))
    unit = (m.group(3) or "C").upper().replace("°","")
    if unit == "F":
        c = (num - 32.0) * 5.0 / 9.0
    else:
        c = num
    return f"{round(c,1)}"

# ───────── UI bits ─────────
def patient_header(name: str, mrn: str, default_view="Clinicians"):
    left, right = st.columns([0.70, 0.30], vertical_alignment="center")
    with left:
        st.markdown(f"## {name}")
        st.caption(f"MRN: {mrn}")
    with right:
        st.markdown("")
        view = st.radio("View", ["Patient","Clinicians"], horizontal=True,
                        index=0 if default_view=="Patient" else 1,
                        label_visibility="collapsed")
    st.divider()
    return view

def general_info_block(state: dict) -> dict:
    c1, c2 = st.columns([0.23, 0.77])
    with c1:
        st.markdown("""
        <div style="width:140px;height:140px;border-radius:50%;
             background:linear-gradient(180deg,#edf2f7,#e2e8f0);
             display:flex;align-items:center;justify-content:center;
             font-size:48px;color:#6b7280;margin-bottom:8px;">👤</div>
        """, unsafe_allow_html=True)
        st.caption("Photo placeholder")
    with c2:
        a1,a2,a3,a4 = st.columns(4)
        state["Patient"] = a1.text_input("Patient", str(state.get("Patient","")))
        state["Age"]     = a2.text_input("Age", str(state.get("Age","")))
        state["Sex"]     = a3.text_input("Sex", str(state.get("Sex","")))
        state["ESI"]     = a4.text_input("ESI", str(state.get("ESI","")))

        b1,b2,b3,b4 = st.columns(4)
        state["HR"]   = b1.text_input("HR", str(state.get("HR","")))
        state["SBP"]  = b2.text_input("SBP", str(state.get("SBP","")))
        state["SpO₂"] = b3.text_input("SpO₂", str(state.get("SpO₂","")))
        # Temperature (°C) with auto-convert from any input
        temp_c_val = temp_to_celsius(state.get("Temp", state.get("TempC","")))
        state["TempC"] = b4.text_input("Temp (°C)", "" if temp_c_val=="" else temp_c_val)
        state["Temp"]  = f"{state['TempC']}°C" if state.get("TempC") not in (None,"","None") else ""

        cL,cR = st.columns(2)
        state["ECG"] = cL.text_input("ECG", str(state.get("ECG","")))
        state["hs-cTn (ng/L)"] = cR.text_input("hs-cTn (ng/L)",
            "" if state.get("hs-cTn (ng/L)") in [None,"None"] else str(state.get("hs-cTn (ng/L)")) )

        d1,d2,d3 = st.columns(3)
        state["OnsetMin"] = d1.text_input("Onset (min)", str(state.get("OnsetMin","")))
        state["Arrival"]  = d2.text_input("Arrival", str(state.get("Arrival","")))
        state["DOB"]      = d3.text_input("DOB", str(state.get("DOB","—")))
        state["CC"]       = st.text_area("Chief complaint", str(state.get("CC","")), height=70)
    return state

# ───────── Page renderer (controller) ─────────
def render_patient_chart():
    st.set_page_config(page_title="Patient Chart", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")
    _init_state(); _enter_page("Patient Chart"); _render_sidebar(__file__); _show_back("← Back")

    # If coming from Track Board, prefer that data
    track_df = st.session_state.get("trackboard_df")
    st.session_state.setdefault("selected_patient_id", "CP-1000")

   # ───────── Selector (robust against None/empty) ─────────
    st.caption("Search / select a patient (type ID or name to filter)")

    # Build string options in the form "ID — Name"
    options: list[str] = []
    id_list: list[str] = []

    if track_df is not None and not track_df.empty:
        # Ensure PatientID/Patient are strings; drop rows without an ID
        tmp = track_df.copy()
        if "PatientID" in tmp.columns:
            tmp["PatientID"] = tmp["PatientID"].astype(str)
        if "Patient" in tmp.columns:
            tmp["Patient"] = tmp["Patient"].astype(str).fillna("—")
        tmp = tmp[tmp["PatientID"].notna() & (tmp["PatientID"] != "None")]

        for _, r in tmp.iterrows():
            pid = str(r["PatientID"])
            nm  = str(r.get("Patient", "—"))
            options.append(f"{pid} — {nm}")
            id_list.append(pid)

    else:
        # Fallback to demo profiles
        for pid, d in FIXED_PROFILES.items():
            nm = str(d.get("Patient", ""))
            options.append(f"{pid} — {nm}")
            id_list.append(pid)

    if not options:
        st.error("No patients available.")
        return

    # Resolve a safe default id
    current_id = st.session_state.get("selected_patient_id")
    if not current_id or str(current_id) not in id_list:
        current_id = id_list[0]
        st.session_state["selected_patient_id"] = current_id

    # Preselect label by matching ID exactly (safe even if None earlier)
    try:
        default_index = id_list.index(str(current_id))
    except ValueError:
        default_index = 0

    selected_label = st.selectbox("", options, index=default_index)
    sel_id = selected_label.split(" — ")[0].strip()
    st.session_state["selected_patient_id"] = sel_id

    # Build the patient dict
    if track_df is not None and not track_df.empty:
        row = track_df.loc[track_df["PatientID"] == sel_id]
        if row.empty:
            # fallback to first
            row = track_df.iloc[[0]]
        patient = make_patient_from_row(row.iloc[0].to_dict())
        name = str(row.iloc[0].get("Patient","—"))
    else:
        base = dict(FIXED_PROFILES.get(sel_id, next(iter(FIXED_PROFILES.values()))))
        name = base.get("Patient","—")
        # Minimal mapping to patient dict expected by the components
        patient = make_patient_from_row({
            "PatientID": base.get("MRN"), "Patient": base.get("Patient"),
            "Age": base.get("Age"), "Sex": base.get("Sex"),
            "Arrival": base.get("Arrival"), "OnsetMin": base.get("OnsetMin"),
            "CC": base.get("CC"),
            "SBP": base.get("SBP"), "DBP": 80, "HR": base.get("HR"), "RR": 18,
            "SpO₂": base.get("SpO₂"), "ECG": base.get("ECG"),
            "hs-cTn (ng/L)": base.get("hs-cTn (ng/L)"), "TempF": 98.6,
        })

    # Header + editable general info
    view = patient_header(name, patient["mrn"], default_view="Clinicians")
    edited = general_info_block({
        "Patient": name, "MRN": patient["mrn"],
        "Age": patient["age"], "Sex": patient["sex"], "ESI": "",
        "HR": patient["vitals"]["HR"], "SBP": patient["vitals"]["BP"].split("/")[0],
        "SpO₂": patient["vitals"]["SpO2"], "Temp": f"{patient['vitals']['TempF']}",
        "ECG": "Normal" if not patient["risk_inputs"]["ecg_abnormal"] else "Abnormal",
        "hs-cTn (ng/L)": patient["risk_inputs"].get("troponin"),
        "OnsetMin": patient["data_quality"]["time_from_onset_min"],
        "Arrival": patient["arrival_mode"], "DOB": "—",
        "CC": patient["chief_complaint"]
    })

    # Risk summary (from the component toolkit)
    summary = compute_summary(patient)

    # Top risk ribbon + colored band
    st.progress(min(0.99, summary["base"]))
    st.markdown(f"**AI Estimated Risk:** {round(100*summary['base'])}%  "
                f"_[{round(100*summary['lo'])}–{round(100*summary['hi'])}%]_ • "
                f"Confidence: {summary['conf_dot']} {summary['conf_txt']}")
    risk_badge(100*summary["base"], palette=TRACK_BOARD_PALETTE)

    # Suggested steps (simple expander—kept)
    with st.expander("🕒 Suggested next steps", expanded=(view == "Clinicians")):
        for s in summary["steps"]: st.markdown(f"- {s}")

    # View-specific panels
    if view == "Patient":
        render_patient_panel(summary)
    else:
        render_clinician_panel(summary)

    # Document agreement / override (unchanged)
    st.divider()
    st.markdown("#### ✅ Document agreement / override")
    st.radio("Do you agree with the AI suggestion?", ["Yes", "No / Override"], horizontal=True, key="agree_radio")
    st.text_area("Notes (optional)", placeholder="Rationale, serial lab plan, shared decision details…", height=120)
    st.button("Save to audit log", type="primary")


if __name__ == "__main__":
    render_patient_chart()
else:
    render_patient_chart()