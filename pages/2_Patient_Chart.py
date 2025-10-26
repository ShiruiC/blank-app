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
from components.patient_view import (
    make_patient_from_row, compute_summary,
    triage_level_from_summary, disposition_from_summary,
    render_patient_panel, render_clinician_panel,
    decompose_uncertainty, sens_table, pct_tier
)

# ───────── Demo fallback profiles (only used if no trackboard_df available) ─────────
FIXED_PROFILES = {
    "CP-1000": {"MRN":"CP-1000","Patient":"Weber, Charlotte","Age":27,"Sex":"Female","ESI":2,
                "HR":78,"SBP":164,"SpO₂":96,"TempC": 37.0, "Temp": "37.0°C","ECG":"Normal","hs-cTn (ng/L)":None,
                "OnsetMin":90,"CC":"Chest pain at rest, mild SOB.","Arrival":"10:39", "DOB":"—"},
    "CP-1001": {"MRN":"CP-1001","Patient":"Green, Gary","Age":59,"Sex":"Male","ESI":3,
                "HR":102,"SBP":138,"SpO₂":95,"TempC": 37.0, "Temp": "37.0°C","ECG":"Borderline","hs-cTn (ng/L)":0.03,
                "OnsetMin":60,"CC":"Intermittent chest tightness with exertion.","Arrival":"11:07","DOB":"—"},
}

# ───────── Small helpers ─────────
def risk_band(point_risk: float) -> str:
    try: r = float(point_risk)
    except: return "—"
    if r < 10: return "Low"
    if r < 30: return "Moderate"
    return "High"

TRACK_BOARD_PALETTE = {"High":"#DC2626","Moderate":"#F59E0B","Low":"#16A34A"}

def risk_badge(point_risk: float, palette=None):
    band = risk_band(point_risk)
    colors = {"High":"#DC2626","Moderate":"#F59E0B","Low":"#16A34A"}
    if palette: colors.update(palette)
    color = colors.get(band, "#9CA3AF")
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:.5rem;">
          <span style="font-weight:600;">Risk band:</span>
          <span style="padding:.15rem .55rem;border-radius:999px;background:{color};color:white;font-weight:700;">
            {band}
          </span>
        </div>
        """, unsafe_allow_html=True
    )

def temp_to_celsius(value):
    if value is None: return ""
    s = str(value).strip().replace(" ", "")
    m = re.match(r"^(-?\d+(\.\d+)?)(°?[CFcf])?$", s)
    if not m:
        try: return f"{round(float(s),1)}"
        except: return ""
    num = float(m.group(1))
    unit = (m.group(3) or "C").upper().replace("°","")
    c = (num - 32.0) * 5.0 / 9.0 if unit == "F" else num
    return f"{round(c,1)}"

# ───────── UI bits (header + unchanged basic info) ─────────
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
        seed_temp_c = state.get("TempC")
        state["TempC"] = b4.text_input("Temp (°C)", "" if seed_temp_c in (None,"") else f"{float(seed_temp_c):.1f}")
        state["Temp"]  = f"{state['TempC']}°C" if state.get("TempC") else ""

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

def _pill(text, tone="#2563EB", inverse=False):
    return f"<span style='border-radius:999px;padding:.25rem .6rem;border:1px solid {tone};" \
           f"margin:.15rem;display:inline-block;" \
           f"background:{tone if inverse else 'transparent'};color:{'white' if inverse else tone};font-weight:600'>{text}</span>"

def _card(title: str, body: str):
    st.markdown(
        f"""
        <div style="border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;margin:.25rem 0;background:#fff;">
          <div style="font-weight:600;margin-bottom:6px;">{title}</div>
          <div style="line-height:1.5">{body}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ───────── Main page ─────────
def render_patient_chart():
    _init_state(); _enter_page("Patient Chart"); _render_sidebar(__file__); _show_back("← Back")

    # If coming from Track Board, prefer that data
    track_df = st.session_state.get("trackboard_df")
    st.session_state.setdefault("selected_patient_id", "CP-1000")

    # Patient selector (robust)
    st.caption("Search / select a patient (type ID or name to filter)")
    options, id_list = [], []
    if track_df is not None and not track_df.empty:
        tmp = track_df.copy()
        if "PatientID" in tmp.columns: tmp["PatientID"] = tmp["PatientID"].astype(str)
        if "Patient" in tmp.columns:   tmp["Patient"]   = tmp["Patient"].astype(str).fillna("—")
        tmp = tmp[tmp["PatientID"].notna() & (tmp["PatientID"] != "None")]
        for _, r in tmp.iterrows():
            pid = str(r["PatientID"]); nm = str(r.get("Patient","—"))
            options.append(f"{pid} — {nm}"); id_list.append(pid)
    else:
        for pid, d in FIXED_PROFILES.items():
            options.append(f"{pid} — {d.get('Patient','')}"); id_list.append(pid)

    if not options:
        st.error("No patients available."); return

    current_id = st.session_state.get("selected_patient_id")
    if not current_id or str(current_id) not in id_list:
        current_id = id_list[0]; st.session_state["selected_patient_id"] = current_id
    try: default_index = id_list.index(str(current_id))
    except ValueError: default_index = 0

    selected_label = st.selectbox("", options, index=default_index)
    sel_id = selected_label.split(" — ")[0].strip()
    st.session_state["selected_patient_id"] = sel_id

    # Build the patient dict
    if track_df is not None and not track_df.empty:
        row = track_df.loc[track_df["PatientID"] == sel_id]
        if row.empty: row = track_df.iloc[[0]]
        patient = make_patient_from_row(row.iloc[0].to_dict())
        name = str(row.iloc[0].get("Patient","—"))
    else:
        base = dict(FIXED_PROFILES.get(sel_id, next(iter(FIXED_PROFILES.values()))))
        name = base.get("Patient","—")
        patient = make_patient_from_row({
            "PatientID": base.get("MRN"), "Patient": base.get("Patient"),
            "Age": base.get("Age"), "Sex": base.get("Sex"),
            "Arrival": base.get("Arrival"), "OnsetMin": base.get("OnsetMin"),
            "CC": base.get("CC"),
            "SBP": base.get("SBP"), "DBP": 80, "HR": base.get("HR"), "RR": 18,
            "SpO₂": base.get("SpO₂"), "ECG": base.get("ECG"),
            "hs-cTn (ng/L)": base.get("hs-cTn (ng/L)"),
            "TempC": base.get("TempC", 37.0), "Temp":  f"{base.get('TempC', 37.0):.1f}°C",
        })

    # derive Celsius for the UI (safe)
    try:
        v = patient["vitals"]
        tempC = float(v["TempC"]) if v.get("TempC") not in (None,"","None") else None
    except Exception:
        tempC = None

    # Header + BASIC INFO (unchanged structure)
    view = patient_header(name, patient["mrn"], default_view="Clinicians")
    _ = general_info_block({
        "Patient": name, "MRN": patient["mrn"],
        "Age": patient["age"], "Sex": patient["sex"], "ESI": "",
        "HR": patient["vitals"]["HR"], "SBP": patient["vitals"]["BP"].split("/")[0],
        "SpO₂": patient["vitals"]["SpO2"],
        "TempC": "" if tempC is None else f"{tempC:.1f}", "Temp":  "" if tempC is None else f"{tempC:.1f}°C",
        "ECG": "Normal" if not patient["risk_inputs"]["ecg_abnormal"] else "Abnormal",
        "hs-cTn (ng/L)": patient["risk_inputs"].get("troponin"),
        "OnsetMin": patient["data_quality"]["time_from_onset_min"],
        "Arrival": patient["arrival_mode"], "DOB": "—", "CC": patient["chief_complaint"]
    })

    # Compute model summary
    summary = compute_summary(patient)
    triage = triage_level_from_summary(summary)
    default_disp = disposition_from_summary(summary)

    # ======== TOP: AI RECOMMENDATIONS (Triage -> Disposition -> Next steps) ========
    st.markdown("### AI Recommendations")
    with st.container(border=True):
        # Row kept balanced 1:1
        cA, cB = st.columns(2)
        # TRIAGE (T1–T5)
        with cA:
            t_color = {"T1":"#DC2626","T2":"#F97316","T3":"#F59E0B","T4":"#22C55E","T5":"#10B981"}[triage["code"]]
            st.markdown(
                "<div style='font-weight:700;margin-bottom:6px'>Triage recommendation</div>"
                + f"<div style='font-size:18px'>{_pill('TriageGO: ' + triage['code'], t_color, True)} "
                + f"<span style='color:#6b7280'>{triage['label']}</span></div>",
                unsafe_allow_html=True
            )
            st.caption(triage["desc"])
        # DISPOSITION (4 tags)
        with cB:
            st.markdown("<div style='font-weight:700;margin-bottom:6px'>Disposition recommendation</div>", unsafe_allow_html=True)
            opts = ["Confirm/Admit","Observe","Consult","Defer/Discharge"]
            try:
                choice = st.segmented_control("disp", opts, selection_mode="single",
                                              default=default_disp, label_visibility="collapsed")
            except Exception:
                choice = st.radio("disp", opts, index=opts.index(default_disp), horizontal=True, label_visibility="collapsed")
            st.session_state["disp_choice"] = choice

        # NEXT STEPS
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        cols = st.columns(2)
        suggested = summary["steps"]
        half = (len(suggested)+1)//2
        with cols[0]:
            st.markdown("**Next steps (AI):**")
            for s in suggested[:half]: st.markdown(f"- {s}")
        with cols[1]:
            st.markdown("&nbsp;")
            for s in suggested[half:]: st.markdown(f"- {s}")
        st.text_area("Add/adjust next steps (free text)", placeholder="hs-Troponin now; repeat per rule-out; observe with continuous vitals; reassess pain…", key="next_steps_free")

    # ======== RISK (clear & compact) ========
    st.markdown("### Risk Estimate")
    c1, c2, c3 = st.columns([0.35, 0.35, 0.30])
    with c1:
        _card("Point risk", f"<div style='font-size:22px;font-weight:700;'>{round(100*summary['base'])}%</div>")
    with c2:
        lo, hi = round(100*summary["lo"]), round(100*summary["hi"])
        width = round(100*summary["width"])
        _card("Interval", f"<div style='font-size:15px'><b>{lo}% – {hi}%</b> (width {width}%)</div>"
                          "<div style='color:#6b7280;margin-top:4px'>Point = best estimate; Interval = uncertainty range.</div>")
    with c3:
        risk_badge(100*summary["base"], palette=TRACK_BOARD_PALETTE)
    st.progress(min(0.99, summary["base"]))  # quick glance

    # ======== CONFIDENCE & UNCERTAINTY (with composition + two layers) ========
    st.markdown("### Confidence & Uncertainty")
    conf_col, det_col = st.columns([0.30, 0.70])
    alea_pct, epis_pct, conf_score, conf_tier = decompose_uncertainty(summary, patient)

    with conf_col:
        _card("Confidence score",
              f"<div style='font-size:22px;font-weight:700;'>{pct_tier(conf_score, conf_tier)}</div>"
              "<div style='color:#6b7280;margin-top:4px'>Computed internally; shown in tiers for readability.</div>")
        bars = f"""
        <div style="display:flex;gap:.5rem;align-items:center;margin-top:.5rem">
          <div style="min-width:92px">Aleatoric</div>
          <div style="height:10px;background:#e5e7eb;border-radius:999px;flex:1;position:relative;">
            <div style="height:10px;border-radius:999px;background:#60A5FA;width:{int(alea_pct)}%"></div>
          </div><div style="min-width:42px;text-align:right">{int(alea_pct)}%</div>
        </div>
        <div style="display:flex;gap:.5rem;align-items:center;margin-top:.4rem">
          <div style="min-width:92px">Epistemic</div>
          <div style="height:10px;background:#e5e7eb;border-radius:999px;flex:1;position:relative;">
            <div style="height:10px;border-radius:999px;background:#A78BFA;width:{int(epis_pct)}%"></div>
          </div><div style="min-width:42px;text-align:right">{int(epis_pct)}%</div>
        </div>
        """
        _card("Uncertainty composition", bars)

    with det_col:
        # Clinical reasoning layer
        _card("Clinical reasoning (why high/low risk?)",
              "<b>Risk stratification</b>: combines simple clinical rule + model estimate.<br/>"
              f"• Clinical rule: HEART-like score — <i>illustrative</i>.<br/>"
              f"• Model estimate: {round(summary['base']*100)}% "
              f"({round(summary['lo']*100)}–{round(summary['hi']*100)}%)<br/>"
              f"<b>Contributing factors</b>: {', '.join(summary['drivers'])}.")
        # Model reasoning layer
        sens_html = sens_table()
        _card("Model reliability (how reliable right now?)",
              "• Data completeness: "
              f"<b>{'High' if 'troponin' not in patient['data_quality']['missing'] else 'Medium/Low'}</b><br/>"
              f"• Model familiarity: <b>{'Lower' if patient['data_quality'].get('ood') else 'Typical'}</b><br/>"
              "• Prediction stability: <b>High</b> (small input changes → small output changes)."
              f"<div style='margin-top:.25rem'>{sens_html}</div>")

    # Role-specific explanation blocks (kept lean)
    if view == "Patient":
        render_patient_panel(summary)
    else:
        render_clinician_panel(summary)

    # ======== AGREEMENT / OVERRIDE ========
    st.divider()
    st.markdown("#### ✅ Document agreement / override")
    st.radio("Do you agree with the AI suggestion?", ["Yes", "No / Override"], horizontal=True, key="agree_radio")
    st.text_area("Notes (optional)", placeholder="Rationale, serial lab plan, shared decision details…", height=120)
    st.button("Save to audit log", type="primary")


if __name__ == "__main__":
    render_patient_chart()
else:
    render_patient_chart()