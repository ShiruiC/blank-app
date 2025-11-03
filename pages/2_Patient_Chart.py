# pages/2_Patient_Chart.py
import streamlit as st
from html import escape

# ── Uncertainty side-panel (pure Streamlit, no iframe)
def _render_uncertainty_panel():
    opened = bool(st.session_state.get("why_open", False))
    scope = st.session_state.get("why_scope", "triage")

    # Harmonize fonts with the main page
    st.markdown("""
    <style>
      .ua-panel * { font-family: inherit !important; }
      .ua-panel h3, .ua-panel .stHeading, .ua-panel .stSubheader { font-weight: 800; }
      .ua-panel .stMetric label { font-weight: 600; }
      .ua-panel .stProgress > div > div { transition: width .25s ease; }
      .ua-panel .ua-handle { font-size: 18px; font-weight: 900; }
    </style>
    """, unsafe_allow_html=True)

    # Handle button (always visible in the slim right column)
    handle_label = "≪" if opened else "≫"
    if opened:
        if st.button(handle_label, key="ua_collapse", help="Hide details", type="secondary"):
            _close_drawer()
            st.stop()
    else:
        if st.button(handle_label, key="ua_expand", help="Show uncertainty details", type="secondary"):
            _open_drawer(st.session_state.get("why_scope","triage"),
                         st.session_state.get("why_triage_label",""),
                         st.session_state.get("why_dispo_label",""))
            st.stop()

    if not opened:
        return  # nothing else in the slim state

    # Panel content
    with st.container(border=True):
        st.markdown('<div class="ua-panel">', unsafe_allow_html=True)

        summary  = st.session_state.get("_ua_summary")
        patient  = st.session_state.get("_ua_patient")
        tri_lbl  = st.session_state.get("why_triage_label","")
        disp_lbl = st.session_state.get("why_dispo_label","")
        if not summary or not patient:
            st.info("No uncertainty data available.")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        from components.patient_view import decompose_uncertainty  # local import to avoid circulars
        alea_pct, epis_pct, conf_score, conf_tier = decompose_uncertainty(summary, patient)
        conf_pct = int(round(conf_score*100))

        st.subheader("Confidence & Uncertainty")
        st.metric("Confidence score", f"{conf_pct}%", conf_tier)

        st.caption("Uncertainty composition")
        st.progress(min(100, int(alea_pct)), text="Aleatoric")
        st.progress(min(100, int(epis_pct)), text="Epistemic")

        st.divider()

        # Scope-specific details
        if scope == "triage":
            base = int(round(100*summary.get("base",0)))
            lo, hi = int(round(100*summary.get("lo",0))), int(round(100*summary.get("hi",0)))
            st.subheader(f"Why Triage: {tri_lbl}")
            st.write(f"**Risk & interval**: point {base}% • range {lo}%–{hi}%")

            drivers = summary.get("drivers", []) or []
            if drivers:
                st.caption("Contributing factors")
                for d in drivers: st.write(f"• {d}")

        elif scope == "disposition":
            st.subheader(f"Why Disposition: {disp_lbl}")
            dq = patient.get("data_quality",{}) or {}
            missing = dq.get("missing", [])
            wid_pct = int(round(100*summary.get("width",0.0)))

            st.write("**Data completeness**")
            if missing:
                for m in missing: st.write(f"• Missing: {m}")
            else:
                st.write("• No missing key data")

            st.write(f"**Risk interval width** ≈ {wid_pct}% — narrower ranges mean clearer split between Admit vs Observe.")
            st.write("**Model familiarity**: " + ("Outside typical training range" if dq.get("ood") else "Typical for training distribution"))

        else:
            st.subheader("Why These Next Steps")
            st.write("**Prediction stability (local sensitivity)**")
            st.table({
                "Vital": ["HR","SBP","SpO₂"],
                "Change": ["+5%","+5%","±5%"],
                "Output ↑": ["+2%","+0%","+3%"],
                "Output ↓": ["-1%","+1%","—"],
            })

        st.divider()
        st.caption("Aleatoric = patient variability • Epistemic = model/data limits")
        st.markdown('</div>', unsafe_allow_html=True)

# ── Drawer helpers (no iframe, no JS)
def _drawer_cols(opened: bool):
    """Return (left, right) columns. When closed, right is a slim handle."""
    if opened:
        return st.columns([0.68, 0.32], vertical_alignment="top")
    else:
        # keep a tiny right column for the handle button
        return st.columns([0.98, 0.02], vertical_alignment="center")

def _open_drawer(scope: str, triage_label: str = "", dispo_label: str = ""):
    st.session_state["why_open"] = True
    st.session_state["why_scope"] = scope
    if triage_label: st.session_state["why_triage_label"] = triage_label
    if dispo_label:  st.session_state["why_dispo_label"]  = dispo_label

def _close_drawer():
    st.session_state["why_open"] = False

# ── Safe utils import (optional)
def _no_op(*a, **k): return None
try:
    from utils import init_state as _init_state, enter_page as _enter_page, show_back_top_right as _show_back, render_sidebar as _render_sidebar
except Exception:
    _init_state = _no_op; _enter_page = _no_op; _show_back = _no_op; _render_sidebar = _no_op

from components.patient_view import (
    make_patient_from_row, compute_summary,
    triage_level_from_summary, disposition_from_summary,
)

# ── Demo data
FIXED_PROFILES = {
    "CP-1000": {"MRN":"CP-1000","Patient":"Weber, Charlotte","Age":27,"Sex":"Female","ESI":2,
                "HR":78,"SBP":164,"SpO₂":96,"TempC":37.0,"ECG":"Normal","hs-cTn (ng/L)":None,
                "OnsetMin":90,"CC":"Chest pain at rest, mild SOB.","Arrival":"10:39"},
    "CP-1001": {"MRN":"CP-1001","Patient":"Green, Gary","Age":59,"Sex":"Male","ESI":3,
                "HR":102,"SBP":138,"SpO₂":95,"TempC":37.0,"ECG":"Borderline","hs-cTn (ng/L)":0.03,
                "OnsetMin":60,"CC":"Intermittent chest tightness with exertion.","Arrival":"11:07"},
}

# ── Drawer state
st.session_state.setdefault("why_open", False)
st.session_state.setdefault("why_scope", "triage")
st.session_state.setdefault("why_triage_label", "")
st.session_state.setdefault("why_dispo_label", "")

TRIAGE_LEVELS = {
    "T1": ("Immediate",     "Highest probability for intensive care, emergency procedure, or mortality.", "#DC2626"),
    "T2": ("Very Urgent",   "Elevated probability for intensive care, emergency procedure, or mortality.", "#F97316"),
    "T3": ("Urgent",        "Moderate probability of hospital admission or very low probability of intensive care, emergency procedure, or mortality.", "#F59E0B"),
    "T4": ("Less Urgent",   "Low probability of hospital admission.", "#22C55E"),
    "T5": ("Non-Urgent",    "Fast turnaround and low probability of hospital admission.", "#10B981"),
}
DISP_LEVELS = {
    "Confirm/Admit":  "Admit or confirm acute management plan — likely inpatient treatment.",
    "Observe":        "Monitor in observation unit — reassess after a short period.",
    "Consult":        "Seek specialist input before final decision.",
    "Defer/Discharge":"Safe for discharge — provide safety-net and follow-up.",
}

def _chip(text, bg="#3B82F6"):
    return f"<span style='border-radius:999px;padding:.25rem .6rem;background:{bg};color:#fff;margin-right:.35rem;margin-bottom:.35rem;display:inline-block;font-weight:600'>{escape(text)}</span>"

def _pill(text, tone="#2563EB", inverse=False):
    return (
        "<span style='border-radius:999px;padding:.25rem .6rem;border:1px solid {tone};"
        "margin:.15rem .25rem .15rem 0;display:inline-block;"
        "background:{bg};color:{fg};font-weight:700'>{t}</span>"
    ).format(tone=tone, bg=(tone if inverse else "transparent"), fg=("white" if inverse else tone), t=text)

def patient_header(name: str, mrn: str, ui: dict) -> None:
    left, right = st.columns([0.70, 0.30], vertical_alignment="center")
    with left:
        st.markdown(f"## {name}")
        st.caption(f"MRN: {mrn}")
    with right:
        st.radio("View", ["Patient", "Clinicians"], horizontal=True, index=1, label_visibility="collapsed")

def render_dispo_list(ai_dispo: str):
    for k, desc in DISP_LEVELS.items():
        is_ai = (k == ai_dispo)
        st.markdown(
            f"""
            <div style="border:1px solid {'#3B82F6' if is_ai else '#e5e7eb'};border-radius:10px;padding:10px 12px;margin:.35rem 0;
                        background:{'#EEF2FF' if is_ai else 'white'};display:flex;gap:.6rem;align-items:flex-start">
              <div style="min-width:120px;font-weight:700">{escape(k)}</div>
              <div style="color:#374151">{escape(desc)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

def render_patient_chart():
    _init_state(); _enter_page("Patient Chart"); _render_sidebar(__file__); _show_back("← Back")

    # Patient picker
    track_df = st.session_state.get("trackboard_df")
    st.session_state.setdefault("selected_patient_id", "CP-1000")
    st.caption("Search / select a patient (type ID or name to filter)")

    options, ids = [], []
    if track_df is not None and not track_df.empty:
        tmp = track_df.copy()
        if "PatientID" in tmp.columns: tmp["PatientID"] = tmp["PatientID"].astype(str)
        if "Patient" in tmp.columns:   tmp["Patient"]   = tmp["Patient"].astype(str).fillna("—")
        tmp = tmp[tmp["PatientID"].notna() & (tmp["PatientID"] != "None")]
        for _, r in tmp.iterrows():
            pid = str(r["PatientID"]); nm = str(r.get("Patient","—"))
            options.append(f"{pid} — {nm}"); ids.append(pid)
    else:
        for pid, d in FIXED_PROFILES.items():
            options.append(f"{pid} — {d.get('Patient','')}"); ids.append(pid)

    if not options:
        st.error("No patients available."); return

    cur = st.session_state.get("selected_patient_id")
    if not cur or cur not in ids:
        cur = ids[0]; st.session_state["selected_patient_id"] = cur
    sel_label = st.selectbox("", options, index=ids.index(cur))
    sel_id = sel_label.split(" — ")[0].strip()
    st.session_state["selected_patient_id"] = sel_id

    # Build patient
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
            "TempC": base.get("TempC", 37.0),
        })

    # Prepare UI dict
    ui = {
        "Patient": name, "Age": patient["age"], "Sex": patient["sex"],
        "OnsetMin": patient["data_quality"]["time_from_onset_min"],
        "Arrival": patient["arrival_mode"], "CC": patient["chief_complaint"],
        "HR": patient["vitals"]["HR"], "SBP": patient["vitals"]["BP"].split("/")[0],
        "SpO₂": patient["vitals"]["SpO2"], "TempC": f"{patient['vitals'].get('TempC', 37.0):.1f}",
        "ECG": "Normal" if not patient["risk_inputs"]["ecg_abnormal"] else "Abnormal",
        "hs-cTn (ng/L)": patient["risk_inputs"].get("troponin"),
    }

    # Header + basic info
    patient_header(name, patient["mrn"], ui)

    # Shared UI styles (labels, muted text, line, badge)
    st.markdown("""
    <style>
      .ua-infocard{background:#f3f4f6;border:1px solid #e5e7eb;border-radius:10px;padding:10px 12px;margin-top:.25rem}
      .ua-kv{display:grid;grid-template-columns:120px 1fr;row-gap:6px;column-gap:10px;font-size:14px;color:#374151}
      .ua-kv b{color:#111827}
      .ua-line{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
      .ua-label{font-weight:800}
      .ua-muted{color:#6b7280;font-size:13px}
      .ua-badge{border-radius:999px;background:#EEF2FF;color:#1e40af;border:1px solid #93C5FD;padding:.2rem .6rem;font-weight:700}
      .ua-slim-hr{border:none;border-top:1px solid #e5e7eb;margin:0}
    </style>
    """, unsafe_allow_html=True)

    # — Grey info card (read-only)
    st.markdown(
        f"""
        <div class="ua-infocard">
          <div class="ua-kv">
            <div><b>Age</b></div><div>{ui['Age']}</div>
            <div><b>Sex</b></div><div>{escape(str(ui['Sex']))}</div>
            <div><b>Arrival</b></div><div>{escape(str(ui['Arrival']))}</div>
            <div><b>Onset (min)</b></div><div>{ui['OnsetMin']}</div>
            <div><b>Chief complaint</b></div><div>{escape(str(ui['CC']))}</div>
            <div><b>ECG</b></div><div>{escape(str(ui['ECG']))}</div>
            <div><b>hs-cTn (ng/L)</b></div><div>{'—' if ui['hs-cTn (ng/L)'] is None else ui['hs-cTn (ng/L)']}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Vitals tabs
    tabs = st.tabs(["Current", "History", "Results"])
    with tabs[0]:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("HR", f"{ui['HR']} bpm"); c2.metric("SBP", f"{ui['SBP']} mmHg")
        c3.metric("SpO₂", f"{ui['SpO₂']} %"); c4.metric("Temp", f"{ui['TempC']} °C")
        st.caption("Updated • stability/variance inform aleatoric uncertainty")
    with tabs[1]:
        st.caption("No prior ED visits in this demo.")
    with tabs[2]:
        st.caption("No lab/imaging results beyond vitals in this demo.")

    # Compute + defaults
    summary = compute_summary(patient)
    st.session_state["_ua_summary"] = summary
    st.session_state["_ua_patient"] = patient
    tri = triage_level_from_summary(summary)
    t_code, t_label, t_desc = tri["code"], tri["label"], tri["desc"]
    color_map = {"T1":"#DC2626","T2":"#F97316","T3":"#F59E0B","T4":"#22C55E","T5":"#10B981"}
    t_color = color_map[t_code]
    disp_default = disposition_from_summary(summary)

    # ============ LAYOUT WITH RIGHT DRAWER ============
    left_col, right_col = _drawer_cols(bool(st.session_state.get("why_open", False)))

    # LEFT COLUMN CONTENT
    with left_col:
        st.markdown("### AI Recommendations")

        # TRIAGE — compact row + ❓
        r1, r2 = st.columns([0.86, 0.14])
        with r1:
            st.markdown(
                f"""
                <div class="ua-line">
                  <span class="ua-label">AI Triage Recommendation:</span>
                  {_pill(t_code, tone=t_color, inverse=True)}
                  <div><b>{escape(t_label)}</b><br><span class="ua-muted">{escape(t_desc)}</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with r2:
            st.button("❓", key="why_triage_btn", help="How confident is the AI triage recommendation?",
                      on_click=_open_drawer, args=("triage", f"{t_code} — {t_label}", ""))

        # All triage levels (AI highlighted)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        for k in ["T1","T2","T3","T4","T5"]:
            lab, desc, colhex = TRIAGE_LEVELS[k]
            is_ai = (k == t_code)
            st.markdown(
                f"""
                <div style="display:flex;align-items:flex-start;gap:.6rem;margin:.35rem 0;
                            border:1px solid {'#3B82F6' if is_ai else '#e5e7eb'};
                            background:{'#EEF2FF' if is_ai else 'white'};
                            border-radius:10px;padding:10px 12px;">
                  <div style="margin-top:1px">{_pill(k, tone=colhex, inverse=True)}</div>
                  <div style="flex:1">
                    <b>{escape(lab)}</b><br>
                    <span style='color:#6b7280;font-size:13px'>{escape(desc)}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Clinician TRIAGE
        left, right = st.columns([0.18, 0.82], vertical_alignment="center")
        with left:
            st.markdown("<div class='ua-label'>Clinician Triage</div>", unsafe_allow_html=True)
        with right:
            tri_opts = ["T1","T2","T3","T4","T5"]
            try:
                st.segmented_control("tri_final", tri_opts, selection_mode="single",
                                     default=t_code, label="", label_visibility="collapsed")
            except Exception:
                st.radio(label="", options=tri_opts, index=tri_opts.index(t_code),
                         horizontal=True, key="tri_final_radio", label_visibility="collapsed")
        st.markdown('<div style="height:0;margin-top:-12px;"><hr class="ua-slim-hr"></div>', unsafe_allow_html=True)

        # DISPOSITION — compact row + ❓
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        dleft, dright = st.columns([0.86, 0.14])
        with dleft:
            st.markdown(
                f"""
                <div class="ua-line">
                  <span class="ua-label">AI Disposition Recommendation:</span>
                  <span class="ua-badge">{escape(disp_default)}</span>
                  <span class="ua-muted">{escape(DISP_LEVELS[disp_default])}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        with dright:
            st.button("❓", key="why_dispo_btn", help="How confident is the AI disposition recommendation?",
                      on_click=_open_drawer, args=("disposition","",str(disp_default)))

        # Full dispositions list (AI highlighted)
        render_dispo_list(disp_default)

        # Clinician DISPOSITION
        left, right = st.columns([0.18, 0.82], vertical_alignment="center")
        with left:
            st.markdown("<div class='ua-label'>Clinician Disposition</div>", unsafe_allow_html=True)
        with right:
            disp_opts = list(DISP_LEVELS.keys())
            try:
                st.segmented_control("disp_final", disp_opts, selection_mode="single",
                                     default=disp_default, label="", label_visibility="collapsed")
            except Exception:
                st.radio(label="", options=disp_opts, index=disp_opts.index(disp_default),
                         horizontal=True, key="disp_final_radio", label_visibility="collapsed")
        st.markdown('<div style="height:0;margin-top:-20px;"><hr class="ua-slim-hr"></div>', unsafe_allow_html=True)

        # NEXT STEPS — ❓ + checkboxes
        s1, s2 = st.columns([0.86, 0.14])
        with s1:
            st.markdown("**AI Next-Steps Suggestions**")
        with s2:
            st.button("❓", key="why_steps_btn", help="How confident is the AI next-steps suggestion?",
                      on_click=_open_drawer, args=("steps","",""))

        ai_steps = list(summary.get("steps") or [
            "Possible NSTEMI — obtain hs-Troponin now",
            "Repeat hs-Troponin per rule-out protocol",
            "Continuous ECG & vitals",
            "Reassess chest pain in 1–2 h",
        ])
        for i, step in enumerate(ai_steps):
            c = st.columns([0.06, 0.94])
            with c[0]:
                st.checkbox("", key=f"agree_step_{i}")
            with c[1]:
                st.markdown(_chip(step), unsafe_allow_html=True)

        # Notes + Save
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        st.text_area("Notes (optional)", placeholder="Rationale, serial lab plan, shared decision…", height=140, key="clin_notes")
        st.button("Save", type="primary", key="save_btn")

    # RIGHT COLUMN CONTENT (IMPORTANT: not nested under left_col)
    with right_col:
        _render_uncertainty_panel()

if __name__ == "__main__":
    render_patient_chart()
else:
    render_patient_chart()