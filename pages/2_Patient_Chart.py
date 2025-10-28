# pages/2_Patient_Chart.py
import re
import streamlit as st
from datetime import datetime
from html import escape

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
    render_confidence_uncertainty,  # 仍可用作“一行渲染”的替代
    render_patient_panel, render_clinician_panel,
    decompose_uncertainty, sens_table, pct_tier,
    pattern_similarity, alea_reason, epi_reason,
    band_from_risk,  # 导入组件中的阈值函数（0..1）
)

# ───────── Demo fallback profiles ─────────
FIXED_PROFILES = {
    "CP-1000": {"MRN":"CP-1000","Patient":"Weber, Charlotte","Age":27,"Sex":"Female","ESI":2,
                "HR":78,"SBP":164,"SpO₂":96,"TempC": 37.0, "Temp": "37.0°C","ECG":"Normal","hs-cTn (ng/L)":None,
                "OnsetMin":90,"CC":"Chest pain at rest, mild SOB.","Arrival":"10:39", "DOB":"—"},
    "CP-1001": {"MRN":"CP-1001","Patient":"Green, Gary","Age":59,"Sex":"Male","ESI":3,
                "HR":102,"SBP":138,"SpO₂":95,"TempC": 37.0, "Temp": "37.0°C","ECG":"Borderline","hs-cTn (ng/L)":0.03,
                "OnsetMin":60,"CC":"Intermittent chest tightness with exertion.","Arrival":"11:07","DOB":"—"},
}

# ───────── Small helpers ─────────
def risk_band_pct(point_risk_pct: float) -> str:
    """把百分数输入(0..100)转换成组件阈值函数(0..1)"""
    try:
        return band_from_risk(float(point_risk_pct) / 100.0)
    except Exception:
        return "—"

TRACK_BOARD_PALETTE = {"High":"#DC2626","Moderate":"#F59E0B","Low-Moderate":"#22C55E","Low":"#16A34A"}

def risk_badge(point_risk_pct: float, palette=None):
    band = risk_band_pct(point_risk_pct)
    colors = {"High":"#DC2626","Moderate":"#F59E0B","Low-Moderate":"#22C55E","Low":"#16A34A"}
    if palette: colors.update(palette)
    color = colors.get(band, "#9CA3AF")
    return (
        "<span style='padding:.15rem .55rem;border-radius:999px;"
        f"background:{color};color:white;font-weight:700'>{band}</span>"
    )

def _pill(text, tone="#2563EB", inverse=False):
    return (
        "<span style='border-radius:999px;padding:.25rem .6rem;border:1px solid {tone};"
        "margin:.15rem .25rem .15rem 0;display:inline-block;"
        "background:{bg};color:{fg};font-weight:600'>{t}</span>"
    ).format(tone=tone, bg=(tone if inverse else "transparent"), fg=("white" if inverse else tone), t=text)

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

# ───────── Header + basic info ─────────
def patient_header(name: str, mrn: str, default_view="Clinicians"):
    left, right = st.columns([0.70, 0.30], vertical_alignment="center")
    with left:
        st.markdown(f"## {name}")
        st.caption(f"MRN: {mrn}")
    with right:
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

# ───────── Main page ─────────
def render_patient_chart():
    _init_state(); _enter_page("Patient Chart"); _render_sidebar(__file__); _show_back("← Back")

    # Selector
    track_df = st.session_state.get("trackboard_df")
    st.session_state.setdefault("selected_patient_id", "CP-1000")

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

    # Build patient dict
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

    # Header + BASIC INFO
    view = patient_header(name, patient["mrn"], default_view="Clinicians")
    _ = general_info_block({
        "Patient": name, "MRN": patient["mrn"],
        "Age": patient["age"], "Sex": patient["sex"], "ESI": "",
        "HR": patient["vitals"]["HR"], "SBP": patient["vitals"]["BP"].split("/")[0],
        "SpO₂": patient["vitals"]["SpO2"],
        "TempC": f"{patient['vitals'].get('TempC', 37.0):.1f}",
        "Temp":  f"{patient['vitals'].get('TempC', 37.0):.1f}°C",
        "ECG": "Normal" if not patient["risk_inputs"]["ecg_abnormal"] else "Abnormal",
        "hs-cTn (ng/L)": patient["risk_inputs"].get("troponin"),
        "OnsetMin": patient["data_quality"]["time_from_onset_min"],
        "Arrival": patient["arrival_mode"], "DOB": "—", "CC": patient["chief_complaint"]
    })

    # Compute summary
    summary = compute_summary(patient)
    triage = triage_level_from_summary(summary)
    default_disp = disposition_from_summary(summary)

    # ======== AI RECOMMENDATIONS ========
    st.markdown("### AI Recommendations")
    st.caption("Decision support only — clinicians should exercise their own clinical judgment based on available information.")

    try:
        box = st.container(border=True)
    except TypeError:
        box = st.container()

    with box:
        # ---------- TRIAGE ----------
        st.markdown("**Triage Recommendation**")
        triage_levels = {
            "T1: Immediate": "Highest probability for intensive care, emergency procedure, or mortality.",
            "T2: Very Urgent": "Elevated probability for intensive care, emergency procedure, or mortality.",
            "T3: Urgent": "Moderate probability of hospital admission or very low probability of intensive care, emergency procedure, or mortality.",
            "T4: Less Urgent": "Low probability of hospital admission.",
            "T5: Non-Urgent": "Fast turnaround and low probability of hospital admission."
        }
        triage_colors = {"T1":"#DC2626","T2":"#F97316","T3":"#F59E0B","T4":"#22C55E","T5":"#10B981"}
        tri_opts = ["T1","T2","T3","T4","T5"]
        try:
            tri_sel = st.segmented_control("tri_palette", tri_opts, selection_mode="single",
                                           default=triage.get("code","T3") if isinstance(triage, dict) else "T3",
                                           label_visibility="collapsed")
        except Exception:
            tri_sel = st.radio("tri_palette", tri_opts, index=tri_opts.index(triage.get("code","T3") if isinstance(triage, dict) else "T3"),
                               horizontal=True, label_visibility="collapsed")
        selected_key = next((k for k in triage_levels if k.startswith(f"{tri_sel}:")), "T3: Urgent")
        sel_code = selected_key.split(":")[0]; sel_label = selected_key.split(":")[1].strip()
        sel_color = triage_colors.get(sel_code, "#9CA3AF")
        st.markdown(f"<div style='margin-top:6px'>{_pill(f'Acuity Level: {sel_code} — {sel_label}', tone=sel_color, inverse=True)}</div>", unsafe_allow_html=True)
        st.markdown("<div style='color:#6b7280;font-size:13px;margin-top:6px'>AI Confidence: <b>High</b></div>", unsafe_allow_html=True)

        left, right = st.columns(2)
        all_keys = list(triage_levels.keys())
        def _mini_item(k: str):
            c = triage_colors.get(k.split(':')[0], "#9CA3AF")
            return (
                f"<div style='display:flex;align-items:center;margin:.35rem 0'>"
                f"{_pill(k.split(':')[0], tone=c, inverse=True)}"
                f"<div style='margin-left:.6rem'><b>{k.split(':')[1].strip()}</b><br>"
                f"<span style='color:#6b7280;font-size:12px'>{triage_levels[k]}</span></div></div>"
            )
        with left:
            st.markdown(_mini_item(all_keys[0]) + _mini_item(all_keys[2]) + _mini_item(all_keys[4]), unsafe_allow_html=True)
        with right:
            st.markdown(_mini_item(all_keys[1]) + _mini_item(all_keys[3]), unsafe_allow_html=True)

        st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
        st.checkbox("Adopt this acuity level", key="adopt_triage", value=True)

        st.markdown("<hr style='margin-top:-10px;margin-bottom:0;border:none;border-top:1px solid #e5e7eb'/>", unsafe_allow_html=True)

        # ---------- DISPOSITION ----------
        st.markdown("**Disposition Recommendation**")
        disp_levels = {
            "Confirm/Admit": "Admit or confirm acute management plan — likely inpatient treatment.",
            "Observe": "Monitor in observation unit — reassess after a short period.",
            "Consult": "Seek specialist input before final decision.",
            "Defer/Discharge": "Safe for discharge — provide safety-net and follow-up."
        }
        disp_options = list(disp_levels.keys())
        default_disp_idx = disp_options.index(default_disp) if default_disp in disp_options else 0
        try:
            disp_selected = st.segmented_control("disp", disp_options, selection_mode="single",
                                                 default=disp_options[default_disp_idx], label_visibility="collapsed")
        except Exception:
            disp_selected = st.radio("disp", disp_options, index=default_disp_idx, horizontal=True, label_visibility="collapsed")
        st.markdown(f"<div style='margin-top:6px;color:#6b7280'>{disp_levels[disp_selected]}</div>", unsafe_allow_html=True)
        st.markdown("<div style='color:#6b7280;font-size:13px;margin-top:4px'>AI Confidence: <b>Moderate</b></div>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
        st.checkbox("Adopt this disposition", key="adopt_disp", value=True)

        st.markdown("<hr style='margin-top:-10px;margin-bottom:0;border:none;border-top:1px solid #e5e7eb'/>", unsafe_allow_html=True)

        # ---------- NEXT STEPS ----------
        st.markdown("**Next Steps Suggestion**")
        suggested = summary.get("steps", []) or [
            "Possible NSTEMI — obtain hs-Troponin now",
            "Repeat hs-Troponin per rule-out protocol",
            "Continuous ECG and vitals monitoring",
            "Reassess chest pain and risk factors in 1–2 h"
        ]
        bubble_html = "<div style='display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.4rem;'>"
        for s in suggested: bubble_html += _pill(s, tone="#3B82F6", inverse=True)
        bubble_html += "</div>"
        st.markdown(bubble_html, unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ======== CONFIDENCE & UNCERTAINTY ========
    def _badge(text: str, tone: str = "info") -> str:
        colors = {
            "success": ("#065f46", "#d1fae5"),
            "warn":    ("#92400e", "#fef3c7"),
            "danger":  ("#7f1d1d", "#fee2e2"),
            "info":    ("#1e40af", "#dbeafe"),
            "neutral": ("#374151", "#e5e7eb"),
        }
        fg, bg = colors.get(tone, colors["neutral"])
        return (
            f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;'
            f'font-weight:600;color:{fg};background:{bg};font-size:12px">{text}</span>'
        )

    def _tier_badge(tier: str) -> str:
            t = (tier or "").lower()
            if t == "high":
                return _badge("High", "success")
            if t == "medium":
                return _badge("Medium", "warn")
            return _badge("Low", "danger")

    def _info_icon(tooltip: str) -> str:
        tip = escape(tooltip or "", quote=True)  # 安全转义 & < > " '
        return f'<span title="{tip}" style="color:#6b7280;margin-left:6px">ℹ️</span>'

    def _h4(txt: str) -> str:
        return f'<div style="font-weight:700;font-size:15px;margin-bottom:.4rem">{txt}</div>'

    # 容器，方便独立定制样式（可选）
    st.markdown('<div id="cu-block">', unsafe_allow_html=True)

    with st.expander("Confidence & Uncertainty", expanded=False):
        # 关键数值
        alea_pct, epis_pct, conf_score, conf_tier = decompose_uncertainty(summary, patient)
        conf_pct = int(round(conf_score * 100))
        # 与顶层 Confidence 同向：High/Medium/Low
        unc_intensity = {"High":"High", "Medium":"Moderate", "Low":"Low"}.get(conf_tier, "Moderate")
        dom_type = "Aleatoric" if alea_pct >= epis_pct else "Epistemic"

        # 第一行：Confidence + Composition
        c1, c2 = st.columns([0.40, 0.60])
        with c1:
            st.markdown(
                _h4("Confidence score") +
                f"<div style='display:flex;align-items:baseline;gap:.5rem'>"
                f"<div style='font-size:24px;font-weight:800'>{conf_pct}%</div>"
                f"{_tier_badge(conf_tier)}</div>",
                unsafe_allow_html=True
            )
        with c2:
            bars = f"""
            <div style="display:flex;gap:.5rem;align-items:center;margin-top:.2rem">
              <div style="min-width:92px">Aleatoric</div>
              <div style="height:10px;background:#e5e7eb;border-radius:999px;flex:1;position:relative;">
                <div style="height:10px;border-radius:999px;background:#60A5FA;width:{int(alea_pct)}%"></div>
              </div><div style="min-width:42px;text-align:right">{int(alea_pct)}%</div>
            </div>
            <div style="color:#6b7280;margin:4px 0 10px 92px">patient variability</div>
            <div style="display:flex;gap:.5rem;align-items:center;">
              <div style="min-width:92px">Epistemic</div>
              <div style="height:10px;background:#e5e7eb;border-radius:999px;flex:1;position:relative;">
                <div style="height:10px;border-radius:999px;background:#A78BFA;width:{int(epis_pct)}%"></div>
              </div><div style="min-width:42px;text-align:right">{int(epis_pct)}%</div>
            </div>
            <div style="color:#6b7280;margin:4px 0 2px 92px">model limitation</div>
            """
            st.markdown(_h4("Uncertainty composition") + bars, unsafe_allow_html=True)

        st.divider()

        # Clinical reasoning
        st.markdown(
            "### Clinical reasoning layer "
            "<span style='color:#6b7280;font-weight:500;font-size:1rem'>— why this patient’s risk is high/low</span>",
            unsafe_allow_html=True
        )
        a1, a2 = st.columns(2)

        with a1:
            lo, hi = int(round(100*summary.get("lo",0))), int(round(100*summary.get("hi",0)))
            base   = int(round(100*summary.get("base",0)))
            chips = " ".join(_badge(str(d), "neutral") for d in (summary.get("drivers",[]) or [])) or "<span style='color:#6b7280'>—</span>"
            st.markdown(
                _h4("Risk") +
                f"<div style='display:flex;align-items:center;gap:.5rem'>"
                f"<div style='font-size:22px;font-weight:800'>{base}%</div>"
                f"{risk_badge(base, palette=TRACK_BOARD_PALETTE)}</div>"
                # 去掉“Point risk”行
                f"<div style='margin-top:.5rem'><b>Interval:</b> {lo}% – {hi}% "
                f"<span style='color:#6b7280'>(uncertainty range)</span></div>"
                f"<div style='margin-top:.5rem'><b>Contributing factors:</b> {chips}</div>",
                unsafe_allow_html=True
            )

        with a2:
            pr_sim = pattern_similarity(patient) or 0.0
            suspected = summary.get("suspected_condition", "ACS-like")
            st.markdown(
                _h4("Detected diagnosis / pattern recognition") +
                f"<div><b>Similarity:</b> <span style='font-weight:800'>{int(pr_sim*100)}%</span>"
                f" to <span style='font-weight:600'>{suspected}</span> cluster.</div>",
                unsafe_allow_html=True
            )

        clinical_reason_raw = alea_reason(summary, patient) if dom_type=="Aleatoric" else epi_reason(summary, patient)
        clinical_reason = re.sub(r'^\s*drivers:\s*', '', clinical_reason_raw, flags=re.I)
        st.markdown(
            f"<div style='margin-top:.75rem;padding:.6rem .8rem;border:1px solid #e5e7eb;border-radius:10px;background:#fafafa'>"
            f"<div style='display:flex;align-items:center;gap:.5rem;font-weight:700'>"
            f"Uncertainty type — {dom_type} "
            f"{_badge(unc_intensity, {'High':'danger','Moderate':'warn','Low':'success'}[unc_intensity])}"
            f"</div>"
            f"<div style='margin-top:.35rem;color:#374151'><b>Drivers:</b> {clinical_reason}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.divider()

        # Model reasoning
        st.markdown(
            "### Model reasoning layer "
            "<span style='color:#6b7280;font-weight:500;font-size:1rem'>— how much the model trusts itself</span>",
            unsafe_allow_html=True
        )
        m1, m2, m3 = st.columns(3)

        with m1:
            missing = patient.get("data_quality", {}).get("missing", [])
            if not missing:
                tag = _tier_badge("High"); info = _info_icon("No missing key data")
                detail = "<span style='color:#6b7280'>No missing data</span>"
            else:
                tag = _tier_badge("Medium" if len(missing) <= 2 else "Low")
                info = _info_icon("Missing inputs may reduce certainty")
                detail = "Missing: " + " ".join(_badge(m, "neutral") for m in missing)
            st.markdown(_h4("Data completeness") + f"<div>{tag}{info}</div><div style='margin-top:.4rem'>{detail}</div>",
                        unsafe_allow_html=True)

        with m2:
            dq = patient.get("data_quality", {}) or {}
            ood = dq.get("ood"); note = dq.get("age_note")
            if ood:
                tag = _tier_badge("Low"); info = _info_icon(ood if isinstance(ood,str) else "Slight distribution shift vs. training data")
                detail = "<span style='color:#6b7280'>Outside typical training range</span>"
            else:
                if note:
                    tag = _tier_badge("Medium"); info = _info_icon(str(note)); detail = f"<span>{note}</span>"
                else:
                    tag = _tier_badge("High"); info = _info_icon("Typical for training distribution"); detail = "<span style='color:#6b7280'>Typical</span>"
            st.markdown(_h4("Model familiarity") + f"<div>{tag}{info}</div><div style='margin-top:.4rem'>{detail}</div>",
                        unsafe_allow_html=True)

        with m3:
            stab_tag = _tier_badge("High"); info = _info_icon("Consistent results across runs; small input changes → small output changes")
            st.markdown(_h4("Prediction stability") + f"<div>{stab_tag}{info}</div>", unsafe_allow_html=True)
            st.caption("Details:")
            st.markdown(f"<div style='margin-top:.2rem'>{sens_table()}</div>", unsafe_allow_html=True)

        model_reason_raw = epi_reason(summary, patient) if dom_type=="Epistemic" else alea_reason(summary, patient)
        model_reason = re.sub(r'^\s*drivers:\s*', '', model_reason_raw, flags=re.I)
        st.markdown(
            f"<div style='margin-top:.75rem;padding:.6rem .8rem;border:1px solid #e5e7eb;border-radius:10px;background:#fafafa'>"
            f"<div style='display:flex;align-items:center;gap:.5rem;font-weight:700'>"
            f"Uncertainty type — {dom_type} "
            f"{_badge(unc_intensity, {'High':'danger','Moderate':'warn','Low':'success'}[unc_intensity])}"
            f"</div>"
            f"<div style='margin-top:.35rem;color:#374151'><b>Drivers:</b> {model_reason}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ======== DOCUMENT AGREEMENT / OVERRIDE ========
    st.divider()
    st.markdown("#### Document agreement / override")
    st.radio("Do you agree with the AI suggestion?", ["Yes", "No / Override"], horizontal=True, key="agree_radio")
    st.text_area("Notes (optional)", placeholder="Rationale, serial lab plan, shared decision details…", height=120)
    st.button("Save to audit log", type="primary")

    # ======== CLINICIAN DECISIONS ========
    st.divider()
    st.markdown("### Clinician Decisions")
    c1, c2 = st.columns([0.45, 0.55])
    with c1:
        st.markdown("**Triage & Patient Acuity**")
        tri_opts = ["T1","T2","T3","T4","T5"]
        try:
            tri_sel = st.segmented_control("tri_final", tri_opts, selection_mode="single",
                                           default=triage["code"], label_visibility="collapsed")
        except Exception:
            tri_sel = st.radio("tri_final", tri_opts, index=tri_opts.index(triage["code"]), horizontal=True, label_visibility="collapsed")
    with c2:
        st.markdown("**Disposition**")
        disp_opts = ["Confirm/Admit","Observe","Consult","Defer/Discharge"]
        try:
            disp_sel = st.segmented_control("disp_final", disp_opts, selection_mode="single",
                                            default=st.session_state.get("disp_choice", default_disp), label_visibility="collapsed")
        except Exception:
            disp_sel = st.radio("disp_final", disp_opts, index=disp_opts.index(st.session_state.get("disp_choice", default_disp)), horizontal=True, label_visibility="collapsed")

    st.markdown("**Next steps**")
    st.text_area("steps_final", placeholder="hs-Troponin now; repeat per rule-out algorithm; observe; reassess pain…", label_visibility="collapsed", height=90)

    b1, b2 = st.columns([0.20, 0.80])
    with b1:
        st.button("Save", type="primary")
    with b2:
        st.button("Discard")

if __name__ == "__main__":
    render_patient_chart()
else:
    render_patient_chart()
