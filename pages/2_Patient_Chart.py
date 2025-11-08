# pages/2_Patient_Chart.py
# Patient Chart — keeps prior layout, three fixed patients, search bar
# Adds: Evidence / Reasoning section (with decisive inputs chips) inside the Uncertainty drawer (triage scope)

import streamlit as st
from html import escape

# ── Drawer helpers & uncertainty panel (pure Streamlit, no iframe) ─────────────────────────────────────

def _drawer_cols(opened: bool):
    """Return (left, right) columns. When closed, right is eliminated so content spans full width."""
    if opened:
        return st.columns([0.68, 0.32], vertical_alignment="top")
    else:
        left = st.container()
        right = st.container()  # placeholder
        return left, right

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

# Core patient helpers
from components.patient_view import (
    make_patient_from_row, compute_summary,
    triage_level_from_summary, disposition_from_summary,
    decompose_uncertainty
)

# ── Fixed interview-friendly patient seeds (aligned to Track Board) ─────────────────────────────────────
# (Only these 3 will have detailed charts; we still keep the search bar UI.)
FIXED_PROFILES = {
    "CP-1000": {"MRN":"CP-1000","Patient":"Weber, Charlotte","Age":28,"Sex":"Female","ESI":2,
                "HR":82,"SBP":158,"SpO₂":97,"TempC":37.1,"ECG":"Normal","hs-cTn (ng/L)":None,
                "OnsetMin":85,"CC":"Chest pain at rest, mild shortness of breath.","Arrival":"Ambulance"},
    "CP-1001": {"MRN":"CP-1001","Patient":"Green, Gary","Age":60,"Sex":"Male","ESI":3,
                "HR":106,"SBP":136,"SpO₂":95,"TempC":36.9,"ECG":"Nonspecific","hs-cTn (ng/L)":12.0,
                "OnsetMin":40,"CC":"Severe chest pressure radiating to left arm, nausea, diaphoresis.","Arrival":"Walk-in"},
    "CP-1002": {"MRN":"CP-1002","Patient":"Lopez, Mariah","Age":44,"Sex":"Female","ESI":3,
                "HR":94,"SBP":128,"SpO₂":98,"TempC":37.3,"ECG":"ST/T abn","hs-cTn (ng/L)":58.0,
                "OnsetMin":25,"CC":"Acute chest tightness with diaphoresis during activity.","Arrival":"Ambulance"},
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

def _pill(text, tone="#2563EB", inverse=False):
    return (
        "<span style='border-radius:999px;padding:.25rem .6rem;border:1px solid {tone};"
        "margin:.15rem .25rem .15rem 0;display:inline-block;"
        "background:{bg};color:{fg};font-weight:700'>{t}</span>"
    ).format(tone=tone, bg=(tone if inverse else "transparent"), fg=("white" if inverse else tone), t=text)

def _chip(text, bg="#3B82F6"):
    return f"<span style='border-radius:999px;padding:.25rem .6rem;background:{bg};color:#fff;margin-right:.35rem;margin-bottom:.35rem;display:inline-block;font-weight:600'>{escape(text)}</span>"

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

# ── Uncertainty side-panel (with the Evidence / Reasoning block you requested) ─────────────────────────

def _render_uncertainty_panel(scope_here: str):
    """
    Scope-aware panel. Opens ONLY from the ❓ button for this scope.
    The right-edge handle (≪) is shown only when OPEN and only closes the panel.
    """
    is_open   = bool(st.session_state.get("why_open"))
    scope_now = st.session_state.get("why_scope")
    open_here = bool(is_open and scope_now == scope_here)
    if not open_here:
        return

    # Shared CSS
    st.markdown("""
    <style>
      .ua-panel * { font-family: inherit !important; }
      .ua-panel h3, .ua-panel .stHeading, .ua-panel .stSubheader { font-weight: 800; }
      .ua-panel .stMetric label { font-weight: 600; }
      .ua-panel .stProgress > div > div { transition: width .25s ease; }
      .ua-handle { font-size: 18px; font-weight: 900; }
    </style>
    """, unsafe_allow_html=True)

    # Close-only handle
    c_top = st.columns([0.92, 0.08])
    with c_top[1]:
        if st.button("≪", key=f"ua_collapse_{scope_here}", help="Hide details", type="secondary"):
            _close_drawer()
            st.stop()

    with st.container(border=True):
        st.markdown('<div class="ua-panel">', unsafe_allow_html=True)

        summary  = st.session_state.get("_ua_summary")
        patient  = st.session_state.get("_ua_patient")
        if not summary or not patient:
            st.info("No uncertainty data available.")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        alea_pct, epis_pct, conf_score, conf_tier = decompose_uncertainty(summary, patient)
        conf_pct = int(round(conf_score*100))

        if scope_here == "triage":
            # ---------- Prep ----------
            base     = float(summary.get("base",0.0))
            lo, hi   = float(summary.get("lo",0.0)), float(summary.get("hi",0.0))
            width    = float(summary.get("width", hi-lo))
            base_pct = int(round(100*base)); lo_pct = int(round(100*lo)); hi_pct = int(round(100*hi))

            def _badge_html(text, fg="#111827", bd="#e5e7eb", bg="#fff"):
                return (f"<span style='display:inline-block;border:1px solid {bd};background:{bg};color:{fg};"
                        f"border-radius:999px;padding:2px 8px;margin-left:8px;font-weight:800'>{escape(text)}</span>")

            def _risk_tier(r):
                if r >= 0.40: return ("High", "#DC2626")
                if r >= 0.20: return ("Medium", "#F59E0B")
                if r >= 0.10: return ("Low", "#22C55E")
                return ("Very Low", "#10B981")
            risk_tier, risk_color = _risk_tier(base)

            ri = dict(patient.get("risk_inputs", {}))  # local copy we can enrich for chips
            vitals = patient.get("vitals", {})

            # Interview-oriented enrichment for decisive chips:
            # - put chief complaint text and symptoms/pain_features into ri
            ri["chief_complaint"] = patient.get("chief_complaint", "")
            # treat pain_features as 'symptoms' too for chip detection
            pf_list = ri.get("pain_features") or []
            ri["symptoms"] = list(pf_list)
            # provide a textual ecg for chip logic
            ri["ecg"] = ("Abnormal" if ri.get("ecg_abnormal") else "Normal")

            # similarity proxy (kept)
            sim_pct = int(round(100 * (0.3
                                       + (0.3 if ri.get('ecg_abnormal') else 0)
                                       + (min(0.4, (ri.get('troponin') or 0)/0.04 * 0.4)))))

            # Bring in onset minutes for chips
            onset_min = patient.get("data_quality",{}).get("time_from_onset_min", None)

            # ---------- 1) Evidence / Reasoning ----------
            st.subheader("Evidence / Reasoning")

            # Risk estimate with inline tag
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:.25rem;'>"
                f"<div><b>Risk estimate:</b> {base_pct}%</div>"
                f"{_badge_html(risk_tier, fg=risk_color, bd=risk_color+'33')}"
                f"</div>",
                unsafe_allow_html=True
            )
            st.caption("How likely near-term deterioration is and how sick the patient is right now.")

            # Recognized diagnosis
            st.markdown(f"**Recognized diagnosis:** similarity **{sim_pct}%** to {escape(summary.get('suspected_condition','ACS-like'))} pattern.")

            # Decisive inputs only (chips) — your requested block
            def _chips_line(items):
                def _chip_html(text, fg="#1f2937", bg="#f3f4f6", bd="#e5e7eb"):
                    return (f"<span style='display:inline-block;border:1px solid {bd};background:{bg};color:{fg};"
                            f"border-radius:999px;padding:2px 8px;margin:4px 6px 0 0;font-weight:700;font-size:12px'>{escape(text)}</span>")
                return "".join(_chip_html(t) for t in (items or ["—"]))

            def _decisive_inputs():
                chips = []

                # Vitals — only if materially abnormal
                try:
                    sbp = int(str(vitals.get("BP","").split("/")[0]))
                    if sbp >= 140: chips.append("SBP ≥140 (hypertensive)")
                    elif sbp <= 90: chips.append("SBP ≤90 (hypotension)")
                except Exception:
                    pass
                try:
                    hr = int(vitals.get("HR", 0))
                    if hr >= 100: chips.append("HR ≥100 (tachycardia)")
                    elif hr <= 50: chips.append("HR ≤50 (bradycardia)")
                except Exception:
                    pass
                try:
                    spo2 = int(vitals.get("SpO₂", vitals.get("SpO2", 0)))
                    if spo2 and spo2 < 94: chips.append("SpO₂ <94%")
                except Exception:
                    pass
                try:
                    t = float(str(vitals.get("TempC", vitals.get("Temp", 0))))
                    if t >= 38.0: chips.append("Fever (≥38°C)")
                    elif t <= 35.5: chips.append("Hypothermia (≤35.5°C)")
                except Exception:
                    pass

                # Chief complaint / symptoms — highlight classic high-risk features
                cc = (ri.get("chief_complaint") or "").lower()
                symps = " ".join(ri.get("symptoms") or []).lower() + " " + cc
                def _has(s): return s in symps
                if any(_has(k) for k in ["chest pain", "pressure"]):
                    chips.append("Chest pain/pressure")
                if _has("rest"):
                    chips.append("Pain at rest")
                if any(_has(k) for k in ["shortness of breath","sob","dyspnea"]):
                    chips.append("Dyspnea")
                # High-risk pain qualifiers gathered earlier
                pf = {str(p).lower() for p in (ri.get("pain_features") or [])}
                if "radiating" in pf:  chips.append("Radiating pain")
                if "crushing"  in pf:  chips.append("Crushing pain")
                if "diaphoresis" in pf: chips.append("Diaphoresis")

                # ECG / Lab — only if abnormal or positive
                ecg = (ri.get("ecg") or vitals.get("ECG") or "").lower()
                if ecg and ecg not in ("normal","normal sinus","nsr"):
                    chips.append(f"ECG: {ecg.capitalize()}")
                tro = ri.get("troponin")
                if isinstance(tro,(int,float)) and tro is not None:
                    if tro >= 0.01: chips.append("hs-cTn positive/raised")
                elif isinstance(tro,str) and tro.lower() in ["positive","elevated","high"]:
                    chips.append("hs-cTn positive/raised")

                # Context — time/arrival that bumps acuity
                if isinstance(onset_min, int) and onset_min <= 90:
                    chips.append("Early presentation (≤90 min)")
                am_lc = (patient.get("arrival_mode") or "").lower()
                if ("ambul" in am_lc) or ("ems" in am_lc):
                    chips.append("Arrived by ambulance")

                # Risk factors — only if clustered
                rf_count = sum(1 for k in ("rf_htn","rf_dm","rf_smoker") if ri.get(k))
                if rf_count >= 2:
                    chips.append("Multiple CV risk factors")

                # Demographics — only when extreme/impactful
                try:
                    age = int(patient.get("Age") or patient.get("age") or 0)
                    if age >= 65: chips.append("Age ≥65")
                    elif 0 < age <= 18: chips.append("Age ≤18")
                except Exception:
                    pass

                # Pattern match — only if strong
                try:
                    if sim_pct >= 60:
                        chips.append(f"Pattern match {sim_pct}%")
                except Exception:
                    pass

                # De-dupe while preserving order
                seen = set(); uniq = []
                for c in chips:
                    if c not in seen:
                        seen.add(c); uniq.append(c)
                return uniq

            decisive = _decisive_inputs()
            st.markdown("**Decisive inputs for this prediction**")
            if decisive:
                st.markdown(_chips_line(decisive), unsafe_allow_html=True)
            else:
                st.caption("No decisive findings beyond baseline; inputs were within normal ranges.")

            st.divider()

            # ---------- 2) Confidence & Uncertainty ----------
            # Compact one-row confidence with traffic-light pill
            pill_bg = {"High":"#16a34a","Medium":"#f59e0b","Low":"#f97316"}[conf_tier]
            st.subheader("Confidence & Uncertainty")
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:.5rem;flex-wrap:wrap">
                  <div style="font-size:22px;font-weight:900">Confidence score: {conf_pct}%</div>
                  <span style="display:inline-block;border-radius:999px;padding:3px 10px;
                               background:{pill_bg};color:white;font-weight:800">
                    {conf_tier}
                  </span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.caption("How confident is the AI in the triage recommendation?")

            # Five-aspect composition (interval now lives under Prediction stability)
            dq = patient.get("data_quality", {}) or {}
            missing = dq.get("missing", []) or []
            ood = 1.0 if dq.get("ood") else 0.0

            # Human ambiguity: simple proxy by symptom text length
            cc_txt = (ri.get("chief_complaint") or "")
            symp_txt_len = len(" ".join([cc_txt] + [s for s in (ri.get("pain_features") or [])]))
            is_clear = bool(cc_txt) and symp_txt_len >= 12
            human_vague = 0.0 if is_clear else 1.0

            # Weights
            w_data = min(1.0, 0.45 * len(missing))         # data gaps
            w_fam  = 0.60 * ood                            # model familiarity
            w_stab = max(0.0, (width - 0.10) / 0.25)       # prediction stability (interval)
            w_chain = 0.0                                   # not applicable at triage
            w_human = 0.40 * human_vague

            raw = [w_data, w_fam, w_stab, w_human, w_chain]
            total = sum(raw) or 1.0
            pct5 = [int(round(100 * x / total)) for x in raw]
            delta = 100 - sum(pct5)
            if delta != 0: pct5[0] += delta  # rounding fix

            # Small pie
            start2 = pct5[0]
            start3 = start2 + pct5[1]
            start4 = start3 + pct5[2]
            start5 = start4 + pct5[3]
            pie_css = (
                f"background: conic-gradient(#f59e0b 0 {pct5[0]}%, "
                f"#6366f1 {pct5[0]}% {start2 + pct5[1]}%, "
                f"#22c55e {start3}% {start3 + pct5[2]}%, "
                f"#e11d48 {start4}% {start4 + pct5[3]}%, "
                f"#0ea5e9 {start5}% 100%);"
            )
            st.markdown(
                f"<div style='width:100px;height:100px;border-radius:50%;margin:6px 0;{pie_css}'></div>",
                unsafe_allow_html=True
            )

            # Legend — with interval text
            labels = ["Data gaps","Model familiarity","Prediction stability","Human ambiguity","Chained decisions"]
            expl = [
                ("Missing: " + ", ".join(missing)) if missing else "No key inputs missing",
                "Outside training distribution" if ood else "Typical for training distribution",
                f"Risk interval ≈ {lo_pct}%–{hi_pct}% (width {int(round(100*width))} pts) — from (a) small input perturbations and (b) cross-model spread.",
                "Unclear/vague narrative" if not is_clear else "Clear symptom description (0%)",
                "Not applicable at triage (0%)",
            ]
            colors = ["#f59e0b","#6366f1","#22c55e","#e11d48","#0ea5e9"]
            for label, pct, note, col in zip(labels, pct5, expl, colors):
                st.markdown(
                    f"<div style='display:flex;gap:.5rem;align-items:flex-start;margin:.15rem 0'>"
                    f"<span style='width:10px;height:10px;background:{col};border-radius:2px;margin-top:4px'></span>"
                    f"<div><b>{escape(label)}</b> — {pct}%"
                    f"<div style='color:#6b7280;font-size:12px'>{escape(note)}</div></div></div>",
                    unsafe_allow_html=True
                )

            # Prediction stability detail
            st.markdown("**Prediction stability — details**")
            with st.container(border=True):
                st.markdown(
                    f"<div style='font-size:15px;margin-bottom:6px'><b>Overall risk interval:</b> "
                    f"{lo_pct}%–{hi_pct}% "
                    f"<span style='color:#6b7280'>(width {int(round(100*width))} pts)</span></div>",
                    unsafe_allow_html=True
                )
                st.caption("Interval combines two views of stability: (A) small input changes with the same model, and (B) cross-model spread with the same inputs.")

                # A) Same model, small input changes (local sensitivity — simple illustrative table)
                delta_up   = max(1, int(round(100*min(0.04, width/3))))
                delta_down = max(0, int(round(100*min(0.03, width/4))))
                st.table({
                    "Vital / Input": ["HR +5%", "SBP -5%", "SpO₂ -2 pts", "Temp +0.5°C"],
                    "Δ Risk (points)": [f"+{delta_up}", f"+{max(0,delta_up-1)}", f"+{max(1,delta_up)}", f"+{max(0,delta_down)}"],
                })

                # B) Same inputs, different models (cross-model variance — illustrative)
                def clip01(x): return max(0, min(1, x))
                m1 = clip01(base + 0.5*width - 0.03)   # Logistic (proxy)
                m2 = clip01(base - 0.4*width + 0.01)   # XGBoost (proxy)
                m3 = clip01(base + 0.2*width + 0.00)   # Neural (proxy)
                st.table({
                    "Model": ["Logistic (proxy)", "XGBoost (proxy)", "Neural (proxy)"],
                    "Risk %": [int(round(100*m1)), int(round(100*m2)), int(round(100*m3))],
                })

        elif scope_here == "disposition":
            st.subheader(f"Why Disposition: {st.session_state.get('why_dispo_label','')}")
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

        else:  # steps
            st.subheader("Why These Next Steps")
            st.write("**Prediction stability (local sensitivity)**")
            st.table({
                "Vital": ["HR","SBP","SpO₂"],
                "Change": ["+5%","+5%","±5%"],
                "Output ↑": ["+2%","+0%","+3%"],
                "Output ↓": ["-1%","+1%","—"],
            })

        st.markdown('</div>', unsafe_allow_html=True)

# ── Page renderer ──────────────────────────────────────────────────────────────────────────────────────

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
        # Only allow our three canonical profiles in chart
        tmp = tmp[tmp["PatientID"].isin(list(FIXED_PROFILES.keys()))]
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

    # Build patient dict
    if track_df is not None and not track_df.empty and sel_id in track_df["PatientID"].values:
        row = track_df.loc[track_df["PatientID"] == sel_id].iloc[0].to_dict()
        base = FIXED_PROFILES.get(sel_id, {})
        # Prefer interview-friendly seeds for Arrival/CC if present
        row.setdefault("Arrival", base.get("Arrival","—"))
        row.setdefault("CC", base.get("CC","Chest pain."))
        patient = make_patient_from_row(row)
        name = str(row.get("Patient","—"))
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

    # Prepare simple UI dict
    ui = {
        "Patient": name, "Age": patient["age"], "Sex": patient["sex"],
        "OnsetMin": patient["data_quality"]["time_from_onset_min"],
        "Arrival": patient["arrival_mode"], "CC": patient["chief_complaint"],
        "HR": patient["vitals"]["HR"], "SBP": patient["vitals"]["BP"].split("/")[0],
        "SpO₂": patient["vitals"]["SpO2"], "TempC": f"{patient['vitals'].get('TempC', 37.0):.1f}",
        "ECG": "Normal" if not patient["risk_inputs"]["ecg_abnormal"] else (base.get("ECG","Abnormal") if isinstance(base, dict) else "Abnormal"),
        "hs-cTn (ng/L)": patient["risk_inputs"].get("troponin"),
    }

    # Header + basic info
    patient_header(name, patient["mrn"], ui)

    # Shared UI styles
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

    # ============ AI SECTION ============
    st.markdown("### AI Recommendations")

    # ----- TRIAGE -----
    tri_open = bool(st.session_state.get("why_open") and st.session_state.get("why_scope")=="triage")
    left_col, right_col = _drawer_cols(tri_open)
    with left_col:
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
            st.button("❓", key="why_triage_btn",
                      help="How confident is the AI triage recommendation?",
                      on_click=_open_drawer, args=("triage", f"{t_code} — {t_label}", ""))

        # All triage levels
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
        lt, rt = st.columns([0.18, 0.82], vertical_alignment="center")
        with lt:  st.markdown("<div class='ua-label'>Clinician Triage</div>", unsafe_allow_html=True)
        with rt:
            tri_opts = ["T1","T2","T3","T4","T5"]
            try:
                st.segmented_control("tri_final", tri_opts, selection_mode="single",
                                     default=t_code, label="", label_visibility="collapsed")
            except Exception:
                st.radio(label="", options=tri_opts, index=tri_opts.index(t_code),
                         horizontal=True, key="tri_final_radio", label_visibility="collapsed")
        st.markdown('<div style="height:0;margin-top:-12px;"><hr class="ua-slim-hr"></div>', unsafe_allow_html=True)
    with right_col:
        _render_uncertainty_panel("triage")

    # ----- DISPOSITION -----
    disp_open = bool(st.session_state.get("why_open") and st.session_state.get("why_scope")=="disposition")
    left_col, right_col = _drawer_cols(disp_open)
    with left_col:
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
            st.button("❓", key="why_dispo_btn",
                      help="How confident is the AI disposition recommendation?",
                      on_click=_open_drawer, args=("disposition","",str(disp_default)))

        render_dispo_list(disp_default)

        lcdl, lcdr = st.columns([0.18, 0.82], vertical_alignment="center")
        with lcdl: st.markdown("<div class='ua-label'>Clinician Disposition</div>", unsafe_allow_html=True)
        with lcdr:
            disp_opts = list(DISP_LEVELS.keys())
            try:
                st.segmented_control("disp_final", disp_opts, selection_mode="single",
                                     default=disp_default, label="", label_visibility="collapsed")
            except Exception:
                st.radio(label="", options=disp_opts, index=disp_opts.index(disp_default),
                         horizontal=True, key="disp_final_radio", label_visibility="collapsed")
        st.markdown('<div style="height:0;margin-top:-20px;"><hr class="ua-slim-hr"></div>', unsafe_allow_html=True)
    with right_col:
        _render_uncertainty_panel("disposition")

    # ----- NEXT STEPS -----
    steps_open = bool(st.session_state.get("why_open") and st.session_state.get("why_scope")=="steps")
    left_col, right_col = _drawer_cols(steps_open)
    with left_col:
        s1, s2 = st.columns([0.86, 0.14])
        with s1:
            st.markdown("**AI Next-Steps Suggestions**")
            st.caption("Check a box to indicate you agree/accept that AI suggestion. Unchecked = not accepted.")
        with s2:
            st.button("❓", key="why_steps_btn",
                    help="How confident is the AI next-steps suggestion?",
                    on_click=_open_drawer, args=("steps","",""))

        ai_steps = list(summary.get("steps") or [
            "Possible NSTEMI — obtain hs-Troponin now",
            "Repeat hs-Troponin per rule-out protocol",
            "Continuous ECG & vitals",
            "Reassess chest pain in 1–2 h",
        ])
        for i, step in enumerate(ai_steps):
            c = st.columns([0.06, 0.94])
            with c[0]: st.checkbox("", key=f"agree_step_{i}")
            with c[1]: st.markdown(_chip(step), unsafe_allow_html=True)

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

        notes_ph = (
            "If you disagree with the AI triage or disposition recommendation, please explain your rationale here. "
            "If additional next steps are needed, list them here."
        )
        st.text_area("Notes (optional)", placeholder=notes_ph, height=160, key="clin_notes")
        st.button("Save", type="primary", key="save_btn")
    with right_col:
        _render_uncertainty_panel("steps")

if __name__ == "__main__":
    render_patient_chart()
else:
    render_patient_chart()