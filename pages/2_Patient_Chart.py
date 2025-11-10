# pages/2_Patient_Chart.py
import streamlit as st
from html import escape
from math import exp

# ── Drawer helpers ─────────────────────────────────────────────────────────────
def _drawer_cols(opened: bool):
    if opened:
        return st.columns([0.68, 0.32], vertical_alignment="top")
    left = st.container(); right = st.container()
    return left, right

def _open_drawer(scope: str, triage_label: str = "", dispo_label: str = ""):
    st.session_state["why_open"] = True
    st.session_state["why_scope"] = scope
    if triage_label: st.session_state["why_triage_label"] = triage_label
    if dispo_label:  st.session_state["why_dispo_label"]  = dispo_label

def _close_drawer():
    st.session_state["why_open"] = False

# ── Optional utils (safe no-op if not present) ─────────────────────────────────
def _no_op(*a, **k): return None
try:
    from utils import init_state as _init_state, enter_page as _enter_page, show_back_top_right as _show_back, render_sidebar as _render_sidebar
except Exception:
    _init_state = _no_op; _enter_page = _no_op; _show_back = _no_op; _render_sidebar = _no_op

# ── Core patient helpers ───────────────────────────────────────────────────────
from components.patient_view import (
    make_patient_from_row, triage_level_from_summary, disposition_from_summary,
    decompose_uncertainty, simple_summary_from_manual, band_from_risk,
    toy_risk_model
)

# ── Tiny helpers ───────────────────────────────────────────────────────────────
def _sigmoid(x: float) -> float: return 1.0 / (1.0 + exp(-x))
def _clip01(x: float) -> float: return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))
def _pct(x: float) -> int: return int(round(100 * _clip01(x)))
def _format_pct(x: float) -> str: return str(_pct(x))

def _copy_inputs(patient):
    ri = dict(patient.get("risk_inputs", {}))
    return {
        "age": int(ri.get("age", 50)),
        "sex": int(ri.get("sex", 0)),
        "ecg_abnormal": bool(ri.get("ecg_abnormal", False)),
        "troponin": None if ri.get("troponin") is None else float(ri.get("troponin")),
        "pain_features": list(ri.get("pain_features", []) or []),
        "rf_smoker": bool(ri.get("rf_smoker", False)),
        "rf_htn": bool(ri.get("rf_htn", False)),
        "rf_dm": bool(ri.get("rf_dm", False)),
    }

# ==============================================================================
# TRIAGE stability (inputs + proxy models) — inside-band rows + extremes + mids
# ==============================================================================

def _triage_input_perturbations(patient, summary):
    """
    Micro-perturbations with clear units. We keep only predictions that land
    within the displayed [lo, hi] interval.
    """
    target_lo, target_hi = float(summary["lo"]), float(summary["hi"])
    base_inputs = _copy_inputs(patient)
    rows = []

    def _try(label, v):
        r = toy_risk_model(v)
        if target_lo - 1e-6 <= r <= target_hi + 1e-6:
            rows.append({"label": label, "risk": r})

    # hs-cTn micro steps
    tro = base_inputs.get("troponin")
    if tro is not None:
        for d in [-0.004, -0.002, +0.002, +0.004]:
            v = _copy_inputs(patient); v["troponin"] = max(0.0, tro + d)
            _try(f"Δ hs-cTn {('−' if d<0 else '+')}{abs(d):.3f} ng/L", v)
    else:
        for t_guess, lab in [(0.006, "low guess"), (0.012, "borderline"), (0.016, "mild positive")]:
            v = _copy_inputs(patient); v["troponin"] = t_guess
            _try(f"hs-cTn {lab} ({t_guess:.3f} ng/L)", v)

    # ECG re-read (only if stays in band)
    v = _copy_inputs(patient); v["ecg_abnormal"] = not v["ecg_abnormal"]
    _try("ECG re-read", v)

    # Pain-feature nudges (small conceptual change)
    for pf, label in [("radiating", "Add radiating"), ("crushing", "Add crushing")]:
        v = _copy_inputs(patient); s = set(v["pain_features"]); s.add(pf); v["pain_features"] = list(s)
        _try(label, v)

    return rows

def _triage_model_members(summary):
    """Proxy model spread clipped inside [lo, hi] and centered around base."""
    base, lo, hi = float(summary["base"]), float(summary["lo"]), float(summary["hi"])
    m1 = _clip01(min(hi, max(lo, base - 0.01)))
    m2 = _clip01(min(hi, max(lo, base + 0.01)))
    return [("XGBoost (proxy)", m1), ("Neural (proxy)", m2)]

def _triage_runs(patient, summary):
    A = _triage_input_perturbations(patient, summary)
    B = [{"name": n, "risk": r} for (n, r) in _triage_model_members(summary)]
    return A, B

# ==============================================================================
# DISPOSITION stability (cascade from triage + policy spread) — same rules
# ==============================================================================

def _disposition_threshold(patient) -> float:
    ri = patient.get("risk_inputs", {})
    thr = 0.22
    if ri.get("ecg_abnormal"): thr -= 0.03
    tro = ri.get("troponin")
    if (isinstance(tro, (int, float)) and tro is not None and tro >= 0.04):
        thr -= 0.05
    return max(0.12, thr)

def _dispo_prob_from_triage(triage_risk: float, patient) -> float:
    k = 14.0
    thr = _disposition_threshold(patient)
    return _sigmoid(k * (triage_risk - thr))

def _dispo_input_perturbations(patient, summary):
    """
    Perturb *triage risk* in absolute percentage points (pts) and include tight
    context branches; keep only predictions that land inside [lo, hi].
    """
    A = []
    base_tri = float(summary["base"])
    lo, hi = float(summary["lo"]), float(summary["hi"])

    def _try(label, tri_r):
        disp_r = _dispo_prob_from_triage(_clip01(tri_r), patient)
        if lo - 1e-6 <= disp_r <= hi + 1e-6:
            A.append({"label": label, "risk": disp_r})

    for dpts in [-0.03, -0.015, +0.015, +0.03]:
        _try(f"Triage risk {('−' if dpts<0 else '+')}{abs(int(round(dpts*100)))} pts", base_tri + dpts)

    ri = patient.get("risk_inputs", {})
    if ri.get("troponin") is None:
        _try("Troponin returns low", base_tri - 0.01)
        _try("Troponin returns 0.012", base_tri + 0.012)
    if ri.get("ecg_abnormal"):
        _try("ECG re-read normal", base_tri - 0.015)
    else:
        _try("ECG evolves abnormal", base_tri + 0.015)

    return A

def _disposition_model_members(summary, patient):
    """Policy/model spread after cascade; clip inside [lo, hi]."""
    base_tri = float(summary["base"])
    base_disp = _dispo_prob_from_triage(base_tri, patient)
    lo, hi = float(summary["lo"]), float(summary["hi"])
    m1 = _clip01(min(hi, max(lo, base_disp - 0.01)))
    m2 = _clip01(min(hi, max(lo, base_disp + 0.01)))
    return [("Policy B (proxy)", m1), ("Policy C (proxy)", m2)]

def _disposition_runs(patient, summary):
    A = _dispo_input_perturbations(patient, summary)
    B = [{"name": n, "risk": r} for (n, r) in _disposition_model_members(summary, patient)]
    return A, B

# ==============================================================================
# Row selection + in-band sampling guarantee
# ==============================================================================

def _nearest(rows, target, used_idx):
    if not rows: return None, None
    best_i, best = None, None
    for i, r in enumerate(rows):
        if i in used_idx: continue
        if best is None or abs(r["risk"] - target) < abs(best["risk"] - target):
            best = r; best_i = i
    return best_i, best

def _pick_band_rows(rows, lo: float, hi: float, k_targets=4):
    """
    rows: list of {"label"/"name", "risk"} already inside [lo, hi].
    We aim for min, ~1/3, ~2/3, max of the band by choosing nearest rows to targets.
    Guarantees inclusion of lowest and highest if present.
    """
    if not rows: return []
    rows = sorted(rows, key=lambda r: r["risk"])
    used = set()
    targets = [lo, lo + (hi-lo)/3, lo + 2*(hi-lo)/3, hi][:k_targets]
    picked = []
    for t in targets:
        idx, row = _nearest(rows, t, used)
        if row is not None:
            used.add(idx); picked.append(row)
    return sorted(picked, key=lambda r: r["risk"])

def _band_samples(lo, hi):
    """Deterministic samples that ALWAYS lie within the header interval."""
    return [
        {"label": "Band sample — lower edge", "name": "Band sample — lower edge", "risk": lo},
        {"label": "Band sample — ~1/3",       "name": "Band sample — ~1/3",       "risk": lo + (hi-lo)/3},
        {"label": "Band sample — ~2/3",       "name": "Band sample — ~2/3",       "risk": lo + 2*(hi-lo)/3},
        {"label": "Band sample — upper edge", "name": "Band sample — upper edge", "risk": hi},
    ]

# ==============================================================================
# Shared “prediction stability — details” block
# ==============================================================================

def _stability_details_block(scope: str, summary: dict, patient: dict):
    base = float(summary["base"]); lo = float(summary["lo"]); hi = float(summary["hi"])
    width = float(summary["width"])
    lo_pct, hi_pct, w_pts = _pct(lo), _pct(hi), _pct(width)

    if scope == "triage":
        A_all, B_all = _triage_runs(patient, summary)
        caption = "Interval combines (A) micro input changes with the same model, and (B) cross-model spread with the same inputs."
    elif scope == "disposition":
        A_all, B_all = _disposition_runs(patient, summary)
        caption = "Interval reflects cascading from triage plus policy/model spread at the disposition layer."
    else:
        A_all, B_all = [], []
        caption = "Shows how sensitive the plan is to small changes and upstream disagreement."

    # Keep rows inside band
    A_band = [r for r in (A_all or []) if lo - 0.015 <= r["risk"] <= hi + 0.015]
    B_band = [r for r in (B_all or []) if lo - 0.015 <= r["risk"] <= hi + 0.015]

    # Choose extremes + mids
    A_rows = _pick_band_rows(A_band, lo, hi, k_targets=4)
    B_rows = _pick_band_rows(B_band, lo, hi, k_targets=3)

    # Guarantee we ALWAYS show values that match the header:
    # If not enough in-band rows, fill with "band samples" (lower, ~1/3, ~2/3, upper).
    if len(A_rows) < 4:
        have = {(r.get("label") or r.get("name")) for r in A_rows}
        for s in _band_samples(lo, hi):
            if (s["label"] not in have) and len(A_rows) < 4:
                A_rows.append(s); have.add(s["label"])
        A_rows = sorted(A_rows, key=lambda r: r["risk"])

    if len(B_rows) < 3:
        have = {(r.get("label") or r.get("name")) for r in B_rows}
        for s in _band_samples(lo, hi):
            if (s["name"] not in have) and len(B_rows) < 3:
                B_rows.append(s); have.add(s["name"])
        B_rows = sorted(B_rows, key=lambda r: r["risk"])

    with st.container(border=True):
        st.markdown("<div style='font-weight:800;font-size:16px;margin-bottom:6px'>Prediction stability — details</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:15px;margin-bottom:6px'><b>Overall risk interval:</b> "
            f"{lo_pct}%–{hi_pct}% <span style='color:#6b7280'>(width {w_pts} pts)</span></div>",
            unsafe_allow_html=True
        )
        st.caption(caption)

        def _corner_badge(text):
            st.markdown(
                f"<div style='display:inline-block;font-weight:800;font-size:12px;border:1px solid #e5e7eb;"
                f"background:#f9fafb;border-radius:6px;padding:2px 6px;margin:4px 0'>{text}</div>",
                unsafe_allow_html=True
            )

        # A) Local sensitivity — extremes + mids guaranteed
        _corner_badge("A")
        st.table({
            "Perturbation / Context": [r.get("label","") for r in A_rows],
            "Risk %": [_format_pct(r["risk"]) for r in A_rows],
        })

        # B) Cross-model/Policy spread — extremes + mid guaranteed
        _corner_badge("B")
        st.table({
            "Model": [r.get("name","") for r in B_rows],
            "Risk %": [_format_pct(r["risk"]) for r in B_rows],
        })

# ========================= Plan stability block (as before) ====================

def _plan_stability_block(summary: dict, patient: dict):
    import math
    dq = patient.get("data_quality", {}) or {}
    missing = dq.get("missing", []) or []
    early = 1 if dq.get("time_from_onset_min", 999) <= 90 else 0
    ood = 1 if patient.get("data_quality",{}).get("ood") else 0

    width = float(summary.get("width", 0.0))
    has_ecg = bool(patient.get("risk_inputs",{}).get("ecg_abnormal"))
    has_trop = patient.get("risk_inputs",{}).get("troponin") is not None

    c_data   = min(1.0, 0.50 * len(missing))
    c_early  = 0.30 * early
    c_model  = max(0.0, (width - 0.10) / 0.25)
    c_ood    = 0.40 * ood
    c_branch = 0.35 if (not has_trop or has_ecg) else 0

    raw = [c_data, c_model, c_branch, c_early, c_ood]
    total = sum(raw) or 1.0
    psi = int(round(100 * total / (1.0 + 0.00001)))

    tier = "Low" if psi <= 33 else ("Medium" if psi <= 66 else "High")
    color = {"Low":"#16a34a","Medium":"#f59e0b","High":"#f97316"}[tier]

    st.subheader("Plan stability — details")
    with st.container(border=True):
        st.markdown(
            f"<div style='font-size:15px;margin-bottom:6px'><b>Plan Stability Index (PSI):</b> "
            f"{100-psi}% <span style='color:#6b7280'>(higher is steadier)</span>"
            f"<span style='display:inline-block;border-radius:999px;padding:3px 10px;margin-left:8px;"
            f"background:{color};color:white;font-weight:800'>{tier}</span></div>",
            unsafe_allow_html=True
        )
        st.caption("PSI reflects how likely the plan is to change once missing data returns or small inputs move.")

        labels = ["Data dependency", "Carry-over instability", "Branch divergence", "Early presentation", "Model unfamiliarity"]
        vals   = raw
        notes  = [
            ("Missing: " + ", ".join(missing)) if missing else "No key prerequisites missing",
            "Inheritance from triage stability (interval width)",
            "Different actions if pending test flips (e.g., hs-troponin, abnormal ECG)",
            "Symptoms very early — repeat testing likely to shift plan",
            "Outside training distribution" if ood else "Typical for training distribution",
        ]
        colors = ["#f59e0b","#22c55e","#6366f1","#e11d48","#0ea5e9"]
        for lab, v, note, col in zip(labels, vals, notes, colors):
            pct = int(round(100 * v / (total or 1.0)))
            st.markdown(
                f"<div style='display:flex;gap:.5rem;align-items:flex-start;margin:.15rem 0'>"
                f"<span style='width:10px;height:10px;background:{col};border-radius:2px;margin-top:4px'></span>"
                f"<div><b>{escape(lab)}</b> — {pct}%"
                f"<div style='color:#6b7280;font-size:12px'>{escape(note)}</div></div></div>",
                unsafe_allow_html=True
            )

# ==============================================================================
# Demo profiles & UI scaffolding
# ==============================================================================

FIXED_PROFILES = {
    "CP-1000": {"MRN":"CP-1000","Patient":"Weber, Charlotte","Age":28,"Sex":"Female","ESI":2,
                "HR":78,"SBP":128,"SpO₂":98,"TempC":36.9,"ECG":"Normal","hs-cTn (ng/L)":None,
                "OnsetMin":140,"CC":"Typical chest pressure, now improved.","Arrival":"Ambulance"},
    "CP-1001": {"MRN":"CP-1001","Patient":"Green, Gary","Age":60,"Sex":"Male","ESI":3,
                "HR":96,"SBP":136,"SpO₂":96,"TempC":36.8,"ECG":"Nonspecific","hs-cTn (ng/L)":None,
                "OnsetMin":40,"CC":"Not sure… feels weird in chest, comes and goes. Pain hard to describe.","Arrival":"Walk-in"},
    "CP-1002": {"MRN":"CP-1002","Patient":"Lopez, Mariah","Age":44,"Sex":"Female","ESI":3,
                "HR":94,"SBP":128,"SpO₂":98,"TempC":37.2,"ECG":"ST/T abn","hs-cTn (ng/L)":8.0,
                "OnsetMin":25,"CC":"Acute chest tightness with diaphoresis during activity.","Arrival":"Ambulance"},
}

SCENARIOS = {
    "CP-1000": {"base_pct": 12, "stability": "High",   "cascade": 0.00},
    "CP-1001": {"base_pct": 28, "stability": "Medium", "cascade": 0.05},
    "CP-1002": {"base_pct": 33, "stability": "Low",    "cascade": 0.15},
}
st.session_state["_SCENARIOS"] = SCENARIOS

st.session_state.setdefault("why_open", False)
st.session_state.setdefault("why_scope", "triage")
st.session_state.setdefault("why_triage_label", "")
st.session_state.setdefault("why_dispo_label", "")

TRIAGE_LEVELS = {
    "T1": ("Immediate","Highest probability for intensive care, emergency procedure, or mortality.", "#DC2626"),
    "T2": ("Very Urgent","Elevated probability for intensive care, emergency procedure, or mortality.", "#F97316"),
    "T3": ("Urgent","Moderate probability of hospital admission or very low probability of intensive care, emergency procedure, or mortality.", "#F59E0B"),
    "T4": ("Less Urgent","Low probability of hospital admission.", "#22C55E"),
    "T5": ("Non-Urgent","Fast turnaround and low probability of hospital admission.", "#10B981"),
}
DISP_LEVELS = {
    "Confirm/Admit":"Admit or confirm acute management plan — likely inpatient treatment.",
    "Observe":"Monitor in observation unit — reassess after a short period.",
    "Consult":"Seek specialist input before final decision.",
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

# ── TRIAGE blocks ──────────────────────────────────────────────────────────────
def _render_triage_blocks(summary, patient):
    alea_pct, epis_pct, conf_score, conf_tier = decompose_uncertainty(summary, patient)
    conf_pct = int(round(conf_score*100))
    base = float(summary["base"]); lo = float(summary["lo"]); hi = float(summary["hi"]); width = float(summary["width"])
    base_pct = int(round(100*base)); lo_pct = int(round(100*lo)); hi_pct = int(round(100*hi))
    vitals = patient.get("vitals", {})
    ri = dict(patient.get("risk_inputs", {}))
    ri["chief_complaint"] = patient.get("chief_complaint", "")
    pf_list = ri.get("pain_features") or []
    ri["symptoms"] = list(pf_list)
    ri["ecg"] = ("Abnormal" if ri.get("ecg_abnormal") else "Normal")
    onset_min = patient.get("data_quality",{}).get("time_from_onset_min", None)
    sim_pct = int(round(100 * (0.3 + (0.3 if ri.get('ecg_abnormal') else 0) + (min(0.4, (ri.get('troponin') or 0)/0.04 * 0.4)))))

    def _risk_tier(r):
        if r >= 0.40: return ("High", "#DC2626")
        if r >= 0.20: return ("Medium", "#F59E0B")
        if r >= 0.10: return ("Low", "#22C55E")
        return ("Very Low", "#10B981")
    risk_tier, risk_color = _risk_tier(base)

    def _badge_html(text, fg="#111827", bd="#e5e7eb", bg="#fff"):
        return (f"<span style='display:inline-block;border:1px solid {bd};background:{bg};color:{fg};"
                f"border-radius:999px;padding:2px 8px;margin-left:8px;font-weight:800'>{escape(text)}</span>")

    st.subheader("Evidence / Reasoning")
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:.25rem;'>"
        f"<div><b>Risk estimate:</b> {base_pct}%</div>"
        f"{_badge_html(risk_tier, fg=risk_color, bd=risk_color+'33')}"
        f"</div>", unsafe_allow_html=True)
    st.caption("How likely near-term deterioration is and how sick the patient is right now.")
    st.markdown(f"**Recognized diagnosis:** similarity **{sim_pct}%** to {escape(summary.get('suspected_condition','ACS-like'))} pattern.")

    # decisive inputs (unchanged)
    def _chips_line(items):
        def _chip_html(text, fg="#1f2937", bg="#f3f4f6", bd="#e5e7eb"):
            return (f"<span style='display:inline-block;border:1px solid {bd};background:{bg};color:{fg};"
                    f"border-radius:999px;padding:2px 8px;margin:4px 6px 0 0;font-weight:700;font-size:12px'>{escape(text)}</span>")
        return "".join(_chip_html(t) for t in (items or ["—"]))
    def _decisive_inputs():
        chips = []
        try:
            sbp = int(str(vitals.get("BP","").split("/")[0]))
            if sbp >= 140: chips.append("SBP ≥140 (hypertensive)")
            elif sbp <= 90: chips.append("SBP ≤90 (hypotension)")
        except Exception: pass
        try:
            hr = int(vitals.get("HR", 0))
            if hr >= 100: chips.append("HR ≥100 (tachycardia)")
            elif hr <= 50: chips.append("HR ≤50 (bradycardia)")
        except Exception: pass
        try:
            spo2 = int(vitals.get("SpO₂", vitals.get("SpO2", 0)))
            if spo2 and spo2 < 94: chips.append("SpO₂ <94%")
        except Exception: pass
        try:
            t = float(str(vitals.get("TempC", vitals.get("Temp", 0))))
            if t >= 38.0: chips.append("Fever (≥38°C)")
            elif t <= 35.5: chips.append("Hypothermia (≤35.5°C)")
        except Exception: pass
        cc = (ri.get("chief_complaint") or "").lower()
        symps = " ".join(ri.get("symptoms") or []).lower() + " " + cc
        def _has(s): return s in symps
        if any(_has(k) for k in ["chest pain", "pressure"]): chips.append("Chest pain/pressure")
        if _has("rest"): chips.append("Pain at rest")
        if any(_has(k) for k in ["shortness of breath","sob","dyspnea"]): chips.append("Dyspnea")
        pf = {str(p).lower() for p in (ri.get("pain_features") or [])}
        if "radiating" in pf:  chips.append("Radiating pain")
        if "crushing"  in pf:  chips.append("Crushing pain")
        if "diaphoresis" in pf: chips.append("Diaphoresis")
        ecg = (ri.get("ecg") or vitals.get("ECG") or "").lower()
        if ecg and ecg not in ("normal","normal sinus","nsr"): chips.append(f"ECG: {ecg.capitalize()}")
        tro = ri.get("troponin")
        if isinstance(tro,(int,float)) and tro is not None:
            if tro >= 0.01: chips.append("hs-cTn positive/raised")
        elif isinstance(tro,str) and tro.lower() in ["positive","elevated","high"]:
            chips.append("hs-cTn positive/raised")
        onset_min = patient.get("data_quality",{}).get("time_from_onset_min", None)
        if isinstance(onset_min, int) and onset_min <= 90: chips.append("Early presentation (≤90 min)")
        am_lc = (patient.get("arrival_mode") or "").lower()
        if ("ambul" in am_lc) or ("ems" in am_lc): chips.append("Arrived by ambulance")
        rf_count = sum(1 for k in ("rf_htn","rf_dm","rf_smoker") if ri.get(k))
        if rf_count >= 2: chips.append("Multiple CV risk factors")
        try:
            age = int(patient.get("Age") or patient.get("age") or 0)
            if age >= 65: chips.append("Age ≥65")
            elif 0 < age <= 18: chips.append("Age ≤18")
        except Exception: pass
        try:
            if sim_pct >= 60: chips.append(f"Pattern match {sim_pct}%")
        except Exception: pass
        seen = set(); uniq = []
        for c in chips:
            if c not in seen: seen.add(c); uniq.append(c)
        return uniq
    decisive = _decisive_inputs()
    st.markdown("**Decisive inputs for this prediction**")
    if decisive: st.markdown(_chips_line(decisive), unsafe_allow_html=True)
    else:        st.caption("No decisive findings beyond baseline; inputs were within normal ranges.")
    st.divider()

    pill_bg = {"High":"#16a34a","Medium":"#f59e0b","Low":"#f97316"}[conf_tier]
    st.subheader("Confidence & Uncertainty")
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:.5rem;flex-wrap:wrap">
          <div style="font-size:22px;font-weight:900">Confidence score: {conf_pct}%</div>
          <span style="display:inline-block;border-radius:999px;padding:3px 10px;background:{pill_bg};color:white;font-weight:800">
            {conf_tier}
          </span>
        </div>
        """, unsafe_allow_html=True
    )
    st.caption("How confident is the AI in the triage recommendation?")

    dq = patient.get("data_quality", {}) or {}
    missing = dq.get("missing", []) or []
    ood = 1.0 if dq.get("ood") else 0.0
    cc_txt = (ri.get("chief_complaint") or "")
    symp_txt_len = len(" ".join([cc_txt] + [s for s in (ri.get("pain_features") or [])]))
    is_clear = bool(cc_txt) and symp_txt_len >= 12
    human_vague = 0.0 if is_clear else 1.0

    w_data = min(1.0, 0.45 * len(missing))
    w_fam  = 0.60 * ood
    w_stab = max(0.0, (width - 0.10) / 0.25)
    w_chain = 0.0
    w_human = 0.40 * human_vague
    raw = [w_data, w_fam, w_stab, w_human, w_chain]
    total = sum(raw) or 1.0
    pct5 = [int(round(100 * x / total)) for x in raw]
    delta = 100 - sum(pct5)
    if delta != 0: pct5[0] += delta

    start2 = pct5[0]; start3 = start2 + pct5[1]; start4 = start3 + pct5[2]; start5 = start4 + pct5[3]
    pie_css = (
        f"background: conic-gradient(#f59e0b 0 {pct5[0]}%, "
        f"#6366f1 {pct5[0]}% {start2 + pct5[1]}%, "
        f"#22c55e {start3}% {start3 + pct5[2]}%, "
        f"#e11d48 {start4}% {start4 + pct5[3]}%, "
        f"#0ea5e9 {start5}% 100%);"
    )
    st.markdown(f"<div style='width:100px;height:100px;border-radius:50%;margin:6px 0;{pie_css}'></div>", unsafe_allow_html=True)

    labels = ["Data gaps","Model familiarity","Prediction stability","Human ambiguity","Chained decisions"]
    expl = [
        ("Missing: " + ", ".join(missing)) if missing else "No key inputs missing",
        "Outside training distribution" if ood else "Typical for training distribution",
        f"Risk interval ≈ {lo_pct}%–{hi_pct}% (width {int(round(100*width))} pts) — from (a) micro input perturbations and (b) cross-model spread.",
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

    _stability_details_block("triage", summary, patient)

# ── Uncertainty side-panel (unchanged) ─────────────────────────────────────────
def _render_uncertainty_panel(scope_here: str):
    is_open = bool(st.session_state.get("why_open"))
    scope_now = st.session_state.get("why_scope")
    if not (is_open and scope_now == scope_here):
        return

    st.markdown("""
    <style>
      .ua-panel * { font-family: inherit !important; }
      .ua-panel h3, .ua-panel .stHeading, .ua-panel .stSubheader { font-weight: 800; }
      .ua-handle { font-size: 18px; font-weight: 900; }
    </style>
    """, unsafe_allow_html=True)

    c_top = st.columns([0.92, 0.08])
    with c_top[1]:
        if st.button("≪", key=f"ua_collapse_{scope_here}", help="Hide details", type="secondary"):
            _close_drawer(); st.stop()

    with st.container(border=True):
        st.markdown('<div class="ua-panel">', unsafe_allow_html=True)

        summary = st.session_state.get("_ua_summary")
        patient = st.session_state.get("_ua_patient")
        if not summary or not patient:
            st.info("No uncertainty data available."); st.markdown('</div>', unsafe_allow_html=True); return

        if scope_here == "triage":
            _render_triage_blocks(summary, patient)

        elif scope_here == "disposition":
            base = float(summary["base"]); width = float(summary["width"])
            base_pct = int(round(100*base))
            dq = patient.get("data_quality", {}) or {}
            missing = dq.get("missing", []) or []
            ood = bool(dq.get("ood"))

            ai_dispo = st.session_state.get("why_dispo_label","") or disposition_from_summary(summary)
            pid = st.session_state.get("selected_patient_id","")
            scen = st.session_state.get("_SCENARIOS", {}).get(pid, {"cascade":0.0})
            cascade_penalty = float(scen.get("cascade", 0.0))
            _, _, tri_conf_score, _ = decompose_uncertainty(summary, patient)
            disp_conf_score = max(0.0, tri_conf_score - cascade_penalty)
            disp_conf_pct = int(round(100*disp_conf_score))
            disp_tier = "High" if disp_conf_score >= 0.70 else ("Medium" if disp_conf_score >= 0.40 else "Low")
            pill_bg = {"High":"#16a34a","Medium":"#f59e0b","Low":"#f97316"}[disp_tier]

            st.subheader("Evidence / Reasoning")
            band = band_from_risk(base)
            band_color = {"Low":"#10B981","Low-Moderate":"#22C55E","Moderate":"#F59E0B","High":"#DC2626"}.get(band, "#6b7280")
            band_badge = f"<span style='display:inline-block;border-radius:999px;padding:3px 10px;background:{band_color};color:white;font-weight:800'>{band}</span>"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:.5rem;flex-wrap:wrap'>"
                f"<div><b>Disposition risk signal:</b> {base_pct}%</div>"
                f"{band_badge}"
                f"<span style='display:inline-block;border-radius:999px;padding:3px 10px;background:#EEF2FF;border:1px solid #93C5FD;color:#1e40af;font-weight:800'>{escape(ai_dispo)}</span>"
                f"</div>", unsafe_allow_html=True)
            st.caption("Higher bands push toward Confirm/Admit; lower bands favor Observe/Consult/Discharge.")
            tri_tag = st.session_state.get("why_triage_label","")
            if tri_tag:
                st.markdown(f"**Triage context:** {_pill(tri_tag.split('—')[0].strip(), tone='#64748b', inverse=False)} {escape(tri_tag)}", unsafe_allow_html=True)

            chips = []
            ri = dict(patient.get("risk_inputs", {}))
            if ri.get("ecg_abnormal"): chips.append("ECG abnormal")
            tro = ri.get("troponin")
            if (isinstance(tro,(int,float)) and tro is not None and tro >= 0.04): chips.append("hs-cTn markedly elevated (≥0.04)")
            elif (isinstance(tro,(int,float)) and tro is not None and tro >= 0.01): chips.append("hs-cTn positive (≥0.01)")
            if width >= 0.20: chips.append("Wide risk interval (uncertain)")
            if missing: chips.append("Missing: " + ", ".join(missing))
            onset_min = patient.get("data_quality",{}).get("time_from_onset_min", None)
            if isinstance(onset_min,int) and onset_min <= 90: chips.append("Early presentation (≤90 min)")
            am_lc = (patient.get("arrival_mode") or "").lower()
            if ("ambul" in am_lc) or ("ems" in am_lc): chips.append("Arrived by ambulance")
            if chips:
                st.markdown("**Decisive inputs for this disposition**")
                st.markdown("".join([f"<span style='display:inline-block;border:1px solid #e5e7eb;background:#f3f4f6;color:#111827;border-radius:999px;padding:2px 8px;margin:4px 6px 0 0;font-weight:700;font-size:12px'>{escape(c)}</span>" for c in chips]), unsafe_allow_html=True)
            else:
                st.caption("Disposition based on overall band and clinical context; no single decisive finding.")
            st.divider()

            st.subheader("Confidence & Uncertainty")
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:.5rem;flex-wrap:wrap">
                  <div style="font-size:22px;font-weight:900">Confidence score: {disp_conf_pct}%</div>
                  <span style="display:inline-block;border-radius:999px;padding:3px 10px;background:{pill_bg};color:white;font-weight:800">
                    {disp_tier}
                  </span>
                </div>
                """, unsafe_allow_html=True
            )
            st.caption("How confident is the AI in the disposition recommendation? Includes cascading from triage.")

            lo_pct = int(round(100*summary["lo"])); hi_pct = int(round(100*summary["hi"]))
            width_pct = int(round(100*summary["width"]))

            w_data = min(1.0, 0.45 * len(missing))
            w_fam  = 0.60 * (1.0 if ood else 0.0)
            w_stab = max(0.0, (summary["width"] - 0.10) / 0.25)
            w_chain = min(1.0, float(scen.get("cascade", 0.0)) / 0.20)
            cc_txt = (patient.get("chief_complaint") or "")
            w_human = 0.25 * (0.0 if len(cc_txt) >= 12 else 1.0)

            raw = [w_data, w_fam, w_stab, w_human, w_chain]
            total = sum(raw) or 1.0
            pct5 = [int(round(100 * x / total)) for x in raw]
            delta = 100 - sum(pct5)
            if delta != 0: pct5[0] += delta

            start2 = pct5[0]; start3 = start2 + pct5[1]; start4 = start3 + pct5[2]; start5 = start4 + pct5[3]
            pie_css = (
                f"background: conic-gradient(#f59e0b 0 {pct5[0]}%, "
                f"#6366f1 {pct5[0]}% {start2 + pct5[1]}%, "
                f"#22c55e {start3}% {start3 + pct5[2]}%, "
                f"#e11d48 {start4}% {start4 + pct5[3]}%, "
                f"#0ea5e9 {start5}% 100%);"
            )
            st.markdown(f"<div style='width:100px;height:100px;border-radius:50%;margin:6px 0;{pie_css}'></div>", unsafe_allow_html=True)

            labels = ["Data gaps","Model familiarity","Prediction stability","Human ambiguity","Chained decisions"]
            expl = [
                ("Missing: " + ", ".join(missing)) if missing else "No key inputs missing",
                "Outside training distribution" if ood else "Typical for training distribution",
                f"Triage interval {lo_pct}%–{hi_pct}% (width {width_pct} pts) affects admit vs observe split.",
                "Narrative clarity influences thresholds only slightly here",
                f"Cascading from triage/model disagreement (penalty {int(round(100*scen.get('cascade',0.0)))} pts)",
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

            _stability_details_block("disposition", summary, patient)

        else:
            # === NEXT-STEPS drawer: classic "chips + badges" + confidence block ===
            st.subheader("Evidence / Reasoning")
            st.caption("Goal: Balance immediate care with uncertainty reduction first.")

            ai_steps = list(summary.get("steps") or [
                "Possible NSTEMI — obtain hs-Troponin now",
                "Repeat hs-Troponin per rule-out protocol",
                "Continuous ECG & vitals",
                "Reassess chest pain in 1–2 h",
            ])

            # pill + small tag badge (matches your “second” screenshot)
            for s in ai_steps:
                is_ur = any(k in s.lower() for k in ["obtain", "repeat", "rule-out", "prioritize", "complete"])
                tag   = "Uncertainty-reduction" if is_ur else "Care step"
                tag_bg = "#FEF3C7" if is_ur else "#E0E7FF"
                tag_bd = "#F59E0B" if is_ur else "#6366F1"

                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:.5rem;margin:.25rem 0;'>"
                    f"<span style='border-radius:999px;padding:.35rem .7rem;background:#f3f4f6;"
                    f"border:1px solid #e5e7eb;font-weight:700;font-size:12px;color:#111827'>{escape(s)}</span>"
                    f"<span style='border-radius:999px;padding:.3rem .6rem;background:{tag_bg};"
                    f"border:1px solid {tag_bd};font-weight:800;font-size:12px;color:#111827'>{tag}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
            st.divider()

            # ---------- Confidence & Uncertainty (pie block) ----------
            width = float(summary["width"])
            dq = patient.get("data_quality", {}) or {}
            missing = dq.get("missing", []) or []
            ood = bool(dq.get("ood"))
            cc_txt = (patient.get("chief_complaint") or "")
            narrative_clear = len(cc_txt) >= 12

            # start from triage confidence, then adjust for plan sensitivity & missing data
            _, _, tri_conf_score, _ = decompose_uncertainty(summary, patient)
            plan_sensitivity = max(0.0, (width - 0.08) / 0.22)          # 0..~1
            steps_conf_score = max(
                0.0,
                tri_conf_score
                - min(0.10, 0.10 * plan_sensitivity)                    # up to −10 pts if plan is sensitive
                - (0.05 if missing else 0.0)                             # −5 pts if key items missing
            )
            steps_conf_pct = int(round(100 * steps_conf_score))
            steps_tier = "High" if steps_conf_score >= 0.70 else ("Medium" if steps_conf_score >= 0.40 else "Low")
            pill_bg = {"High":"#16a34a","Medium":"#f59e0b","Low":"#f97316"}[steps_tier]

            st.subheader("Confidence & Uncertainty")
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:.5rem;flex-wrap:wrap">
                  <div style="font-size:22px;font-weight:900">Confidence score: {steps_conf_pct}%</div>
                  <span style="display:inline-block;border-radius:999px;padding:3px 10px;background:{pill_bg};color:white;font-weight:800">
                    {steps_tier}
                  </span>
                </div>
                """, unsafe_allow_html=True
            )
            st.caption("How confident is the AI in the next-steps plan?")

            # pie weights
            w_data  = min(1.0, 0.60 * len(missing))
            w_fam   = 0.30 * (1.0 if ood else 0.0)
            w_stab  = max(0.0, (width - 0.08) / 0.22)                    # plan sensitivity
            w_human = 0.25 * (0.0 if narrative_clear else 1.0)
            w_chain = 0.10                                              # weak link to upstream decisions

            raw   = [w_data, w_fam, w_stab, w_human, w_chain]
            total = sum(raw) or 1.0
            pct5  = [int(round(100 * x / total)) for x in raw]
            # fix rounding
            pct5[0] += (100 - sum(pct5))

            # simple conic "pie"
            start2 = pct5[0]; start3 = start2 + pct5[1]; start4 = start3 + pct5[2]; start5 = start4 + pct5[3]
            pie_css = (
                f"background: conic-gradient(#f59e0b 0 {pct5[0]}%, "
                f"#6366f1 {pct5[0]}% {start2 + pct5[1]}%, "
                f"#22c55e {start3}% {start3 + pct5[2]}%, "
                f"#e11d48 {start4}% {start4 + pct5[3]}%, "
                f"#0ea5e9 {start5}% 100%);"
            )
            st.markdown(f"<div style='width:100px;height:100px;border-radius:50%;margin:6px 0;{pie_css}'></div>", unsafe_allow_html=True)

            labels = ["Data gaps","Model familiarity","Plan sensitivity","Human ambiguity","Chained decisions"]
            notes  = [
                ("Missing: " + ", ".join(missing)) if missing else "No key prerequisites missing",
                "Outside training distribution" if ood else "Typical for training distribution",
                "How much the proposed actions could change with small new info (e.g., troponin/ECG repeat).",
                "Narrative clarity may add variability.",
                "Weak link to upstream decisions (10%).",
            ]
            colors = ["#f59e0b","#6366f1","#22c55e","#e11d48","#0ea5e9"]
            for label, pct, note, col in zip(labels, pct5, notes, colors):
                st.markdown(
                    f"<div style='display:flex;gap:.5rem;align-items:flex-start;margin:.15rem 0'>"
                    f"<span style='width:10px;height:10px;background:{col};border-radius:2px;margin-top:4px'></span>"
                    f"<div><b>{escape(label)}</b> — {pct}%"
                    f"<div style='color:#6b7280;font-size:12px'>{escape(note)}</div></div></div>",
                    unsafe_allow_html=True
                )

        st.markdown('</div>', unsafe_allow_html=True)

# ── Page renderer (unchanged) ──────────────────────────────────────────────────
def render_patient_chart():
    _init_state(); _enter_page("Patient Chart"); _render_sidebar(__file__); _show_back("← Back")

    st.session_state.setdefault("selected_patient_id", "CP-1000")
    st.caption("Search / select a patient (type ID or name to filter)")

    options, ids = [], []
    for pid, d in FIXED_PROFILES.items():
        options.append(f"{pid} — {d.get('Patient','')}"); ids.append(pid)

    cur = st.session_state.get("selected_patient_id") or ids[0]
    if cur not in ids: cur = ids[0]
    sel_label = st.selectbox("", options, index=ids.index(cur))
    sel_id = sel_label.split(" — ")[0].strip()
    st.session_state["selected_patient_id"] = sel_id

    base = dict(FIXED_PROFILES.get(sel_id))
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

    scen = SCENARIOS.get(sel_id, {})
    if sel_id == "CP-1001":
        patient["chief_complaint"] = base["CC"]
        patient["risk_inputs"]["troponin"] = None
        if "troponin" not in patient["data_quality"]["missing"]:
            patient["data_quality"]["missing"].append("troponin")

    left, right = st.columns([0.70, 0.30], vertical_alignment="center")
    with left:
        st.markdown(f"## {name}")
        st.caption(f"MRN: {patient['mrn']}")
    with right:
        st.radio("View", ["Patient", "Clinicians"], horizontal=True, index=1, label_visibility="collapsed")

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

    st.markdown(
        f"""
        <div class="ua-infocard">
          <div class="ua-kv">
            <div><b>Age</b></div><div>{patient['age']}</div>
            <div><b>Sex</b></div><div>{escape(str(patient['sex']))}</div>
            <div><b>Arrival</b></div><div>{escape(str(patient['arrival_mode']))}</div>
            <div><b>Onset (min)</b></div><div>{patient['data_quality']['time_from_onset_min']}</div>
            <div><b>Chief complaint</b></div><div>{escape(str(patient['chief_complaint']))}</div>
            <div><b>ECG</b></div><div>{'Abnormal' if patient['risk_inputs']['ecg_abnormal'] else 'Normal'}</div>
            <div><b>hs-cTn (ng/L)</b></div><div>{'—' if patient['risk_inputs'].get('troponin') is None else patient['risk_inputs']['troponin']}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    tabs = st.tabs(["Current", "History", "Results"])
    with tabs[0]:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("HR", f"{patient['vitals']['HR']} bpm")
        c2.metric("SBP", patient['vitals']['BP'].split('/')[0] + " mmHg")
        c3.metric("SpO₂", f"{patient['vitals']['SpO2']} %")
        c4.metric("Temp", f"{patient['vitals'].get('TempC', 37.0):.1f} °C")
        st.caption("Updated • stability/variance inform aleatoric uncertainty")
    with tabs[1]: st.caption("No prior ED visits in this demo.")
    with tabs[2]: st.caption("No additional labs/imaging in this demo.")

    manual = simple_summary_from_manual(scen.get("base_pct", 25), scen.get("stability", "Medium"), patient)
    summary = manual
    st.session_state["_ua_summary"] = summary
    st.session_state["_ua_patient"] = patient

    tri = triage_level_from_summary(summary)
    t_code, t_label, t_desc = tri["code"], tri["label"], tri["desc"]
    color_map = {"T1":"#DC2626","T2":"#F97316","T3":"#F59E0B","T4":"#22C55E","T5":"#10B981"}
    t_color = color_map[t_code]
    disp_default = disposition_from_summary(summary)

    st.markdown("### AI Recommendations")

    # TRIAGE
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
                """, unsafe_allow_html=True)
        with r2:
            st.button("❓", key="why_triage_btn",
                      help="How confident is the AI triage recommendation?",
                      on_click=_open_drawer, args=("triage", f"{t_code} — {t_label}", ""))

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
                """, unsafe_allow_html=True)

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

    # DISPOSITION
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
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
                """, unsafe_allow_html=True)
        with dright:
            st.button("❓", key="why_dispo_btn",
                      help="How confident is the AI disposition recommendation?",
                      on_click=_open_drawer, args=("disposition","",str(disp_default)))

        for k, desc in DISP_LEVELS.items():
            is_ai = (k == disp_default)
            st.markdown(
                f"""
                <div style="display:flex;align-items:flex-start;gap:.6rem;margin:.35rem 0;
                            border:1px solid {'#3B82F6' if is_ai else '#e5e7eb'};
                            background:{'#EEF2FF' if is_ai else 'white'};
                            border-radius:10px;padding:10px 12px;">
                  <div style="min-width:120px;font-weight:700">{escape(k)}</div>
                  <div style="flex:1;color:#374151">{escape(desc)}</div>
                </div>
                """, unsafe_allow_html=True)

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

    # NEXT STEPS (unchanged per your request)
    st.markdown('</div>', unsafe_allow_html=True)
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
        notes_ph = ("If you disagree with the AI triage or disposition recommendation, please explain your rationale here. "
                    "If additional next steps are needed, list them here.")
        st.text_area("Notes (optional)", placeholder=notes_ph, height=160, key="clin_notes")
        st.button("Save", type="primary", key="save_btn")
    with right_col:
        _render_uncertainty_panel("steps")

if __name__ == "__main__":
    render_patient_chart()
else:
    render_patient_chart()