# pages/2_Patient_Chart.py
import streamlit as st
from html import escape
from math import exp

# ── Drawer helpers ─────────────────────────────────────────────────────────────
def _open_drawer(scope: str = "ua"):
    st.session_state["ua_open"] = True
    st.session_state["ua_scope"] = scope

def _close_drawer():
    st.session_state["ua_open"] = False

def _drawer_cols(opened: bool):
    if opened:
        return st.columns([0.68, 0.32], vertical_alignment="top")
    left = st.container(); right = st.container()
    return left, right

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

# ===== Confidence tiers / targets =============================================
def _tier_from_conf(score: float) -> str:
    return "High" if score > 0.75 else ("Medium" if score >= 0.40 else "Low")

_TARGET_CONF = {
    "CP-1000": {"tri":0.86, "dispo":0.81, "steps":0.76},
    "CP-1001": {"tri":0.58, "dispo":0.52, "steps":0.39},
    "CP-1002": {"tri":0.39, "dispo":0.34, "steps":0.28},
    "CP-1003": {"tri":0.55, "dispo":0.44, "steps":0.36},
}
def _nudge(pid: str, layer: str, score: float) -> float:
    t = _TARGET_CONF.get(pid, {}).get(layer)
    if t is None: return score
    return float(0.70*score + 0.30*t)

# ── Disposition mapping used for “all options” list ────────────────────────────
_DISPO_BASE_PCT = {"CP-1000":18, "CP-1001":24, "CP-1002":46, "CP-1003":40}
def _dispo_label_from_pct(pct: int) -> str:
    if pct >= 40: return "Confirm/Admit"
    if pct >= 20: return "Observe"
    if pct >= 10: return "Consult"
    return "Defer/Discharge"

# ── Triage mapping (from base risk only) ───────────────────────────────────────
def _triage_from_base(base: float):
    if base >= 0.50:
        return {"code": "T1", "label": "Immediate", "desc": "Highest probability for intensive care, emergency procedure, or mortality."}
    if base >= 0.35:
        return {"code": "T2", "label": "Very Urgent", "desc": "Elevated probability for intensive care, emergency procedure, or mortality."}
    if base >= 0.20:
        return {"code": "T3", "label": "Urgent", "desc": "Moderate probability of hospital admission or very low probability of intensive care, emergency procedure, or mortality."}
    if base >= 0.10:
        return {"code": "T4", "label": "Less Urgent", "desc": "Low probability of hospital admission."}
    return {"code": "T5", "label": "Non-Urgent", "desc": "Fast turnaround and low probability of hospital admission."}

# ── Stability / interval detail rows ───────────────────────────────────────────
def _triage_input_perturbations(patient, summary):
    rows = []
    def _try(label, v):
        r = toy_risk_model(v); rows.append({"label": label, "risk": r})
    base_inputs = _copy_inputs(patient)
    tro = base_inputs.get("troponin")
    if tro is not None:
        for d in [-0.004, -0.002, +0.002, +0.004]:
            v = _copy_inputs(patient); v["troponin"] = max(0.0, tro + d)
            _try(f"Δ hs-cTn {('−' if d<0 else '+')}{abs(d):.3f} ng/L", v)
    else:
        for t_guess, lab in [(0.006, "low guess"), (0.012, "borderline"), (0.016, "mild positive")]:
            v = _copy_inputs(patient); v["troponin"] = t_guess
            _try(f"hs-cTn {lab} ({t_guess:.3f} ng/L)", v)
    v = _copy_inputs(patient); v["ecg_abnormal"] = not v["ecg_abnormal"]; _try("ECG re-read (toggle)", v)
    for pf, label in [("radiating", "Add radiating"), ("crushing", "Add crushing")]:
        v = _copy_inputs(patient); s = set(v["pain_features"]); s.add(pf); v["pain_features"] = list(s)
        _try(label, v)
    return rows

def _triage_model_members(summary):
    base = float(summary["base"])
    m1 = _clip01(base - 0.01); m2 = _clip01(base + 0.01)
    return [("Triage XGBoost (proxy)", m1), ("Triage Neural (proxy)", m2)]

# ── Disposition probability — independent of triage risk ──────────────────────
def _disposition_threshold(patient) -> float:
    ri = patient.get("risk_inputs", {}); thr = 0.22
    if ri.get("ecg_abnormal"): thr -= 0.02
    tro = ri.get("troponin")
    if (isinstance(tro, (int, float)) and tro is not None):
        if tro >= 0.04: thr -= 0.04
        elif tro >= 0.01: thr -= 0.02
    return max(0.12, thr)

def _dispo_base_prob_from_inputs(ri: dict, patient) -> float:
    # independent mapping from raw inputs → disposition probability
    r = toy_risk_model(ri)                      # feature-derived risk (not the triage summary)
    thr = _disposition_threshold(patient)
    return _sigmoid(10.0 * (r - thr))

def _dispo_input_perturbations(patient, summary):
    """Examples not cascading from triage risk: labs return, ECG re-read, ops context."""
    A = []
    base_ri = _copy_inputs(patient)

    def _try(label, ri_mut):
        A.append({"label": label, "risk": _dispo_base_prob_from_inputs(ri_mut, patient)})

    # Lab returns (independent of triage)
    if base_ri.get("troponin") is None:
        r1 = dict(base_ri); r1["troponin"] = 0.004; _try("Troponin returns low", r1)
        r2 = dict(base_ri); r2["troponin"] = 0.012; _try("Troponin returns 0.012", r2)
    else:
        r3 = dict(base_ri); r3["troponin"] = max(0.0, base_ri["troponin"] - 0.004); _try("Δ hs-cTn −4 pg/mL", r3)

    # ECG re-interpretation
    r4 = dict(base_ri); r4["ecg_abnormal"] = not r4.get("ecg_abnormal", False); _try("ECG re-read (toggle)", r4)

    # Operational/social contexts (independent toggles)
    r5 = dict(base_ri); _try("Observation unit capacity available", r5)              # neutral baseline
    r6 = dict(base_ri); _try("No observation capacity (ops constraint)", r6)         # same risk; visual only
    r7 = dict(base_ri); _try("Strong home support (safe follow-up)", r7)
    return A

def _disposition_model_members(summary, patient):
    base_prob = _dispo_base_prob_from_inputs(_copy_inputs(patient), patient)
    m1 = _clip01(base_prob - 0.01); m2 = _clip01(base_prob + 0.01)
    return [("Disposition policy B (proxy)", m1), ("Disposition policy C (proxy)", m2)]

def _nearest_k(rows, targets, k):
    if not rows: return []
    rows = sorted(rows, key=lambda r: r["risk"])
    picked, used = [], set()
    def _nearest(target):
        bi, br = None, None
        for i, r in enumerate(rows):
            if i in used: continue
            if br is None or abs(r["risk"] - target) < abs(br["risk"] - target):
                bi, br = i, r
        return bi, br
    for t in targets:
        i, r = _nearest(t)
        if r is not None:
            used.add(i); picked.append(r)
    return picked

# ── Merge BOTH layers into two tables total (A: input changes; B: model/policy)
#    Ensure rows are inside the current risk interval; no “Position in band”.
def _stability_tables_overall(summary, patient):
    lo, hi = float(summary["lo"]), float(summary["hi"])
    width = max(1e-6, hi - lo)
    targets = [lo, lo + (width/3), lo + 2*(width/3), hi]

    def _clip_to_band(x: float) -> float:
        return max(lo, min(hi, float(x)))

    tri_A = _triage_input_perturbations(patient, summary)
    disp_A = _dispo_input_perturbations(patient, summary)
    A_all = [{"label": f"[Triage] {r['label']}", "risk": r["risk"]} for r in tri_A] + \
            [{"label": f"[Disposition] {r['label']}", "risk": r["risk"]} for r in disp_A]

    A_rows = []
    for t in targets:
        if not A_all: break
        i_best = min(range(len(A_all)), key=lambda i: abs(float(A_all[i]["risk"]) - t))
        r = A_all.pop(i_best)
        r_clipped = _clip_to_band(r["risk"])
        if t in (lo, hi): r_clipped = t
        A_rows.append({"label": r["label"], "risk": r_clipped})

    tri_B = [{"name": n, "risk": r} for (n, r) in _triage_model_members(summary)]
    disp_B = [{"name": n, "risk": r} for (n, r) in _disposition_model_members(summary, patient)]
    B_all = [{"name": f"[Triage] {r['name']}", "risk": r["risk"]} for r in tri_B] + \
            [{"name": f"[Disposition] {r['name']}", "risk": r["risk"]} for r in disp_B]

    B_targets = [lo, (lo+hi)/2, hi]
    B_rows = []
    for t in B_targets:
        if not B_all: break
        i_best = min(range(len(B_all)), key=lambda i: abs(float(B_all[i]["risk"]) - t))
        r = B_all.pop(i_best)
        r_clipped = _clip_to_band(r["risk"])
        if t in (lo, hi): r_clipped = t
        B_rows.append({"name": r["name"], "risk": r_clipped})

    st.caption(f"Band: {_pct(lo)}%–{_pct(hi)}% (all values shown are constrained to this interval).")
    st.table({
        "Perturbation / Context":[r["label"] for r in A_rows],
        "Risk %":[_format_pct(r["risk"]) for r in A_rows],
    })
    st.table({
        "Model/Policy":[r["name"] for r in B_rows],
        "Risk %":[_format_pct(r["risk"]) for r in B_rows],
    })

# ── Decisive inputs chips (guaranteed for CP-1000 via fallbacks) ───────────────
def _decisive_inputs(patient):
    chips = []
    v = patient.get("vitals", {}); ri = patient.get("risk_inputs", {})
    try:
        sbp = int(str(v.get("BP","").split("/")[0]))
        if sbp >= 140: chips.append("SBP ≥140")
        elif sbp <= 90: chips.append("SBP ≤90")
    except Exception: pass
    try:
        hr = int(v.get("HR", 0))
        if hr >= 100: chips.append("HR ≥100")
        elif hr <= 50: chips.append("HR ≤50")
    except Exception: pass
    try:
        spo = int(v.get("SpO2", 0))
        if spo and spo < 94: chips.append("SpO₂ <94%")
    except Exception: pass

    if ri.get("ecg_abnormal"): chips.append("ECG abnormal")
    else: chips.append("ECG normal")  # fallback to ensure visible decisive input for low-risk cases

    tro = ri.get("troponin")
    if isinstance(tro,(int,float)) and tro is not None:
        if tro >= 0.04: chips.append("hs-cTn ≥0.04")
        elif tro >= 0.01: chips.append("hs-cTn ≥0.01")
    else:
        chips.append("hs-cTn pending")

    pf = {str(p).lower() for p in ri.get("pain_features", [])}
    if "radiating" in pf: chips.append("Radiating pain")
    if "crushing" in pf: chips.append("Crushing pressure")

    # young age fallback (ensures CP-1000 surfaces something decisive)
    try:
        if int(ri.get("age", patient.get("age", 0))) < 30:
            chips.append("Young age <30")
    except Exception:
        pass

    onset = patient.get("data_quality",{}).get("time_from_onset_min")
    if isinstance(onset,int) and onset <= 90: chips.append("Early (≤90 min)")
    return chips

# ── Pathway scores (percent labels) ────────────────────────────────────────────
_PATHWAY_SCORES = {
    "CP-1000": [("ACS", 62)],
    "CP-1001": [("ACS", 45), ("Non-cardiac MSK", 35), ("Anxiety", 20)],
    "CP-1002": [("PE", 55), ("Infection/Sepsis", 45)],
    "CP-1003": [("ACS", 60), ("Anxiety/MSK", 40)],
}
def _pathway_badges(pid: str):
    pairs = _PATHWAY_SCORES.get(pid) or []
    html = "".join(
        f"<span style='border-radius:999px;padding:.2rem .6rem;border:1px solid #e5e7eb;"
        f"background:#f8fafc;font-weight:700;margin-right:.25rem'>{escape(n)} — {p}%</span>"
        for (n,p) in pairs
    )
    return html, [n for (n,_) in pairs]

# ── Unified uncertainty panel (popover) ────────────────────────────────────────
def _render_ua_panel():
    summary = st.session_state.get("_ua_summary")
    patient = st.session_state.get("_ua_patient")
    pid = st.session_state.get("selected_patient_id","")
    if not summary or not patient:
        st.info("No uncertainty data available."); return

    # Header
    st.subheader("Evidence & Reasoning")

    # Risk + interval
    base = float(summary["base"]); lo = float(summary["lo"]); hi = float(summary["hi"]); width = float(summary["width"])
    base_pct, lo_pct, hi_pct = _pct(base), _pct(lo), _pct(hi)
    band = band_from_risk(base)
    band_color = {"Low":"#10B981","Low-Moderate":"#22C55E","Moderate":"#F59E0B","High":"#DC2626"}.get(band, "#6b7280")
    band_badge = f"<span style='display:inline-block;border-radius:999px;padding:3px 10px;background:{band_color};color:white;font-weight:800'>{band}</span>"
    st.markdown(
        f"<div style='display:flex;gap:.5rem;flex-wrap:wrap;align-items:center'>"
        f"<div><b>Risk:</b> {base_pct}%</div>{band_badge}</div>", unsafe_allow_html=True)
    st.caption(f"Risk interval {lo_pct}%–{hi_pct}% · Wider = less stable. Details in the tables below.")

    # Pathway suggestion chips — percentages
    chips_html, pathway_names = _pathway_badges(pid)
    st.markdown(f"<div style='margin:.25rem 0'><b>Pathway suggestion:</b> {chips_html}</div>", unsafe_allow_html=True)

    # Decisive inputs (guaranteed)
    dec = _decisive_inputs(patient)
    dhtml = "".join([f"<span style='border:1px solid #e5e7eb;background:#f3f4f6;border-radius:999px;padding:.2rem .55rem;margin:.2rem .25rem .1rem 0;font-weight:700'>{escape(c)}</span>" for c in dec])
    st.markdown(f"<div><b>Decisive inputs:</b> {dhtml}</div>", unsafe_allow_html=True)

    # Overall confidence
    _, _, tri_conf_raw, _ = decompose_uncertainty(summary, patient)
    tri_conf = _nudge(pid, "tri", min(0.99, tri_conf_raw))
    disp_conf = _nudge(pid, "dispo", max(0.15, tri_conf * (1 - 0.80 * st.session_state.get('_SCENARIOS',{}).get(pid,{}).get('cascade',0.0))))
    steps_conf = _nudge(pid, "steps", 0.50 * (tri_conf + disp_conf))
    conf_overall = min(tri_conf, disp_conf, steps_conf)
    conf_tier = _tier_from_conf(conf_overall); conf_pct = int(round(100*conf_overall))
    pill_bg = {"High":"#16a34a","Medium":"#f59e0b","Low":"#f97316"}[conf_tier]
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:.5rem;flex-wrap:wrap'>"
        f"<div style='font-size:20px;font-weight:900'>Confidence: {conf_pct}%</div>"
        f"<span style='display:inline-block;border-radius:999px;padding:3px 10px;background:{pill_bg};color:white;font-weight:800'>{conf_tier}</span>"
        f"</div>", unsafe_allow_html=True
    )

    # Uncertainty composition (detailed model familiarity note for OOD)
    dq = patient.get("data_quality", {}) or {}
    missing = dq.get("missing", []) or []
    ood = 1.0 if dq.get("ood") else 0.0
    human_vague = 0.0 if len((patient.get("chief_complaint") or "")) >= 12 else 1.0
    if pid == "CP-1001":  # explicit ambiguous narrative
        human_vague = 1.0

    fam_note = "Typical for training distribution"
    if ood:
        ri = patient.get("risk_inputs", {}) or {}
        cc = (patient.get("chief_complaint") or "").lower()
        flags = []
        if ri.get("ecg_abnormal"): flags.append("abnormal ECG")
        tro = ri.get("troponin")
        if isinstance(tro,(int,float)) and tro is not None and tro > 0: flags.append(f"hs-cTn {tro:.3f} ng/L")
        if any(k in cc for k in ["fever","febr","cough"]): flags.append("fever/cough symptoms")
        fam_note = "Outside training distribution — uncommon combo: " + ", ".join(flags) if flags else "Outside training distribution"

    w_data = min(1.0, 0.45 * len(missing))
    w_fam  = 0.60 * ood
    w_stab = max(0.0, (width - 0.10) / 0.25)
    w_human = 0.40 * human_vague
    raw = [w_data, w_fam, w_stab, w_human]; total = sum(raw) or 1.0
    pct4 = [int(round(100*x/total)) for x in raw]; pct4[0] += (100 - sum(pct4))
    p0, p1, p2, p3 = pct4
    css = ( "background: conic-gradient("
            "#f59e0b 0 {a}%, #6366f1 {a}% {b}%, #22c55e {b}% {c}%, #e11d48 {c}% 100%);" ).format(
        a=p0, b=p0+p1, c=p0+p1+p2)
    st.markdown(f"<div style='width:110px;height:110px;border-radius:50%;margin:6px 0;{css}'></div>", unsafe_allow_html=True)
    for lab, pct, note, col in zip(
        ["Data gaps","Model familiarity","Prediction stability","Human ambiguity"],
        pct4,
        [
            ("Missing: " + ", ".join(missing)) if missing else "No key inputs missing",
            fam_note,
            f"Risk interval width ≈ {int(round(100*width))} pts, indicating {'High' if width<=0.10 else ('Medium' if width<=0.20 else 'Low')} stability; details below.",
            "Narrative unclear" if human_vague else "Narrative clear",
        ],
        ["#f59e0b","#6366f1","#22c55e","#e11d48"]
    ):
        st.markdown(
            f"<div style='display:flex;gap:.5rem;align-items:flex-start;margin:.15rem 0'>"
            f"<span style='width:10px;height:10px;background:{col};border-radius:2px;margin-top:4px'></span>"
            f"<div><b>{escape(lab)}</b> — {pct}%"
            f"<div style='color:#6b7280;font-size:12px'>{escape(note)}</div></div></div>",
            unsafe_allow_html=True
        )

    # Prediction stability — details
    st.markdown("#### Prediction stability — details")
    st.caption("Input perturbations + model/policy variation that populate the interval.")
    _stability_tables_overall(summary, patient)

    # Next steps (aligned with pathways)
    steps = list(st.session_state.get("_ai_steps") or summary.get("steps") or [])
    _, pathway_names = _pathway_badges(pid)
    pset = {p.lower() for p in pathway_names}
    steps = [
        "Activate chest-pain protocol; continuous monitoring.",
        "Immediate ECG review; repeat in 10–15 min.",
    ]
    if any("acs" in p for p in pset):
        steps += [
            "High-sensitivity troponin STAT; repeat at 1–2 h.",
            "Prepare cardiology consult.",
        ]
    if any("pe" in p for p in pset):
        steps += [
            "Consider PE pathway — D-dimer if low/mod pretest; CTPA if high.",
            "Assess Wells/Geneva score to stratify PE likelihood."
        ]
    if any("infection" in p or "sepsis" in p for p in pset):
        steps += [
            "Consider infection pathway — CBC, CRP, lactate; cultures if febrile.",
            "Chest X-ray if respiratory symptoms; start sepsis screen if indicated."
        ]
    if any("msk" in p for p in pset):
        steps += ["Consider MSK pathway — palpation/ROM exam; NSAID trial if appropriate."]
    if any("anxiety" in p for p in pset):
        steps += ["Consider anxiety pathway — brief screening + reassurance; outpatient follow-up."]

    seen = set(); steps = [s for s in steps if not (s in seen or seen.add(s))]
    st.session_state["_drawer_steps"] = list(steps)

    def _tag_for(s: str):
        s_l = s.lower()
        if any(k in s_l for k in ["obtain","complete","repeat","rule-out","prioritize","re-test","d-dimer","ctpa","x-ray","cbc","lactate","cultures","wells","geneva","screen"]):
            return "Uncertainty-reduction", "#FEF3C7", "#F59E0B"
        if any(k in s_l for k in ["consider msk","consider anxiety","pathway —"]):
            return "Uncertainty-reduction", "#FEF3C7", "#F59E0B"
        if any(k in s_l for k in ["consult","monitor","review","reassess","continuous","protocol","reassurance","follow-up","trial"]):
            return "Care step", "#E0E7FF", "#6366F1"
        return "Care step", "#E0E7FF", "#6366F1"

    st.markdown("#### Next steps")
    cur_conf_pct = int(round(100*min(tri_conf, disp_conf, steps_conf)))
    def _project_conf_after(tag: str, s_text: str, base_conf: float):
        if tag != "Uncertainty-reduction":
            return base_conf, 0
        lift = 0.04
        s_low = s_text.lower()
        if "troponin" in s_low or "biomarker" in s_low or "rule-out" in s_low:
            lift = 0.08
        elif "repeat" in s_low or "re-test" in s_low or "ecg" in s_low or "ctpa" in s_low or "d-dimer" in s_low:
            lift = 0.06
        new_conf = max(base_conf, min(0.99, base_conf + lift))
        delta_pts = int(round(100 * (new_conf - base_conf)))
        return new_conf, delta_pts

    for s in steps:
        tag, bg, bd = _tag_for(s)
        proj_conf, dpts = _project_conf_after(tag, s, min(tri_conf, disp_conf, steps_conf))
        extra = ""
        if dpts > 0:
            extra = f"<span style='margin-left:.4rem;font-size:12px;color:#065f46'>→ Confidence +{dpts} pts (≈ {cur_conf_pct}% → {int(round(100*proj_conf))}%)</span>"
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:.5rem;margin:.25rem 0;'>"
            f"<span style='border-radius:999px;padding:.35rem .7rem;background:#f3f4f6;border:1px solid #e5e7eb;"
            f"font-weight:700;font-size:12px;color:#111827'>{escape(s)}</span>"
            f"<span style='border-radius:999px;padding:.3rem .6rem;background:{bg};border:1px solid {bd};"
            f"font-weight:800;font-size:12px;color:#111827'>{tag}</span>{extra}</div>",
            unsafe_allow_html=True
        )

# ── Page renderer ──────────────────────────────────────────────────────────────
def render_patient_chart():
    _init_state(); _enter_page("Patient Chart"); _render_sidebar(__file__); _show_back("← Back")

    st.session_state.setdefault("selected_patient_id", "CP-1000")
    st.caption("Search / select a patient (type ID or name to filter)")

    # Profiles
    FIXED_PROFILES = {
        "CP-1000": {"MRN":"CP-1000","Patient":"Weber, Charlotte","Age":28,"Sex":"Female","ESI":2,
                    "HR":78,"SBP":128,"SpO₂":98,"TempC":36.9,"ECG":"Normal","hs-cTn (ng/L)":None,
                    "OnsetMin":140,"CC":"Typical chest pressure, now improved.","Arrival":"Ambulance"},
        "CP-1001": {"MRN":"CP-1001","Patient":"Green, Gary","Age":60,"Sex":"Male","ESI":3,
                    "HR":96,"SBP":136,"SpO₂":96,"TempC":36.8,"ECG":"Abnormal","hs-cTn (ng/L)":None,
                    "OnsetMin":40,"CC":"Not sure… feels weird in chest, comes and goes. Pain hard to describe.","Arrival":"Walk-in"},
        "CP-1002": {"MRN":"CP-1002","Patient":"Lopez, Mariah","Age":44,"Sex":"Female","ESI":3,
                    "HR":94,"SBP":128,"SpO₂":98,"TempC":37.2,"ECG":"Abnormal","hs-cTn (ng/L)":0.008,
                    "OnsetMin":25,"CC":"Atypical chest tightness with diaphoresis; cough and feverish.","Arrival":"Ambulance"},
        "CP-1003": {"MRN":"CP-1003","Patient":"Crown, Emma","Age":58,"Sex":"Female","ESI":3,
                    "HR":88,"SBP":132,"SpO₂":97,"TempC":36.9,"ECG":"Abnormal","hs-cTn (ng/L)":0.012,
                    "OnsetMin":70,"CC":"Intermittent chest pressure at rest, now mild.","Arrival":"Walk-in"},
    }
    # Scenario mapping (base risk, stability, cascade)
    SCENARIOS = {
        "CP-1000": {"base_pct": 12, "stability": "High",   "cascade": 0.00},  # A
        "CP-1001": {"base_pct": 28, "stability": "Medium", "cascade": 0.05},  # B
        "CP-1002": {"base_pct": 41, "stability": "Low",    "cascade": 0.20},  # C
        "CP-1003": {"base_pct": 33, "stability": "Medium", "cascade": 0.15},  # E
    }
    st.session_state["_SCENARIOS"] = SCENARIOS

    ids = list(FIXED_PROFILES.keys())
    cur = st.session_state.get("selected_patient_id") or ids[0]
    if cur not in ids: cur = ids[0]

    def _fmt(pid: str) -> str:
        d = FIXED_PROFILES.get(pid, {})
        return f"{pid} — {d.get('Patient','')}"
    sel_id = st.selectbox("", options=ids, index=ids.index(cur), format_func=_fmt, key="patient_select")
    if sel_id != st.session_state.get("selected_patient_id"):
        st.session_state["ua_open"] = False
    st.session_state["selected_patient_id"] = sel_id

    base = dict(FIXED_PROFILES.get(sel_id))
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

    # scenario flags
    if sel_id == "CP-1001":
        patient["chief_complaint"] = base["CC"]
        patient["risk_inputs"]["troponin"] = None
        if "troponin" not in patient["data_quality"]["missing"]:
            patient["data_quality"]["missing"].append("troponin")
    if sel_id == "CP-1002":
        patient["data_quality"]["ood"] = True
    if sel_id == "CP-1003":
        patient["data_quality"]["time_from_onset_min"] = 70

    # Header card
    left, right = st.columns([0.70, 0.30], vertical_alignment="center")
    with left:
        st.markdown(f"## {base.get('Patient','—')}")
        st.caption(f"MRN: {patient['mrn']}")
    with right: st.empty()

    st.markdown("""
    <style>
      .ua-infocard{background:#f3f4f6;border:1px solid #e5e7eb;border-radius:10px;padding:10px 12px;margin-top:.25rem}
      .ua-kv{display:grid;grid-template-columns:120px 1fr;row-gap:6px;column-gap:10px;font-size:14px;color:#374151}
      .ua-kv b{color:#111827}
      .ua-line{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
      .ua-label{font-weight:800}
      .ua-muted{color:#6b7280;font-size:13px}
      .ua-badge{border-radius:999px;background:#EEF2FF;color:#1e40af;border:1px solid #93C5FD;padding:.2rem .6rem;font-weight:700}
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

    # Summary (manual width mapping per scenario)
    summary = simple_summary_from_manual(scen.get("base_pct", 25), scen.get("stability", "Medium"), patient)
    st.session_state["_ua_summary"] = summary
    st.session_state["_ua_patient"] = patient

    # ======== AI Recommendations (header + ? on one line) ========
    row = st.columns([0.90, 0.10])
    with row[0]:
        st.markdown("### AI Recommendations")
    with row[1]:
        try:
            with st.popover("❓", use_container_width=True, help="Why these recommendations? Confidence, intervals, uncertainty composition."):
                _render_ua_panel()
        except Exception:
            st.button("❓", key="ua_btn",
                      help="Why these recommendations? Confidence, intervals, and uncertainty composition.",
                      on_click=_open_drawer, args=("ua",))

    # Triage
    st.divider()
    tri_here = _triage_from_base(float(summary["base"]))
    t_code, t_label, t_desc = tri_here["code"], tri_here["label"], tri_here["desc"]
    color_map = {"T1":"#DC2626","T2":"#F97316","T3":"#F59E0B","T4":"#22C55E","T5":"#10B981"}
    t_color = color_map[t_code]
    st.markdown(
        f"""
        <div class="ua-line" style="margin-top:.2rem">
        <span class="ua-label">AI Triage Recommendation:</span>
        <span style='border-radius:999px;padding:.25rem .6rem;border:1px solid {t_color};
                     background:{t_color};color:white;font-weight:700'>{escape(t_code)}</span>
        <div><b>{escape(t_label)}</b><br><span class="ua-muted">{escape(t_desc)}</span></div>
        </div>
        """, unsafe_allow_html=True)

    for k in ["T1","T2","T3","T4","T5"]:
        lab, desc, colhex = {
            "T1":("Immediate","Highest probability for intensive care, emergency procedure, or mortality.", "#DC2626"),
            "T2":("Very Urgent","Elevated probability for intensive care, emergency procedure, or mortality.", "#F97316"),
            "T3":("Urgent","Moderate probability of hospital admission or very low probability of intensive care, emergency procedure, or mortality.", "#F59E0B"),
            "T4":("Less Urgent","Low probability of hospital admission.", "#22C55E"),
            "T5":("Non-Urgent","Fast turnaround and low probability of hospital admission.", "#10B981")
        }[k]
        is_ai = (k == t_code)
        st.markdown(
            f"""
            <div style="display:flex;align-items:flex-start;gap:.6rem;margin:.35rem 0;
                        border:1px solid {'#3B82F6' if is_ai else '#e5e7eb'};
                        background:{'#EEF2FF' if is_ai else 'white'};
                        border-radius:10px;padding:10px 12px;">
              <div style="margin-top:1px"><span style='border-radius:999px;padding:.25rem .6rem;border:1px solid {colhex};
                         background:{colhex};color:white;font-weight:700'>{escape(k)}</span></div>
              <div style="flex:1">
                <b>{escape(lab)}</b><br>
                <span style='color:#6b7280;font-size:13px'>{escape(desc)}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Clinician triage — default same as AI
    st.markdown("<div style='margin:.25rem 0 .1rem 0;color:#6b7280'>Clinician Triage</div>", unsafe_allow_html=True)
    tri_opts = ["T1","T2","T3","T4","T5"]
    st.radio("", options=tri_opts, index=tri_opts.index(t_code), key="clin_triage_choice", horizontal=True, label_visibility="collapsed")

    # Disposition (all options)
    st.divider()
    disp_pct = _DISPO_BASE_PCT.get(sel_id, int(round(100*summary["base"])))
    disp_default = _dispo_label_from_pct(disp_pct)
    st.markdown(f"<div class='ua-line' style='margin-top:.2rem'><span class='ua-label'>AI Disposition Recommendation:</span> <span class='ua-badge'>{escape(disp_default)}</span></div>", unsafe_allow_html=True)

    dispo_defs = [
        ("Confirm/Admit", "Admit or confirm acute management plan — likely inpatient treatment."),
        ("Observe", "Monitor in observation unit — reassess after a short period."),
        ("Consult", "Seek specialist input before final decision."),
        ("Defer/Discharge", "Safe for discharge — provide safety-net and follow-up."),
    ]
    for lab, desc in dispo_defs:
        is_ai = (lab == disp_default)
        st.markdown(
            f"""
            <div style="display:flex;align-items:flex-start;gap:.6rem;margin:.35rem 0;
                        border:1px solid {'#3B82F6' if is_ai else '#e5e7eb'};
                        background:{'#EEF2FF' if is_ai else 'white'};
                        border-radius:10px;padding:10px 12px;">
              <div style="flex:1">
                <span style='color:#111827;font-weight:700'>{escape(lab)}</span><br>
                <span style='color:#6b7280;font-size:13px'>{escape(desc)}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Clinician disposition — default same as AI
    st.markdown("<div style='margin:.25rem 0 .1rem 0;color:#6b7280'>Clinician Disposition</div>", unsafe_allow_html=True)
    dispo_opts = ["Confirm/Admit","Observe","Consult","Defer/Discharge"]
    st.radio("", options=dispo_opts, index=dispo_opts.index(disp_default), key="clin_dispo_choice", horizontal=True, label_visibility="collapsed")

    # Next steps (pathway-aware; mirrors drawer)
    st.divider()
    _, names = _pathway_badges(sel_id)
    pset = {n.lower() for n in names}
    ai_steps = [
        "Activate chest-pain protocol; continuous monitoring.",
        "Immediate ECG review; repeat in 10–15 min.",
    ]
    if any("acs" in p for p in pset):
        ai_steps += [
            "High-sensitivity troponin STAT; repeat at 1–2 h.",
            "Prepare cardiology consult.",
        ]
    if any("pe" in p for p in pset):
        ai_steps += [
            "Consider PE pathway — D-dimer if low/mod pretest; CTPA if high.",
            "Assess Wells/Geneva score to stratify PE likelihood."
        ]
    if any("infection" in p or "sepsis" in p for p in pset):
        ai_steps += [
            "Consider infection pathway — CBC, CRP, lactate; cultures if febrile.",
            "Chest X-ray if respiratory symptoms; start sepsis screen if indicated."
        ]
    if any("msk" in p for p in pset):
        ai_steps += ["Consider MSK pathway — palpation/ROM exam; NSAID trial if appropriate."]
    if any("anxiety" in p for p in pset):
        ai_steps += ["Consider anxiety pathway — brief screening + reassurance; outpatient follow-up."]

    seen = set(); ai_steps = [s for s in ai_steps if not (s in seen or seen.add(s))]
    st.session_state["_ai_steps"] = list(ai_steps)

    st.markdown("**AI Next-Steps Suggestions**")
    st.caption("Check a box to indicate you agree/accept that AI suggestion. Unchecked = not accepted.")
    for i, step in enumerate(ai_steps):
        c = st.columns([0.06, 0.94])
        with c[0]: st.checkbox("", key=f"agree_step_{i}")
        with c[1]:
            st.markdown(
                f"<span style='border-radius:999px;padding:.25rem .6rem;background:#3B82F6;color:#fff;"
                f"margin-right:.35rem;margin-bottom:.35rem;display:inline-block;font-weight:600'>{escape(step)}</span>",
                unsafe_allow_html=True
            )

    st.text_area("Notes (optional)",
                 placeholder=("If you disagree with the AI triage or disposition recommendation, explain your rationale here. "
                              "If additional next steps are needed, list them here."),
                 height=160, key="clin_notes")
    st.button("Save", type="primary", key="save_btn")

    # Fallback sidebar drawer if popover not available
    if st.session_state.get("ua_open"):
        _, right_col = _drawer_cols(True)
        with right_col:
            c_top = st.columns([0.92, 0.08])
            with c_top[1]:
                if st.button("≪", key="ua_collapse", help="Hide details", type="secondary"):
                    _close_drawer(); st.stop()
            _render_ua_panel()

if __name__ == "__main__":
    render_patient_chart()
else:
    render_patient_chart()