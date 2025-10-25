# components/patient_view.py
import math
from datetime import datetime
import streamlit as st

# ───────────────────────── Uncertainty-aware toy model ─────────────────────────
def toy_risk_model(inputs: dict) -> float:
    score = 0.0
    score += 0.015 * (inputs["age"] - 40)               # age
    score += 0.15 if inputs["sex"] == 1 else 0.0        # 1=male, 0=female
    score += 0.25 if inputs["ecg_abnormal"] else 0.0    # abnormal ECG
    if inputs.get("troponin") is not None:
        score += 0.35 * min(1.0, max(0.0, (inputs["troponin"] / 0.04)))
    pf = inputs.get("pain_features", [])
    score += 0.1 if "radiating" in pf else 0.0
    score += 0.08 if "crushing" in pf else 0.0
    score += 0.05 if inputs.get("rf_smoker") else 0.0
    score += 0.03 if inputs.get("rf_htn") else 0.0
    score += 0.04 if inputs.get("rf_dm") else 0.0
    risk = 1 / (1 + math.exp(- (score - 0.5)))
    return float(min(0.98, max(0.01, risk)))

def widen_interval(base_risk: float, data_quality: dict):
    width = 0.08
    width += 0.07 * len(data_quality.get("missing", []))
    width += 0.08 if data_quality.get("ood") else 0.0
    if data_quality.get("time_from_onset_min", 999) < 90:
        width += 0.05
    width = float(min(0.35, max(0.05, width)))
    lo = max(0.0, base_risk - width/2)
    hi = min(1.0, base_risk + width/2)
    if (hi - lo) <= 0.10: conf = ("High", "🟢")
    elif (hi - lo) <= 0.20: conf = ("Medium", "🟡")
    else: conf = ("Low", "🟠")
    return lo, hi, conf

def pct(p: float) -> str:
    return f"{round(100*p)}%"

def band_from_risk(r: float) -> str:
    if r < 0.10: return "Low"
    if r < 0.20: return "Low-Moderate"
    if r < 0.40: return "Moderate"
    return "High"

def drivers_from_inputs(p: dict):
    ri = p["risk_inputs"]; d = []
    d += ["Abnormal ECG"] if ri.get("ecg_abnormal") else []
    if ri.get("troponin") is None: d += ["Troponin pending"]
    elif ri["troponin"] >= 0.01: d += [f"Troponin {ri['troponin']:.3f} ng/mL"]
    if "radiating" in ri.get("pain_features", []): d += ["Radiating pain"]
    if "crushing" in ri.get("pain_features", []): d += ["Crushing pressure"]
    if "diaphoresis" in ri.get("pain_features", []): d += ["Diaphoresis"]
    if ri.get("rf_smoker"): d += ["Smoker"]
    if ri.get("rf_htn"): d += ["Hypertension"]
    if ri.get("rf_dm"): d += ["Diabetes"]
    return d or ["All vitals & ECG reassuring"]

def next_steps_plan(base: float, width: float, critical: bool):
    steps = []
    if base >= 0.40 or critical:
        steps += [
            "Activate chest-pain protocol; continuous monitoring.",
            "Immediate ECG review; repeat in 10–15 min.",
            "High-sensitivity troponin STAT; repeat at 1–2 h.",
            "Prepare cardiology consult.",
        ]
    elif base >= 0.20:
        steps += [
            "hs-Troponin now; repeat per rule-out algorithm.",
            "Observe with continuous vitals; reassess pain.",
        ]
    else:
        steps += [
            "Obtain/complete troponin for rule-out.",
            "If serial biomarkers & ECG remain normal → consider discharge with rapid follow-up.",
        ]
    if width > 0.20:
        steps.insert(0, "⚠️ Estimate uncertain → prioritize missing data / repeat tests before disposition.")
    return steps

def make_patient_from_row(row: dict):
    def _get(k, default=None):
        v = row.get(k, default)
        return default if v in [None, ""] else v

    name = _get("Patient", "Unknown")
    age  = int(_get("Age", 50))
    sex  = _get("Sex", "Unknown")
    ecg_normal = str(_get("ECG", "Normal")).lower().strip() == "normal"
    troponin = _get("hs-cTn (ng/L)", None) or _get("hs-cTn", None)
    try:
        troponin = float(troponin) if troponin not in [None, "—", ""] else None
    except Exception:
        troponin = None

    return {
        "name": name,
        "mrn": _get("PatientID", _get("ID", "—")),
        "dob": "—",
        "age": age,
        "sex": sex,
        "arrival_mode": _get("Arrival", "—"),
        "onset": _get("Onset", _get("OnsetMin", "—")),
        "chief_complaint": _get("CC", "Chest pain."),
        "vitals": {
            "BP": f"{_get('SBP', 120)}/{_get('DBP', 80)}",
            "HR": int(_get("HR", 80)),
            "RR": int(_get("RR", 18)),
            "SpO2": int(_get("SpO₂", _get("SpO2", 96))),
            "TempF": float(_get("TempF", 98.6)),
        },
        "risk_inputs": {
            "age": age,
            "sex": 1 if str(sex).lower().startswith("m") else 0,
            "ecg_abnormal": (not ecg_normal),
            "troponin": troponin,
            "pain_features": ["pressure"],
            "rf_smoker": bool(_get("Smoker", False)),
            "rf_htn": bool(_get("HTN", False)),
            "rf_dm": bool(_get("DM", False)),
        },
        "data_quality": {
            "missing": [] if troponin is not None else ["troponin"],
            "ood": bool(_get("OOD", False)),
            "time_from_onset_min": int(_get("OnsetMin", 120)) if str(_get("OnsetMin", "")).isdigit() else 120,
        },
    }

def compute_summary(p: dict):
    base = toy_risk_model(p["risk_inputs"])
    lo, hi, (conf_txt, conf_dot) = widen_interval(base, p["data_quality"])
    width = hi - lo
    critical = p["risk_inputs"]["ecg_abnormal"] or (
        p["risk_inputs"].get("troponin") is not None and p["risk_inputs"]["troponin"] >= 0.04
    )
    steps = next_steps_plan(base, width, critical)
    drivers = drivers_from_inputs(p)
    return {
        "base": base, "lo": lo, "hi": hi, "width": width,
        "conf_txt": conf_txt, "conf_dot": conf_dot,
        "band": band_from_risk(base),
        "drivers": drivers, "steps": steps
    }

# Role-specific panels (no duplicate metrics/risk/confidence here)
def render_patient_panel(summary: dict):
    st.markdown("**What this means for you**")
    nf_mid, nf_lo, nf_hi = round(summary["base"]*100), round(summary["lo"]*100), round(summary["hi"]*100)
    st.markdown(
        f"- Out of **100** people like you, about **{nf_mid}** might have a serious heart problem.\n"
        f"- A reasonable range for now is **{nf_lo}–{nf_hi} out of 100** until we get more results.\n"
        "- We’ll follow the steps above (blood test, repeat ECG, monitoring)."
    )
    st.markdown("**What is affecting your risk:** " + ", ".join(summary["drivers"]))

def render_clinician_panel(summary: dict):
    st.markdown("**Primary drivers:** " + ", ".join(summary["drivers"]))
    with st.expander("Uncertainty explanation: quick list → detailed"):
        st.markdown(
            "- **Quick list:** early presentation, missing key features, potential distribution shift.\n"
            "- **Detailed:** interval widens with missing hs-troponin, early window (<90 min), and OOD flag; "
            "confidence tiers: width ≤10% = high; ≤20% = medium; >20% = low. "
            "This is a triage risk for **serious cardiac cause**; apply clinical judgment and local pathways."
        )