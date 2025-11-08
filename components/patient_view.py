# components/patient_view.py
# Utilities for building patient objects and UI-friendly summaries.
# This keeps the original layout helpers, AND adds a simple "Approach B" summary
# that maps a manual base risk (%) + stability label (High/Medium/Low)
# to an interval (lo/hi) and confidence — purely for display.

import math

def toy_risk_model(inputs: dict) -> float:
    """(Kept for compatibility) A toy logistic risk model. Not used in Approach B."""
    score = 0.0
    score += 0.015 * (inputs["age"] - 40)
    score += 0.15 if inputs["sex"] == 1 else 0.0
    score += 0.25 if inputs["ecg_abnormal"] else 0.0
    if inputs.get("troponin") is not None:
        score += 0.35 * min(1.0, max(0.0, float(inputs["troponin"]) / 0.04))
    pf = inputs.get("pain_features", [])
    score += 0.1 if "radiating" in pf else 0.0
    score += 0.08 if "crushing" in pf else 0.0
    score += 0.05 if inputs.get("rf_smoker") else 0.0
    score += 0.03 if inputs.get("rf_htn") else 0.0
    score += 0.04 if inputs.get("rf_dm") else 0.0
    risk = 1 / (1 + math.exp(-(score - 0.5)))
    return float(min(0.98, max(0.01, risk)))

def widen_interval(base_risk: float, data_quality: dict):
    """(Kept for compatibility) Interval widening by data quality. Not used in Approach B."""
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
    """Simple, UI-level suggested steps based on base risk + interval width."""
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
    """Normalize a track-board row dict into a patient dict used by the chart UI."""
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
            "TempC": float(_get("TempC", 37.0)),
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
    """(Kept for compatibility) Original path using toy model + widen_interval."""
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
        "drivers": drivers, "steps": steps, "critical": critical,
        "suspected_condition": "ACS-like"
    }

def triage_level_from_summary(s: dict):
    if s["critical"] or s["base"] >= 0.50:
        return {"code": "T1", "label": "Immediate", "desc": "Highest probability for intensive care, emergency procedure, or mortality."}
    if s["base"] >= 0.35:
        return {"code": "T2", "label": "Very Urgent", "desc": "Elevated probability for intensive care, emergency procedure, or mortality."}
    if s["base"] >= 0.20:
        return {"code": "T3", "label": "Urgent", "desc": "Moderate probability of hospital admission or very low probability of intensive care, emergency procedure, or mortality."}
    if s["base"] >= 0.10:
        return {"code": "T4", "label": "Less Urgent", "desc": "Low probability of hospital admission."}
    return {"code": "T5", "label": "Non-Urgent", "desc": "Fast turnaround and low probability of hospital admission."}

def disposition_from_summary(s: dict):
    if s["critical"] or s["base"] >= 0.40: return "Confirm/Admit"
    if s["base"] >= 0.20: return "Observe"
    if s["base"] >= 0.10: return "Consult"
    return "Defer/Discharge"

# ---------- Approach B: UI-only mapping (NEW) ----------

def width_from_stability_label(stab: str) -> int:
    """Map stability label to interval width in points (0–100 scale)."""
    m = {"High": 6, "Medium": 18, "Low": 30}
    return m.get(str(stab).title(), 18)

def confidence_from_width(width_pts: int) -> str:
    """Confidence category derived from interval width."""
    if width_pts <= 6: return "High"
    if width_pts <= 12: return "Medium"
    return "Low"

def simple_summary_from_manual(base_pct: float, stability_label: str, patient: dict):
    """
    Summary for display only:
      - base_pct: manual risk point (0–100)
      - stability_label: 'High'/'Medium'/'Low' → width points 6/18/30
      - interval is centered on base and clipped to [0,100]
      - confidence derives from width
    """
    width_pts = width_from_stability_label(stability_label)
    lo = max(0.0, base_pct - width_pts / 2.0)
    hi = min(100.0, base_pct + width_pts / 2.0)
    conf_txt = confidence_from_width(width_pts)
    conf_dot = {"High": "🟢", "Medium": "🟡", "Low": "🟠"}[conf_txt]

    ri = patient.get("risk_inputs", {})
    critical = bool(ri.get("ecg_abnormal")) or (
        ri.get("troponin") is not None and float(ri["troponin"]) >= 0.04
    )

    steps = next_steps_plan(base_pct / 100.0, (hi - lo) / 100.0, critical)

    return {
        "base": base_pct / 100.0,
        "lo": lo / 100.0,
        "hi": hi / 100.0,
        "width": (hi - lo) / 100.0,
        "conf_txt": conf_txt,
        "conf_dot": conf_dot,
        "band": band_from_risk(base_pct / 100.0),
        "drivers": drivers_from_inputs(patient),
        "steps": steps,
        "critical": critical,
        "suspected_condition": "ACS-like",
    }

# ---------- Uncertainty helpers used by the drawers ----------

def decompose_uncertainty(summary: dict, patient: dict):
    """
    Return (aleatoric_pct, epistemic_pct, conf_score_0_1, tier).
    Uses interval width as the main driver for confidence; adds small signals for data quality.
    """
    width = summary["width"]
    conf_score = max(0.0, min(1.0, 1.0 - (width / 0.35)))
    miss = len(patient["data_quality"].get("missing", []))
    early = 1 if patient["data_quality"].get("time_from_onset_min", 999) < 90 else 0
    ood = 1 if patient["data_quality"].get("ood") else 0
    epi_signal = 0.18*miss + 0.12*early + 0.20*ood
    epi = min(0.80, max(0.0, epi_signal))
    alea_raw = min(0.90, max(0.05, width / 0.35))
    total = alea_raw + epi
    alea = alea_raw / total if total > 0 else 0.5
    epis = epi / total if total > 0 else 0.5
    tier = "High" if conf_score >= 0.70 else ("Medium" if conf_score >= 0.40 else "Low")
    return alea*100, epis*100, conf_score, tier