# components/patient_view.py
import math
import streamlit as st
from typing import Dict

# ───────────────────────── Uncertainty-aware toy model ─────────────────────────
def toy_risk_model(inputs: dict) -> float:
    score = 0.0
    score += 0.015 * (inputs["age"] - 40)
    score += 0.15 if inputs["sex"] == 1 else 0.0
    score += 0.25 if inputs["ecg_abnormal"] else 0.0
    if inputs.get("troponin") is not None:
        score += 0.35 * min(1.0, max(0.0, (inputs["troponin"] / 0.04)))
    pf = inputs.get("pain_features", [])
    score += 0.1 if "radiating" in pf else 0.0
    score += 0.08 if "crushing" in pf else 0.0
    score += 0.05 if inputs.get("rf_smoker") else 0.0
    score += 0.03 if inputs.get("rf_htn") else 0.0
    score += 0.04 if inputs.get("rf_dm") else 0.0
    risk = 1 / (1 + math.exp(-(score - 0.5)))
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
    if (hi - lo) <= 0.10:
        conf = ("High", "🟢")
    elif (hi - lo) <= 0.20:
        conf = ("Medium", "🟡")
    else:
        conf = ("Low", "🟠")
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
    if ri.get("troponin") is None:
        d += ["Troponin pending"]
    elif ri["troponin"] >= 0.01:
        d += [f"Troponin {ri['troponin']:.3f} ng/mL"]
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

# ====== Triage + Disposition rules (simple, tunable) ======
def triage_level_from_summary(s: dict):
    if s["critical"] or s["base"] >= 0.50:
        return {"code": "T1", "label": "Immediate", "desc": "High risk for intensive care, emergency procedure, or mortality."}
    if s["base"] >= 0.35:
        return {"code": "T2", "label": "Emergent", "desc": "Elevated risk for intensive care, emergency procedure, or mortality."}
    if s["base"] >= 0.20:
        return {"code": "T3", "label": "Urgent", "desc": "Moderate risk of hospital admission; very low risk of intensive care."}
    if s["base"] >= 0.10:
        return {"code": "T4", "label": "Less urgent", "desc": "Low risk of hospital admission."}
    return {"code": "T5", "label": "Non-urgent", "desc": "Fast-track; very low risk of admission."}

def disposition_from_summary(s: dict):
    if s["critical"] or s["base"] >= 0.40: return "Confirm/Admit"
    if s["base"] >= 0.20: return "Observe"
    if s["base"] >= 0.10: return "Consult"
    return "Defer/Discharge"

# ====== Confidence score, aleatoric/epistemic split, tiers ======
def decompose_uncertainty(summary: dict, patient: dict):
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
    if conf_score < 0.40: tier = "Low"
    elif conf_score < 0.70: tier = "Medium"
    else: tier = "High"
    return alea*100, epis*100, conf_score, tier

def pct_tier(score: float, tier: str):
    return f"{round(score*100)}% <span style='color:#6b7280'>(<b>{tier}</b>)</span>"

# ====== Extras used in the new layout ======
def sens_table():
    rows = [
        ("HR",  "+5%", "+2%",  "-1%"),
        ("SBP", "+5%", "+0%",  "+1%"),
        ("SpO₂", "±5%", "+3%", "—"),
    ]
    html = [
        "<table style='width:100%;border-collapse:collapse;font-size:12px'>",
        "<thead><tr>",
        "<th style='text-align:left;border-bottom:1px solid #e5e7eb;padding:4px 0'>Vital</th>",
        "<th style='text-align:left;border-bottom:1px solid #e5e7eb;padding:4px 0'>Change</th>",
        "<th style='text-align:left;border-bottom:1px solid #e5e7eb;padding:4px 0'>Output ↑</th>",
        "<th style='text-align:left;border-bottom:1px solid #e5e7eb;padding:4px 0'>Output ↓</th>",
        "</tr></thead><tbody>",
    ]
    for v, delta, up, dn in rows:
        html.append(
            f"<tr><td style='padding:4px 0'>{v}</td>"
            f"<td style='padding:4px 0'>{delta}</td>"
            f"<td style='padding:4px 0'>{up}</td>"
            f"<td style='padding:4px 0'>{dn}</td></tr>"
        )
    html.append("</tbody></table>")
    return "".join(html)

def pattern_similarity(patient: dict) -> float:
    sim = 0.3
    if patient["risk_inputs"].get("ecg_abnormal"): sim += 0.3
    t = patient["risk_inputs"].get("troponin")
    if t is not None:
        sim += min(0.4, t / 0.04 * 0.4)
    return float(min(0.98, max(0.02, sim)))

def alea_reason(summary: dict, patient: dict) -> str:
    w = summary["width"]
    hints = []
    if patient["data_quality"].get("time_from_onset_min", 999) < 90:
        hints.append("early presentation (<90 min)")
    if "troponin" in patient["data_quality"].get("missing", []):
        hints.append("troponin pending")
    if not hints: hints.append("physiologic variability in symptoms")
    tier = "low" if w <= 0.10 else ("medium" if w <= 0.20 else "high")
    return f"interval width suggests {tier} aleatoric uncertainty; drivers: {', '.join(hints)}."

def epi_reason(summary: dict, patient: dict) -> str:
    ood = patient["data_quality"].get("ood", False)
    miss = patient["data_quality"].get("missing", [])
    parts = []
    parts.append("no OOD signals" if not ood else "slight distribution shift vs. training")
    parts.append("no missing key data" if not miss else f"missing: {', '.join(miss)}")
    return "Model familiarity & limits: " + "; ".join(parts)

# ── Role-specific panels (concise; no duplicate metrics) ──
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
    return

# ───────────────────── Confidence & Uncertainty renderer ──────────────────────
def render_confidence_uncertainty(*, sum_dict: dict, patient: dict):
    def _pct_int(x) -> int:
        try: return int(round(100*float(x)))
        except: return 0

    def _badge(text, tone="info"):
        colors = {
            "success": ("#065f46", "#d1fae5"),
            "warn":    ("#92400e", "#fef3c7"),
            "danger":  ("#7f1d1d", "#fee2e2"),
            "info":    ("#1e40af", "#dbeafe"),
            "neutral": ("#374151", "#e5e7eb"),
        }
        fg, bg = colors.get(tone, colors["neutral"])
        return f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;font-weight:600;color:{fg};background:{bg};font-size:12px'>{text}</span>"

    def _tier_badge(tier: str):
        t = (tier or "").lower()
        if t == "high":   return _badge("High", "success")
        if t == "medium": return _badge("Medium", "warn")
        return _badge("Low", "danger")

    def _info_icon(tip: str):
        safe = (tip or "").replace("'", "&#39;")
        return f"<span title='{safe}' style='color:#6b7280;margin-left:6px'>ℹ️</span>"

    def _h4(txt): 
        return f"<div style='font-weight:700;font-size:15px;margin-bottom:.4rem'>{txt}</div>"

    def _uncertainty_intensity(conf_score: float) -> str:
        try: cs = float(conf_score)
        except: cs = 0.5
        if cs < 0.35:  return "High"
        if cs < 0.70:  return "Moderate"
        return "Low"

    def _tone(level: str):
        return {"High":"danger","Moderate":"warn","Low":"success"}.get(level, "neutral")

    def _risk_band_badge(base_pct: int):
        band = band_from_risk(base_pct/100.0)
        tone = "success" if band in ["Low","Low-Moderate"] else ("warn" if band=="Moderate" else "danger")
        return _badge(band, tone)

    with st.expander("Confidence & Uncertainty", expanded=False):

        alea_pct, epis_pct, conf_score, conf_tier = decompose_uncertainty(sum_dict, patient)
        conf_pct = _pct_int(conf_score)
        unc_intensity = _uncertainty_intensity(conf_score)
        dom_type = "Aleatoric" if alea_pct >= epis_pct else "Epistemic"

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

        st.markdown("### Clinical reasoning layer — why this patient’s risk is high/low")
        a1, a2 = st.columns(2)

        with a1:
            lo, hi = _pct_int(sum_dict.get("lo", 0)), _pct_int(sum_dict.get("hi", 0))
            base = _pct_int(sum_dict.get("base", 0))
            drivers = sum_dict.get("drivers", []) or []
            chips = " ".join(_badge(str(d), "neutral") for d in drivers) if drivers else "<span style='color:#6b7280'>—</span>"
            st.markdown(
                _h4("Risk") +
                f"<div style='display:flex;align-items:center;gap:.5rem'>"
                f"<div style='font-size:22px;font-weight:800'>{base}%</div>{_risk_band_badge(base)}</div>"
                f"<div style='color:#6b7280;margin-top:2px'>Point risk</div>"
                f"<div style='margin-top:.5rem'><b>Interval:</b> {lo}% – {hi}% <span style='color:#6b7280'>(uncertainty range)</span></div>"
                f"<div style='margin-top:.5rem'><b>Contributing factors:</b> {chips}</div>",
                unsafe_allow_html=True
            )

        with a2:
            pr_sim = pattern_similarity(patient) or 0.0
            suspected = sum_dict.get("suspected_condition", "ACS-like")
            st.markdown(
                _h4("Detected diagnosis / pattern recognition") +
                f"<div><b>Similarity:</b> <span style='font-weight:800'>{_pct_int(pr_sim)}%</span>"
                f" to <span style='font-weight:600'>{suspected}</span> cluster.</div>",
                unsafe_allow_html=True
            )

        clinical_reason = alea_reason(sum_dict, patient) if dom_type == "Aleatoric" else epi_reason(sum_dict, patient)
        st.markdown(
            f"<div style='margin-top:.75rem;padding:.6rem .8rem;border:1px solid #e5e7eb;border-radius:10px;background:#fafafa'>"
            f"<div style='display:flex;align-items:center;gap:.5rem;font-weight:700'>"
            f"Uncertainty type — {dom_type} "
            f"{_badge(unc_intensity, _tone(unc_intensity))}</div>"
            f"<div style='margin-top:.35rem;color:#374151'><b>Drivers:</b> {clinical_reason}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.divider()

        st.markdown("### Model reasoning layer — how much the model trusts itself")
        m1, m2, m3 = st.columns(3)

        with m1:
            missing = (patient.get("data_quality", {}) or {}).get("missing", []) or []
            if not missing:
                tag = _tier_badge("High"); detail = "<span style='color:#6b7280'>No missing data</span>"
            else:
                tag = _tier_badge("Medium" if len(missing) <= 2 else "Low")
                chips = " ".join(_badge(m, "neutral") for m in missing)
                detail = f"Missing: {chips}"
            st.markdown(_h4("Data completeness") + f"<div>{tag}</div><div style='margin-top:.4rem'>{detail}</div>",
                        unsafe_allow_html=True)

        with m2:
            dq = patient.get("data_quality", {}) or {}
            ood = dq.get("ood", False)
            if ood:
                tag = _tier_badge("Low"); detail = "<span style='color:#6b7280'>Outside typical training range</span>"
            else:
                tag = _tier_badge("High"); detail = "<span style='color:#6b7280'>Typical</span>"
            st.markdown(_h4("Model familiarity") + f"<div>{tag}</div><div style='margin-top:.4rem'>{detail}</div>",
                        unsafe_allow_html=True)

        with m3:
            stab_tag = _tier_badge("High")
            st.markdown(_h4("Prediction stability") + f"<div>{stab_tag}</div>", unsafe_allow_html=True)
            st.caption("Details:")
            st.markdown(f"<div style='margin-top:.2rem'>{sens_table()}</div>", unsafe_allow_html=True)

        model_reason = epi_reason(sum_dict, patient) if dom_type == "Epistemic" else alea_reason(sum_dict, patient)
        st.markdown(
            f"<div style='margin-top:.75rem;padding:.6rem .8rem;border:1px solid #e5e7eb;border-radius:10px;background:#fafafa'>"
            f"<div style='display:flex;align-items:center;gap:.5rem;font-weight:700'>"
            f"Uncertainty type — {dom_type} "
            f"{_badge(unc_intensity, _tone(unc_intensity))}</div>"
            f"<div style='margin-top:.35rem;color:#374151'><b>Drivers:</b> {model_reason}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

# ─────────────────────────── Tabs (NEW) ───────────────────────────
def status_tabs(state: Dict) -> None:
    """
    Render overview tabs below the header: Current | History | Results.
    Read-only; values are pulled from `state`.
    """
    tab1, tab2, tab3 = st.tabs(["Current", "History", "Results"])

    with tab1:
        st.subheader("Current Vitals")
        c1, c2, c3, c4 = st.columns(4)
        hr   = state.get("HR", "—")
        sbp  = state.get("SBP", "—")
        spo2 = state.get("SpO₂", state.get("SpO2", "—"))
        tmp  = state.get("TempC", "—")

        c1.metric("HR",   f"{hr} bpm"   if hr  != "—" else "—")
        c2.metric("SBP",  f"{sbp} mmHg" if sbp != "—" else "—")
        c3.metric("SpO₂", f"{spo2} %"   if spo2!= "—" else "—")
        c4.metric("Temp", f"{tmp} °C"   if tmp != "—" else "—")

        last = state.get("VitalsUpdated") or state.get("Arrival") or "—"
        st.caption(f"Updated {last} • Stability/variance → aleatoric uncertainty")

    with tab2:
        st.subheader("Vital Trends")
        st.info("Trend plots for HR / SBP / SpO₂ / Temp can be shown here. "
                "Higher variance ⇒ ↑ aleatoric uncertainty; monitoring gaps ⇒ ↑ epistemic uncertainty.")

    with tab3:
        st.subheader("Results")
        ecg = state.get("ECG", "—")
        tro = state.get("hs-cTn (ng/L)", state.get("hs_cTn", "—"))
        tro_time = state.get("TroponinTime", "—")

        r1, r2 = st.columns(2)
        with r1:
            st.text_input("ECG (summary)", value=str(ecg), disabled=True)
        with r2:
            st.text_input("hs-cTn (ng/L)", value="" if tro in [None, "None"] else str(tro), disabled=True)
            st.caption(f"Result time: {tro_time}")

        missing = state.get("MissingKeyResults", [])
        if missing:
            st.warning("Missing for pathway: " + ", ".join(missing) + "  → ↑ epistemic uncertainty")
        else:
            st.success("Key results complete  → ↓ epistemic uncertainty")