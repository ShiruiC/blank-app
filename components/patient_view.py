# components/patient_view.py
import math
from html import escape
import uuid

def toy_risk_model(inputs: dict) -> float:
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

# Drawer HTML — no anchors/links; page provides the fixed Streamlit close button
def _badge(text, tone="info"):
    colors = {"success":("#065f46","#d1fae5"), "warn":("#92400e","#fef3c7"),
              "danger":("#7f1d1d","#fee2e2"), "info":("#1e40af","#dbeafe"),
              "neutral":("#374151","#e5e7eb")}
    fg,bg = colors.get(tone, colors["neutral"])
    return f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;font-weight:700;color:{fg};background:{bg};font-size:12px'>{text}</span>"

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
    tier = "High" if conf_score >= 0.70 else ("Medium" if conf_score >= 0.40 else "Low")
    return alea*100, epis*100, conf_score, tier

def uncertainty_drawer_html(scope: str, *, sum_dict: dict, patient: dict,
                            triage_label: str = "", dispo_label: str = "", opened: bool = True) -> str:
    from html import escape as esc
    import uuid
    uid = "ua_" + uuid.uuid4().hex

    # ---- existing computation (kept) ----
    alea_pct, epis_pct, conf_score, conf_tier = decompose_uncertainty(sum_dict, patient)
    conf_pct = int(round(conf_score*100))
    intensity = "Low" if conf_score >= 0.70 else ("Moderate" if conf_score >= 0.35 else "High")
    dom_type = "Aleatoric" if alea_pct >= epis_pct else "Epistemic"
    tone_map = {"High":"danger","Moderate":"warn","Low":"success"}

    def _badge(text, tone="info"):
        colors = {"success":("#065f46","#d1fae5"), "warn":("#92400e","#fef3c7"),
                  "danger":("#7f1d1d","#fee2e2"), "info":("#1e40af","#dbeafe"),
                  "neutral":("#374151","#e5e7eb")}
        fg,bg = colors.get(tone, colors["neutral"])
        return f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;font-weight:700;color:{fg};background:{bg};font-size:12px'>{esc(text)}</span>"

    def h4(t): return f"<div style='font-weight:800;font-size:15px;margin:10px 0 6px'>{esc(t)}</div>"

    title = {
        "triage": f"Why Triage: {esc(triage_label or '—')}",
        "disposition": f"Why Disposition: {esc(dispo_label or '—')}",
        "steps": "Why These Next Steps",
    }.get(scope, "Why?")

    # bars + sections (same as your version; trimmed for brevity)
    bars = (
        "<div style='display:flex;gap:.5rem;align-items:center;margin-top:.2rem'>"
        "<div style='min-width:92px'>Aleatoric</div>"
        "<div style='height:10px;background:#e5e7eb;border-radius:999px;flex:1;position:relative'>"
        f"<div style='height:10px;border-radius:999px;background:#60A5FA;width:{int(alea_pct)}%'></div>"
        f"</div><div style='min-width:42px;text-align:right'>{int(alea_pct)}</div></div>"
        "<div style='color:#6b7280;margin:4px 0 10px 92px'>patient variability</div>"
        "<div style='display:flex;gap:.5rem;align-items:center'>"
        "<div style='min-width:92px'>Epistemic</div>"
        "<div style='height:10px;background:#e5e7eb;border-radius:999px;flex:1;position:relative'>"
        f"<div style='height:10px;border-radius:999px;background:#A78BFA;width:{int(epis_pct)}%'></div>"
        f"</div><div style='min-width:42px;text-align:right'>{int(epis_pct)}</div></div>"
        "<div style='color:#6b7280;margin:4px 0 2px 92px'>model limitation</div>"
    )

    if scope == "triage":
        # ---------- Evidence/Reasoning + Confidence/Uncertainty (triage only) ----------
        base = float(sum_dict.get("base", 0))
        lo, hi = float(sum_dict.get("lo", 0)), float(sum_dict.get("hi", 0))
        base_pct = int(round(100 * base))
        lo_pct, hi_pct = int(round(100 * lo)), int(round(100 * hi))
        width = float(sum_dict.get("width", hi - lo))

        # Risk band → tag (distinct from confidence tag)
        def _risk_tier(r):
            if r >= 0.40: return ("High", "#DC2626")       # red
            if r >= 0.20: return ("Medium", "#F59E0B")     # amber
            if r >= 0.10: return ("Low", "#22C55E")        # green
            return ("Very Low", "#10B981")

        risk_tier, risk_color = _risk_tier(base)

        # Simple similarity proxy already used in your original code
        pr = int(round(100 * (0.3
                              + (0.3 if patient['risk_inputs'].get('ecg_abnormal') else 0)
                              + (min(0.4, (patient['risk_inputs'].get('troponin') or 0) / 0.04 * 0.4)))))

        # --- Needed inputs (compact tags) ---
        ri = patient.get("risk_inputs", {})
        vitals = patient.get("vitals", {})
        demo_tags = [f"Age {patient.get('age','—')}", f"Sex {patient.get('sex','—')}"]
        if ri.get("rf_htn"): demo_tags.append("Comorbidity: HTN")
        if ri.get("rf_dm"):  demo_tags.append("Comorbidity: DM")
        if ri.get("rf_smoker"): demo_tags.append("Smoker")

        symptom_tags = []
        for pf in ri.get("pain_features", []) or []:
            symptom_tags.append(pf.capitalize())
        cc = (patient.get("chief_complaint") or "").strip()
        if cc: symptom_tags.append("CC: " + (cc[:28] + "…" if len(cc) > 28 else cc))

        lab_ecg_tags = []
        lab_ecg_tags.append("ECG: Abnormal" if ri.get("ecg_abnormal") else "ECG: Normal")
        if ri.get("troponin") is None:
            lab_ecg_tags.append("Troponin: pending")
        else:
            lab_ecg_tags.append(f"hs-cTn: {ri['troponin']:.3f} ng/mL")

        vit_tags = [f"HR {vitals.get('HR','—')}",
                    f"RR {vitals.get('RR','—')}",
                    f"SpO₂ {vitals.get('SpO2', vitals.get('SpO₂','—'))}",
                    f"BP {vitals.get('BP','—')}",
                    f"T {vitals.get('TempC','—')}°C"]

        pattern_tags = []
        if ri.get("ecg_abnormal"): pattern_tags.append("Chest-pain + abn. ECG")
        sbp = str(vitals.get("BP","")).split("/")
        if len(sbp) == 2 and sbp[0]:
            try:
                if int(sbp[0]) < 100: pattern_tags.append("Low BP pattern")
            except Exception:
                pass
        if (ri.get("troponin") or 0) >= 0.01: pattern_tags.append("Troponin↑ pattern")
        if not pattern_tags: pattern_tags.append("Pattern: non-specific")

        ctxt = patient.get("arrival_mode","—")
        ctx_tags = []
        if ctxt and ctxt != "—": ctx_tags.append(f"Arrival: {ctxt}")
        onset_min = patient.get("data_quality",{}).get("time_from_onset_min", None)
        if isinstance(onset_min, int): ctx_tags.append(f"Onset {onset_min} min")
        mental = patient.get("mental_status","")
        if mental: ctx_tags.append(f"Mental: {mental}")

        def _chip(text, fg="#1f2937", bg="#f3f4f6", bd="#e5e7eb"):
            return (f"<span style='display:inline-block;border:1px solid {bd};"
                    f"background:{bg};color:{fg};border-radius:999px;padding:2px 8px;"
                    f"margin:2px 6px 0 0;font-weight:700;font-size:12px'>{esc(text)}</span>")

        # ---------- Five-aspect uncertainty mix (simple, readable heuristics) ----------
        dq = patient.get("data_quality", {}) or {}
        missing = dq.get("missing", []) or []
        ood = 1.0 if dq.get("ood") else 0.0

        # Heuristics:
        # - Data gaps → count of missing key fields
        w_data = min(1.0, 0.45 * len(missing))
        # - Model familiarity → OOD flag
        w_fam = 0.60 * ood
        # - Prediction stability → wider interval => more uncertainty
        w_stab = max(0.0, (width - 0.10) / 0.25)  # 0 if <=0.10, up to ~1 if very wide
        # - Human ambiguity → vague/short CC or no symptom detail
        human_vague = 1.0 if (not symptom_tags or len(" ".join(symptom_tags)) < 10 or len(cc) < 8) else 0.25
        w_human = 0.40 * human_vague
        # - Chained decisions (cascading) → mid/high width contributes
        w_chain = 0.35 if width > 0.18 else (0.15 if width > 0.12 else 0.05)

        raw = [w_data, w_fam, w_stab, w_human, w_chain]
        total = sum(raw) or 1.0
        pct5 = [int(round(100 * x / total)) for x in raw]
        # Normalize to exactly 100%
        delta = 100 - sum(pct5)
        if delta != 0:
            pct5[0] += delta

        labels5 = ["Data gaps", "Model familiarity", "Prediction stability",
                   "Human ambiguity", "Chained decisions"]
        expl5 = [
            ("Missing: " + ", ".join(missing)) if missing else "No key inputs missing",
            "Outside training distribution" if ood else "Typical for training distribution",
            f"Interval width ≈ {int(round(100*width))} pts — wider ⇒ less stable",
            "Unclear/vague narrative or limited symptom detail" if human_vague >= 1.0 else "Clear symptom description",
            "Uncertainty propagates from triage → disposition choices" if width > 0.12 else "Minimal cross-step carryover",
        ]

        # Conic pie using CSS (compact) + legend
        pie_css = (f"background: conic-gradient(#f59e0b 0 {pct5[0]}%, "
                   f"#6366f1 {pct5[0]}% {pct5[0]+pct5[1]}%, "
                   f"#22c55e {pct5[0]+pct5[1]}% {pct5[0]+pct5[1]+pct5[2]}%, "
                   f"#e11d48 {pct5[0]+pct5[1]+pct5[2]}% {pct5[0]+pct5[1]+pct5[2]+pct5[3]}%, "
                   f"#0ea5e9 {pct5[0]+pct5[1]+pct5[2]+pct5[3]}% 100%);")

        def legend_row(color, label, pct, note):
            return (f"<div style='display:flex;align-items:flex-start;gap:.5rem;margin:.15rem 0'>"
                    f"<span style='width:10px;height:10px;background:{color};border-radius:2px;margin-top:4px'></span>"
                    f"<div style='flex:1'><b>{esc(label)}</b> — {pct}%"
                    f"<div style='color:#6b7280;font-size:12px'>{esc(note)}</div></div></div>")

        legend = (
            legend_row("#f59e0b", labels5[0], pct5[0], expl5[0]) +
            legend_row("#6366f1", labels5[1], pct5[1], expl5[1]) +
            legend_row("#22c55e", labels5[2], pct5[2], expl5[2]) +
            legend_row("#e11d48", labels5[3], pct5[3], expl5[3]) +
            legend_row("#0ea5e9", labels5[4], pct5[4], expl5[4])
        )

        # ---------- Sections ----------
        section = (
            # 1) Evidence / Reasoning
            h4("Evidence / Reasoning") +
            "<div style='display:flex;align-items:center;gap:.6rem;margin:.15rem 0 .35rem'>"
            f"<div><b>Risk estimate:</b> {base_pct}%</div>"
            f"{_badge(risk_tier, tone='neutral')}"
            "</div>"
            f"<div style='color:#6b7280;font-size:12px;margin:-2px 0 6px'>"
            "Risk tag reflects probability band (not confidence).</div>"
            + h4("Recognized diagnosis") +
            f"<div>Similarity: <b>{pr}%</b> to {esc(sum_dict.get('suspected_condition','ACS-like'))} pattern.</div>"

            + h4("Needed inputs for this prediction") +
            "<div style='margin:.25rem 0 .1rem;color:#374151;font-weight:700'>Vital signs</div>"
            + "".join(_chip(t) for t in vit_tags) +
            "<div style='margin:.6rem 0 .1rem;color:#374151;font-weight:700'>Chief complaint & symptoms</div>"
            + ("".join(_chip(t) for t in symptom_tags) or _chip("—")) +
            "<div style='margin:.6rem 0 .1rem;color:#374151;font-weight:700'>Lab / ECG</div>"
            + "".join(_chip(t) for t in lab_ecg_tags) +
            "<div style='margin:.6rem 0 .1rem;color:#374151;font-weight:700'>Demographics</div>"
            + "".join(_chip(t) for t in demo_tags) +
            "<div style='margin:.6rem 0 .1rem;color:#374151;font-weight:700'>Clinical pattern recognition</div>"
            + "".join(_chip(t) for t in pattern_tags) +
            "<div style='margin:.6rem 0 .1rem;color:#374151;font-weight:700'>Context</div>"
            + ("".join(_chip(t) for t in ctx_tags) or _chip("—"))
            + "<hr style='border:none;border-top:1px solid #e5e7eb;margin:12px 0'/>"

            # 2) Confidence / Uncertainty
            + h4("Confidence & Uncertainty")
            + "<div style='display:flex;align-items:baseline;gap:.6rem'>"
            f"<div style='font-size:24px;font-weight:900'>{int(round(conf_score*100))}%</div>"
            f"{_badge(conf_tier, 'success' if conf_tier=='High' else ('warn' if conf_tier=='Medium' else 'danger'))}"
            "<div style='font-size:12px;color:#6b7280'>"
            "Confidence reflects how tight the risk range is.</div></div>"
            + h4("Risk interval")
            + f"<div><b>Range:</b> {lo_pct}%–{hi_pct}% "
            f"(width {int(round(100*width))} pts). "
            "Intervals come from perturbing inputs & data quality cues; "
            "narrower ranges generally mean higher confidence.</div>"
            + h4("Uncertainty composition")
            + "<div style='display:flex;gap:1rem;align-items:center;margin:.5rem 0'>"
            f"<div style='width:96px;height:96px;border-radius:50%;{pie_css}'></div>"
            f"<div style='flex:1'>{legend}</div>"
            "</div>"
            + "<div style='color:#6b7280;font-size:12px;margin-top:.25rem'>"
            "If an aspect contributes 0%, it will still be listed as ‘None’ for completeness.</div>"
        )
    elif scope == "disposition":
        dq = patient.get("data_quality",{}) or {}
        missing = dq.get("missing", [])
        miss = (" ".join(
            f"<span style='border:1px solid #e5e7eb;border-radius:6px;padding:2px 6px;margin-right:4px'>{esc(m)}</span>"
            for m in missing)) if missing else "<span style='color:#6b7280'>No missing key data</span>"
        wid_pct = int(round(100*sum_dict.get("width",0.0)))
        fam = "Outside typical training range" if dq.get("ood") else "Typical for training distribution"
        section = (
            h4("Data completeness") + f"<div>{miss}</div>" +
            h4("Model familiarity") + f"<div>{esc(fam)}</div>" +
            h4("How certainty affects disposition") +
            f"<div>Risk interval width ≈ <b>{wid_pct}%</b>. Narrower ranges → clearer split between Admit vs Observe.</div>"
        )
    else:
        section = (
            h4("Prediction stability (local sensitivity)") +
            "<table style='width:100%;border-collapse:collapse;font-size:12px'>"
            "<thead><tr><th style='text-align:left;border-bottom:1px solid #e5e7eb;padding:4px 0'>Vital</th>"
            "<th style='text-align:left;border-bottom:1px solid #e5e7eb;padding:4px 0'>Change</th>"
            "<th style='text-align:left;border-bottom:1px solid #e5e7eb;padding:4px 0'>Output ↑</th>"
            "<th style='text-align:left;border-bottom:1px solid #e5e7eb;padding:4px 0'>Output ↓</th></tr></thead>"
            "<tbody><tr><td>HR</td><td>+5%</td><td>+2%</td><td>-1%</td></tr>"
            "<tr><td>SBP</td><td>+5%</td><td>+0%</td><td>+1%</td></tr>"
            "<tr><td>SpO₂</td><td>±5%</td><td>+3%</td><td>—</td></tr></tbody></table>"
        )

    callout = (
        "<div style='margin-top:.75rem;padding:.6rem .8rem;border:1px solid #e5e7eb;border-radius:10px;background:#fafafa'>"
        f"<div style='font-weight:800'>Uncertainty type — {dom_type} "
        f"{_badge(intensity, tone='success' if intensity=='Low' else ('warn' if intensity=='Moderate' else 'danger'))}</div>"
        "<div style='margin-top:.35rem;color:#374151'><b>Drivers:</b> " +
        esc('interval width suggests ' + ('low' if sum_dict['width']<=0.10 else ('medium' if sum_dict['width']<=0.20 else 'high')) +
            ' aleatoric uncertainty' if dom_type=='Aleatoric' else 'model familiarity & limits') +
        "</div></div>"
    )

    start_hidden = " hidden" if not opened else ""
    return f"""
<style>
/* iframe-local, so we need full height via component height in Python */
#{uid} .ua-shell{{position:fixed;top:0;right:0;height:100vh;z-index:2147483602;pointer-events:none}}
#{uid} .ua-drawer{{position:absolute;top:0;right:0;height:100%;width:420px;max-width:92vw;background:#fff;
  border-left:1px solid #e5e7eb;box-shadow:-8px 0 24px rgba(0,0,0,.08);
  transform:translateX(0);transition:transform .22s ease;pointer-events:auto;display:flex;flex-direction:column}}
#{uid} .ua-handle{{position:absolute;top:50%;right:420px;transform:translateY(-50%);width:28px;height:64px;
  border:1px solid #d1d5db;background:#fff;border-radius:8px 0 0 8px;display:flex;align-items:center;justify-content:center;
  font-weight:900;cursor:pointer;pointer-events:auto;box-shadow:-2px 2px 8px rgba(0,0,0,.06)}}
#{uid}.hidden .ua-drawer{{transform:translateX(100%)}}
#{uid}.hidden .ua-handle{{right:0}} /* when hidden, the grip sits at the page edge */

#{uid} .ua-head{{position:sticky;top:0;background:#fff;z-index:2;padding:12px 16px 10px 16px;border-bottom:1px solid #e5e7eb}}
#{uid} .ua-title{{font-weight:800;font-size:18px;margin:0}}
#{uid} .ua-x{{position:absolute;top:10px;right:10px;display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:8px;
  border:1px solid #d1d5db;font-weight:800;color:#111827;background:#fff;cursor:pointer}}
#{uid} .ua-x:hover{{background:#f3f4f6}}
#{uid} .ua-body{{flex:1 1 auto;overflow:auto;-webkit-overflow-scrolling:touch;padding:16px 18px 24px 18px}}
#{uid} .ua-body pre{{display:none!important}}
</style>

<div id="{uid}" class="{('' if opened else 'hidden')}">
  <div class="ua-shell">
    <div class="ua-handle" title="Toggle details">{"≪" if opened else "≫"}</div>
    <div class="ua-drawer" role="dialog" aria-label="Confidence & Uncertainty">
      <div class="ua-head">
        <div class="ua-title">{title}</div>
        <button class="ua-x" title="Close">✕</button>
      </div>
      <div class="ua-body">
        {h4("Confidence score")}
        <div style="display:flex;align-items:baseline;gap:.5rem">
          <div style="font-size:24px;font-weight:900">{conf_pct}%</div>
          {_badge(conf_tier, 'success' if conf_tier=='High' else ('warn' if conf_tier=='Medium' else 'danger'))}
        </div>
        {h4("Uncertainty composition")}
        {bars}
        <hr style="border:none;border-top:1px solid #e5e7eb;margin:12px 0"/>
        {section}
        <hr style="border:none;border-top:1px solid #e5e7eb;margin:12px 0"/>
        {callout}
      </div>
    </div>
  </div>
</div>

<script>
(function(){{
  const root = document.getElementById("{uid}");
  const handle = root.querySelector(".ua-handle");
  const closeBtn = root.querySelector(".ua-x");
  const setOpen = (open)=>{{
    if(open) {{
      root.classList.remove("hidden");
      handle.textContent = "≪";
    }} else {{
      root.classList.add("hidden");
      handle.textContent = "≫";
    }}
  }};
  handle?.addEventListener("click", ()=>{{
    setOpen(root.classList.contains("hidden"));
  }});
  closeBtn?.addEventListener("click", ()=> setOpen(false));
}})();
</script>
"""