# pages/1_ED_Track_Board.py
# ED Track Board — Chest Pain Triage (AgGrid list; no auto-jump to chart)
# - Exactly three fixed demo patients (Weber / Green / Lopez)
# - English comments & labels
# - Same layout & filters as before
# - Selecting a row only stores it in session_state for Patient Chart to read later

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from st_aggrid.shared import GridUpdateMode, DataReturnMode

st.set_page_config(
    page_title="ED Track Board — Chest Pain",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optional helpers; safely no-op if utils is unavailable
def _no_op(*a, **k): return None
try:
    from utils import init_state, enter_page, show_back_top_right, render_sidebar
except Exception:
    init_state = _no_op; enter_page = _no_op; show_back_top_right = _no_op; render_sidebar = _no_op

init_state()
enter_page("ED Track Board")
render_sidebar(__file__)
show_back_top_right("← Back")

# ───────── Fixed demo patients (NO randomness) ─────────
now = datetime.now().replace(second=0, microsecond=0)

# Manual base risk (%) and stability label drive interval & confidence.
# Approach B mapping:
#   High -> width 6 pts; Medium -> width 18 pts; Low -> width 30 pts
def _width_from_stability(label: str) -> int:
    return {"High": 6, "Medium": 18, "Low": 30}.get(str(label).title(), 18)

def _conf_from_width(width_pts: int) -> str:
    if width_pts <= 6: return "high"
    if width_pts <= 12: return "medium"
    return "low"

def _interval_from_base(base_pct: float, width_pts: int):
    lo = max(0.0, base_pct - width_pts/2)
    hi = min(100.0, base_pct + width_pts/2)
    return round(lo, 1), round(hi, 1)

# Three canonical profiles
# Keep IDs aligned with Patient Chart (CP-1000 / CP-1001 / CP-1002)
PATIENTS = [
    {
        "PatientID": "CP-1000",
        "Patient": "Weber, Charlotte",
        "Age": 28, "Sex": "Female", "ESI": 2,
        "HR": 82, "SBP": 158, "SpO₂": 97,
        "ECG": "Normal", "hs-cTn (ng/L)": None,
        "OnsetMin": 85,
        "CC": "Chest pain at rest, mild shortness of breath.",
        "Arrival": (now - timedelta(minutes=41)).strftime("%H:%M"),
        "Room": "Pod A",
        "TempC": 37.1, "Temp": "37.1°C",
        # Manual risk UI inputs
        "RiskBase%": 10.0, "Stability": "Medium",
        # Unc. reason examples for display only
        "UncReasonSeed": ["troponin pending"]
    },
    {
        "PatientID": "CP-1001",
        "Patient": "Green, Gary",
        "Age": 60, "Sex": "Male", "ESI": 3,
        "HR": 106, "SBP": 136, "SpO₂": 95,
        "ECG": "Nonspecific", "hs-cTn (ng/L)": 12.0,
        "OnsetMin": 40,
        "CC": "Severe chest pressure radiating to left arm, nausea, diaphoresis.",
        "Arrival": (now - timedelta(minutes=22)).strftime("%H:%M"),
        "Room": "Pod B",
        "TempC": 36.9, "Temp": "36.9°C",
        "RiskBase%": 22.0, "Stability": "Medium",
        "UncReasonSeed": ["nonspecific ECG"]
    },
    {
        "PatientID": "CP-1002",
        "Patient": "Lopez, Mariah",
        "Age": 44, "Sex": "Female", "ESI": 3,
        "HR": 94, "SBP": 128, "SpO₂": 98,
        "ECG": "ST/T abn", "hs-cTn (ng/L)": 58.0,
        "OnsetMin": 25,
        "CC": "Acute chest tightness with diaphoresis during activity.",
        "Arrival": (now - timedelta(minutes=7)).strftime("%H:%M"),
        "Room": "WRM",
        "TempC": 37.3, "Temp": "37.3°C",
        "RiskBase%": 68.0, "Stability": "Low",
        "UncReasonSeed": ["high troponin", "abnormal ECG"]
    },
]

def _build_df() -> pd.DataFrame:
    rows = []
    for p in PATIENTS:
        base = float(p["RiskBase%"])
        width_pts = _width_from_stability(p["Stability"])
        lo, hi = _interval_from_base(base, width_pts)
        conf = _conf_from_width(width_pts)
        # Next Steps (very lightweight)
        next_steps = []
        if p["hs-cTn (ng/L)"] is None: next_steps.append("Confirm: draw hs-cTn now")
        if conf == "low": next_steps.append("Consult: senior/ACS pathway")
        if base >= 20:   next_steps.append("Observe: continuous ECG + vitals")
        if base < 5 and conf != "low": next_steps.append("Defer: discharge w/ next-day FU")

        rows.append({
            "PatientID": p["PatientID"],
            "Arrival": p["Arrival"],
            "Patient": p["Patient"],
            "Age": p["Age"], "Sex": p["Sex"],
            "ESI": p["ESI"], "HR": p["HR"], "SBP": p["SBP"], "SpO₂": p["SpO₂"],
            "ECG": p["ECG"], "hs-cTn (ng/L)": p["hs-cTn (ng/L)"],
            "OnsetMin": p["OnsetMin"], "CC": p["CC"], "Room": p["Room"],
            "TempC": p["TempC"], "Temp": p["Temp"],
            # Risk / uncertainty
            "Risk %": round(base, 1),
            "Risk Lo": lo, "Risk Hi": hi,
            "Confidence": conf,
            "Uncertainty Reason": ", ".join(p.get("UncReasonSeed") or []) or "—",
            "Next Steps": " · ".join(next_steps) if next_steps else "—",
        })
    return pd.DataFrame(rows)

df = _build_df()

# Save the master in session_state so Patient Chart can read it
st.session_state["trackboard_df"] = df.copy()
st.session_state.setdefault("selected_patient_id", None)
st.session_state.setdefault("selected_patient_name", None)

# ───────── Header  ─────────
st.title("ED Track Board — Chest Pain Triage")
st.caption("List view with arrival, ESI, vitals, ECG/troponin, risk with interval, confidence badge, and actionable next steps.")

# ───────── Filters  ─────────
colf1, colf2, colf3 = st.columns([1,1,2])
with colf1:
    sort_by = st.selectbox("Sort by", ["Arrival","Risk %","ESI"])
with colf2:
    min_conf = st.selectbox("Min confidence", ["any","medium","high"])
with colf3:
    show_cols = st.multiselect(
        "Visible columns",
        ["Room","Arrival","Patient","ESI","HR","SBP","SpO₂","ECG","hs-cTn (ng/L)",
         "Risk %","Risk Lo","Risk Hi","Confidence","Uncertainty Reason","Next Steps"],
        default=["Room","Arrival","Patient","ESI","HR","SBP","SpO₂","ECG",
                 "hs-cTn (ng/L)","Risk %","Confidence","Next Steps"]
    )

work = df.copy()
if min_conf != "any":
    work = work[work["Confidence"].isin(["high"] if min_conf=="high" else ["high","medium"])]
ascending = True if sort_by in ["Arrival","ESI"] else False
work = work.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

# ───────── Derived display columns ─────────
def risk_badge_text(r, lo, hi):
    band = f"{int(round(r))}% [{int(round(lo))}–{int(round(hi))}%]"
    if r >= 20: return f"🟥 {band}"
    if r >= 10: return f"🟧 {band}"
    if r >= 5:  return f"🟨 {band}"
    return f"🟩 {band}"

def conf_badge_text(c):
    return {"high":"🔵 High","medium":"🟣 Medium","low":"⚪ Low"}[c]

disp = work.copy()
disp["Risk"] = disp.apply(lambda x: risk_badge_text(x["Risk %"], x["Risk Lo"], x["Risk Hi"]), axis=1)
disp["Conf"] = disp["Confidence"].map(conf_badge_text)

base_order = ["Room","Arrival","Patient","ESI","HR","SBP","SpO₂","ECG","hs-cTn (ng/L)","Risk","Conf","Next Steps","Uncertainty Reason"]
final_cols = [c for c in base_order if c in show_cols or c in ["Risk","Conf"]]
if "Risk %" in show_cols and "Risk" not in final_cols:
    final_cols.insert(final_cols.index("ECG")+1, "Risk %")
if "Confidence" in show_cols and "Conf" not in final_cols:
    final_cols.insert(final_cols.index("Risk")+1, "Confidence")

disp_for_grid = disp[["PatientID"] + final_cols].copy()

# ───────── AgGrid config (no auto-jump; selection only) ─────────
gb = GridOptionsBuilder.from_dataframe(disp_for_grid)

# 0) lock editing
gb.configure_default_column(editable=False)

# 1) official selection API
gb.configure_selection(selection_mode="single", use_checkbox=False)

# 2) stable row id
gb.configure_grid_options(
    suppressClickEdit=True,
    stopEditingWhenCellsLoseFocus=True,
    domLayout="autoHeight",
    getRowId=JsCode("function(p){ return p.data.PatientID; }"),
)

# 3) "Open" button column — only selects row (no navigation)
open_btn_renderer = JsCode("""
class BtnCellRenderer {
  init(params){
    this.params = params;
    const e = document.createElement('button');
    e.innerText = 'Open';
    e.style.padding = '4px 10px';
    e.style.cursor = 'pointer';
    e.addEventListener('click', () => {
      params.api.deselectAll();
      params.api.selectNode(params.node, true); // selection changed
    });
    this.eGui = e;
  }
  getGui(){ return this.eGui; }
}
""")

if "Open" not in disp_for_grid.columns:
    disp_for_grid.insert(1, "Open", "Open")

gb.configure_column("Open", header_name="", width=90,
                    editable=False, cellRenderer=open_btn_renderer)

# 4) visual tweaks
gb.configure_column("Patient", cellStyle={"color":"#1f77b4","textDecoration":"underline","cursor":"pointer"})
gb.configure_column("PatientID", header_name="ID", width=90)

grid = AgGrid(
    disp_for_grid,
    gridOptions=gb.build(),
    update_mode=GridUpdateMode.SELECTION_CHANGED,      # return selection events
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    allow_unsafe_jscode=True,
    theme="streamlit",
    fit_columns_on_grid_load=False,
    key="track_board_grid_fixed3"
)

resp = grid
sel_raw = resp.get("selected_rows")
if sel_raw is None:
    sel = []
elif isinstance(sel_raw, pd.DataFrame):
    sel = sel_raw.to_dict("records")
else:
    sel = sel_raw

# Store the selected row for the Patient Chart page (no auto-switch)
if sel:
    row  = sel[0]
    pid  = row["PatientID"]
    name = row["Patient"]
    st.session_state["selected_patient_id"] = pid
    st.session_state["selected_patient_name"] = name
    st.session_state["selected_patient"] = row
    st.success(f"Selected **{name}** ({pid}). Open the Patient Chart page to view details.")
else:
    st.info("Select a row to make it available in the Patient Chart page.")

# ───────── Legend ─────────
with st.expander("Legend • Uncertainty & Action Rules", expanded=False):
    st.markdown("""
**Risk badge** shows a **point estimate + interval** (e.g., *12% [8–18%]*).  
The **interval width** is a visual encoding of **prediction stability**:
**High → 6 pts**, **Medium → 18 pts**, **Low → 30 pts**.

**Confidence** maps from interval width: **High / Medium / Low**.  
When **Low**, human oversight is recommended (**Consult**).

**Next Steps** use four slots:  
- **Confirm** — obtain missing tests (e.g., draw hs-cTn).  
- **Observe** — continuous ECG and vital monitoring.  
- **Consult** — senior review / ACS pathway.  
- **Defer** — if risk is very low and confidence acceptable, safe discharge with follow-up.
""")