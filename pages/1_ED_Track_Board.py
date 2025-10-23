# pages/1_ED_Track_Board.py
# ED Track Board — Chest Pain Triage (AgGrid: click Patient to open chart)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode, DataReturnMode

st.set_page_config(
    page_title="ED Track Board — Chest Pain",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

from utils import init_state, enter_page, show_back_top_right, render_sidebar
init_state()
enter_page("ED Track Board")
render_sidebar(__file__)
show_back_top_right("← Back")

# ───────── Demo data ─────────
np.random.seed(7)
now = datetime.now().replace(second=0, microsecond=0)

def demo_rows(n=10):
    names = ["Weber, Charlotte","Green, Gary","Brown, Beverly","Busch, Amelia",
             "Johnson, Sally","Adams, Devin","Morales, Miles","Crown, Emma",
             "Pink, Patsy","Schmidt, Jonas"]
    
    FIXED_PROFILES = {
        "CP-1000": {  # Olivia 对应的示例（可改名）
            "Patient": "Weber, Charlotte",
            "Age": 27, "Sex": "Female",
            "ESI": 2, "HR": 78, "SBP": 164, "SpO₂": 96,
            "ECG": "Normal", "hs-cTn (ng/L)": None,
            "OnsetMin": 90, "CC": "Chest pain at rest, mild SOB."
        },
        "CP-1001": {  # Mark 对应的示例（可改名）
            "Patient": "Green, Gary",
            "Age": 59, "Sex": "Male",
            "ESI": 3, "HR": 141, "SBP": 109, "SpO₂": 91,
            "ECG": "Normal", "hs-cTn (ng/L)": 0.0,  # 初到院未升高（演示）
            "OnsetMin": 30, "CC": "Severe chest pressure radiating to left arm, nausea, diaphoresis."
        },
    }

    rows = []
    for i in range(n):
        pid = f"CP-{1000+i}"
        troponin_missing = np.random.rand() < 0.35
        hr  = int(np.random.normal(84, 18))
        sbp = int(np.random.normal(132, 22))
        spo2 = int(np.clip(np.random.normal(96, 3), 85, 100))
        esis = np.random.choice([2,3,4], p=[0.2,0.55,0.25])
        ecg_flags = np.random.choice(["ST/T abn","Normal","Nonspecific"], p=[0.25,0.5,0.25])
        hsctn = None if troponin_missing else round(max(0, np.random.normal(12, 18)),1)

        row = {
            "PatientID": pid,
            "Arrival": (now - timedelta(minutes=np.random.randint(3, 120))).strftime("%H:%M"),
            "Patient": names[i % len(names)],
            "Age": int(np.random.normal(52, 18)),
            "Sex": np.random.choice(["Male","Female"]),
            "ESI": esis, "HR": hr, "SBP": sbp, "SpO₂": spo2,
            "ECG": ecg_flags, "hs-cTn (ng/L)": hsctn,
            "OnsetMin": np.random.randint(15, 180),
            "CC": "Chest pain.",
            "Room": np.random.choice(["WRM","Pod A","Pod B","Fast Track"], p=[0.4,0.3,0.2,0.1])
        }

        # ✅ 如果在固定名单里，就用固定资料覆盖随机值
        if pid in FIXED_PROFILES:
            row.update(FIXED_PROFILES[pid])

        rows.append(row)

    return pd.DataFrame(rows)

df = demo_rows(10)

# ───────── Risk & Uncertainty (prototype) ─────────
def compute_risk_and_uncertainty(row: pd.Series):
    score = 0
    if row.HR >= 100: score += 1.0
    if row.SBP < 100 or row.SBP > 180: score += 1.0
    if row["SpO₂"] < 93: score += 1.0
    if row.ECG == "ST/T abn": score += 2.0
    elif row.ECG == "Nonspecific": score += 0.5
    if row.ESI == 2: score += 1.0

    reasons = []
    if pd.notna(row["hs-cTn (ng/L)"]):
        tn = row["hs-cTn (ng/L)"]
        if tn >= 52: score += 3.0; reasons.append("high troponin")
        elif tn >= 14: score += 1.5; reasons.append("borderline troponin")
    else:
        reasons.append("missing troponin")

    risk = 100 * (1 / (1 + np.exp(-(score - 2.5))))
    risk = float(np.clip(risk, 1, 95))

    width = 6
    if pd.isna(row["hs-cTn (ng/L)"]): width += 10
    if row.ECG == "Nonspecific": width += 4
    if row["SpO₂"] < 93 or row.SBP < 100: width += 3
    width = min(width, 30)

    lo = max(0.0, risk - width/2); hi = min(100.0, risk + width/2)
    conf = "high" if width <= 6 else ("medium" if width <= 12 else "low")

    next_steps = []
    if pd.isna(row["hs-cTn (ng/L)"]): next_steps.append("Confirm: draw hs-cTn now")
    if conf == "low": next_steps.append("Consult: senior/ACS pathway")
    if risk >= 20:   next_steps.append("Observe: continuous ECG + vitals")
    if risk < 5 and conf != "low": next_steps.append("Defer: discharge w/ next-day FU")

    return round(risk,1), round(lo,1), round(hi,1), conf, ", ".join(reasons) if reasons else "—", " · ".join(next_steps) if next_steps else "—"

calc = df.apply(compute_risk_and_uncertainty, axis=1)
df[["Risk %","Risk Lo","Risk Hi","Confidence","Uncertainty Reason","Next Steps"]] = pd.DataFrame(calc.tolist(), index=df.index)

# ───────── Header  ─────────
st.title("ED Track Board — Chest Pain Triage")
st.caption("List view with arrival, ESI, vitals, ECG/troponin, risk with interval, confidence badge, and actionable next steps.")

# ───────── Filters under header ─────────
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

# ───────── Build derived display columns ─────────
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

# ───────── AgGrid config: click Patient to open ─────────
# ✅ 让患者页能读到当前板上的数据
st.session_state["trackboard_df"] = df          # 或者存 work/disp，看患者页需要的列
st.session_state.setdefault("selected_patient_id", None)
st.session_state.setdefault("selected_patient_name", None)

gb = GridOptionsBuilder.from_dataframe(disp_for_grid)
gb.configure_grid_options(
    rowSelection="single",
    suppressRowClickSelection=False,   # clicking a cell selects the row
    domLayout="autoHeight",
)
# make Patient look like a link
gb.configure_column("Patient", cellStyle={"color":"#1f77b4","textDecoration":"underline","cursor":"pointer"})
# shrink internal ID
gb.configure_column("PatientID", header_name="ID", width=90)

# column widths: a bit nicer
for col, w in [("Room",110),("Arrival",100),("ESI",70),("HR",80),("SBP",90),("SpO₂",90),("ECG",120),
               ("hs-cTn (ng/L)",130),("Risk",140),("Conf",120),("Next Steps",340)]:
    if col in disp_for_grid.columns:
        gb.configure_column(col, width=w)

# ... gb config ...

grid = AgGrid(
    disp_for_grid,
    gridOptions=gb.build(),
    update_mode=GridUpdateMode.MODEL_CHANGED | GridUpdateMode.SELECTION_CHANGED,
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,  # selection returns a list
    allow_unsafe_jscode=False,
    theme="streamlit",
    fit_columns_on_grid_load=False,
)

sel = grid.get("selected_rows", [])
if isinstance(sel, list) and len(sel) > 0:
    pid = sel[0]["PatientID"]
    name = sel[0]["Patient"]
    st.session_state["selected_patient_id"] = pid
    st.session_state["selected_patient_name"] = name

    # Navigate if your Streamlit has switch_page; otherwise show a link
    try:
        st.switch_page("pages/2_Patient_Chart.py")
    except Exception:
        st.success(f"Selected **{name}** ({pid}).")
        st.page_link("pages/2_Patient_Chart.py", label="➡️ Open Patient Chart")

# ───────── Legend ─────────
with st.expander("Legend • Uncertainty & Action Rules", expanded=False):
    st.markdown("""
**Risk badge** shows a **point estimate + interval** (e.g., *12% [8–18%]*).  
The **interval widens automatically** when data quality is weak (e.g., **missing hs-cTn**, nonspecific ECG, unstable vitals).

**Confidence** is mapped from interval width: **High / Medium / Low**.  
When **Low**, human oversight is recommended (**Consult**).

**Next Steps** (DP3) use four slots:  
- **Confirm** — obtain missing tests (e.g., draw hs-cTn).  
- **Observe** — continuous ECG and vital monitoring.  
- **Consult** — senior review / ACS pathway.  
- **Defer** — if risk is very low and confidence acceptable, safe discharge with follow-up.
""")