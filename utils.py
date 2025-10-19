# utils.py
import streamlit as st
import pandas as pd
from datetime import datetime
import os
from streamlit_option_menu import option_menu


PAGES = [
    ("streamlit_app.py",            "house",        "Landing"),
    ("pages/1_ED_Track_Board.py",   "kanban",       "ED Track Board"),
    ("pages/2_Patient_Chart.py",    "heart-pulse",  "Patient Chart"),
    ("pages/3_Registration.py",     "receipt",      "Registration"),
    ("pages/4_Visits_and_Complaints.py", "folder2", "Visits & Complaints"),
    ("pages/5_Patient_Acuity.py",   "sliders",      "Patient Acuity"),
    ("pages/6_ED_Disposition.py",   "check2-circle","ED Disposition"),
    ("pages/7_ED_Triage_Notes.py",  "journal-text", "ED Triage Notes"),
]

def init_state():
    st.session_state.setdefault("selected_patient_id", None)
    st.session_state.setdefault("sidebar_collapsed", False)

def header(title: str, subtitle: str = ""):
    col1, col2 = st.columns([0.75, 0.25])
    with col1:
        st.markdown(f"### {title}")
        if subtitle: st.caption(subtitle)
    with col2:
        st.caption(datetime.now().strftime("Time: %H:%M  |  Date: %Y-%m-%d"))

def _current_base(current_file: str) -> str:
    # 传进来时有可能是完整路径或相对路径，统一成 basename
    return os.path.basename(current_file)

def render_sidebar(current_file: str):
    # 侧栏只显示“图标 + 下面能完全显示的文字”，不再有任何多余 title
    paths, icons, labels = zip(*PAGES)
    label2path = {lbl: p for p, _, lbl in PAGES}

    # 算哪个是当前激活项（用于默认选中）
    current_base = _current_base(current_file)
    try:
        default_idx = [os.path.basename(p) for p in paths].index(current_base)
    except ValueError:
        default_idx = 0  # 找不到时落到第一个

    with st.sidebar:
        selected = option_menu(
            None,                       # 不显示菜单标题 → 去掉多余文字
            list(labels),
            icons=list(icons),
            menu_icon=None,
            default_index=default_idx,
            styles={
                "container": {
                    "padding": "0",
                    "background-color": "#2F66F6",   # 你的蓝色
                    "width": "100%",
                },
                "icon": {
                    "color": "white",
                    "font-size": "22px",
                },
                "nav-link": {
                    "display": "flex",
                    "flex-direction": "column",      # 图标在上、文字在下
                    "gap": "6px",
                    "align-items": "center",
                    "justify-content": "center",
                    "color": "white",
                    "font-size": "12px",
                    "text-align": "center",
                    "padding": "16px 8px",
                    "margin": "8px 10px",
                    "border-radius": "12px",
                },
                "nav-link-selected": {
                    "background-color": "rgba(255,255,255,0.18)",
                    "color": "white",
                },
            },
            orientation="vertical",
        )

    # 如果选择变化，就跳转到对应页面
    target_path = label2path.get(selected)
    if target_path and os.path.basename(target_path) != current_base:
        # 允许写 'pages/xxx.py' 或 'streamlit_app.py'
        st.switch_page(target_path)

# ---------------- 下面保留你的数据与其它函数 ----------------

def make_trackboard_data():
    rows = [
        dict(Room="WRM", Patient="Brown, Beverly", AgeSex="18 y/o • F", Complaint="Knee pain", Acuity="Level 4",
             RN="MEE; NEE", MD="APE", Lab="V [0/4/4]", Rad="-", Dispo="-", Comments="-",
             Unack="●", New="-", MRN="A001"),
        dict(Room="WRM", Patient="James, Sally", AgeSex="73 y/o • F", Complaint="Fatigue", Acuity="Level 4",
             RN="SJP", MD="KEM", Lab="-", Rad="-", Dispo="-", Comments="-",
             Unack="●", New="-", MRN="A002"),
        dict(Room="WRM", Patient="Green, Gary", AgeSex="48 y/o • M", Complaint="Chest pain", Acuity="Level 2",
             RN="NPT", MD="KEM", Lab="003", Rad="✓ [2/2]", Dispo="Admit / Observation", Comments="-",
             Unack="●", New="▪", MRN="A003"),
        dict(Room="WRM", Patient="Adams, Devin", AgeSex="18 y/o • M", Complaint="Altered mental status", Acuity="Level 2",
             RN="MEE; NEE", MD="APE", Lab="-", Rad="-", Dispo="-", Comments="-",
             Unack="●", New="-", MRN="A004"),
        dict(Room="WRM", Patient="Weber, Charlotte", AgeSex="37 y/o • F", Complaint="Chest pain", Acuity="Level 3",
             RN="-", MD="-", Lab="-", Rad="-", Dispo="-", Comments="-",
             Unack="●", New="-", MRN="A005"),
        dict(Room="WRM", Patient="Busch, Amelia", AgeSex="37 y/o • F", Complaint="Chest pain", Acuity="Level 3",
             RN="-", MD="-", Lab="-", Rad="-", Dispo="-", Comments="-",
             Unack="●", New="-", MRN="A006"),
    ]
    df = pd.DataFrame(rows)
    df["Indicators"] = (
        df["Unack"].replace({"-": ""}) + "  " +
        df["New"].replace({"-": ""})
    ).str.strip()
    return df

TRACKBOARD_DF = make_trackboard_data()

PATIENT_DB = {
    "A003": {
        "name": "Green, Gary",
        "age": 48, "sex": "M", "mrn": "A003", "arrival_mode": "Walk-in",
        "vitals": {"HR": 118, "BP": "165/98", "RR": 24, "SpO2": "97%", "Temp": "36.9°C"},
        "chief_complaint": "Chest pain",
        "history": ["Hypertension", "Smoker (10 pack-years)"],
        "orders": ["ECG (done)", "High-Sensitivity Troponin x2", "Chest X-ray", "Aspirin 300 mg chewed"],
        "notes": "Pain 1h ago radiates to left arm; diaphoresis.",
        "acuity_suggestion": "T2 — Elevated risk; consider ACS pathway.",
        "uncertainty": {"confidence": 0.72, "pending": ["Second troponin", "No prior ECG"]},
        "disposition": "TBD",
    }
}