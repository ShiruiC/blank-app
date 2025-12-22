# utils.py
import os
from datetime import datetime
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# —— 导航常量：分成两份 —— #
# 1) 侧边栏用的「列表」（路径, 图标, 标签）
NAV_PAGES = [
    ("streamlit_app.py",                 "house",         "Landing"),
    ("pages/1_ED_Track_Board.py",        "kanban",        "ED Track Board"),
    ("pages/2_Patient_Chart.py",         "heart-pulse",   "Patient Chart"),
    ("pages/3_Registration.py",          "receipt",       "Registration"),
    ("pages/4_Visits_and_Complaints.py", "folder2",       "Visits & Complaints"),
    ("pages/5_Clinical_Decision_Log.py", "clipboard-check",       "Clinical Decision Log"),
    ("pages/6_Audit_Log.py",             "clock-history", "Audit Log"),
]

# 2) 路由查找用的「字典」 label -> path
PAGE_ROUTES = {
    "Landing": "streamlit_app.py",
    "ED Track Board": "pages/1_ED_Track_Board.py",
    "Patient Chart": "pages/2_Patient_Chart.py",
    "Registration": "pages/3_Registration.py",
    "Visits & Complaints": "pages/4_Visits_and_Complaints.py",
    "Clinical Decision Log": "pages/5_Clinical_Decision_Log.py",
    "Audit Log": "pages/6_Audit_Log.py",
}

# --------- 状态 & 通用组件 --------- #
def init_state():
    st.session_state.setdefault("nav_history", ["Landing"])
    st.session_state.setdefault("selected_patient_id", None)
    st.session_state.setdefault("sidebar_collapsed", False)

def header(title: str, subtitle: str = ""):
    col1, col2 = st.columns([0.75, 0.25])
    with col1:
        st.markdown(f"### {title}")
        if subtitle:
            st.caption(subtitle)
    with col2:
        st.caption(datetime.now().strftime("Time: %H:%M  |  Date: %Y-%m-%d"))

def _current_base(current_file: str) -> str:
    return os.path.basename(current_file)

def enter_page(page_label: str):
    """每个页面一开始调用，记录当前页面到历史栈。"""
    hist = st.session_state["nav_history"]
    if not hist or hist[-1] != page_label:
        hist.append(page_label)

def go_back():
    """返回上一页（不新开 tab，用 st.switch_page 内部跳转）。"""
    hist = st.session_state.get("nav_history", [])
    if len(hist) > 1:
        hist.pop()
        target_label = hist[-1]
        st.switch_page(PAGE_ROUTES[target_label])

def show_back_top_right(label: str = "⬅ Back"):
    """右上角显示 Back 按钮。Landing 页不显示。"""
    if len(st.session_state.get("nav_history", [])) <= 1:
        return
    col_a, col_b, col_c, col_right = st.columns([1, 1, 8, 0.8])
    with col_right:
        if st.button(label, type="secondary"):
            go_back()

def reset_on_landing():
    """回到 Landing 时自动‘刷新记忆’：清除除 nav_history 外的状态。"""
    keep = {"nav_history"}
    for k in list(st.session_state.keys()):
        if k not in keep:
            del st.session_state[k]
    st.session_state["nav_history"] = ["Landing"]

def render_sidebar(current_file: str):
    """蓝色侧栏：图标在上，文字在下；点击跳转不新开页。"""
    paths, icons, labels = zip(*NAV_PAGES)
    current_base = _current_base(current_file)

    try:
        default_idx = [os.path.basename(p) for p in paths].index(current_base)
    except ValueError:
        default_idx = 0

    with st.sidebar:
        selected = option_menu(
            None,
            list(labels),
            icons=list(icons),
            menu_icon=None,
            default_index=default_idx,
            styles={
                "container": {"padding": "0", "background-color": "#2F66F6", "width": "100%"},
                "icon": {"color": "white", "font-size": "22px"},
                "nav-link": {
                    "display": "flex",
                    "flex-direction": "column",
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
                "nav-link-selected": {"background-color": "rgba(255,255,255,0.18)", "color": "white"},
            },
            orientation="vertical",
        )

    target_path = PAGE_ROUTES.get(selected)
    if target_path and os.path.basename(target_path) != current_base:
        st.switch_page(target_path)

# ---------------- 保留的数据与其它函数 ----------------
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
    df["Indicators"] = (df["Unack"].replace({"-": ""}) + "  " + df["New"].replace({"-": ""})).str.strip()
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
