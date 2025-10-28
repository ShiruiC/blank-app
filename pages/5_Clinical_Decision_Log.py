# pages/5_Clinical_Decision_Log.py
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any

# ───────── Safe utils import (optional in your project) ─────────
def _no_op(*a, **k): return None
try:
    from utils import render_sidebar as _render_sidebar, enter_page as _enter_page, show_back_top_right as _show_back
except Exception:
    _render_sidebar = _no_op; _enter_page = _no_op; _show_back = _no_op

PAGE_TITLE = "Clinical Decision Log"

# ===== Helpers =====

AGREE_FULL = "Agree"
AGREE_PARTIAL = "Partial"
AGREE_NO = "Disagree"

def _norm(s: Optional[str]) -> str:
    return (s or "").strip()

def _agreement(ai_triage: str, ai_dispo: str, ai_steps: str,
               triage: str, dispo: str, steps: str) -> str:
    ai_t, ai_d, ai_s = _norm(ai_triage).upper(), _norm(ai_dispo).upper(), _norm(ai_steps)
    t, d, s = _norm(triage).upper(), _norm(dispo).upper(), _norm(steps)
    triage_match = (ai_t == t) if ai_t or t else True
    dispo_match  = (ai_d == d) if ai_d or d else True
    steps_match  = (ai_s == s) if ai_s or s else True
    if triage_match and dispo_match and steps_match:
        return AGREE_FULL
    if triage_match and dispo_match:
        return AGREE_PARTIAL
    return AGREE_NO

def _ensure_state():
    if "clinical_decisions" not in st.session_state:
        st.session_state["clinical_decisions"] = []
    # expose a callable other pages can use
    def log_clinical_decision(*, patient_id: str, patient_name: str,
                              clinician_id: str, clinician_name: str,
                              ai_triage: str, ai_disposition: str, ai_next_steps: str = "",
                              triage: str, disposition: str, next_steps: str = "",
                              meta: Optional[Dict[str, Any]] = None,
                              timestamp: Optional[str] = None):
        entry = {
            "timestamp": timestamp or datetime.now().isoformat(timespec="seconds"),
            "patient_id": patient_id,
            "patient_name": patient_name,
            "clinician_id": clinician_id,
            "clinician_name": clinician_name,
            "ai_triage": ai_triage, "ai_disposition": ai_disposition, "ai_next_steps": ai_next_steps or "",
            "triage": triage, "disposition": disposition, "next_steps": next_steps or "",
            "meta": meta or {},
        }
        entry["agreement"] = _agreement(
            entry["ai_triage"], entry["ai_disposition"], entry["ai_next_steps"],
            entry["triage"], entry["disposition"], entry["next_steps"]
        )
        st.session_state.clinical_decisions.append(entry)
    st.session_state.log_clinical_decision = log_clinical_decision

def _demo_seed():
    if st.session_state.clinical_decisions:
        return
    seed = [
        dict(patient_id="CP-1000", patient_name="Weber, Charlotte",
             clinician_id="U-001", clinician_name="Dr. Demo",
             ai_triage="T3", ai_disposition="Observe", ai_next_steps="hs-Troponin now; repeat per rule-out algorithm; observe…",
             triage="T3", disposition="Observe", next_steps="hs-Troponin now; repeat per rule-out algorithm; observe…"),
        dict(patient_id="CP-1001", patient_name="Müller, Jonas",
             clinician_id="U-001", clinician_name="Dr. Demo",
             ai_triage="T2", ai_disposition="Consult", ai_next_steps="Urgent cardiology consult; continuous monitor",
             triage="T2", disposition="Confirm/Admit", next_steps="Admit to CCU; continuous monitor"),
    ]
    for e in seed:
        st.session_state.log_clinical_decision(**e, meta={"seed": True})

def _badge(text: str):
    if text == AGREE_FULL:
        st.markdown(f"<span class='tag ok'>{text}</span>", unsafe_allow_html=True)
    elif text == AGREE_PARTIAL:
        st.markdown(f"<span class='tag warn'>{text}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='tag no'>{text}</span>", unsafe_allow_html=True)

def _css():
    st.markdown(
        """
        <style>
          .metric-chips { display:flex; gap:.5rem; flex-wrap:wrap; margin:.25rem 0 1rem 0;}
          .chip {border-radius:999px; padding:.25rem .6rem; font-size:.85rem; border:1px solid rgba(0,0,0,.08); background:#fafafa;}
          .chip strong {margin-left:.3rem}
          .tag {border-radius:8px; padding:.15rem .5rem; font-size:.80rem; border:1px solid transparent;}
          .tag.ok {background:#ECFDF5; color:#047857; border-color:#A7F3D0;}
          .tag.warn {background:#FFFBEB; color:#92400E; border-color:#FDE68A;}
          .tag.no {background:#FEF2F2; color:#B91C1C; border-color:#FECACA;}
          .mini {font-size:.8rem; color:#666;}
          .tbl .stMarkdown { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 24rem; }
        </style>
        """,
        unsafe_allow_html=True
    )

def _filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.container():
        c1, c2, c3, c4 = st.columns([1.3,1.3,1,1])
        patient = c1.selectbox(
            "Patient", ["All"] + sorted((df["patient_id"].astype(str)+" · "+df["patient_name"].astype(str)).unique().tolist()),
            index=0
        )
        clinician = c2.selectbox("Clinician", ["All"] + sorted(df["clinician_name"].unique().tolist()), index=0)
        agree = c3.selectbox("Agreement", ["All", AGREE_FULL, AGREE_PARTIAL, AGREE_NO], index=0)
        date_range = c4.date_input("Date", value=None)

    mask = pd.Series(True, index=df.index)

    if patient != "All":
        pid = patient.split(" · ")[0]
        mask &= (df["patient_id"] == pid)
    if clinician != "All":
        mask &= (df["clinician_name"] == clinician)
    if agree != "All":
        mask &= (df["agreement"] == agree)
    if date_range:
        if not isinstance(date_range, (list, tuple)):
            date_range = [date_range, date_range]
        start = datetime.combine(date_range[0], datetime.min.time())
        end = datetime.combine(date_range[-1], datetime.max.time())
        t = pd.to_datetime(df["timestamp"])
        mask &= (t >= start) & (t <= end)

    return df[mask].copy()

def _summary_chips(df: pd.DataFrame):
    cols = st.columns(4)
    cols[0].markdown("**Entries**")
    cols[0].markdown(f"<div class='metric-chips'><div class='chip'>Total <strong>{len(df):,}</strong></div></div>", unsafe_allow_html=True)
    for i, (label, color) in enumerate([(AGREE_FULL,"ok"), (AGREE_PARTIAL,"warn"), (AGREE_NO,"no")], start=1):
        n = int((df["agreement"] == label).sum())
        cols[i].markdown(f"**{label}**")
        cols[i].markdown(f"<div class='metric-chips'><div class='chip'><span class='tag {color}'>{label}</span> <strong>{n:,}</strong></div></div>", unsafe_allow_html=True)

def _render_table(df: pd.DataFrame):
    if df.empty:
        st.info("No entries match the current filters.")
        return
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    show_cols = {
        "timestamp": "Time",
        "patient_id": "MRN",
        "patient_name": "Patient",
        "clinician_name": "Clinician",
        "ai_triage": "AI Triage",
        "triage": "Chosen Triage",
        "ai_disposition": "AI Disposition",
        "disposition": "Chosen Disposition",
        "agreement": "Agreement",
        "ai_next_steps": "AI Next Steps",
        "next_steps": "Chosen Next Steps",
    }
    slim = df[list(show_cols.keys())].copy()
    slim.rename(columns=show_cols, inplace=True)

    st.markdown("#### Log entries")
    st.dataframe(
        slim.style.hide(axis="index"),
        use_container_width=True, height=min(560, 48 + 32*max(3, len(slim)))
    )

    st.markdown("#### Details")
    for _, row in df.iterrows():
        with st.expander(f"{row['timestamp']} • {row['patient_id']} · {row['patient_name']} • {row['clinician_name']}"):
            cols = st.columns(4)
            cols[0].markdown("**AI Triage**"); cols[0].markdown(row["ai_triage"])
            cols[1].markdown("**Chosen Triage**"); cols[1].markdown(row["triage"])
            cols[2].markdown("**AI Disposition**"); cols[2].markdown(row["ai_disposition"])
            cols[3].markdown("**Chosen Disposition**"); cols[3].markdown(row["disposition"])
            st.markdown("**Agreement**"); _badge(row["agreement"])
            st.markdown("**Next steps**")
            st.write(f"**AI:** {row['ai_next_steps'] or '—'}")
            st.write(f"**Chosen:** {row['next_steps'] or '—'}")
            meta = row.get("meta") or {}
            if meta:
                st.markdown("<div class='mini'>Meta:</div>", unsafe_allow_html=True)
                st.json(meta, expanded=False)

def _download(df: pd.DataFrame):
    filename = f"clinical_decision_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Export CSV", data=csv, file_name=filename, mime="text/csv")

def _render_sidebar_safe():
    # utils.render_sidebar may be defined as render_sidebar(current_file: str)
    # or without args. Try both.
    try:
        _render_sidebar(__file__)
    except TypeError:
        try:
            _render_sidebar(current_file=__file__)
        except TypeError:
            _render_sidebar()

# ===== Page =====
def main():
    _enter_page(PAGE_TITLE)
    _ensure_state()
    _css()

    # Use the imported blue sidebar renderer (from utils)
    _render_sidebar_safe()

    st.title("Clinical Decision Log")
    st.caption("Records of clinician decisions vs. AI recommendations, per patient.")

    _demo_seed()

    df = pd.DataFrame(st.session_state.clinical_decisions or [])
    if df.empty:
        st.info("No clinical decisions have been recorded yet.")
        _show_back()
        return

    df_filtered = _filters(df)
    _summary_chips(df_filtered)
    _render_table(df_filtered)
    _download(df_filtered)

    # Top-right back button (from utils)
    _show_back()

if __name__ == "__main__":
    main()