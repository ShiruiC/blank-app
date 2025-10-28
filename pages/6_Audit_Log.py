# pages/6_Audit_Log.py
import hashlib, json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import streamlit as st
import os

# ───────── Safe utils import (optional in your project) ─────────
def _no_op(*a, **k): return None
try:
    from utils import render_sidebar as _render_sidebar, enter_page as _enter_page, show_back_top_right as _show_back
except Exception:
    _render_sidebar = _no_op; _enter_page = _no_op; _show_back = _no_op

PAGE_TITLE = "Audit Log"
CURRENT_FILE = __file__ if "__file__" in globals() else "pages/6_Audit_Log.py"

ACTION_CHOICES = [
    "login","logout","view_chart","edit_chart","triage_change","disposition_change",
    "ai_inference","ai_adoption_toggle","export","registration_update","note_add",
    "consent_update","access_denied","config_change"
]
ROLE_CHOICES = ["Attending","Resident","Nurse","Admin","System"]
STATUS_CHOICES = ["success","failure","warning"]
SEVERITY_CHOICES = ["info","low","medium","high","critical"]

# ====== Hashing helpers ======
def _canonical_payload(entry: Dict[str, Any]) -> str:
    payload = {k: v for k, v in entry.items() if k != "hash"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))

def compute_hash(entry: Dict[str, Any]) -> str:
    m = hashlib.sha256()
    m.update(_canonical_payload(entry).encode("utf-8"))
    return m.hexdigest()

def chain_and_hash(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prev = ""
    for e in entries:
        e["prev_hash"] = prev
        e["hash"] = compute_hash(e)
        prev = e["hash"]
    return entries

def verify_chain(df: pd.DataFrame) -> Dict[str, Any]:
    ok = True
    broken = []
    prev = ""
    for i, row in df.iterrows():
        if row.get("prev_hash","") != prev:
            ok = False; broken.append(i)
        entry = row.to_dict()
        expected = compute_hash({k: entry[k] for k in entry if k != "hash"})
        if entry.get("hash","") != expected:
            ok = False
            if i not in broken: broken.append(i)
        prev = entry.get("hash","")
    return {"ok": ok, "broken_indices": broken}

# ====== Demo data ======
def _mk_entry(ts: datetime, actor_id: str, role: str, action: str,
              target_type: str, target_id: str, mrn: Optional[str],
              status: str, severity: str, ip: str, sid: str,
              metadata: Dict[str, Any], before: Optional[Dict[str, Any]], after: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "ts": ts.isoformat(timespec="seconds"),
        "actor_id": actor_id, "actor_role": role, "action": action,
        "target_type": target_type, "target_id": target_id, "mrn": mrn or "",
        "status": status, "severity": severity,
        "client_ip": ip, "session_id": sid,
        "metadata": metadata, "before": before, "after": after,
        "prev_hash": "", "hash": ""
    }

def make_demo_logs(n_days: int = 2) -> pd.DataFrame:
    now = datetime.now()
    ts = now - timedelta(days=n_days)
    rows = []
    rows.append(_mk_entry(ts, "u123","Resident","login","user","u123","", "success","info","10.0.0.21","S-1",{},None,None))
    ts += timedelta(minutes=1)
    rows.append(_mk_entry(ts,"u123","Resident","view_chart","patient","CP-1000","CP-1000","success","info","10.0.0.21","S-1",{"page":"Patient Chart"},None,None))
    ts += timedelta(minutes=1)
    rows.append(_mk_entry(ts,"u123","Resident","ai_inference","patient","CP-1000","CP-1000","success","info","10.0.0.21","S-1",{"model":"TriageNet v0.3","latency_ms":182},None,{"triage":"T2","dispo":"Observe"}))
    ts += timedelta(minutes=2)
    rows.append(_mk_entry(ts,"u123","Resident","triage_change","patient","CP-1000","CP-1000","success","medium","10.0.0.21","S-1",{"reason":"Chest pain + ECG abn"},{"triage":"T3"},{"triage":"T2"}))
    ts += timedelta(minutes=1)
    rows.append(_mk_entry(ts,"u123","Resident","ai_adoption_toggle","patient","CP-1000","CP-1000","success","info","10.0.0.21","S-1",{"adopted":True},{"adopted":False},{"adopted":True}))
    ts += timedelta(minutes=3)
    rows.append(_mk_entry(ts,"u987","Nurse","view_chart","patient","CP-2000","CP-2000","success","info","10.0.0.31","S-2",{"page":"ED Track Board"},None,None))
    ts += timedelta(minutes=2)
    rows.append(_mk_entry(ts,"u001","Admin","export","report","audit-csv","", "success","low","10.0.0.9","S-ADM",{"rows":124},None,None))
    ts += timedelta(minutes=5)
    rows.append(_mk_entry(ts,"u123","Resident","access_denied","patient","CP-3000","CP-3000","failure","high","10.0.0.21","S-1",{"policy":"need-to-know"},None,None))
    chain_and_hash(rows)
    return pd.DataFrame(rows)

# ====== UI helpers ======
def _enter():
    # Try both signatures: (title, current_file) and (title)
    try:
        _enter_page(PAGE_TITLE, CURRENT_FILE)
    except TypeError:
        try:
            _enter_page(PAGE_TITLE)
        except Exception:
            pass
    except Exception:
        pass

def _render_sidebar_safe():
    # Your utils likely needs current_file; try with it, then fall back.
    try:
        _render_sidebar(CURRENT_FILE)
    except TypeError:
        try:
            _render_sidebar()
        except Exception:
            pass
    except Exception:
        pass

def _back_fallback_right():
    with st.container():
        st.markdown(
            "<div style='text-align:right'>"
            "<button onclick='history.back()' "
            "style='padding:6px 10px;border:1px solid #e5e7eb;border-radius:10px;background:#fff;cursor:pointer'>← Back</button>"
            "</div>",
            unsafe_allow_html=True
        )

def render_header(integrity: Dict[str, Any]):
    col_l, col_r = st.columns([1, 0.2], vertical_alignment="center")
    with col_l:
        st.title(PAGE_TITLE)
    with col_r:
        try:
            _show_back()
        except Exception:
            _back_fallback_right()

    st.caption("Read-only, append-only trail of actions across the app.")
    if integrity["ok"]:
        st.success("Integrity: OK — hash chain verified.")
    else:
        st.error(f"Integrity: BROKEN — {len(integrity['broken_indices'])} suspicious entries detected.")

def render_filters(df: pd.DataFrame) -> Dict[str, Any]:
    st.subheader("Filters")

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        date_min = pd.to_datetime(df["ts"]).min().date()
        date_max = pd.to_datetime(df["ts"]).max().date()
        start, end = st.date_input("Date range", value=(date_min, date_max))
    with c2:
        actor = st.multiselect("Actor", sorted(df["actor_id"].unique().tolist()))
        role = st.multiselect("Role", ROLE_CHOICES)
    with c3:
        action = st.multiselect("Action", ACTION_CHOICES)
        status = st.multiselect("Status", STATUS_CHOICES)

    c4, c5, c6 = st.columns([1,1,1])
    with c4:
        severity = st.multiselect("Severity", SEVERITY_CHOICES)
    with c5:
        mrn = st.text_input("Patient MRN contains")
    with c6:
        search = st.text_input("Free-text search (actor/action/target/metadata)")

    return {
        "date_start": start, "date_end": end,
        "actor": actor, "role": role, "action": action,
        "status": status, "severity": severity,
        "mrn": mrn.strip(), "search": search.strip()
    }

def apply_filters(df: pd.DataFrame, f: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out["ts_dt"] = pd.to_datetime(out["ts"])
    out = out[(out["ts_dt"].dt.date >= f["date_start"]) & (out["ts_dt"].dt.date <= f["date_end"])]
    if f["actor"]:   out = out[out["actor_id"].isin(f["actor"])]
    if f["role"]:    out = out[out["actor_role"].isin(f["role"])]
    if f["action"]:  out = out[out["action"].isin(f["action"])]
    if f["status"]:  out = out[out["status"].isin(f["status"])]
    if f["severity"]:out = out[out["severity"].isin(f["severity"])]
    if f["mrn"]:     out = out[out["mrn"].str.contains(f["mrn"], case=False, na=False)]
    if f["search"]:
        needle = f["search"].lower()
        def _match(r):
            hay = " ".join([
                r.get("actor_id",""), r.get("actor_role",""), r.get("action",""),
                r.get("target_type",""), r.get("target_id",""), r.get("mrn",""),
                json.dumps(r.get("metadata", {}))
            ]).lower()
            return needle in hay
        out = out[out.apply(_match, axis=1)]
    return out.drop(columns=["ts_dt"])

def download_button(df: pd.DataFrame):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered CSV", data=csv,
                       file_name="audit_log_filtered.csv", mime="text/csv",
                       use_container_width=True)

def render_table(df: pd.DataFrame):
    st.subheader("Results")
    page_size = st.select_slider("Rows per page", options=[10,25,50,100], value=25)
    total = len(df)
    pages = max(1, (total + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1)
    start = (page - 1) * page_size
    end = min(start + page_size, total)

    slim = df.iloc[start:end].copy()
    show_cols = ["ts","actor_id","actor_role","action","mrn","target_type","target_id","status","severity","hash"]
    st.dataframe(slim[show_cols], use_container_width=True, hide_index=True)

    with st.expander("View selected row details"):
        row_idx = st.number_input("Row index (absolute)", min_value=0, max_value=total-1, value=0, step=1)
        try:
            r = df.iloc[int(row_idx)].to_dict()
            st.json({
                "timestamp": r["ts"],
                "actor": {"id": r["actor_id"], "role": r["actor_role"], "ip": r["client_ip"], "session": r["session_id"]},
                "event": {"action": r["action"], "status": r["status"], "severity": r["severity"]},
                "target": {"type": r["target_type"], "id": r["target_id"], "mrn": r.get("mrn","")},
                "metadata": r.get("metadata", {}),
                "before": r.get("before", None),
                "after": r.get("after", None),
                "integrity": {"prev_hash": r.get("prev_hash",""), "hash": r.get("hash","")}
            }, expanded=True)
        except Exception:
            st.info("Choose a valid row index within the filtered result set.")

def main():
    _enter()
    _render_sidebar_safe()  # ✅ pass CURRENT_FILE if your util requires it
    df = make_demo_logs(n_days=2)
    integrity = verify_chain(df)
    render_header(integrity)

    with st.container(border=True):
        filters = render_filters(df)
        filtered = apply_filters(df, filters)
        ctop1, ctop2 = st.columns([3,1])
        with ctop1:
            st.caption(f"{len(filtered)} of {len(df)} events match the filters.")
        with ctop2:
            download_button(filtered)

    render_table(filtered)

    st.divider()
    with st.expander("How this Audit Log works (for admins)", expanded=False):
        st.markdown("""
- **Append-only**: new events are appended; existing rows are never mutated.
- **Hash chain**: each row stores `prev_hash` and `hash` to detect tampering.
- **Scope**: spans the whole app (login, chart views/edits, triage/disposition changes, AI runs, exports, denials, config).
- **Fields**: timestamps, actor (id/role/ip/session), action, target (type/id/MRN), status, severity, metadata, before/after diffs.
- **Filters & export**: investigative workflow with date, actor, role, action, MRN, status, severity, free text, and CSV export.
        """)

if __name__ == "__main__":
    main()