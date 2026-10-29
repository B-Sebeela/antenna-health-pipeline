"""Alerts page – multi-user monitoring."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard_auth import (
    require_login,
    require_role,
    sync_alerts_from_results,
    list_alerts,
    acknowledge_alert,
    count_open_alerts,
    get_db,
)

st.set_page_config(page_title="Alerts", page_icon="🚨", layout="wide")

username, role = require_login()
sync_alerts_from_results()

st.title("Alerts")
st.caption("Active and historical anomaly alerts for monitored antenna scenarios")

high_open, total_open = count_open_alerts()
c1, c2, c3 = st.columns(3)
c1.metric("Open / Acknowledged", total_open)
c2.metric("HIGH severity (active)", high_open)
c3.metric("Your role", role)

st.divider()

filter_status = st.selectbox(
    "Filter status",
    ["ACTIVE", "OPEN", "ACKNOWLEDGED", "RESOLVED", "ALL"],
    index=0,
)

if filter_status == "ACTIVE":
    alerts = [a for a in list_alerts() if a["status"] in ("OPEN", "ACKNOWLEDGED")]
elif filter_status == "ALL":
    alerts = list_alerts()
else:
    alerts = list_alerts(status=filter_status)

if not alerts:
    st.success("No alerts for this filter.")
else:
    for a in alerts:
        sev = a["severity"]
        colour = {"HIGH": "red", "MODERATE": "orange", "LOW": "blue"}.get(sev, "gray")
        left, right = st.columns([4, 1])
        with left:
            st.markdown(
                f"**{a['scenario_id']}** · :{colour}[{sev}] · `{a['status']}` · "
                f"score {float(a['anomaly_score'] or 0):.1f}"
            )
            st.caption(f"{a.get('message') or ''} · created {str(a['created_at'])[:19]}")
            if a.get("ack_by"):
                st.caption(f"Acknowledged by {a['ack_by']} at {str(a['ack_at'])[:19]}")
        with right:
            can_ack = require_role(role, "operator") and a["status"] == "OPEN"
            if can_ack:
                if st.button("Acknowledge", key=f"ack_{a['alert_id']}"):
                    acknowledge_alert(int(a["alert_id"]), username)
                    st.rerun()
            elif a["status"] == "OPEN":
                st.caption("View only")
        st.divider()

st.subheader("Audit log (recent)")
conn = get_db()
try:
    audit = pd.read_sql_query(
        """
        SELECT username, action, detail, created_at
        FROM audit_log
        ORDER BY audit_id DESC
        LIMIT 30
        """,
        conn,
    )
finally:
    conn.close()

if audit.empty:
    st.info("No audit events yet.")
else:
    audit["created_at"] = audit["created_at"].astype(str).str[:19]
    st.dataframe(audit, use_container_width=True, hide_index=True)