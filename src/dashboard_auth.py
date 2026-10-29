"""
Authentication, roles, alerts and audit helpers for the dashboard.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import sqlite3

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_FILE = PROJECT_ROOT / "data" / "antenna_health.db"
CONFIG_FILE = PROJECT_ROOT / "config.yaml"

# username -> role
ROLES = {
    "admin": "admin",
    "operator": "operator",
    "viewer": "viewer",
}

ROLE_LEVEL = {"viewer": 1, "operator": 2, "admin": 3}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def load_authenticator() -> tuple[stauth.Authenticate, dict]:
    with open(CONFIG_FILE) as f:
        config = yaml.load(f, Loader=SafeLoader)

    # Hash plain passwords on first load if still plain (streamlit-authenticator API)
    # Prefer pre-hashed passwords in production; for demo we hash at runtime once.
    credentials = config["credentials"]
    authenticator = stauth.Authenticate(
        credentials,
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )
    return authenticator, config


def require_login() -> tuple[str, str]:
    """
    Gate the page. Returns (username, role) or stops.
    Call at the top of every page.
    """
    authenticator, config = load_authenticator()

    authenticator.login(location="main")

    auth_status = st.session_state.get("authentication_status")
    name = st.session_state.get("name")
    username = st.session_state.get("username")

    if auth_status is False:
        st.error("Username or password is incorrect.")
        st.stop()
    if auth_status is None:
        st.warning("Please log in to access the Antenna Health Monitoring System.")
        st.stop()

    role = ROLES.get(username, "viewer")

    # Sidebar identity + logout
    with st.sidebar:
        st.markdown(f"**Signed in:** {name}")
        st.caption(f"Role: `{role}`")
        authenticator.logout("Logout", location="sidebar")

    # Audit login once per session
    if not st.session_state.get("_login_audited"):
        write_audit(username, "LOGIN", f"role={role}")
        st.session_state["_login_audited"] = True

    return username, role


def require_role(role: str, minimum: str = "operator") -> bool:
    """Return True if role is at least minimum."""
    return ROLE_LEVEL.get(role, 0) >= ROLE_LEVEL.get(minimum, 99)


def write_audit(username: str, action: str, detail: str = "") -> None:
    conn = get_db()
    try:
        conn.execute(
            """
            INSERT INTO audit_log (username, action, detail, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (username, action, detail, _now()),
        )
        conn.commit()
    finally:
        conn.close()


def sync_alerts_from_results() -> int:
    """
    Create OPEN alerts for HIGH (and MODERATE) anomalies.
    Resolve alerts when scenario is HEALTHY again.
    Returns number of new OPEN alerts created.
    """
    conn = get_db()
    created = 0
    try:
        rows = conn.execute(
            """
            SELECT scenario_id, severity, anomaly_score_100, health_status
            FROM anomaly_results
            """
        ).fetchall()

        for row in rows:
            sid = row["scenario_id"]
            severity = row["severity"]
            score = row["anomaly_score_100"]
            health = row["health_status"]

            open_alert = conn.execute(
                """
                SELECT alert_id, status FROM alerts
                WHERE scenario_id = ?
                  AND status IN ('OPEN', 'ACKNOWLEDGED')
                ORDER BY alert_id DESC LIMIT 1
                """,
                (sid,),
            ).fetchone()

            should_alert = health == "ANOMALY_DETECTED" and severity in ("HIGH", "MODERATE")

            if should_alert and open_alert is None:
                msg = f"{severity} anomaly on {sid} (score={score:.1f})"
                conn.execute(
                    """
                    INSERT INTO alerts
                        (scenario_id, severity, anomaly_score, health_status,
                         status, message, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'OPEN', ?, ?, ?)
                    """,
                    (sid, severity, score, health, msg, _now(), _now()),
                )
                created += 1
            elif not should_alert and open_alert is not None:
                conn.execute(
                    """
                    UPDATE alerts
                    SET status = 'RESOLVED', updated_at = ?
                    WHERE alert_id = ?
                    """,
                    (_now(), open_alert["alert_id"]),
                )

        conn.commit()
    finally:
        conn.close()
    return created


def list_alerts(status: str | None = None) -> list[dict]:
    conn = get_db()
    try:
        if status:
            rows = conn.execute(
                """
                SELECT * FROM alerts
                WHERE status = ?
                ORDER BY
                    CASE severity WHEN 'HIGH' THEN 0 WHEN 'MODERATE' THEN 1 ELSE 2 END,
                    created_at DESC
                """,
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM alerts
                ORDER BY
                    CASE status WHEN 'OPEN' THEN 0 WHEN 'ACKNOWLEDGED' THEN 1 ELSE 2 END,
                    CASE severity WHEN 'HIGH' THEN 0 WHEN 'MODERATE' THEN 1 ELSE 2 END,
                    created_at DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def count_open_alerts() -> tuple[int, int]:
    """Returns (open_high, open_total)."""
    conn = get_db()
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE status IN ('OPEN', 'ACKNOWLEDGED')"
        ).fetchone()[0]
        high = conn.execute(
            """
            SELECT COUNT(*) FROM alerts
            WHERE status IN ('OPEN', 'ACKNOWLEDGED') AND severity = 'HIGH'
            """
        ).fetchone()[0]
        return int(high), int(total)
    finally:
        conn.close()


def acknowledge_alert(alert_id: int, username: str) -> None:
    conn = get_db()
    try:
        conn.execute(
            """
            UPDATE alerts
            SET status = 'ACKNOWLEDGED',
                ack_by = ?,
                ack_at = ?,
                updated_at = ?
            WHERE alert_id = ? AND status = 'OPEN'
            """,
            (username, _now(), _now(), alert_id),
        )
        conn.commit()
    finally:
        conn.close()
    write_audit(username, "ACK_ALERT", f"alert_id={alert_id}")