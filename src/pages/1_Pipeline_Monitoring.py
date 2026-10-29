"""
Pipeline Monitoring page – Antenna Health Monitoring System
"""

from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st

# File is at: src/pages/pipeline_monitoring.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_FILE = PROJECT_ROOT / "data" / "antenna_health.db"
FEATURE_FILE = PROJECT_ROOT / "data" / "features" / "scenario_features_refined.parquet"
ANOMALY_PARQUET = PROJECT_ROOT / "data" / "features" / "anomaly_results.parquet"

st.set_page_config(page_title="Pipeline Monitoring", page_icon="⚙️", layout="wide")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard_auth import require_login, sync_alerts_from_results

username, role = require_login()
sync_alerts_from_results()

st.title("Pipeline Monitoring")
st.caption("Data-engineering visibility – run history, stage health and data quality")


@st.cache_data(ttl=30, show_spinner=False)
def load_pipeline_runs(limit: int = 30) -> pd.DataFrame:
    conn = sqlite3.connect(DATABASE_FILE)
    try:
        df = pd.read_sql_query(
            f"""
            SELECT run_id, stage, status, started_at, finished_at,
                   scenarios_in, scenarios_out, error_message, notes
            FROM pipeline_runs
            ORDER BY run_id DESC
            LIMIT {limit}
            """,
            conn,
        )
    finally:
        conn.close()
    return df


@st.cache_data(ttl=30, show_spinner=False)
def load_data_quality() -> dict:
    conn = sqlite3.connect(DATABASE_FILE)
    try:
        n_results = conn.execute("SELECT COUNT(*) FROM anomaly_results").fetchone()[0]
        has_scenarios = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='scenarios'"
        ).fetchone()
        n_scenarios = (
            conn.execute("SELECT COUNT(*) FROM scenarios").fetchone()[0]
            if has_scenarios else 0
        )
        last_ingest = conn.execute(
            """
            SELECT MAX(started_at) FROM pipeline_runs
            WHERE stage IN ('ingest', 'features') AND status = 'ok'
            """
        ).fetchone()[0]
        last_anomaly = conn.execute(
            """
            SELECT MAX(started_at) FROM pipeline_runs
            WHERE stage IN ('anomaly', 'store') AND status = 'ok'
            """
        ).fetchone()[0]
    finally:
        conn.close()

    try:
        n_features = len(pd.read_parquet(FEATURE_FILE))
    except Exception:
        n_features = None
    try:
        n_anomaly_pq = len(pd.read_parquet(ANOMALY_PARQUET)) if ANOMALY_PARQUET.exists() else None
    except Exception:
        n_anomaly_pq = None

    return {
        "n_results": n_results,
        "n_scenarios": n_scenarios,
        "n_features": n_features,
        "n_anomaly_pq": n_anomaly_pq,
        "last_ingest": last_ingest,
        "last_anomaly": last_anomaly,
    }


def parse_ts(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except Exception:
        return None


def compute_duration(row) -> str:
    start = parse_ts(row["started_at"])
    end = parse_ts(row["finished_at"])
    if start is None or end is None:
        return "—"
    seconds = (end - start).total_seconds()
    if seconds < 0:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f} min"
    return f"{minutes/60:.1f} h"


def colour_status(val: str) -> str:
    v = str(val).lower()
    if v == "ok":
        return "background-color: #e6f4ea; color: #137333; font-weight: 700; text-align: center"
    if v == "error":
        return "background-color: #fce8e6; color: #c5221f; font-weight: 700; text-align: center"
    if v == "started":
        return "background-color: #fef7e0; color: #b06000; font-weight: 700; text-align: center"
    return "text-align: center"


runs = load_pipeline_runs()
dq = load_data_quality()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Results in SQLite", dq["n_results"])
c2.metric("Features (parquet)", dq["n_features"] if dq["n_features"] is not None else "—")
c3.metric("Anomaly results (parquet)", dq["n_anomaly_pq"] if dq["n_anomaly_pq"] is not None else "—")

last = dq["last_anomaly"] or dq["last_ingest"]
if last:
    try:
        ts = pd.to_datetime(last, utc=True)
        age_h = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600
        freshness = f"{age_h:.1f} h ago" if age_h < 48 else f"{age_h/24:.1f} d ago"
    except Exception:
        freshness = str(last)[:19]
else:
    freshness = "No runs yet"
c4.metric("Last successful run", freshness)

st.divider()
st.subheader("Latest stage status")

if runs.empty:
    st.info("No pipeline runs recorded yet.")
else:
    latest = (
        runs.sort_values("run_id", ascending=False)
        .groupby("stage", as_index=False)
        .first()
        .sort_values("stage")
    )
    cols = st.columns(len(latest))
    for i, (_, row) in enumerate(latest.iterrows()):
        with cols[i]:
            status = str(row["status"]).lower()
            icon = "🟢" if status == "ok" else ("🔴" if status == "error" else "🟡")
            st.markdown(f"{icon} **{row['stage']}** — `{row['status']}`")
            finished = parse_ts(row["finished_at"])
            st.caption(finished.strftime("%Y-%m-%d %H:%M:%S") if finished is not None else "—")

st.divider()
st.subheader("Recent pipeline runs")

if runs.empty:
    st.info("Run `python run_pipeline.py` to generate history.")
else:
    display = runs.copy()
    display["Duration"] = display.apply(compute_duration, axis=1)
    display["Started"] = display["started_at"].apply(
        lambda x: parse_ts(x).strftime("%Y-%m-%d %H:%M:%S") if parse_ts(x) else "—"
    )
    display["Finished"] = display["finished_at"].apply(
        lambda x: parse_ts(x).strftime("%Y-%m-%d %H:%M:%S") if parse_ts(x) else "—"
    )
    display = display.rename(columns={
        "run_id": "ID",
        "stage": "Stage",
        "status": "Status",
        "scenarios_in": "In",
        "scenarios_out": "Out",
        "error_message": "Error",
    })
    display["Status"] = display["Status"].astype(str).str.upper()
    display["Error"] = display["Error"].fillna("").astype(str).str.slice(0, 60)

    show_cols = ["ID", "Stage", "Status", "Started", "Finished", "Duration", "In", "Out", "Error"]
    styled = display[show_cols].style.map(colour_status, subset=["Status"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Antenna Health Monitoring Pipeline | "
    "FEKO → Ingestion → Parquet → Features → "
    "Anomaly Detection → Classification → SQLite → Dashboard"
)