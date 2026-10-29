"""
Antenna Health Monitoring – Pipeline Orchestrator (Phase 2)

Logs every stage into pipeline_runs so the dashboard can show
execution status, duration and data freshness.
"""

from pathlib import Path
import subprocess
import sys
import time
import sqlite3
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_FILE = PROJECT_ROOT / "data" / "antenna_health.db"

STAGES = [
    {
        "name": "features",
        "display": "Feature Extraction",
        "command": [sys.executable, "-m", "src.preprocessing.features_refined"],
    },
    {
        "name": "anomaly",
        "display": "Anomaly Detection",
        "command": [sys.executable, "-m", "src.model.anomaly_detection"],
    },
    {
        "name": "store",
        "display": "Store Results in SQLite",
        "command": [sys.executable, "-m", "src.model.store_results"],
    },
]


def log_run_start(stage: str, scenarios_in: int = 0, notes: str | None = None) -> int:
    conn = sqlite3.connect(DATABASE_FILE)
    try:
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            """
            INSERT INTO pipeline_runs
                (stage, status, started_at, scenarios_in, notes)
            VALUES (?, 'started', ?, ?, ?)
            """,
            (stage, now, scenarios_in, notes),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def log_run_finish(
    run_id: int,
    status: str,
    scenarios_out: int = 0,
    error_message: str | None = None,
) -> None:
    conn = sqlite3.connect(DATABASE_FILE)
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            UPDATE pipeline_runs
            SET status = ?,
                finished_at = ?,
                scenarios_out = ?,
                error_message = ?
            WHERE run_id = ?
            """,
            (status, now, scenarios_out, error_message, run_id),
        )
        conn.commit()
    finally:
        conn.close()


def count_controlled_scenarios() -> int:
    """Best-effort count of controlled scenarios currently in the results table."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        n = conn.execute(
            "SELECT COUNT(*) FROM anomaly_results"
        ).fetchone()[0]
        conn.close()
        return int(n)
    except Exception:
        return 0


def run_stage(stage: dict) -> bool:
    name = stage["name"]
    display = stage["display"]
    command = stage["command"]

    print()
    print("=" * 70)
    print(f"STARTING: {display}")
    print("=" * 70)
    print()

    scenarios_in = count_controlled_scenarios()
    run_id = log_run_start(
        stage=name,
        scenarios_in=scenarios_in,
        notes=f"Orchestrated run of {display}",
    )

    start_time = time.time()
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    elapsed = time.time() - start_time

    print()
    print("-" * 70)

    if result.returncode == 0:
        scenarios_out = count_controlled_scenarios()
        log_run_finish(run_id, status="ok", scenarios_out=scenarios_out)
        print(f"COMPLETED: {display}")
        print(f"Time: {elapsed:.2f} seconds")
        print("-" * 70)
        return True

    log_run_finish(
        run_id,
        status="error",
        scenarios_out=0,
        error_message=f"Exit code {result.returncode}",
    )
    print(f"FAILED: {display}")
    print(f"Exit code: {result.returncode}")
    print("-" * 70)
    return False


def main():
    print()
    print("=" * 70)
    print("ANTENNA HEALTH MONITORING PIPELINE")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print()
    print("Pipeline:")
    for i, s in enumerate(STAGES, 1):
        print(f"  {i}. {s['display']}")
    print()

    total_start = time.time()

    for stage in STAGES:
        success = run_stage(stage)
        if not success:
            print()
            print("=" * 70)
            print("PIPELINE FAILED")
            print("=" * 70)
            print(f"Stopped after: {stage['display']}")
            print()
            sys.exit(1)

    total_elapsed = time.time() - total_start

    print()
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Total execution time: {total_elapsed:.2f} seconds")
    print()
    print("Outputs:")
    print("  - data/features/scenario_features_refined.parquet")
    print("  - data/features/anomaly_results.parquet")
    print("  - data/antenna_health.db")
    print()
    print("Start the dashboard with:")
    print("  streamlit run src/dashboard.py")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()