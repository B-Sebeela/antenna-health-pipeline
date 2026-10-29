"""Create multi-user monitoring tables: alerts + audit_log."""

from pathlib import Path
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB = PROJECT_ROOT / "data" / "antenna_health.db"

DDL = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario_id     TEXT NOT NULL,
    severity        TEXT NOT NULL,
    anomaly_score   REAL,
    health_status   TEXT,
    status          TEXT NOT NULL DEFAULT 'OPEN',  -- OPEN | ACKNOWLEDGED | RESOLVED
    message         TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT,
    ack_by          TEXT,
    ack_at          TEXT,
    UNIQUE(scenario_id, status)  -- at most one OPEN/ACK per scenario simplified below
);

CREATE TABLE IF NOT EXISTS audit_log (
    audit_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT NOT NULL,
    action          TEXT NOT NULL,
    detail          TEXT,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at);
"""

def main():
    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.executescript(DDL)
    conn.commit()
    conn.close()
    print(f"Monitoring tables ready in {DB}")

if __name__ == "__main__":
    main()