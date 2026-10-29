"""
Store anomaly detection and classification results in SQLite.
"""

from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "anomaly_results.parquet"
)

DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "antenna_health.db"
)


def main():

    print("\n" + "=" * 80)
    print("STORING ANOMALY RESULTS")
    print("=" * 80)

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Anomaly result file not found:\n{FEATURE_FILE}"
        )

    df = pd.read_parquet(FEATURE_FILE)

    print(
        f"Results loaded: {len(df)} scenarios"
    )

    conn = sqlite3.connect(DATABASE_FILE)

    try:

        # ----------------------------------------------------
        # Create table
        # ----------------------------------------------------

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS anomaly_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scenario_id TEXT NOT NULL,
                anomaly_score REAL,
                anomaly_score_100 REAL,
                health_status TEXT,
                severity TEXT,
                predicted_fault_type TEXT,
                predicted_condition TEXT,
                diagnostic_confidence REAL,
                known_health TEXT,
                known_fault_type TEXT,
                known_condition TEXT,
                feed_setting_percent REAL,
                health_correct INTEGER,
                fault_type_correct INTEGER,
                direction_correct INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # ----------------------------------------------------
        # Remove previous controlled results
        # ----------------------------------------------------

        scenario_ids = df["scenario_id"].tolist()

        placeholders = ",".join(
            ["?"] * len(scenario_ids)
        )

        conn.execute(
            f"""
            DELETE FROM anomaly_results
            WHERE scenario_id IN ({placeholders})
            """,
            scenario_ids
        )

        # ----------------------------------------------------
        # Insert
        # ----------------------------------------------------

        insert_sql = """
            INSERT INTO anomaly_results (
                scenario_id,
                anomaly_score,
                anomaly_score_100,
                health_status,
                severity,
                predicted_fault_type,
                predicted_condition,
                diagnostic_confidence,
                known_health,
                known_fault_type,
                known_condition,
                feed_setting_percent,
                health_correct,
                fault_type_correct,
                direction_correct
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        for _, row in df.iterrows():

            conn.execute(
                insert_sql,
                (
                    row["scenario_id"],
                    float(row["anomaly_score"]),
                    float(row["anomaly_score_100"]),
                    row["health_status"],
                    row["severity"],
                    row["predicted_fault_type"],
                    row["predicted_condition"],
                    float(row["diagnostic_confidence"]),
                    row["known_health"],
                    row["known_fault_type"],
                    row["known_condition"],
                    float(row["feed_setting_percent"]),
                    int(row["health_correct"]),
                    int(row["fault_type_correct"]),
                    int(row["direction_correct"]),
                )
            )

        conn.commit()

        # ----------------------------------------------------
        # Verify
        # ----------------------------------------------------

        result = pd.read_sql_query(
            """
            SELECT
                scenario_id,
                anomaly_score_100,
                health_status,
                severity,
                predicted_fault_type,
                predicted_condition
            FROM anomaly_results
            ORDER BY feed_setting_percent
            """,
            conn
        )

        print("\nStored results:")
        print(result.to_string(index=False))

        print(
            f"\nRows in anomaly_results: {len(result)}"
        )

    finally:
        conn.close()

    print("\n" + "=" * 80)
    print("ANOMALY RESULTS STORED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()