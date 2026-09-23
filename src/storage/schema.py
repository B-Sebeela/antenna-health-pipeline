"""
schema.py
---------

SQLite database schema for the Antenna Health Monitoring Pipeline.

The database stores metadata and relationships between:

    FEKO scenario
        |
        +---- Far-field .ffe
        |          |
        |          v
        |    farfield.parquet
        |
        +---- S-parameter .sNp
                   |
                   v
             sparameters.parquet

SQLite stores metadata and links to the Parquet datasets.

Parquet stores the larger numerical datasets.

Main tables
-----------
scenarios
    One row per antenna simulation / measurement scenario.

pipeline_runs
    One row per execution of a pipeline stage.

anomaly_scores
    Model predictions and health scores for scenarios.

Design principle
----------------
Do not store large antenna measurement arrays directly in SQLite.
Instead, SQLite stores metadata and paths to the corresponding
Parquet files.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ===========================================================================
# DATABASE SCHEMA
# ===========================================================================

_DDL = """
-- =========================================================================
-- scenarios
-- =========================================================================
-- One row per antenna simulation / measurement scenario.
--
-- Large numerical datasets are stored in Parquet.
-- SQLite stores metadata and paths to those datasets.
-- =========================================================================

CREATE TABLE IF NOT EXISTS scenarios (

    scenario_id         TEXT PRIMARY KEY,

    -- ---------------------------------------------------------------------
    -- Source provenance
    -- ---------------------------------------------------------------------

    source_file         TEXT NOT NULL,
    ingested_at         TEXT NOT NULL,

    -- ---------------------------------------------------------------------
    -- Far-field simulation metadata
    -- ---------------------------------------------------------------------

    frequency_hz        REAL,
    n_theta             INTEGER,
    n_phi               INTEGER,
    n_samples           INTEGER,

    coordinate_system   TEXT DEFAULT 'Spherical',
    result_type         TEXT,

    -- ---------------------------------------------------------------------
    -- Array / failure configuration
    -- ---------------------------------------------------------------------

    n_elements          INTEGER,
    n_failures          INTEGER,

    failed_elements     TEXT,
    label_vector        TEXT,

    -- ---------------------------------------------------------------------
    -- Far-field Parquet dataset
    -- ---------------------------------------------------------------------

    parquet_path        TEXT,

    -- ---------------------------------------------------------------------
    -- S-parameter metadata
    -- ---------------------------------------------------------------------

    sparameter_path     TEXT,
    sparameter_num_ports INTEGER,
    sparameter_num_frequency_points INTEGER,
    sparameter_reference_impedance_ohm REAL,
    sparameter_data_format TEXT,

    -- ---------------------------------------------------------------------
    -- Data quality
    -- ---------------------------------------------------------------------

    parse_warnings      TEXT DEFAULT '[]',
    is_valid             INTEGER DEFAULT 1
);


-- =========================================================================
-- pipeline_runs
-- =========================================================================
-- One row per execution of a pipeline stage.
-- =========================================================================

CREATE TABLE IF NOT EXISTS pipeline_runs (

    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,

    stage           TEXT NOT NULL,

    status          TEXT NOT NULL,

    started_at      TEXT NOT NULL,

    finished_at     TEXT,

    scenarios_in    INTEGER DEFAULT 0,

    scenarios_out   INTEGER DEFAULT 0,

    error_message   TEXT,

    notes           TEXT
);


-- =========================================================================
-- anomaly_scores
-- =========================================================================
-- Stores anomaly detection, classification and element health results.
-- =========================================================================

CREATE TABLE IF NOT EXISTS anomaly_scores (

    score_id            INTEGER PRIMARY KEY AUTOINCREMENT,

    run_id              INTEGER NOT NULL
                        REFERENCES pipeline_runs(run_id),

    scenario_id         TEXT NOT NULL
                        REFERENCES scenarios(scenario_id),

    scored_at           TEXT NOT NULL,

    -- ---------------------------------------------------------------------
    -- Anomaly detection
    -- ---------------------------------------------------------------------

    anomaly_score       REAL,

    is_anomaly          INTEGER,

    detector_name       TEXT,

    -- ---------------------------------------------------------------------
    -- Failure classification
    -- ---------------------------------------------------------------------

    predicted_failures  TEXT,

    prediction_label    TEXT,

    classifier_name     TEXT,

    confidence          REAL,

    -- ---------------------------------------------------------------------
    -- Per-element health
    -- ---------------------------------------------------------------------

    element_health      TEXT,

    -- ---------------------------------------------------------------------
    -- Ground truth
    -- ---------------------------------------------------------------------

    true_failures       TEXT,

    UNIQUE(run_id, scenario_id, detector_name)
);


-- =========================================================================
-- INDEXES
-- =========================================================================

CREATE INDEX IF NOT EXISTS idx_scenarios_n_failures
    ON scenarios(n_failures);

CREATE INDEX IF NOT EXISTS idx_scenarios_ingested_at
    ON scenarios(ingested_at);

CREATE INDEX IF NOT EXISTS idx_scenarios_sparameter_path
    ON scenarios(sparameter_path);

CREATE INDEX IF NOT EXISTS idx_anomaly_scores_scenario
    ON anomaly_scores(scenario_id);

CREATE INDEX IF NOT EXISTS idx_anomaly_scores_run
    ON anomaly_scores(run_id);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_stage
    ON pipeline_runs(stage, status);
"""


# ===========================================================================
# CONNECTION MANAGEMENT
# ===========================================================================

def get_db(
    db_path: str | Path = "data/antenna_health.db",
) -> sqlite3.Connection:
    """
    Return a SQLite connection.

    The database file is created automatically if it does not exist.

    Parameters
    ----------
    db_path : str | Path
        Location of the SQLite database.

    Returns
    -------
    sqlite3.Connection
        SQLite connection with dictionary-like rows.
    """

    db_path = Path(db_path)

    # Create data directory automatically.
    db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    conn = sqlite3.connect(
        db_path,
        detect_types=sqlite3.PARSE_DECLTYPES,
    )

    conn.row_factory = sqlite3.Row

    # Foreign-key enforcement.
    conn.execute("PRAGMA foreign_keys = ON")

    # WAL improves concurrent read performance.
    conn.execute("PRAGMA journal_mode = WAL")

    # Good balance between performance and durability.
    conn.execute("PRAGMA synchronous = NORMAL")

    logger.debug(
        "Connected to database: %s",
        db_path.resolve(),
    )

    return conn


def init_db(
    conn: sqlite3.Connection,
) -> None:
    """
    Create all database tables and indexes.

    Safe to call repeatedly.
    """

    conn.executescript(_DDL)

    conn.commit()

    logger.info(
        "Database schema initialised (or already up to date)."
    )


# ===========================================================================
# SCENARIO TABLE
# ===========================================================================

def insert_scenario(
    conn: sqlite3.Connection,
    record: dict,
    parquet_path: Optional[str] = None,
    overwrite: bool = False,
) -> bool:
    """
    Insert one antenna scenario into the database.

    The record can originate from the existing FFE parser.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active database connection.

    record : dict
        Parsed FFE record.

    parquet_path : str | None
        Path to the far-field Parquet dataset.

    overwrite : bool
        Replace an existing scenario if True.

    Returns
    -------
    bool
        True if inserted.
        False if scenario already exists and overwrite=False.
    """

    sid = record["scenario_id"]

    if not overwrite:

        existing = conn.execute(
            """
            SELECT 1
            FROM scenarios
            WHERE scenario_id = ?
            """,
            (sid,),
        ).fetchone()

        if existing:

            logger.debug(
                "Scenario '%s' already exists — skipping.",
                sid,
            )

            return False

    failed_elements = record.get(
        "failed_elements"
    )

    label_vector = record.get(
        "label_vector"
    )

    warnings = record.get(
        "parse_warnings",
        [],
    )

    # A scenario with parser warnings is considered invalid.
    is_valid = 0 if warnings else 1

    conn.execute(
        """
        INSERT OR REPLACE INTO scenarios (

            scenario_id,
            source_file,
            ingested_at,

            frequency_hz,
            n_theta,
            n_phi,
            n_samples,

            coordinate_system,
            result_type,

            n_elements,
            n_failures,
            failed_elements,
            label_vector,

            parquet_path,

            sparameter_path,
            sparameter_num_ports,
            sparameter_num_frequency_points,
            sparameter_reference_impedance_ohm,
            sparameter_data_format,

            parse_warnings,
            is_valid

        )
        VALUES (

            :scenario_id,
            :source_file,
            :ingested_at,

            :frequency_hz,
            :n_theta,
            :n_phi,
            :n_samples,

            :coordinate_system,
            :result_type,

            :n_elements,
            :n_failures,
            :failed_elements,
            :label_vector,

            :parquet_path,

            :sparameter_path,
            :sparameter_num_ports,
            :sparameter_num_frequency_points,
            :sparameter_reference_impedance_ohm,
            :sparameter_data_format,

            :parse_warnings,
            :is_valid
        )
        """,
        {
            "scenario_id": sid,

            "source_file": record.get(
                "source_file",
                "",
            ),

            "ingested_at": record.get(
                "ingested_at",
                datetime.now(timezone.utc).isoformat(),
            ),

            "frequency_hz": record.get(
                "frequency_hz"
            ),

            "n_theta": record.get(
                "n_theta"
            ),

            "n_phi": record.get(
                "n_phi"
            ),

            "n_samples": record.get(
                "n_samples"
            ),

            "coordinate_system": record.get(
                "coordinate_system",
                "Spherical",
            ),

            "result_type": record.get(
                "result_type"
            ),

            "n_elements": record.get(
                "n_elements"
            ),

            "n_failures": record.get(
                "n_failures"
            ),

            "failed_elements": (
                json.dumps(failed_elements)
                if failed_elements is not None
                else None
            ),

            "label_vector": (
                json.dumps(label_vector)
                if label_vector is not None
                else None
            ),

            "parquet_path": parquet_path,

            # S-parameter fields are initially empty.
            "sparameter_path": record.get(
                "sparameter_path"
            ),

            "sparameter_num_ports": record.get(
                "sparameter_num_ports"
            ),

            "sparameter_num_frequency_points": record.get(
                "sparameter_num_frequency_points"
            ),

            "sparameter_reference_impedance_ohm": record.get(
                "sparameter_reference_impedance_ohm"
            ),

            "sparameter_data_format": record.get(
                "sparameter_data_format"
            ),

            "parse_warnings": json.dumps(
                warnings
            ),

            "is_valid": is_valid,
        },
    )

    conn.commit()

    logger.info(
        "Inserted scenario '%s' (valid=%d).",
        sid,
        is_valid,
    )

    return True


# ===========================================================================
# FAR-FIELD PARQUET LINK
# ===========================================================================

def update_parquet_path(
    conn: sqlite3.Connection,
    scenario_id: str,
    parquet_path: str,
) -> None:
    """
    Associate a scenario with its far-field Parquet file.
    """

    conn.execute(
        """
        UPDATE scenarios
        SET parquet_path = ?
        WHERE scenario_id = ?
        """,
        (
            parquet_path,
            scenario_id,
        ),
    )

    conn.commit()

    logger.debug(
        "Updated far-field Parquet path for scenario '%s'.",
        scenario_id,
    )


# ===========================================================================
# S-PARAMETER PARQUET LINK
# ===========================================================================

def update_sparameter_path(
    conn: sqlite3.Connection,
    scenario_id: str,
    sparameter_path: str,
) -> None:
    """
    Associate a scenario with its S-parameter Parquet file.
    """

    conn.execute(
        """
        UPDATE scenarios
        SET sparameter_path = ?
        WHERE scenario_id = ?
        """,
        (
            sparameter_path,
            scenario_id,
        ),
    )

    conn.commit()

    logger.debug(
        "Updated S-parameter Parquet path for scenario '%s'.",
        scenario_id,
    )


def update_sparameter_metadata(
    conn: sqlite3.Connection,
    scenario_id: str,
    num_ports: int,
    num_frequency_points: int,
    reference_impedance_ohm: float,
    data_format: str,
) -> None:
    """
    Store S-parameter dataset metadata for a scenario.
    """

    conn.execute(
        """
        UPDATE scenarios

        SET
            sparameter_num_ports = ?,
            sparameter_num_frequency_points = ?,
            sparameter_reference_impedance_ohm = ?,
            sparameter_data_format = ?

        WHERE scenario_id = ?
        """,
        (
            num_ports,
            num_frequency_points,
            reference_impedance_ohm,
            data_format,
            scenario_id,
        ),
    )

    conn.commit()

    logger.debug(
        "Updated S-parameter metadata for scenario '%s'.",
        scenario_id,
    )


# ===========================================================================
# SCENARIO RETRIEVAL
# ===========================================================================

def get_scenario(
    conn: sqlite3.Connection,
    scenario_id: str,
) -> Optional[dict]:
    """
    Fetch one scenario by ID.

    JSON columns are automatically converted back into Python lists.
    """

    row = conn.execute(
        """
        SELECT *
        FROM scenarios
        WHERE scenario_id = ?
        """,
        (scenario_id,),
    ).fetchone()

    if row is None:
        return None

    result = dict(row)

    for column in (
        "failed_elements",
        "label_vector",
        "parse_warnings",
    ):

        if result.get(column) is not None:
            result[column] = json.loads(
                result[column]
            )

    return result


def get_all_scenarios(
    conn: sqlite3.Connection,
    valid_only: bool = True,
    n_failures: Optional[int] = None,
) -> list[dict]:
    """
    Fetch all stored scenarios.
    """

    query = """
        SELECT *
        FROM scenarios
        WHERE 1 = 1
    """

    params: list = []

    if valid_only:

        query += """
            AND is_valid = 1
        """

    if n_failures is not None:

        query += """
            AND n_failures = ?
        """

        params.append(
            n_failures
        )

    query += """
        ORDER BY ingested_at ASC
    """

    rows = conn.execute(
        query,
        params,
    ).fetchall()

    results = []

    for row in rows:

        result = dict(row)

        for column in (
            "failed_elements",
            "label_vector",
            "parse_warnings",
        ):

            if result.get(column) is not None:

                result[column] = json.loads(
                    result[column]
                )

        results.append(result)

    return results


# ===========================================================================
# PIPELINE RUNS
# ===========================================================================

def log_run(
    conn: sqlite3.Connection,
    stage: str,
    status: str = "started",
    scenarios_in: int = 0,
    scenarios_out: int = 0,
    error_message: Optional[str] = None,
    notes: Optional[str] = None,
) -> int:
    """
    Insert a pipeline run record.
    """

    now = datetime.now(
        timezone.utc
    ).isoformat()

    finished_at = (
        now
        if status in ("ok", "error")
        else None
    )

    cursor = conn.execute(
        """
        INSERT INTO pipeline_runs (

            stage,
            status,
            started_at,
            finished_at,
            scenarios_in,
            scenarios_out,
            error_message,
            notes

        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            stage,
            status,
            now,
            finished_at,
            scenarios_in,
            scenarios_out,
            error_message,
            notes,
        ),
    )

    conn.commit()

    run_id = cursor.lastrowid

    logger.info(
        "Logged run #%d | stage=%s | status=%s",
        run_id,
        stage,
        status,
    )

    return run_id


def update_run(
    conn: sqlite3.Connection,
    run_id: int,
    status: str,
    scenarios_out: Optional[int] = None,
    error_message: Optional[str] = None,
) -> None:
    """
    Update a previously started pipeline run.
    """

    finished_at = datetime.now(
        timezone.utc
    ).isoformat()

    fields = [
        "status = ?",
        "finished_at = ?",
    ]

    params: list = [
        status,
        finished_at,
    ]

    if scenarios_out is not None:

        fields.append(
            "scenarios_out = ?"
        )

        params.append(
            scenarios_out
        )

    if error_message is not None:

        fields.append(
            "error_message = ?"
        )

        params.append(
            error_message
        )

    params.append(
        run_id
    )

    conn.execute(
        f"""
        UPDATE pipeline_runs
        SET {", ".join(fields)}
        WHERE run_id = ?
        """,
        params,
    )

    conn.commit()

    logger.info(
        "Updated run #%d → status=%s",
        run_id,
        status,
    )


def get_recent_runs(
    conn: sqlite3.Connection,
    stage: Optional[str] = None,
    limit: int = 20,
) -> list[dict]:
    """
    Retrieve recent pipeline runs.
    """

    query = """
        SELECT *
        FROM pipeline_runs
    """

    params: list = []

    if stage:

        query += """
            WHERE stage = ?
        """

        params.append(
            stage
        )

    query += """
        ORDER BY run_id DESC
        LIMIT ?
    """

    params.append(
        limit
    )

    rows = conn.execute(
        query,
        params,
    ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


# ===========================================================================
# ANOMALY SCORES
# ===========================================================================

def insert_anomaly_score(
    conn: sqlite3.Connection,
    run_id: int,
    scenario_id: str,
    anomaly_score: Optional[float] = None,
    is_anomaly: Optional[int] = None,
    detector_name: Optional[str] = None,
    predicted_failures: Optional[list[int]] = None,
    prediction_label: Optional[list[int]] = None,
    classifier_name: Optional[str] = None,
    confidence: Optional[float] = None,
    element_health: Optional[list[float]] = None,
    true_failures: Optional[list[int]] = None,
) -> int:
    """
    Insert one anomaly / health score record.
    """

    now = datetime.now(
        timezone.utc
    ).isoformat()

    cursor = conn.execute(
        """
        INSERT OR REPLACE INTO anomaly_scores (

            run_id,
            scenario_id,
            scored_at,

            anomaly_score,
            is_anomaly,
            detector_name,

            predicted_failures,
            prediction_label,

            classifier_name,
            confidence,

            element_health,

            true_failures

        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            scenario_id,
            now,

            anomaly_score,
            is_anomaly,
            detector_name,

            (
                json.dumps(predicted_failures)
                if predicted_failures is not None
                else None
            ),

            (
                json.dumps(prediction_label)
                if prediction_label is not None
                else None
            ),

            classifier_name,
            confidence,

            (
                json.dumps(element_health)
                if element_health is not None
                else None
            ),

            (
                json.dumps(true_failures)
                if true_failures is not None
                else None
            ),
        ),
    )

    conn.commit()

    return cursor.lastrowid


def get_scores_for_run(
    conn: sqlite3.Connection,
    run_id: int,
) -> list[dict]:
    """
    Retrieve all anomaly scores for a pipeline run.
    """

    rows = conn.execute(
        """
        SELECT *
        FROM anomaly_scores
        WHERE run_id = ?
        ORDER BY score_id
        """,
        (run_id,),
    ).fetchall()

    results = []

    for row in rows:

        result = dict(row)

        for column in (
            "predicted_failures",
            "prediction_label",
            "element_health",
            "true_failures",
        ):

            if result.get(column) is not None:

                result[column] = json.loads(
                    result[column]
                )

        results.append(result)

    return results


def get_scores_for_scenario(
    conn: sqlite3.Connection,
    scenario_id: str,
) -> list[dict]:
    """
    Retrieve all anomaly scores for a scenario.
    """

    rows = conn.execute(
        """
        SELECT
            s.*,
            r.stage,
            r.started_at AS run_started_at

        FROM anomaly_scores s

        JOIN pipeline_runs r
            ON s.run_id = r.run_id

        WHERE s.scenario_id = ?

        ORDER BY s.scored_at DESC
        """,
        (scenario_id,),
    ).fetchall()

    results = []

    for row in rows:

        result = dict(row)

        for column in (
            "predicted_failures",
            "prediction_label",
            "element_health",
            "true_failures",
        ):

            if result.get(column) is not None:

                result[column] = json.loads(
                    result[column]
                )

        results.append(result)

    return results


# ===========================================================================
# DATABASE SUMMARY
# ===========================================================================

def db_summary(
    conn: sqlite3.Connection,
) -> dict:
    """
    Return a high-level summary of database contents.
    """

    n_scenarios = conn.execute(
        """
        SELECT COUNT(*)
        FROM scenarios
        """
    ).fetchone()[0]

    n_valid = conn.execute(
        """
        SELECT COUNT(*)
        FROM scenarios
        WHERE is_valid = 1
        """
    ).fetchone()[0]

    n_runs = conn.execute(
        """
        SELECT COUNT(*)
        FROM pipeline_runs
        """
    ).fetchone()[0]

    n_scores = conn.execute(
        """
        SELECT COUNT(*)
        FROM anomaly_scores
        """
    ).fetchone()[0]

    n_sparameter_scenarios = conn.execute(
        """
        SELECT COUNT(*)
        FROM scenarios
        WHERE sparameter_path IS NOT NULL
        """
    ).fetchone()[0]

    last_ingest_row = conn.execute(
        """
        SELECT MAX(ingested_at)
        FROM scenarios
        """
    ).fetchone()

    last_ingest = (
        last_ingest_row[0]
        if last_ingest_row
        else None
    )

    failure_rows = conn.execute(
        """
        SELECT
            n_failures,
            COUNT(*) AS cnt

        FROM scenarios

        WHERE
            is_valid = 1
            AND n_failures IS NOT NULL

        GROUP BY n_failures

        ORDER BY n_failures
        """
    ).fetchall()

    failure_counts = {
        row[0]: row[1]
        for row in failure_rows
    }

    summary = {

        "n_scenarios":
            n_scenarios,

        "n_valid_scenarios":
            n_valid,

        "n_runs":
            n_runs,

        "n_scores":
            n_scores,

        "n_sparameter_scenarios":
            n_sparameter_scenarios,

        "last_ingest":
            last_ingest,

        "failure_counts":
            failure_counts,
    }

    logger.info(
        "DB summary: %d scenarios (%d valid) | "
        "%d runs | %d scores | %d S-parameter datasets",
        n_scenarios,
        n_valid,
        n_runs,
        n_scores,
        n_sparameter_scenarios,
    )

    return summary