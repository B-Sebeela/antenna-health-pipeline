"""
database.py
-----------

High-level database service for the Antenna Health Monitoring Pipeline.

This module provides a clean interface between the pipeline stages and
the SQLite schema defined in schema.py.

Responsibilities
----------------
- Initialise the SQLite database.
- Store parsed FEKO scenario metadata.
- Link far-field Parquet files.
- Link S-parameter Parquet files.
- Store S-parameter metadata.
- Record pipeline execution status.
- Retrieve scenario information.
- Produce database summaries.

Architecture
------------

                    FEKO
                     |
          +----------+----------+
          |                     |
          v                     v
       .ffe                    .s7p
          |                     |
          v                     v
    ffe_parser.py        sparam_parser.py
          |                     |
          v                     v
    farfield.parquet     sparameters.parquet
          |                     |
          +----------+----------+
                     |
                     v
               SQLite database
              antenna_health.db
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from src.storage.schema import (
    db_summary,
    get_all_scenarios,
    get_db,
    get_recent_runs,
    get_scenario,
    init_db,
    insert_scenario,
    insert_anomaly_score,
    log_run,
    update_parquet_path,
    update_sparameter_path,
    update_sparameter_metadata,
    update_run,
    get_scores_for_run,
    get_scores_for_scenario,
)


# ===========================================================================
# DATABASE INITIALISATION
# ===========================================================================

def initialise_database(
    db_path: str | Path = "data/antenna_health.db",
) -> sqlite3.Connection:
    """
    Create and initialise the project's SQLite database.

    The database file will be created automatically if it does not exist.

    Parameters
    ----------
    db_path : str | Path
        Location of the SQLite database.

    Returns
    -------
    sqlite3.Connection
        Initialised SQLite connection.
    """

    conn = get_db(
        db_path
    )

    init_db(
        conn
    )

    return conn


# ===========================================================================
# SCENARIO STORAGE
# ===========================================================================

def store_scenario(
    conn: sqlite3.Connection,
    record: dict,
    parquet_path: str | Path | None = None,
    overwrite: bool = False,
) -> bool:
    """
    Store a parsed FEKO scenario in SQLite.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active database connection.

    record : dict
        Parsed record returned by ffe_parser.parse_ffe().

    parquet_path : str | Path | None
        Path to the far-field Parquet file.

    overwrite : bool
        Whether to replace an existing scenario.

    Returns
    -------
    bool
        True if inserted.
        False if already existed and overwrite=False.
    """

    if parquet_path is not None:

        parquet_path = str(
            parquet_path
        )

    return insert_scenario(
        conn=conn,
        record=record,
        parquet_path=parquet_path,
        overwrite=overwrite,
    )


# ===========================================================================
# FAR-FIELD PARQUET
# ===========================================================================

def link_parquet_file(
    conn: sqlite3.Connection,
    scenario_id: str,
    parquet_path: str | Path,
) -> None:
    """
    Associate a scenario with its far-field Parquet file.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active database connection.

    scenario_id : str
        Scenario identifier.

    parquet_path : str | Path
        Relative path to the far-field Parquet file.
    """

    update_parquet_path(
        conn=conn,
        scenario_id=scenario_id,
        parquet_path=str(
            parquet_path
        ),
    )


# ===========================================================================
# S-PARAMETER PARQUET
# ===========================================================================

def link_sparameter_file(
    conn: sqlite3.Connection,
    scenario_id: str,
    sparameter_path: str | Path,
) -> None:
    """
    Associate a scenario with its S-parameter Parquet file.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active database connection.

    scenario_id : str
        Scenario identifier.

    sparameter_path : str | Path
        Relative path to the S-parameter Parquet file.
    """

    update_sparameter_path(
        conn=conn,
        scenario_id=scenario_id,
        sparameter_path=str(
            sparameter_path
        ),
    )


def store_sparameter_metadata(
    conn: sqlite3.Connection,
    scenario_id: str,
    num_ports: int,
    num_frequency_points: int,
    reference_impedance_ohm: float,
    data_format: str,
) -> None:
    """
    Store metadata associated with an S-parameter dataset.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active database connection.

    scenario_id : str
        Scenario identifier.

    num_ports : int
        Number of antenna ports.

    num_frequency_points : int
        Number of frequency samples.

    reference_impedance_ohm : float
        Touchstone reference impedance.

    data_format : str
        Touchstone data format:
            MA
            RI
            DB
    """

    update_sparameter_metadata(
        conn=conn,
        scenario_id=scenario_id,
        num_ports=num_ports,
        num_frequency_points=num_frequency_points,
        reference_impedance_ohm=reference_impedance_ohm,
        data_format=data_format,
    )


# ===========================================================================
# SCENARIO RETRIEVAL
# ===========================================================================

def fetch_scenario(
    conn: sqlite3.Connection,
    scenario_id: str,
) -> Optional[dict]:
    """
    Retrieve a single scenario by ID.
    """

    return get_scenario(
        conn,
        scenario_id,
    )


def fetch_scenarios(
    conn: sqlite3.Connection,
    valid_only: bool = True,
    n_failures: int | None = None,
) -> list[dict]:
    """
    Retrieve stored scenarios.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active database connection.

    valid_only : bool
        If True, return only valid scenarios.

    n_failures : int | None
        Filter by exact number of failed elements.

    Returns
    -------
    list[dict]
        Scenario records.
    """

    return get_all_scenarios(
        conn=conn,
        valid_only=valid_only,
        n_failures=n_failures,
    )


# ===========================================================================
# PIPELINE RUN TRACKING
# ===========================================================================

def start_pipeline_run(
    conn: sqlite3.Connection,
    stage: str,
    scenarios_in: int = 0,
    notes: str | None = None,
) -> int:
    """
    Start a new pipeline run.

    The initial status is 'started'.
    """

    return log_run(
        conn=conn,
        stage=stage,
        status="started",
        scenarios_in=scenarios_in,
        notes=notes,
    )


def complete_pipeline_run(
    conn: sqlite3.Connection,
    run_id: int,
    scenarios_out: int,
) -> None:
    """
    Mark a pipeline run as successfully completed.
    """

    update_run(
        conn=conn,
        run_id=run_id,
        status="ok",
        scenarios_out=scenarios_out,
    )


def fail_pipeline_run(
    conn: sqlite3.Connection,
    run_id: int,
    error_message: str,
) -> None:
    """
    Mark a pipeline run as failed.
    """

    update_run(
        conn=conn,
        run_id=run_id,
        status="error",
        error_message=error_message,
    )


def fetch_recent_runs(
    conn: sqlite3.Connection,
    stage: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """
    Retrieve recent pipeline runs.
    """

    return get_recent_runs(
        conn=conn,
        stage=stage,
        limit=limit,
    )


# ===========================================================================
# ANOMALY SCORES
# ===========================================================================

def store_anomaly_score(
    conn: sqlite3.Connection,
    run_id: int,
    scenario_id: str,
    anomaly_score: float,
    is_anomaly: bool,
    detector_name: str,
    prediction_label: str,
) -> int:
    """
    Store an anomaly-detection result in SQLite.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active database connection.

    run_id : int
        ID of the pipeline run that produced this result.

    scenario_id : str
        Antenna scenario identifier.

    anomaly_score : float
        Calculated anomaly score.

    is_anomaly : bool
        True if the scenario was classified as anomalous.

    detector_name : str
        Name of the detector used.

    prediction_label : str
        Current health classification, e.g.
        'HEALTHY' or 'ANOMALOUS'.

    Returns
    -------
    int
        ID of the inserted anomaly-score record.
    """

    return insert_anomaly_score(
        conn=conn,
        run_id=run_id,
        scenario_id=scenario_id,
        anomaly_score=anomaly_score,
        is_anomaly=int(is_anomaly),
        detector_name=detector_name,
        prediction_label=prediction_label,
    )


def fetch_anomaly_scores_for_run(
    conn: sqlite3.Connection,
    run_id: int,
) -> list[dict]:
    """
    Retrieve all anomaly scores generated by a pipeline run.
    """

    return get_scores_for_run(
        conn=conn,
        run_id=run_id,
    )


def fetch_anomaly_scores_for_scenario(
    conn: sqlite3.Connection,
    scenario_id: str,
) -> list[dict]:
    """
    Retrieve all anomaly scores for a scenario.
    """

    return get_scores_for_scenario(
        conn=conn,
        scenario_id=scenario_id,
    )




# ===========================================================================
# DATABASE SUMMARY
# ===========================================================================

def get_database_summary(
    conn: sqlite3.Connection,
) -> dict:
    """
    Return a high-level summary of the database.

    Intended for:

        - demo.py
        - logging
        - Streamlit dashboard
        - pipeline monitoring
    """

    return db_summary(
        conn
    )


# ===========================================================================
# DEMO-FRIENDLY SUMMARY
# ===========================================================================

def print_database_summary(
    conn: sqlite3.Connection,
) -> None:
    """
    Print a human-readable database summary.

    Useful during development and supervisor demonstrations.
    """

    summary = get_database_summary(
        conn
    )

    print()
    print("=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)

    print(
        f"Scenarios:              "
        f"{summary['n_scenarios']}"
    )

    print(
        f"Valid scenarios:        "
        f"{summary['n_valid_scenarios']}"
    )

    print(
        f"Pipeline runs:           "
        f"{summary['n_runs']}"
    )

    print(
        f"Anomaly scores:          "
        f"{summary['n_scores']}"
    )

    print(
        f"S-parameter datasets:    "
        f"{summary['n_sparameter_scenarios']}"
    )

    print(
        f"Last ingest:             "
        f"{summary['last_ingest']}"
    )

    print(
        f"Failure counts:          "
        f"{summary['failure_counts']}"
    )

    print("=" * 60)