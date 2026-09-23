"""
ingest.py
--------

End-to-end ingestion service for the Antenna Health Monitoring Pipeline.

Supported input files
---------------------

1. FEKO far-field files

        .ffe
          |
          v
        ffe_parser.py
          |
          v
        farfield.parquet
          |
          v
        SQLite metadata


2. Touchstone S-parameter files

        .sNp
          |
          v
        sparam_parser.py
          |
          v
        sparameters.parquet
          |
          v
        SQLite metadata


Pipeline
--------

    data/raw/
        |
        +---- .ffe
        |       |
        |       v
        |   FFE parser
        |       |
        |       v
        |   Far-field Parquet
        |
        +---- .sNp
                |
                v
            S-parameter parser
                |
                v
            S-parameter Parquet

                    |
                    v

             SQLite database
             antenna_health.db


This module is responsible for orchestration only.

Parsing logic belongs to:
    ffe_parser.py
    sparam_parser.py

Storage logic belongs to:
    database.py
    parquet.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from datetime import datetime, timezone

from src.ingest.ffe_parser import parse_ffe

from src.ingest.sparam_parser import (
    parse_sparameter_file,
    validate_sparameter_dataframe,
)

from src.storage.database import (
    initialise_database,
    store_scenario,
    link_parquet_file,
    link_sparameter_file,
    store_sparameter_metadata,
    start_pipeline_run,
    complete_pipeline_run,
    fail_pipeline_run,
    print_database_summary,
)

from src.storage.parquet import (
    save_record_to_parquet,
    save_sparameter_to_parquet,
)


# ===========================================================================
# PROJECT PATHS
# ===========================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "antenna_health.db"
)


# ===========================================================================
# DISPLAY HELPERS
# ===========================================================================

def print_header() -> None:
    """Print the main ingestion header."""

    print()
    print("=" * 64)
    print("          ANTENNA HEALTH MONITORING PIPELINE")
    print("                    DATA INGESTION")
    print("=" * 64)
    print()


def print_section(title: str) -> None:
    """Print a formatted section heading."""

    print()
    print("-" * 64)
    print(title)
    print("-" * 64)


# ===========================================================================
# FFE INGESTION
# ===========================================================================

def ingest_ffe_file(
    conn,
    ffe_path: Path,
) -> bool:
    """
    Parse and store one FEKO .ffe file.

    Returns
    -------
    bool
        True if successful.
    """

    print()
    print(f"  Parsing FFE: {ffe_path.name}")

    try:

        # ---------------------------------------------------------------
        # Parse FEKO far-field file
        # ---------------------------------------------------------------

        record = parse_ffe(
            filepath=ffe_path,
        )

        scenario_id = record["scenario_id"]

        print(
            f"    Scenario      : {scenario_id}"
        )

        print(
            f"    Frequency     : "
            f"{record['frequency_hz'] / 1e6:.0f} MHz"
        )

        print(
            f"    Samples       : "
            f"{record['n_samples']}"
        )

        # ---------------------------------------------------------------
        # Save Parquet
        # ---------------------------------------------------------------

        parquet_path = (
            PROCESSED_DIR
            / f"{scenario_id}.parquet"
        )

        save_record_to_parquet(
            record=record,
            output_path=parquet_path,
        )

        # ---------------------------------------------------------------
        # Store scenario metadata
        # ---------------------------------------------------------------

        inserted = store_scenario(
            conn=conn,
            record=record,
            parquet_path=None,
        )

        if inserted:
            print(
                "    SQLite        : scenario inserted"
            )
        else:
            print(
                "    SQLite        : scenario already exists"
            )

        # ---------------------------------------------------------------
        # Link far-field Parquet
        # ---------------------------------------------------------------

        relative_parquet_path = parquet_path.relative_to(
            PROJECT_ROOT
        )

        link_parquet_file(
            conn=conn,
            scenario_id=scenario_id,
            parquet_path=str(relative_parquet_path),
        )

        print(
            f"    Parquet       : {relative_parquet_path}"
        )

        print(
            "    SQLite link   : created"
        )

        return True

    except Exception as exc:

        print(
            f"    ERROR         : {exc}"
        )

        traceback.print_exc()

        return False


# ===========================================================================
# S-PARAMETER INGESTION
# ===========================================================================

def ingest_sparameter_file(
    conn,
    sparameter_path: Path,
) -> bool:
    """
    Parse and store one Touchstone S-parameter file.

    The scenario ID is derived from the filename stem.

    Example
    -------

        H01_NOM.s7p

    becomes:

        scenario_id = H01_NOM

    Returns
    -------
    bool
        True if successful.
    """

    print()
    print(
        f"  Parsing S-parameter: "
        f"{sparameter_path.name}"
    )

    try:

        # ---------------------------------------------------------------
        # Parse Touchstone file
        # ---------------------------------------------------------------

        result = parse_sparameter_file(
            file_path=sparameter_path,
        )

        dataframe = result.dataframe

        # ---------------------------------------------------------------
        # Validate parser output
        # ---------------------------------------------------------------

        validate_sparameter_dataframe(
            dataframe=dataframe,
            expected_num_ports=result.num_ports,
        )

        scenario_id = sparameter_path.stem

        print(
            f"    Scenario      : {scenario_id}"
        )

        print(
            f"    Ports         : "
            f"{result.num_ports}"
        )

        print(
            f"    Frequencies   : "
            f"{result.num_frequency_points}"
        )

        print(
            f"    Data rows     : "
            f"{len(dataframe)}"
        )

        print(
            f"    Format        : "
            f"{result.header.data_format}"
        )

        print(
            f"    Reference Z0  : "
            f"{result.header.reference_impedance_ohm:.2f} Ω"
        )

        # ---------------------------------------------------------------
        # Save S-parameter Parquet
        # ---------------------------------------------------------------

        parquet_path = (
            PROCESSED_DIR
            / f"{scenario_id}_sparameters.parquet"
        )

        save_sparameter_to_parquet(
            dataframe=dataframe,
            output_path=parquet_path,
        )

        relative_parquet_path = parquet_path.relative_to(
            PROJECT_ROOT
        )

        print(
            f"    Parquet       : "
            f"{relative_parquet_path}"
        )

        # ---------------------------------------------------------------
        # Build minimal scenario record
        # ---------------------------------------------------------------

        #
        # An S-parameter file may not have an accompanying .ffe file.
        #
        # Therefore we create a minimal scenario record containing
        # the information required by the SQLite scenarios table.
        #

        scenario_record = {

            "scenario_id":
                scenario_id,

            "source_file":
                str(sparameter_path.resolve()),

            "frequency_hz":
                float(
                    dataframe["frequency_hz"].iloc[0]
                ),

            "ingested_at":
                datetime.now(timezone.utc).isoformat(),

            "n_theta":
                None,

            "n_phi":
                None,

            "n_samples":
                None,

            "coordinate_system":
                "N/A",

            "result_type":
                "S-parameter",

            "n_elements":
                result.num_ports,

            "n_failures":
                None,

            "failed_elements":
                None,

            "label_vector":
                None,

            "parse_warnings":
                [],
        }

        # ---------------------------------------------------------------
        # Store / update scenario metadata
        # ---------------------------------------------------------------

        inserted = store_scenario(
            conn=conn,
            record=scenario_record,
            parquet_path=None,
        )

        if inserted:
            print(
                "    SQLite        : scenario inserted"
            )
        else:
            print(
                "    SQLite        : scenario already exists"
            )

        # ---------------------------------------------------------------
        # Link S-parameter Parquet
        # ---------------------------------------------------------------

        link_sparameter_file(
            conn=conn,
            scenario_id=scenario_id,
            sparameter_path=str(
                relative_parquet_path
            ),
        )

        print(
            "    SQLite link   : created"
        )

        # ---------------------------------------------------------------
        # Store S-parameter metadata
        # ---------------------------------------------------------------

        store_sparameter_metadata(
            conn=conn,
            scenario_id=scenario_id,
            num_ports=result.num_ports,
            num_frequency_points=(
                result.num_frequency_points
            ),
            reference_impedance_ohm=(
                result.header.reference_impedance_ohm
            ),
            data_format=result.header.data_format,
        )

        print(
            "    Metadata      : stored"
        )

        return True

    except Exception as exc:

        print(
            f"    ERROR         : {exc}"
        )

        traceback.print_exc()

        return False


# ===========================================================================
# DIRECTORY INGESTION
# ===========================================================================

def ingest_directory() -> dict:
    """
    Ingest all supported antenna data files in data/raw/.

    Supported:

        .ffe
        .s1p
        .s2p
        .s3p
        ...
        .sNp

    Returns
    -------
    dict
        Ingestion summary.
    """

    print_header()

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------------
    # Discover files
    # -----------------------------------------------------------------------

    ffe_files = sorted(
        RAW_DIR.glob("*.ffe")
    )

    sparameter_files = sorted(
        [
            path
            for path in RAW_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower().startswith(".s")
            and path.suffix.lower().endswith("p")
        ]
    )

    total_files = (
        len(ffe_files)
        +
        len(sparameter_files)
    )

    print(
        f"Raw directory : "
        f"{RAW_DIR.relative_to(PROJECT_ROOT)}"
    )

    print(
        f"FFE files     : "
        f"{len(ffe_files)}"
    )

    print(
        f"S-parameter   : "
        f"{len(sparameter_files)}"
    )

    # -----------------------------------------------------------------------
    # Initialise database
    # -----------------------------------------------------------------------

    print()

    print(
        "Initialising SQLite database..."
    )

    conn = initialise_database(
        DATABASE_PATH
    )

    print(
        f"✓ Database ready: "
        f"{DATABASE_PATH.relative_to(PROJECT_ROOT)}"
    )

    run_id = start_pipeline_run(
        conn=conn,
        stage="ingest",
        scenarios_in=total_files,
        notes="FFE and Touchstone S-parameter ingestion",
    )

    print(
        f"✓ Pipeline run started: #{run_id}"
    )

    successful = 0
    failed = 0

    try:

        # ================================================================
        # FFE FILES
        # ================================================================

        if ffe_files:

            print_section(
                "FEKO FAR-FIELD INGESTION"
            )

            for ffe_path in ffe_files:

                if ingest_ffe_file(
                    conn=conn,
                    ffe_path=ffe_path,
                ):
                    successful += 1
                else:
                    failed += 1

        # ================================================================
        # S-PARAMETER FILES
        # ================================================================

        if sparameter_files:

            print_section(
                "TOUCHSTONE S-PARAMETER INGESTION"
            )

            for sparameter_path in sparameter_files:

                if ingest_sparameter_file(
                    conn=conn,
                    sparameter_path=sparameter_path,
                ):
                    successful += 1
                else:
                    failed += 1

        # ----------------------------------------------------------------
        # Complete pipeline run
        # ----------------------------------------------------------------

        complete_pipeline_run(
            conn=conn,
            run_id=run_id,
            scenarios_out=successful,
        )

        # ----------------------------------------------------------------
        # Summary
        # ----------------------------------------------------------------

        print()
        print("=" * 64)
        print("                    INGESTION SUMMARY")
        print("=" * 64)

        print(
            f"Files discovered : {total_files}"
        )

        print(
            f"Successful       : {successful}"
        )

        print(
            f"Skipped          : 0"
        )

        print(
            f"Failed           : {failed}"
        )

        print("=" * 64)

        print_database_summary(
            conn
        )

        return {
            "files_discovered":
                total_files,

            "successful":
                successful,

            "failed":
                failed,

            "run_id":
                run_id,
        }

    except Exception as exc:

        fail_pipeline_run(
            conn=conn,
            run_id=run_id,
            error_message=str(exc),
        )

        raise

    finally:

        conn.close()


# ===========================================================================
# COMMAND-LINE ENTRY POINT
# ===========================================================================

def main() -> int:
    """
    Command-line entry point.

    Returns
    -------
    int
        0 for success.
        1 for failure.
    """

    try:

        result = ingest_directory()

        if result["failed"] > 0:
            return 1

        return 0

    except Exception as exc:

        print()
        print("=" * 64)
        print("                 INGESTION FAILED ✗")
        print("=" * 64)

        print()
        print(
            f"Error: {exc}"
        )

        print(
            "Fatal ingestion error."
        )

        traceback.print_exc()

        return 1


if __name__ == "__main__":

    sys.exit(
        main()
    )
