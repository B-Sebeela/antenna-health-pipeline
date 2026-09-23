"""
demo.py
-------

End-to-end demonstration of the Antenna Health Monitoring Pipeline.

Pipeline:

    FEKO .ffe
        |
        v
    1. INGEST
        |
        v
    Parsed FEKO record
        |
        +-------------------+
        |                   |
        v                   v
    2. SQLite           2. Parquet
    metadata            feature data
        |                   |
        |                   v
        |              3. Visualisation
        |                   |
        +---------+---------+
                  |
                  v
            4. DB Summary

This script is intended as the supervisor-facing demonstration of the
first working vertical slice of the project.
"""

from pathlib import Path
import sys
import traceback

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from src.ingest.ffe_parser import parse_ffe

from src.storage.database import (
    initialise_database,
    store_scenario,
    link_parquet_file,
    start_pipeline_run,
    complete_pipeline_run,
    fail_pipeline_run,
    print_database_summary,
)

from src.storage.parquet import (
    save_record_to_parquet,
)

from src.preprocess.visualisation import (
    plot_radiation_pattern,
    plot_far_field_heatmap,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

FFE_FILE = PROJECT_ROOT / "data" / "raw" / "farfield_1.ffe"

DATABASE_PATH = PROJECT_ROOT / "data" / "antenna_health.db"

PARQUET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demo_scenario.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

RADIATION_PATTERN_PATH = (
    OUTPUT_DIR / "demo_radiation_pattern.png"
)

HEATMAP_PATH = (
    OUTPUT_DIR / "demo_far_field_heatmap.png"
)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_header():
    """Print the main demo header."""

    print()
    print("=" * 64)
    print("       ANTENNA HEALTH MONITORING PIPELINE")
    print("                    END-TO-END DEMO")
    print("=" * 64)
    print()


def print_section(number: int, title: str):
    """Print a formatted pipeline stage heading."""

    print()
    print(f"[{number}/4] {title}")
    print("-" * 64)


def print_success(message: str):
    """Print a successful operation."""

    print(f"✓ {message}")


def print_info(label: str, value):
    """Print a formatted metadata field."""

    print(f"  {label:<20}: {value}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_demo() -> int:
    """
    Execute the complete antenna health monitoring demonstration.

    Returns
    -------
    int
        0 for success, 1 for failure.
    """

    print_header()

    # Ensure output directories exist.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = None
    run_id = None

    try:

        # ================================================================
        # 1. INGEST
        # ================================================================

        print_section(1, "INGEST")

        if not FFE_FILE.exists():
            raise FileNotFoundError(
                f"FEKO file not found:\n{FFE_FILE}"
            )

        print("Parsing FEKO simulation...")
        print(f"  Source: {FFE_FILE}")

        # IMPORTANT:
        # This is a nominal scenario, so we do not provide
        # failed_elements or n_elements here.
        record = parse_ffe(
            str(FFE_FILE)
        )

        print_success("FEKO file parsed successfully")
        print()

        print_info(
            "Scenario",
            record["scenario_id"]
        )

        print_info(
            "Frequency",
            f"{record['frequency_hz'] / 1e6:.0f} MHz"
        )

        print_info(
            "Coordinate system",
            record["coordinate_system"]
        )

        print_info(
            "Result type",
            record["result_type"]
        )

        print_info(
            "Theta samples",
            record["n_theta"]
        )

        print_info(
            "Phi samples",
            record["n_phi"]
        )

        print_info(
            "Total samples",
            record["n_samples"]
        )

        if record.get("parse_warnings"):
            print()
            print("  Parser warnings:")
            for warning in record["parse_warnings"]:
                print(f"    - {warning}")
        else:
            print_success("No parser warnings")

        # ================================================================
        # 2. STORAGE
        # ================================================================

        print_section(2, "STORAGE")

        print("Initialising SQLite database...")

        conn = initialise_database(
            DATABASE_PATH
        )

        print_success(
            f"Database initialised: {DATABASE_PATH}"
        )

        # Start an ingest pipeline run.
        run_id = start_pipeline_run(
            conn=conn,
            stage="ingest",
            scenarios_in=1,
            notes="Supervisor end-to-end demonstration",
        )

        print_success(
            f"Pipeline run started: #{run_id}"
        )

        # Store scenario metadata.
        print()
        print("Storing scenario metadata...")

        inserted = store_scenario(
            conn=conn,
            record=record,
            parquet_path=None,
        )

        if inserted:
            print_success(
                f"Scenario '{record['scenario_id']}' stored in SQLite"
            )
        else:
            print(
                f"Scenario '{record['scenario_id']}' already exists"
            )

        # ------------------------------------------------------------
        # Parquet
        # ------------------------------------------------------------

        print()
        print("Writing Parquet feature data...")

        saved_path = save_record_to_parquet(
            record=record,
            output_path=PARQUET_PATH,
        )

        print_success(
            f"Parquet file created: {saved_path}"
        )

        # Link the Parquet file back to the SQLite scenario.
        link_parquet_file(
            conn=conn,
            scenario_id=record["scenario_id"],
            parquet_path=str(
                Path(saved_path).relative_to(PROJECT_ROOT)
            ),
        )

        print_success(
            "SQLite scenario linked to Parquet dataset"
        )

        # Complete ingest run.
        complete_pipeline_run(
            conn=conn,
            run_id=run_id,
            scenarios_out=1,
        )

        print_success(
            "Ingestion/storage pipeline completed"
        )

        # ================================================================
        # 3. VISUALISATION
        # ================================================================

        print_section(3, "VISUALISATION")

        print("Generating radiation pattern...")

        plot_radiation_pattern(
            parquet_path=PARQUET_PATH,
            phi_angle=0.0,
            output_path=RADIATION_PATTERN_PATH,
            show=False,
        )

        print_success(
            f"Radiation pattern saved: {RADIATION_PATTERN_PATH}"
        )

        print()
        print("Generating far-field heatmap...")

        plot_far_field_heatmap(
            parquet_path=PARQUET_PATH,
            output_path=HEATMAP_PATH,
            show=False,
        )

        print_success(
            f"Far-field heatmap saved: {HEATMAP_PATH}"
        )

        # ================================================================
        # 4. SUMMARY
        # ================================================================

        print_section(4, "PIPELINE SUMMARY")

        print_database_summary(
            conn
        )

        # ------------------------------------------------------------
        # Output files
        # ------------------------------------------------------------

        print()
        print("Generated outputs:")
        print(
            f"  • {PARQUET_PATH.relative_to(PROJECT_ROOT)}"
        )
        print(
            f"  • {RADIATION_PATTERN_PATH.relative_to(PROJECT_ROOT)}"
        )
        print(
            f"  • {HEATMAP_PATH.relative_to(PROJECT_ROOT)}"
        )

        # ------------------------------------------------------------
        # Final message
        # ------------------------------------------------------------

        print()
        print("=" * 64)
        print("                 DEMO COMPLETE ✓")
        print("=" * 64)
        print()

        return 0

    except Exception as exc:

        print()
        print("=" * 64)
        print("                 DEMO FAILED ✗")
        print("=" * 64)

        print()
        print(f"Error: {exc}")

        # If a pipeline run was started, mark it as failed.
        if conn is not None and run_id is not None:
            try:
                fail_pipeline_run(
                    conn=conn,
                    run_id=run_id,
                    error_message=str(exc),
                )
            except Exception:
                pass

        print()
        print("Traceback:")
        traceback.print_exc()

        return 1

    finally:

        if conn is not None:
            conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(
        run_demo()
    )
