"""
batch_ingest.py
---------------

Batch ingestion pipeline for FEKO far-field (.ffe) files.

Pipeline:

    data/raw/**/*.ffe
            |
            v
       ffe_parser.py
            |
            v
       validation
            |
            v
       parquet.py
            |
            v
       SQLite database
            |
            v
       pipeline run record

This module currently handles .ffe far-field data.

S-parameter (.s7p) ingestion is handled separately by
sparam_parser.py and will be integrated into the complete
scenario pipeline later.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.ingest.ffe_parser import parse_ffe
from src.storage.parquet import save_record_to_parquet
from src.storage.database import (
    initialise_database,
    store_scenario,
    start_pipeline_run,
    complete_pipeline_run,
    fail_pipeline_run,
)


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================================
# DEFAULT PATHS
# ============================================================================

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_DB_PATH = Path("data/antenna_health.db")


# ============================================================================
# DISCOVER FFE FILES
# ============================================================================

def discover_ffe_files(
    raw_dir: str | Path,
) -> list[Path]:
    """
    Recursively find all .ffe files inside the raw-data directory.

    Example:

        data/raw/
            H01_NOM/
                H01_NOM.ffe

            MF01_L_PLUS/
                MF01_L_PLUS.ffe

            MF02_L_MINUS/
                MF02_L_MINUS.ffe

            M01_FEED_OFFSET/
                M01_FEED_OFFSET.ffe

    Returns
    -------
    list[Path]
        Sorted list of discovered FFE files.
    """

    raw_dir = Path(raw_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory does not exist: {raw_dir}"
        )

    if not raw_dir.is_dir():
        raise NotADirectoryError(
            f"Raw data path is not a directory: {raw_dir}"
        )

    files = sorted(
        path
        for path in raw_dir.rglob("*.ffe")
        if path.is_file()
    )

    return files


# ============================================================================
# SCENARIO ID
# ============================================================================

def derive_scenario_id(
    ffe_path: Path,
    raw_dir: Path,
) -> str:
    """
    Determine the scenario ID from the FFE file location.

    If the file is inside a scenario folder:

        data/raw/H01_NOM/H01_NOM.ffe

    the scenario ID becomes:

        H01_NOM

    If the file is directly inside data/raw:

        data/raw/farfield_1.ffe

    the scenario ID becomes:

        farfield_1
    """

    ffe_path = ffe_path.resolve()
    raw_dir = raw_dir.resolve()

    relative_path = ffe_path.relative_to(
        raw_dir
    )

    # File directly inside data/raw/
    if len(relative_path.parts) == 1:
        return ffe_path.stem

    # File inside a scenario folder
    return relative_path.parts[-2]


# ============================================================================
# BASIC VALIDATION
# ============================================================================

def validate_record(
    record: dict,
) -> tuple[bool, list[str]]:
    """
    Perform basic validation on a parsed FFE record.

    Parser warnings are not treated as Python errors. They remain
    attached to the record so that the database can preserve them.
    """

    errors: list[str] = []

    required_fields = [
        "scenario_id",
        "source_file",
        "frequency_hz",
        "n_theta",
        "n_phi",
        "n_samples",
        "coordinate_system",
        "result_type",
        "theta_deg",
        "phi_deg",
        "re_etheta",
        "im_etheta",
        "re_ephi",
        "im_ephi",
        "gain_theta_dbi",
        "gain_phi_dbi",
        "gain_total_dbi",
        "magnitude_norm_db",
        "parse_warnings",
    ]

    # ------------------------------------------------------------------
    # Required fields
    # ------------------------------------------------------------------

    for field in required_fields:

        if field not in record:

            errors.append(
                f"Missing required field: {field}"
            )

    if errors:
        return False, errors

    # ------------------------------------------------------------------
    # Basic scalar checks
    # ------------------------------------------------------------------

    if not record["scenario_id"]:
        errors.append(
            "scenario_id is empty."
        )

    if record["frequency_hz"] <= 0:
        errors.append(
            "frequency_hz must be greater than zero."
        )

    if record["n_samples"] <= 0:
        errors.append(
            "n_samples must be greater than zero."
        )

    # ------------------------------------------------------------------
    # Check measurement arrays
    # ------------------------------------------------------------------

    array_fields = [
        "theta_deg",
        "phi_deg",
        "re_etheta",
        "im_etheta",
        "re_ephi",
        "im_ephi",
        "gain_theta_dbi",
        "gain_phi_dbi",
        "gain_total_dbi",
        "magnitude_norm_db",
    ]

    lengths = {}

    for field in array_fields:

        try:

            lengths[field] = len(
                record[field]
            )

        except TypeError:

            errors.append(
                f"{field} is not a valid sequence."
            )

    if lengths:

        unique_lengths = set(
            lengths.values()
        )

        if len(unique_lengths) != 1:

            errors.append(
                "Measurement arrays have inconsistent "
                f"lengths: {lengths}"
            )

        else:

            array_length = next(
                iter(unique_lengths)
            )

            if array_length != record["n_samples"]:

                errors.append(
                    "Measurement-array length "
                    "does not match n_samples. "
                    f"Expected {record['n_samples']}, "
                    f"got {array_length}."
                )

    return (
        len(errors) == 0,
        errors,
    )


# ============================================================================
# INGEST ONE FFE FILE
# ============================================================================

def ingest_ffe_file(
    ffe_path: str | Path,
    raw_dir: str | Path,
    processed_dir: str | Path,
    conn,
    overwrite: bool = False,
) -> dict:
    """
    Parse and store one FFE file.
    """

    ffe_path = Path(ffe_path)
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    scenario_id = derive_scenario_id(
        ffe_path,
        raw_dir,
    )

    logger.info(
        "Processing: %s",
        scenario_id,
    )

    # ------------------------------------------------------------------
    # Parse FFE
    # ------------------------------------------------------------------

    try:

        record = parse_ffe(
            filepath=ffe_path,
            scenario_id=scenario_id,
        )

    except Exception as exc:

        logger.error(
            "FAILED | %s | Parser error: %s",
            scenario_id,
            exc,
        )

        return {
            "scenario_id": scenario_id,
            "source_file": str(ffe_path),
            "status": "error",
            "valid": False,
            "parquet_path": None,
            "error": str(exc),
        }

    # ------------------------------------------------------------------
    # Validate parsed record
    # ------------------------------------------------------------------

    is_valid, validation_errors = (
        validate_record(record)
    )

    if not is_valid:

        error_message = "; ".join(
            validation_errors
        )

        logger.error(
            "INVALID | %s | %s",
            scenario_id,
            error_message,
        )

        # Keep invalid scenarios in the database for traceability.
        try:

            store_scenario(
                conn=conn,
                record=record,
                parquet_path=None,
                overwrite=overwrite,
            )

        except Exception as exc:

            logger.error(
                "Could not store invalid scenario "
                "%s: %s",
                scenario_id,
                exc,
            )

        return {
            "scenario_id": scenario_id,
            "source_file": str(ffe_path),
            "status": "invalid",
            "valid": False,
            "parquet_path": None,
            "error": error_message,
        }

    # ------------------------------------------------------------------
    # Detect parser warnings
    # ------------------------------------------------------------------

    parse_warnings = record.get(
        "parse_warnings",
        [],
    )

    has_warnings = bool(
        parse_warnings
    )

    if has_warnings:

        logger.warning(
            "WARNING | %s | Parser warnings: %s",
            scenario_id,
            parse_warnings,
        )

    # ------------------------------------------------------------------
    # Create processed directory
    # ------------------------------------------------------------------

    scenario_dir = (
        processed_dir
        / scenario_id
    )

    scenario_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    parquet_path = (
        scenario_dir
        / "farfield.parquet"
    )

    # ------------------------------------------------------------------
    # Save far-field data to Parquet
    # ------------------------------------------------------------------

    try:

        saved_path = save_record_to_parquet(
            record=record,
            output_path=parquet_path,
        )

    except Exception as exc:

        logger.error(
            "FAILED | %s | Parquet error: %s",
            scenario_id,
            exc,
        )

        return {
            "scenario_id": scenario_id,
            "source_file": str(ffe_path),
            "status": "error",
            "valid": False,
            "parquet_path": None,
            "error": str(exc),
        }

    # ------------------------------------------------------------------
    # Store scenario metadata in SQLite
    # ------------------------------------------------------------------

    try:

        inserted = store_scenario(
            conn=conn,
            record=record,
            parquet_path=saved_path,
            overwrite=overwrite,
        )

    except Exception as exc:

        logger.error(
            "FAILED | %s | Database error: %s",
            scenario_id,
            exc,
        )

        return {
            "scenario_id": scenario_id,
            "source_file": str(ffe_path),
            "status": "error",
            "valid": False,
            "parquet_path": str(saved_path),
            "error": str(exc),
        }

    # ------------------------------------------------------------------
    # Existing scenario
    # ------------------------------------------------------------------

    if not inserted and not overwrite:

        logger.warning(
            "SKIPPED | %s | Scenario already exists "
            "in database.",
            scenario_id,
        )

        return {
            "scenario_id": scenario_id,
            "source_file": str(ffe_path),
            "status": "skipped",
            "valid": not has_warnings,
            "parquet_path": str(saved_path),
            "error": (
                "Scenario already exists "
                "and overwrite=False."
            ),
        }

    # ------------------------------------------------------------------
    # Successful ingestion
    # ------------------------------------------------------------------

    if has_warnings:

        status = "warning"

        logger.info(
            "INGESTED WITH WARNING | %s | %s",
            scenario_id,
            saved_path,
        )

    else:

        status = "ok"

        logger.info(
            "SUCCESS | %s | %s",
            scenario_id,
            saved_path,
        )

    return {
        "scenario_id": scenario_id,
        "source_file": str(ffe_path),
        "status": status,
        "valid": not has_warnings,
        "parquet_path": str(saved_path),
        "error": None,
    }


# ============================================================================
# BATCH INGESTION
# ============================================================================

def batch_ingest(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    db_path: str | Path = DEFAULT_DB_PATH,
    overwrite: bool = False,
) -> dict:
    """
    Run the complete batch ingestion process.
    """

    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    db_path = Path(db_path)

    print()
    print("=" * 70)
    print("ANTENNA HEALTH MONITORING PIPELINE")
    print("BATCH INGESTION")
    print("=" * 70)

    print(
        f"Raw directory:       {raw_dir}"
    )

    print(
        f"Processed directory: {processed_dir}"
    )

    print(
        f"Database:             {db_path}"
    )

    print()

    # ------------------------------------------------------------------
    # Discover FFE files
    # ------------------------------------------------------------------

    ffe_files = discover_ffe_files(
        raw_dir
    )

    print(
        f"FFE files discovered: {len(ffe_files)}"
    )

    if not ffe_files:

        print()
        print(
            "No .ffe files were found."
        )
        print("=" * 70)

        return {
            "run_id": None,
            "scenarios_in": 0,
            "scenarios_out": 0,
            "successful": 0,
            "warnings": 0,
            "skipped": 0,
            "invalid": 0,
            "errors": 0,
            "results": [],
        }

    print()

    # ------------------------------------------------------------------
    # Show discovered files
    # ------------------------------------------------------------------

    print("Discovered files:")

    for index, path in enumerate(
        ffe_files,
        start=1,
    ):

        print(
            f"  {index}. {path}"
        )

    print()

    # ------------------------------------------------------------------
    # Initialise database
    # ------------------------------------------------------------------

    conn = initialise_database(
        db_path
    )

    run_id = None

    results = []

    successful = 0
    warnings = 0
    skipped = 0
    invalid = 0
    errors = 0

    try:

        # --------------------------------------------------------------
        # Start pipeline run
        # --------------------------------------------------------------

        run_id = start_pipeline_run(
            conn=conn,
            stage="ingest",
            scenarios_in=len(ffe_files),
            notes=(
                "Batch ingestion of FEKO "
                ".ffe far-field files."
            ),
        )

        print(
            f"Pipeline run ID: {run_id}"
        )

        print()
        print("-" * 70)

        # --------------------------------------------------------------
        # Process every FFE file
        # --------------------------------------------------------------

        for ffe_path in ffe_files:

            result = ingest_ffe_file(
                ffe_path=ffe_path,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                conn=conn,
                overwrite=overwrite,
            )

            results.append(
                result
            )

            status = result["status"]

            if status == "ok":

                successful += 1

            elif status == "warning":

                warnings += 1

            elif status == "skipped":

                skipped += 1

            elif status == "invalid":

                invalid += 1

            elif status == "error":

                errors += 1

        # --------------------------------------------------------------
        # Number of successfully processed datasets
        # --------------------------------------------------------------

        scenarios_out = (
            successful
            + warnings
        )

        # --------------------------------------------------------------
        # Complete pipeline run
        # --------------------------------------------------------------

        complete_pipeline_run(
            conn=conn,
            run_id=run_id,
            scenarios_out=scenarios_out,
        )

        # --------------------------------------------------------------
        # Final summary
        # --------------------------------------------------------------

        print()
        print("=" * 70)
        print("BATCH INGESTION COMPLETE")
        print("=" * 70)

        print(
            f"Pipeline run ID:       {run_id}"
        )

        print(
            f"Files discovered:      {len(ffe_files)}"
        )

        print(
            f"Successful:             {successful}"
        )

        print(
            f"Warnings:               {warnings}"
        )

        print(
            f"Skipped:                {skipped}"
        )

        print(
            f"Invalid:                {invalid}"
        )

        print(
            f"Errors:                 {errors}"
        )

        print(
            f"Scenarios output:       {scenarios_out}"
        )

        print("=" * 70)

        return {
            "run_id": run_id,
            "scenarios_in": len(ffe_files),
            "scenarios_out": scenarios_out,
            "successful": successful,
            "warnings": warnings,
            "skipped": skipped,
            "invalid": invalid,
            "errors": errors,
            "results": results,
        }

    except Exception as exc:

        logger.exception(
            "Batch ingestion failed."
        )

        # --------------------------------------------------------------
        # If a pipeline run was started, mark it as failed.
        # --------------------------------------------------------------

        if run_id is not None:

            try:

                fail_pipeline_run(
                    conn=conn,
                    run_id=run_id,
                    error_message=str(exc),
                )

            except Exception:

                logger.exception(
                    "Could not mark pipeline run "
                    "as failed."
                )

        raise

    finally:

        conn.close()


# ============================================================================
# COMMAND-LINE ARGUMENTS
# ============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build command-line argument parser.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Batch ingest FEKO .ffe files "
            "into Parquet and SQLite."
        )
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=(
            "Directory containing raw .ffe files. "
            "Default: data/raw"
        ),
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=(
            "Directory for processed Parquet files. "
            "Default: data/processed"
        ),
    )

    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=(
            "SQLite database path. "
            "Default: data/antenna_health.db"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite existing scenarios in the database."
        ),
    )

    return parser


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """
    Main command-line entry point.
    """

    parser = build_argument_parser()

    args = parser.parse_args()

    batch_ingest(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        db_path=args.db_path,
        overwrite=args.overwrite,
    )


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
