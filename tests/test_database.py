from pathlib import Path

from src.ingest.ffe_parser import parse_ffe
from src.storage.database import (
    complete_pipeline_run,
    fetch_scenario,
    get_database_summary,
    initialise_database,
    print_database_summary,
    start_pipeline_run,
    store_scenario,
)


def test_database():

    db_path = Path(
        "data/test_antenna_health.db"
    )

    # Remove previous test database so the test starts clean.
    if db_path.exists():
        db_path.unlink()

    print("Initialising database...")

    conn = initialise_database(
        db_path
    )

    print("✓ Database initialised")

    # ---------------------------------------------------------
    # Parse real FEKO file
    # ---------------------------------------------------------

    print("\nParsing FEKO file...")

    record = parse_ffe(
        "data/raw/farfield_1.ffe",
    )

    print(
        f"Scenario: {record['scenario_id']}"
    )

    # ---------------------------------------------------------
    # Insert scenario
    # ---------------------------------------------------------

    print("\nStoring scenario...")

    inserted = store_scenario(
        conn,
        record,
        parquet_path="data/processed/test_scenario.parquet",
    )

    assert inserted is True

    print("✓ Scenario stored")

    # ---------------------------------------------------------
    # Retrieve scenario
    # ---------------------------------------------------------

    print("\nRetrieving scenario...")

    scenario = fetch_scenario(
        conn,
        record["scenario_id"],
    )

    assert scenario is not None

    print(
        f"Scenario ID: {scenario['scenario_id']}"
    )

    print(
        f"Frequency: "
        f"{scenario['frequency_hz'] / 1e6:.0f} MHz"
    )

    print(
        f"Samples: "
        f"{scenario['n_samples']}"
    )

    print(
        f"Parquet: "
        f"{scenario['parquet_path']}"
    )

    assert scenario["n_samples"] == 2701

    assert (
        scenario["parquet_path"]
        == "data/processed/test_scenario.parquet"
    )

    # ---------------------------------------------------------
    # Pipeline run
    # ---------------------------------------------------------

    print("\nStarting pipeline run...")

    run_id = start_pipeline_run(
        conn,
        stage="ingest",
        scenarios_in=1,
        notes="Database integration test",
    )

    print(
        f"✓ Pipeline run started: #{run_id}"
    )

    complete_pipeline_run(
        conn,
        run_id,
        scenarios_out=1,
    )

    print(
        "✓ Pipeline run completed"
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print_database_summary(
        conn
    )

    summary = get_database_summary(
        conn
    )

    assert summary["n_scenarios"] == 1
    assert summary["n_valid_scenarios"] == 1
    assert summary["n_runs"] == 1

    print("\n✓ Database test PASSED")

    conn.close()


if __name__ == "__main__":
    test_database()