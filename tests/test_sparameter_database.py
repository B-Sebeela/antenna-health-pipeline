from pathlib import Path

from src.storage.database import (
    initialise_database,
    link_sparameter_file,
    store_sparameter_metadata,
    fetch_scenario,
)


def main():

    db_path = Path(
        "data/test_sparameter.db"
    )

    parquet_path = Path(
        "data/processed/H01_NOM_sparameters.parquet"
    )

    print("=" * 60)
    print("S-PARAMETER → SQLITE TEST")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Initialise database
    # ---------------------------------------------------------------

    conn = initialise_database(
        db_path
    )

    print()
    print(f"✓ Database ready: {db_path}")

    # ---------------------------------------------------------------
    # Create a minimal scenario record
    # ---------------------------------------------------------------

    from src.storage.database import store_scenario

    scenario = {
        "scenario_id": "H01_NOM",
        "source_file": "data/raw/H01_NOM/H01_NOM.s7p",
    }

    store_scenario(
        conn,
        scenario,
        overwrite=True,
    )

    print("✓ Scenario created")

    # ---------------------------------------------------------------
    # Link S-parameter Parquet
    # ---------------------------------------------------------------

    link_sparameter_file(
        conn,
        "H01_NOM",
        parquet_path,
    )

    print("✓ S-parameter Parquet linked")

    # ---------------------------------------------------------------
    # Store S-parameter metadata
    # ---------------------------------------------------------------

    store_sparameter_metadata(
        conn=conn,
        scenario_id="H01_NOM",
        num_ports=7,
        num_frequency_points=1,
        reference_impedance_ohm=50.0,
        data_format="MA",
    )

    print("✓ S-parameter metadata stored")

    # ---------------------------------------------------------------
    # Retrieve scenario
    # ---------------------------------------------------------------

    result = fetch_scenario(
        conn,
        "H01_NOM",
    )

    print()
    print("Stored database record:")
    print()

    print(
        f"Scenario ID              : "
        f"{result['scenario_id']}"
    )

    print(
        f"S-parameter path         : "
        f"{result['sparameter_path']}"
    )

    print(
        f"Number of ports          : "
        f"{result['sparameter_num_ports']}"
    )

    print(
        f"Frequency points         : "
        f"{result['sparameter_num_frequency_points']}"
    )

    print(
        f"Reference impedance     : "
        f"{result['sparameter_reference_impedance_ohm']} Ω"
    )

    print(
        f"Data format              : "
        f"{result['sparameter_data_format']}"
    )

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    assert result is not None
    assert result["scenario_id"] == "H01_NOM"

    assert (
        result["sparameter_path"]
        == str(parquet_path)
    )

    assert (
        result["sparameter_num_ports"]
        == 7
    )

    assert (
        result["sparameter_num_frequency_points"]
        == 1
    )

    assert (
        result["sparameter_reference_impedance_ohm"]
        == 50.0
    )

    assert (
        result["sparameter_data_format"]
        == "MA"
    )

    conn.close()

    print()
    print("TEST / VALIDATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()