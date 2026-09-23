from pathlib import Path

from src.ingest.sparam_parser import (
    parse_sparameter_file,
    validate_sparameter_dataframe,
)

from src.storage.parquet import (
    save_sparameter_to_parquet,
    load_sparameter_parquet,
    sparameter_parquet_summary,
)


def main():

    input_file = Path(
        "data/raw/H01_NOM/H01_NOM.s7p"
    )

    output_file = Path(
        "data/processed/H01_NOM_sparameters.parquet"
    )

    print("=" * 60)
    print("S-PARAMETER → PARQUET TEST")
    print("=" * 60)

    print(f"Input : {input_file}")
    print(f"Output: {output_file}")

    # ---------------------------------------------------------------
    # Parse
    # ---------------------------------------------------------------

    result = parse_sparameter_file(
        input_file
    )

    # ---------------------------------------------------------------
    # Validate
    # ---------------------------------------------------------------

    validate_sparameter_dataframe(
        result.dataframe,
        expected_num_ports=result.num_ports,
    )

    print()
    print("Parser:")
    print(f"  Ports            : {result.num_ports}")
    print(f"  Frequency points : {result.num_frequency_points}")
    print(f"  Rows             : {len(result.dataframe)}")

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    saved_path = save_sparameter_to_parquet(
        result.dataframe,
        output_file,
    )

    print()
    print(f"✓ Parquet created: {saved_path}")

    # ---------------------------------------------------------------
    # Load again
    # ---------------------------------------------------------------

    loaded = load_sparameter_parquet(
        saved_path
    )

    print()
    print("Reload test:")
    print(f"  Rows    : {len(loaded)}")
    print(f"  Columns : {len(loaded.columns)}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    summary = sparameter_parquet_summary(
        saved_path
    )

    print()
    print("Parquet summary:")

    for key, value in summary.items():
        print(f"  {key}: {value}")

    print()
    print("TEST / VALIDATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()