from pathlib import Path

from src.ingest.ffe_parser import parse_ffe
from src.storage.parquet import (
    load_parquet,
    parquet_summary,
    save_record_to_parquet,
)


def test_real_ffe_to_parquet():

    ffe_path = Path(
        "data/raw/farfield_1.ffe"
    )

    output_path = Path(
        "data/processed/test_scenario.parquet"
    )

    print("Parsing FEKO file...")

    record = parse_ffe(
        ffe_path,
        failed_elements=[],
    )

    print(f"Scenario: {record['scenario_id']}")
    print(f"Frequency: {record['frequency_hz'] / 1e6:.0f} MHz")
    print(f"Theta samples: {record['n_theta']}")
    print(f"Phi samples: {record['n_phi']}")
    print(f"Total samples: {record['n_samples']}")

    print("\nSaving Parquet file...")

    saved_path = save_record_to_parquet(
        record,
        output_path,
    )

    assert saved_path.exists()

    print(f"Parquet created: {saved_path}")

    print("\nLoading Parquet file...")

    df = load_parquet(saved_path)

    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    assert len(df) == 2701

    assert "theta" in df.columns
    assert "phi" in df.columns
    assert "magnitude_norm_db" in df.columns

    print("\nParquet summary:")

    summary = parquet_summary(saved_path)

    for key, value in summary.items():
        print(f"  {key}: {value}")

    assert summary["rows"] == 2701

    print("\n✓ Parquet test PASSED")


if __name__ == "__main__":
    test_real_ffe_to_parquet()