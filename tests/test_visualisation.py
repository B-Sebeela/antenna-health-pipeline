from pathlib import Path

from src.preprocess.visualisation import (
    load_processed_data,
    plot_far_field_heatmap,
    plot_radiation_pattern,
)


def test_visualisation():

    parquet_path = Path(
        "data/processed/test_scenario.parquet"
    )

    output_directory = Path(
        "data/outputs"
    )

    print("Loading processed Parquet data...")

    df = load_processed_data(
        parquet_path
    )

    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    assert len(df) == 2701

    print("\nGenerating radiation pattern...")

    radiation_path = (
        output_directory /
        "test_radiation_pattern.png"
    )

    plot_radiation_pattern(
        parquet_path=parquet_path,
        phi_angle=0.0,
        output_path=radiation_path,
        show=False,
    )

    assert radiation_path.exists()

    print(
        f"Radiation pattern saved: "
        f"{radiation_path}"
    )

    print("\nGenerating far-field heatmap...")

    heatmap_path = (
        output_directory /
        "test_far_field_heatmap.png"
    )

    plot_far_field_heatmap(
        parquet_path=parquet_path,
        output_path=heatmap_path,
        show=False,
    )

    assert heatmap_path.exists()

    print(
        f"Heatmap saved: "
        f"{heatmap_path}"
    )

    print("\n✓ Visualisation test PASSED")


if __name__ == "__main__":
    test_visualisation()