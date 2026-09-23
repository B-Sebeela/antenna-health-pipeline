
"""
Refined feature extraction for the Antenna Health Monitoring Pipeline.

This module creates a smaller, engineering-focused feature set from
processed far-field Parquet files.

The original features.py module is intentionally left unchanged.
This module is used to develop and evaluate a more compact feature set
before anomaly detection and classification are implemented.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

OUTPUT_FILE = FEATURES_DIR / "scenario_features_refined.parquet"

BASELINE_SCENARIO = "H01_NOM"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_scenario_data(scenario_id: str) -> pd.DataFrame:
    """
    Load processed Parquet data for one scenario.
    """

    parquet_path = (
        PROCESSED_DIR
        / scenario_id
        / "farfield.parquet"
    )

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Processed Parquet file not found for scenario "
            f"'{scenario_id}': {parquet_path}"
        )

    return pd.read_parquet(parquet_path)


def calculate_e_theta_magnitude(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the magnitude of the E-theta electric-field component.
    """

    return np.sqrt(
        df["e_theta_real"] ** 2
        + df["e_theta_imag"] ** 2
    )


# ---------------------------------------------------------------------------
# Global gain features
# ---------------------------------------------------------------------------

def extract_gain_features(df: pd.DataFrame) -> dict:
    """
    Extract global radiation-pattern gain statistics.

    These describe the overall distribution of total gain across
    the far-field measurement grid.
    """

    gain = df["gain_total"]

    return {
        "gain_mean": gain.mean(),
        "gain_std": gain.std(),
        "gain_max": gain.max(),
        "gain_min": gain.min(),
        "gain_range": gain.max() - gain.min(),
        "gain_median": gain.median(),
    }


# ---------------------------------------------------------------------------
# Electric-field features
# ---------------------------------------------------------------------------

def extract_e_theta_features(df: pd.DataFrame) -> dict:
    """
    Extract statistics from the dominant E-theta field component.
    """

    e_theta = calculate_e_theta_magnitude(df)

    return {
        "e_theta_mean": e_theta.mean(),
        "e_theta_std": e_theta.std(),
        "e_theta_max": e_theta.max(),
        "e_theta_median": e_theta.median(),
        "e_theta_range": e_theta.max() - e_theta.min(),
    }


# ---------------------------------------------------------------------------
# Radiation-pattern peak features
# ---------------------------------------------------------------------------

def extract_peak_features(df: pd.DataFrame) -> dict:
    """
    Find the maximum-gain point in the radiation pattern.

    The corresponding theta and phi coordinates describe the
    direction of the strongest radiation.
    """

    peak_index = df["gain_total"].idxmax()
    peak_row = df.loc[peak_index]

    return {
        "peak_gain": peak_row["gain_total"],
        "peak_theta": peak_row["theta"],
        "peak_phi": peak_row["phi"],
    }


# ---------------------------------------------------------------------------
# Baseline deviation features
# ---------------------------------------------------------------------------

def extract_baseline_features(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> dict:
    """
    Calculate deviations from the healthy baseline radiation pattern.

    The comparison is performed point-by-point using the gain_total
    values from the same theta/phi measurement grid.
    """

    current_gain = df["gain_total"].to_numpy()
    baseline_gain = baseline_df["gain_total"].to_numpy()

    if len(current_gain) != len(baseline_gain):
        raise ValueError(
            "Current scenario and baseline scenario have different "
            "numbers of far-field samples."
        )

    difference = current_gain - baseline_gain
    absolute_difference = np.abs(difference)

    return {
        "baseline_mean_difference": np.mean(difference),

        "baseline_mean_absolute_difference": np.mean(
            absolute_difference
        ),

        "baseline_median_absolute_difference": np.median(
            absolute_difference
        ),

        "baseline_std_difference": np.std(
            difference
        ),

        "baseline_max_absolute_difference": np.max(
            absolute_difference
        ),

        "baseline_pct_diff_0_5db": (
            np.mean(absolute_difference >= 0.5) * 100
        ),

        "baseline_pct_diff_1db": (
            np.mean(absolute_difference >= 1.0) * 100
        ),

        "baseline_pct_diff_2db": (
            np.mean(absolute_difference >= 2.0) * 100
        ),

        "baseline_pct_diff_3db": (
            np.mean(absolute_difference >= 3.0) * 100
        ),
    }


# ---------------------------------------------------------------------------
# Complete feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    scenario_id: str,
    baseline_df: pd.DataFrame | None = None,
) -> dict:
    """
    Extract the complete refined feature vector for one scenario.
    """

    df = load_scenario_data(scenario_id)

    features = {
        "scenario_id": scenario_id,
        "n_samples": len(df),
    }

    # Global gain
    features.update(
        extract_gain_features(df)
    )

    # Dominant electric-field component
    features.update(
        extract_e_theta_features(df)
    )

    # Radiation peak
    features.update(
        extract_peak_features(df)
    )

    # Baseline comparison
    if baseline_df is not None:
        features.update(
            extract_baseline_features(
                df,
                baseline_df,
            )
        )

    else:
        # Healthy baseline compared with itself
        features.update({
            "baseline_mean_difference": 0.0,
            "baseline_mean_absolute_difference": 0.0,
            "baseline_median_absolute_difference": 0.0,
            "baseline_std_difference": 0.0,
            "baseline_max_absolute_difference": 0.0,
            "baseline_pct_diff_0_5db": 0.0,
            "baseline_pct_diff_1db": 0.0,
            "baseline_pct_diff_2db": 0.0,
            "baseline_pct_diff_3db": 0.0,
        })

    return features


# ---------------------------------------------------------------------------
# Scenario discovery
# ---------------------------------------------------------------------------

def discover_scenarios() -> list[str]:
    """
    Discover scenarios that contain a processed far-field Parquet file.
    """

    scenarios = []

    if not PROCESSED_DIR.exists():
        return scenarios

    for scenario_dir in sorted(PROCESSED_DIR.iterdir()):

        if not scenario_dir.is_dir():
            continue

        parquet_file = scenario_dir / "farfield.parquet"

        if parquet_file.exists():
            scenarios.append(scenario_dir.name)

    return scenarios


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run refined feature extraction for all available scenarios.
    """

    print("=" * 70)
    print("ANTENNA HEALTH MONITORING PIPELINE")
    print("REFINED FEATURE EXTRACTION")
    print("=" * 70)

    scenarios = discover_scenarios()

    if not scenarios:
        print("\nNo processed scenarios found.")
        return

    print("\nScenarios discovered:")

    for scenario in scenarios:
        print(f"  - {scenario}")

    # ---------------------------------------------------------------
    # Load healthy baseline
    # ---------------------------------------------------------------

    if BASELINE_SCENARIO not in scenarios:
        raise ValueError(
            f"Baseline scenario '{BASELINE_SCENARIO}' "
            f"was not found."
        )

    baseline_df = load_scenario_data(
        BASELINE_SCENARIO
    )

    print(
        f"\nBaseline scenario: {BASELINE_SCENARIO}"
    )

    # ---------------------------------------------------------------
    # Extract features
    # ---------------------------------------------------------------

    feature_rows = []

    for scenario_id in scenarios:

        print(
            f"\nProcessing: {scenario_id}"
        )

        features = extract_features(
            scenario_id=scenario_id,
            baseline_df=baseline_df,
        )

        feature_rows.append(features)

        print(
            f"  Features generated: "
            f"{len(features) - 1}"
        )

    # ---------------------------------------------------------------
    # Create feature dataframe
    # ---------------------------------------------------------------

    feature_df = pd.DataFrame(feature_rows)

    # Keep scenario_id as the first column
    feature_columns = [
        "scenario_id"
    ] + [
        column
        for column in feature_df.columns
        if column != "scenario_id"
    ]

    feature_df = feature_df[
        feature_columns
    ]

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    FEATURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------------
    # Display summary
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("REFINED FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"\nScenarios processed: "
        f"{len(feature_df)}"
    )

    print(
        f"Features generated: "
        f"{len(feature_df.columns) - 1}"
    )

    print(
        f"Output: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}"
    )

    print("\nFeature dataset shape:")

    print(
        f"  {feature_df.shape[0]} rows × "
        f"{feature_df.shape[1]} columns"
    )

    print("\nFeature columns:")

    for column in feature_df.columns:
        if column != "scenario_id":
            print(f"  - {column}")

    # ---------------------------------------------------------------
    # Scenario summary
    # ---------------------------------------------------------------

    print("\nScenario summary:")

    summary_columns = [
        "scenario_id",
        "gain_mean",
        "gain_std",
        "gain_max",
        "e_theta_mean",
        "e_theta_std",
        "peak_gain",
        "peak_theta",
        "peak_phi",
        "baseline_mean_absolute_difference",
        "baseline_max_absolute_difference",
        "baseline_pct_diff_1db",
        "baseline_pct_diff_3db",
    ]

    print(
        feature_df[
            summary_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nDone.")


if __name__ == "__main__":
    main()

