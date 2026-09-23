
"""
Feature extraction for antenna far-field health monitoring.

This module converts scenario-level far-field Parquet data into
compact features suitable for analysis and future machine learning.

Input:
    data/processed/<scenario_id>/farfield.parquet

Output:
    One feature record per scenario.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_FEATURE_DIR = Path("data/features")
DEFAULT_FEATURE_FILE = DEFAULT_FEATURE_DIR / "scenario_features.parquet"

NOMINAL_SCENARIO = "H01_NOM"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scenario_data(
    scenario_id: str,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> pd.DataFrame:
    """
    Load the processed far-field Parquet file for one scenario.
    """

    parquet_path = (
        processed_dir
        / scenario_id
        / "farfield.parquet"
    )

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Processed Parquet file not found: {parquet_path}"
        )

    df = pd.read_parquet(parquet_path)

    required_columns = [
        "theta",
        "phi",
        "e_theta_real",
        "e_theta_imag",
        "e_phi_real",
        "e_phi_imag",
        "gain_theta",
        "gain_phi",
        "gain_total",
        "magnitude_norm_db",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{scenario_id}: missing required columns: {missing}"
        )

    return df


# ---------------------------------------------------------------------------
# Electric-field calculations
# ---------------------------------------------------------------------------

def add_field_magnitudes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the magnitude of E-theta and E-phi.

    |E_theta| = sqrt(Re(E_theta)^2 + Im(E_theta)^2)
    |E_phi|   = sqrt(Re(E_phi)^2 + Im(E_phi)^2)
    """

    result = df.copy()

    result["e_theta_magnitude"] = np.sqrt(
        result["e_theta_real"] ** 2
        + result["e_theta_imag"] ** 2
    )

    result["e_phi_magnitude"] = np.sqrt(
        result["e_phi_real"] ** 2
        + result["e_phi_imag"] ** 2
    )

    return result


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def calculate_statistics(
    series: pd.Series,
    prefix: str,
) -> dict:
    """
    Calculate descriptive statistics for one measurement.
    """

    return {
        f"{prefix}_min": float(series.min()),
        f"{prefix}_max": float(series.max()),
        f"{prefix}_mean": float(series.mean()),
        f"{prefix}_std": float(series.std()),
        f"{prefix}_median": float(series.median()),
        f"{prefix}_p10": float(series.quantile(0.10)),
        f"{prefix}_p25": float(series.quantile(0.25)),
        f"{prefix}_p75": float(series.quantile(0.75)),
        f"{prefix}_p90": float(series.quantile(0.90)),
        f"{prefix}_range": float(series.max() - series.min()),
    }


# ---------------------------------------------------------------------------
# Peak features
# ---------------------------------------------------------------------------

def calculate_peak_features(
    df: pd.DataFrame,
) -> dict:
    """
    Find the maximum total gain and its angular location.
    """

    peak_index = df["gain_total"].idxmax()
    peak = df.loc[peak_index]

    return {
        "peak_gain": float(peak["gain_total"]),
        "peak_theta": float(peak["theta"]),
        "peak_phi": float(peak["phi"]),
    }


# ---------------------------------------------------------------------------
# Scenario feature extraction
# ---------------------------------------------------------------------------

def extract_scenario_features(
    df: pd.DataFrame,
    scenario_id: str,
) -> dict:
    """
    Extract general features from one antenna scenario.
    """

    df = add_field_magnitudes(df)

    features = {
        "scenario_id": scenario_id,
        "n_samples": int(len(df)),
    }

    # Gain statistics
    features.update(
        calculate_statistics(
            df["gain_total"],
            "gain",
        )
    )

    # Theta-polarised gain statistics
    features.update(
        calculate_statistics(
            df["gain_theta"],
            "gain_theta",
        )
    )

    # Phi-polarised gain statistics
    features.update(
        calculate_statistics(
            df["gain_phi"],
            "gain_phi",
        )
    )

    # Normalised magnitude statistics
    features.update(
        calculate_statistics(
            df["magnitude_norm_db"],
            "magnitude_norm",
        )
    )

    # Electric-field statistics
    features.update(
        calculate_statistics(
            df["e_theta_magnitude"],
            "e_theta",
        )
    )

    features.update(
        calculate_statistics(
            df["e_phi_magnitude"],
            "e_phi",
        )
    )

    # Peak radiation features
    features.update(
        calculate_peak_features(df)
    )

    return features


# ---------------------------------------------------------------------------
# Baseline deviation features
# ---------------------------------------------------------------------------

def calculate_baseline_deviation(
    nominal_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
) -> dict:
    """
    Calculate radiation-pattern differences relative to the healthy
    nominal scenario.

    Difference is:

        scenario_gain - nominal_gain
    """

    nominal = nominal_df.set_index(
        ["theta", "phi"]
    )["gain_total"]

    scenario = scenario_df.set_index(
        ["theta", "phi"]
    )["gain_total"]

    comparison = pd.concat(
        [nominal, scenario],
        axis=1,
        keys=["nominal", "scenario"],
        join="inner",
    ).dropna()

    if comparison.empty:
        raise ValueError(
            "No matching theta/phi points found between "
            "nominal and scenario."
        )

    difference = (
        comparison["scenario"]
        - comparison["nominal"]
    )

    absolute_difference = difference.abs()

    return {
        "baseline_mean_difference": float(
            difference.mean()
        ),
        "baseline_mean_absolute_difference": float(
            absolute_difference.mean()
        ),
        "baseline_median_absolute_difference": float(
            absolute_difference.median()
        ),
        "baseline_std_difference": float(
            difference.std()
        ),
        "baseline_max_difference": float(
            difference.max()
        ),
        "baseline_min_difference": float(
            difference.min()
        ),
        "baseline_max_absolute_difference": float(
            absolute_difference.max()
        ),
        "baseline_pct_diff_0_1db": float(
            (absolute_difference >= 0.1).mean() * 100
        ),
        "baseline_pct_diff_0_5db": float(
            (absolute_difference >= 0.5).mean() * 100
        ),
        "baseline_pct_diff_1db": float(
            (absolute_difference >= 1.0).mean() * 100
        ),
        "baseline_pct_diff_2db": float(
            (absolute_difference >= 2.0).mean() * 100
        ),
        "baseline_pct_diff_3db": float(
            (absolute_difference >= 3.0).mean() * 100
        ),
        "baseline_pct_diff_4db": float(
            (absolute_difference >= 4.0).mean() * 100
        ),
        "baseline_pct_diff_5db": float(
            (absolute_difference >= 5.0).mean() * 100
        ),
    }


# ---------------------------------------------------------------------------
# Complete feature extraction
# ---------------------------------------------------------------------------

def build_feature_dataset(
    scenario_ids: list[str],
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    nominal_scenario: str = NOMINAL_SCENARIO,
) -> pd.DataFrame:
    """
    Build one feature row per scenario.

    The nominal scenario is used as the healthy baseline for
    deviation features.
    """

    if nominal_scenario not in scenario_ids:
        scenario_ids = [
            nominal_scenario,
            *scenario_ids,
        ]

    scenario_data = {}

    for scenario_id in scenario_ids:
        scenario_data[scenario_id] = load_scenario_data(
            scenario_id,
            processed_dir,
        )

    nominal_df = scenario_data[nominal_scenario]

    feature_records = []

    for scenario_id in scenario_ids:

        df = scenario_data[scenario_id]

        features = extract_scenario_features(
            df,
            scenario_id,
        )

        # The healthy baseline has zero deviation from itself.
        if scenario_id == nominal_scenario:

            deviation_features = {
                "baseline_mean_difference": 0.0,
                "baseline_mean_absolute_difference": 0.0,
                "baseline_median_absolute_difference": 0.0,
                "baseline_std_difference": 0.0,
                "baseline_max_difference": 0.0,
                "baseline_min_difference": 0.0,
                "baseline_max_absolute_difference": 0.0,
                "baseline_pct_diff_0_1db": 0.0,
                "baseline_pct_diff_0_5db": 0.0,
                "baseline_pct_diff_1db": 0.0,
                "baseline_pct_diff_2db": 0.0,
                "baseline_pct_diff_3db": 0.0,
                "baseline_pct_diff_4db": 0.0,
                "baseline_pct_diff_5db": 0.0,
            }

        else:

            deviation_features = calculate_baseline_deviation(
                nominal_df,
                df,
            )

        features.update(deviation_features)

        feature_records.append(features)

    return pd.DataFrame(feature_records)


# ---------------------------------------------------------------------------
# Save features
# ---------------------------------------------------------------------------

def save_feature_dataset(
    feature_df: pd.DataFrame,
    output_path: Path = DEFAULT_FEATURE_FILE,
) -> Path:
    """
    Save the feature dataset to Parquet.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_df.to_parquet(
        output_path,
        index=False,
    )

    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    scenarios = [
        "H01_NOM",
        "MF01_L_PLUS",
        "MF02_L_MINUS",
        "M01_FEED_OFFSET",
    ]

    print()
    print("=" * 70)
    print("ANTENNA HEALTH MONITORING PIPELINE")
    print("FEATURE EXTRACTION")
    print("=" * 70)

    print()
    print("Scenarios:")

    for scenario in scenarios:
        print(f"  - {scenario}")

    print()

    feature_df = build_feature_dataset(
        scenarios
    )

    output_path = save_feature_dataset(
        feature_df
    )

    print("Feature extraction complete.")
    print()
    print(f"Scenarios processed: {len(feature_df)}")
    print(f"Features generated:  {len(feature_df.columns)}")
    print(f"Output:              {output_path}")

    print()
    print("Feature dataset shape:")
    print(f"  {feature_df.shape[0]} rows × {feature_df.shape[1]} columns")

    print()
    print("Scenario summary:")
    print(
        feature_df[
            [
                "scenario_id",
                "gain_max",
                "gain_mean",
                "gain_std",
                "peak_theta",
                "peak_phi",
                "baseline_mean_absolute_difference",
                "baseline_max_absolute_difference",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

