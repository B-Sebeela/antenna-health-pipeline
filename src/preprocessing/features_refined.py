"""
Refined feature extraction for the Antenna Health Monitoring Pipeline.

Supports two experiment families, each with its own healthy baseline:

  - Feed-offset (H01_*, FO_*): baseline H01_DIPOLE_FEED_50
  - 7-element array (ARR7_*):  baseline ARR7_BASELINE_ALL_ON

Baseline-deviation features are computed over the *meaningful*
part of the sphere only:

1. NULL MASK
   Points where the HEALTHY BASELINE gain is more than
   MASK_DB_BELOW_PEAK dB below the baseline peak are excluded.
   The mask is built from the baseline only, so every scenario is
   compared over exactly the same region.

2. AREA WEIGHTING
   Each retained point is weighted by sin(theta), proportional to
   solid angle on a uniform theta/phi grid.

New audit / metadata columns include experiment_family and
n_elements_off (array family).

This module does not perform anomaly detection or classification.

Pipeline position:

    processed Parquet data
            ->
    refined feature extraction
            ->
    scenario_features_refined.parquet
            ->
    anomaly detection / classification
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

# Per-family healthy references
BASELINE_FEED_OFFSET = "H01_DIPOLE_FEED_50"
BASELINE_ARRAY7 = "ARR7_BASELINE_ALL_ON"

# Legacy alias
BASELINE_SCENARIO = BASELINE_FEED_OFFSET

# Points whose BASELINE gain is below (baseline peak - this value) are
# excluded from the baseline comparison.
MASK_DB_BELOW_PEAK = 30.0

# Points with sin(theta) below this are treated as poles (zero area).
MIN_AREA_WEIGHT = 1e-9

DIFFERENCE_THRESHOLDS_DB = {
    "baseline_pct_diff_0_1db": 0.1,
    "baseline_pct_diff_0_5db": 0.5,
    "baseline_pct_diff_1db": 1.0,
    "baseline_pct_diff_2db": 2.0,
    "baseline_pct_diff_3db": 3.0,
    "baseline_pct_diff_4db": 4.0,
    "baseline_pct_diff_5db": 5.0,
}


# ---------------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
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


# ---------------------------------------------------------------------------
# Family / baseline helpers
# ---------------------------------------------------------------------------

def scenario_family(scenario_id: str) -> str:
    """Return 'array7', 'feed_offset', or 'other'."""
    if scenario_id.startswith("ARR7_"):
        return "array7"
    if scenario_id.startswith("FO_") or scenario_id.startswith("H01_"):
        return "feed_offset"
    return "other"


def baseline_for_scenario(scenario_id: str) -> str:
    """Choose the healthy reference for this scenario."""
    if scenario_family(scenario_id) == "array7":
        return BASELINE_ARRAY7
    return BASELINE_FEED_OFFSET


def extract_array_n_off(scenario_id: str) -> int | None:
    """Map ARR7 scenario IDs to number of elements off (of 7)."""
    mapping = {
        "ARR7_BASELINE_ALL_ON": 0,
        "ARR7_FAIL_1": 1,
        "ARR7_FAIL_2": 2,
        "ARR7_FAIL_3": 3,
        "ARR7_FAIL_5": 5,
        "ARR7_ONLY_1_ON": 6,  # only one on → six off
    }
    return mapping.get(scenario_id)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scenario_data(scenario_id: str) -> pd.DataFrame:
    """Load processed Parquet data for one scenario."""
    parquet_path = PROCESSED_DIR / scenario_id / "farfield.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Processed Parquet file not found for "
            f"scenario '{scenario_id}': {parquet_path}"
        )

    df = pd.read_parquet(parquet_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{scenario_id}: missing required columns: {missing}")

    if df.empty:
        raise ValueError(f"{scenario_id}: Parquet file contains no samples.")

    return df


# ---------------------------------------------------------------------------
# Electric-field calculations
# ---------------------------------------------------------------------------

def calculate_e_theta_magnitude(df: pd.DataFrame) -> pd.Series:
    return np.sqrt(df["e_theta_real"] ** 2 + df["e_theta_imag"] ** 2)


def calculate_e_phi_magnitude(df: pd.DataFrame) -> pd.Series:
    return np.sqrt(df["e_phi_real"] ** 2 + df["e_phi_imag"] ** 2)


# ---------------------------------------------------------------------------
# Global gain features
# ---------------------------------------------------------------------------

def extract_gain_features(df: pd.DataFrame) -> dict:
    gain = df["gain_total"]
    return {
        "gain_mean": float(gain.mean()),
        "gain_std": float(gain.std()),
        "gain_max": float(gain.max()),
        "gain_min": float(gain.min()),
        "gain_range": float(gain.max() - gain.min()),
        "gain_median": float(gain.median()),
    }


# ---------------------------------------------------------------------------
# Electric-field features
# ---------------------------------------------------------------------------

def extract_e_field_features(df: pd.DataFrame) -> dict:
    e_theta = calculate_e_theta_magnitude(df)
    e_phi = calculate_e_phi_magnitude(df)
    return {
        "e_theta_mean": float(e_theta.mean()),
        "e_theta_std": float(e_theta.std()),
        "e_theta_max": float(e_theta.max()),
        "e_theta_range": float(e_theta.max() - e_theta.min()),
        "e_phi_mean": float(e_phi.mean()),
        "e_phi_std": float(e_phi.std()),
        "e_phi_max": float(e_phi.max()),
        "e_phi_range": float(e_phi.max() - e_phi.min()),
    }


# ---------------------------------------------------------------------------
# Radiation-pattern peak features
# ---------------------------------------------------------------------------

def extract_peak_features(df: pd.DataFrame) -> dict:
    peak_index = df["gain_total"].idxmax()
    peak_row = df.loc[peak_index]
    return {
        "peak_gain": float(peak_row["gain_total"]),
        "peak_theta": float(peak_row["theta"]),
        "peak_phi": float(peak_row["phi"]),
    }


# ---------------------------------------------------------------------------
# Weighted statistics helpers
# ---------------------------------------------------------------------------

def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    mean = _weighted_mean(values, weights)
    variance = np.sum(weights * (values - mean) ** 2) / np.sum(weights)
    return float(np.sqrt(variance))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative = cumulative / cumulative[-1]
    index = int(np.searchsorted(cumulative, 0.5))
    return float(sorted_values[index])


def _weighted_pct_at_least(
    values: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> float:
    return float(
        100.0 * np.sum(weights[values >= threshold]) / np.sum(weights)
    )


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def align_with_baseline(
    scenario_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align a scenario with the healthy baseline using theta and phi.
    Inner merge: coarser grids compare on the intersection only.
    """
    scenario = scenario_df[["theta", "phi", "gain_total"]].copy()
    baseline = baseline_df[["theta", "phi", "gain_total"]].copy()

    scenario = scenario.rename(columns={"gain_total": "scenario_gain"})
    baseline = baseline.rename(columns={"gain_total": "baseline_gain"})

    comparison = pd.merge(
        scenario,
        baseline,
        on=["theta", "phi"],
        how="inner",
        validate="one_to_one",
    )

    if comparison.empty:
        raise ValueError(
            "No matching theta/phi points found between "
            "scenario and healthy baseline."
        )

    return comparison


def extract_baseline_features(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> dict:
    """
    Masked, sin(theta)-weighted deviation features vs healthy baseline.
    """
    comparison = align_with_baseline(df, baseline_df)

    baseline_peak = float(baseline_df["gain_total"].max())
    scenario_gain = comparison["scenario_gain"].to_numpy()
    baseline_gain = comparison["baseline_gain"].to_numpy()
    difference_all = scenario_gain - baseline_gain

    area_weight = np.clip(
        np.sin(np.deg2rad(comparison["theta"].to_numpy())),
        0.0,
        None,
    )

    in_main_region = baseline_gain >= baseline_peak - MASK_DB_BELOW_PEAK
    keep = in_main_region & (area_weight > MIN_AREA_WEIGHT)

    if not keep.any():
        raise ValueError(
            "The null mask removed every point; check "
            "MASK_DB_BELOW_PEAK and the baseline data."
        )

    difference = difference_all[keep]
    weights = area_weight[keep]
    absolute_difference = np.abs(difference)

    features = {
        "baseline_n_matching_points": int(len(comparison)),
        "baseline_grid_coverage_pct": float(
            100.0 * len(comparison) / len(baseline_df)
        ),
        "baseline_mask_db_below_peak": float(MASK_DB_BELOW_PEAK),
        "baseline_mask_n_points": int(keep.sum()),
        "baseline_mask_area_fraction_pct": float(
            100.0 * weights.sum() / area_weight.sum()
        ),
        "baseline_mean_difference": _weighted_mean(difference, weights),
        "baseline_mean_absolute_difference": _weighted_mean(
            absolute_difference, weights
        ),
        "baseline_median_absolute_difference": _weighted_median(
            absolute_difference, weights
        ),
        "baseline_std_difference": _weighted_std(difference, weights),
        "baseline_max_difference": float(difference.max()),
        "baseline_min_difference": float(difference.min()),
        "baseline_max_absolute_difference": float(absolute_difference.max()),
        "baseline_unmasked_max_absolute_difference": float(
            np.abs(difference_all).max()
        ),
    }

    for column, threshold in DIFFERENCE_THRESHOLDS_DB.items():
        features[column] = _weighted_pct_at_least(
            absolute_difference, weights, threshold
        )

    return features


# ---------------------------------------------------------------------------
# Scenario metadata
# ---------------------------------------------------------------------------

def extract_feed_setting(scenario_id: str) -> float | None:
    """Extract feed percentage from FO_/H01_ style IDs."""
    parts = scenario_id.split("_")
    if not parts:
        return None
    try:
        return float(parts[-1])
    except ValueError:
        return None


def extract_fault_metadata(scenario_id: str) -> dict:
    """
    Known simulation labels from naming convention.
    Kept separate from data-derived anomaly measurements.
    """
    family = scenario_family(scenario_id)

    if family == "array7":
        n_off = extract_array_n_off(scenario_id)
        is_healthy = scenario_id == BASELINE_ARRAY7

        if is_healthy:
            known_fault = "NONE"
            known_condition = "NOMINAL"
        else:
            known_fault = "ELEMENT_FAILURE"
            if scenario_id == "ARR7_ONLY_1_ON":
                known_condition = "ONLY_ONE_ACTIVE"
            elif n_off is not None:
                known_condition = f"N_OFF_{n_off}"
            else:
                known_condition = "UNKNOWN"

        return {
            "known_health": "HEALTHY" if is_healthy else "FAULT",
            "known_fault_type": known_fault,
            "known_condition": known_condition,
            "feed_setting_percent": None,
            "experiment_family": "array7",
            "n_elements_off": n_off,
        }

    # Feed-offset family
    is_healthy = scenario_id == BASELINE_FEED_OFFSET
    feed_setting = extract_feed_setting(scenario_id)

    if is_healthy:
        known_fault = "NONE"
        known_condition = "NOMINAL"
    else:
        known_fault = "FEED_OFFSET"
        if feed_setting is not None:
            if feed_setting < 50:
                known_condition = "LOWER_THAN_REFERENCE"
            elif feed_setting > 50:
                known_condition = "HIGHER_THAN_REFERENCE"
            else:
                known_condition = "NOMINAL"
        else:
            known_condition = "UNKNOWN"

    return {
        "known_health": "HEALTHY" if is_healthy else "FAULT",
        "known_fault_type": known_fault,
        "known_condition": known_condition,
        "feed_setting_percent": feed_setting,
        "experiment_family": "feed_offset",
        "n_elements_off": None,
    }


# ---------------------------------------------------------------------------
# Complete feature extraction
# ---------------------------------------------------------------------------

def extract_features(scenario_id: str, baseline_df: pd.DataFrame) -> dict:
    """Extract the complete feature vector for one scenario."""
    df = load_scenario_data(scenario_id)

    features = {
        "scenario_id": scenario_id,
        "n_samples": int(len(df)),
    }
    features.update(extract_gain_features(df))
    features.update(extract_e_field_features(df))
    features.update(extract_peak_features(df))
    features.update(extract_baseline_features(df, baseline_df))
    features.update(extract_fault_metadata(scenario_id))

    return features


# ---------------------------------------------------------------------------
# Scenario discovery
# ---------------------------------------------------------------------------

def discover_scenarios() -> list[str]:
    """Discover scenarios containing a processed far-field Parquet file."""
    scenarios = []
    if not PROCESSED_DIR.exists():
        return scenarios

    for scenario_dir in sorted(PROCESSED_DIR.iterdir()):
        if not scenario_dir.is_dir():
            continue
        if (scenario_dir / "farfield.parquet").exists():
            scenarios.append(scenario_dir.name)

    return scenarios


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run refined feature extraction for all available scenarios."""
    print()
    print("=" * 70)
    print("ANTENNA HEALTH MONITORING PIPELINE")
    print("REFINED FEATURE EXTRACTION")
    print("=" * 70)

    scenarios = discover_scenarios()
    if not scenarios:
        print("\nNo processed scenarios found.")
        return

    print(f"\nProcessed scenarios discovered: {len(scenarios)}")
    for scenario in scenarios:
        print(f"  - {scenario}")

    has_fo = any(scenario_family(s) == "feed_offset" for s in scenarios)
    has_arr7 = any(scenario_family(s) == "array7" for s in scenarios)

    if has_fo and BASELINE_FEED_OFFSET not in scenarios:
        raise ValueError(
            f"Healthy baseline '{BASELINE_FEED_OFFSET}' was not found."
        )
    if has_arr7 and BASELINE_ARRAY7 not in scenarios:
        raise ValueError(
            f"Healthy baseline '{BASELINE_ARRAY7}' was not found."
        )

    baseline_cache: dict[str, pd.DataFrame] = {}
    if has_fo:
        baseline_cache[BASELINE_FEED_OFFSET] = load_scenario_data(
            BASELINE_FEED_OFFSET
        )
        print(
            f"\nFeed-offset baseline: {BASELINE_FEED_OFFSET} "
            f"({len(baseline_cache[BASELINE_FEED_OFFSET])} samples)"
        )
    if has_arr7:
        baseline_cache[BASELINE_ARRAY7] = load_scenario_data(BASELINE_ARRAY7)
        print(
            f"Array-7 baseline: {BASELINE_ARRAY7} "
            f"({len(baseline_cache[BASELINE_ARRAY7])} samples)"
        )

    print(
        f"Null mask: baseline gain >= peak - "
        f"{MASK_DB_BELOW_PEAK:.0f} dB, sin(theta) area weighting"
    )

    feature_rows = []

    for scenario_id in scenarios:
        print(f"\nProcessing: {scenario_id}")

        baseline_id = baseline_for_scenario(scenario_id)
        if baseline_id not in baseline_cache:
            print(
                f"  SKIPPED: no baseline configured for family "
                f"'{scenario_family(scenario_id)}'"
            )
            continue

        baseline_df = baseline_cache[baseline_id]
        print(f"  Reference baseline: {baseline_id}")

        features = extract_features(
            scenario_id=scenario_id,
            baseline_df=baseline_df,
        )
        feature_rows.append(features)

        print(f"  Samples: {features['n_samples']}")
        print(f"  Peak gain: {features['peak_gain']:.6f} dBi")
        print(f"  Peak theta: {features['peak_theta']:.2f}")
        print(
            f"  Grid coverage vs baseline: "
            f"{features['baseline_grid_coverage_pct']:.1f}%"
        )
        print(
            f"  Mean abs difference (masked, weighted): "
            f"{features['baseline_mean_absolute_difference']:.6f} dB"
        )
        print(
            f"  Max abs difference: masked "
            f"{features['baseline_max_absolute_difference']:.4f} dB | "
            f"unmasked "
            f"{features['baseline_unmasked_max_absolute_difference']:.4f} dB"
        )
        if features["baseline_grid_coverage_pct"] < 100.0:
            print(
                "  WARNING: scenario grid does not fully cover the "
                "baseline grid; comparison uses the intersection only."
            )

    feature_df = pd.DataFrame(feature_rows)
    feature_columns = ["scenario_id"] + [
        c for c in feature_df.columns if c != "scenario_id"
    ]
    feature_df = feature_df[feature_columns]

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(OUTPUT_FILE, index=False)

    print()
    print("=" * 70)
    print("REFINED FEATURE EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nScenarios processed: {len(feature_df)}")
    print(f"Features generated: {len(feature_df.columns) - 1}")
    print(f"Output: {OUTPUT_FILE}")
    print("\nFeature dataset shape:")
    print(f"  {feature_df.shape[0]} rows x {feature_df.shape[1]} columns")

    summary_columns = [
        "scenario_id",
        "experiment_family",
        "feed_setting_percent",
        "n_elements_off",
        "known_fault_type",
        "peak_gain",
        "peak_theta",
        "baseline_grid_coverage_pct",
        "baseline_mask_area_fraction_pct",
        "baseline_mean_absolute_difference",
        "baseline_max_absolute_difference",
        "baseline_unmasked_max_absolute_difference",
        "baseline_pct_diff_0_5db",
        "baseline_pct_diff_1db",
    ]
    summary_columns = [c for c in summary_columns if c in feature_df.columns]

    print("\nScenario summary:")
    print(
        feature_df[summary_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 70)
    print("Feature extraction stage complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()