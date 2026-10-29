"""
Antenna Health Monitoring Pipeline
Reference-Based Anomaly Detection and Fault Classification

Controlled datasets:
  1) Feed-offset:
        H01_DIPOLE_FEED_50
        FO_DIPOLE_FEED_10 ... FO_DIPOLE_FEED_80
  2) 7-element array element failure:
        ARR7_BASELINE_ALL_ON
        ARR7_FAIL_1 / 2 / 3 / 5
        ARR7_ONLY_1_ON

The detector does not use ground-truth labels to calculate the anomaly
score. Ground-truth fields are only used for validation.

Score components (provisional weights, renormalised):
  masked mean |Δ|, % area > 0.5 dB, % area > 1 dB, peak-elevation tilt.

score_part_* columns are saved for the dashboard.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT / "data" / "features" / "scenario_features_refined.parquet"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_FILE = OUTPUT_DIR / "anomaly_results.parquet"


# ============================================================
# CONFIGURATION (PROVISIONAL until noise-floor study)
# ============================================================

BASELINE_FEED_OFFSET = "H01_DIPOLE_FEED_50"
BASELINE_ARRAY7 = "ARR7_BASELINE_ALL_ON"

NOMINAL_FEED_POSITION = 50.0
NOMINAL_PEAK_THETA_DEG = 90.0

HEALTH_MEAN_ABS_THRESHOLD_DB = 0.02

SCALE_MEAN_ABS_DB = 1.0
SCALE_PCT_0_5_DB = 50.0
SCALE_PCT_1_DB = 25.0
SCALE_TILT_DEG = 5.0

WEIGHT_MEAN = 8.0 / 15.0
WEIGHT_PCT_0_5 = 4.0 / 15.0
WEIGHT_PCT_1 = 2.0 / 15.0
WEIGHT_TILT = 1.0 / 15.0

SEVERITY_LOW_BELOW = 15.0
SEVERITY_MODERATE_BELOW = 40.0

EVIDENCE_MEAN_ABS_DB = 0.1
EVIDENCE_PCT_0_5 = 1.0
EVIDENCE_PCT_1 = 1.0
EVIDENCE_TILT_DEG = 1.0


# ============================================================
# LOAD DATA
# ============================================================

def load_features():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Feature file not found:\n{FEATURE_FILE}")

    df = pd.read_parquet(FEATURE_FILE)

    required = [
        "baseline_mean_absolute_difference",
        "baseline_pct_diff_0_5db",
        "baseline_pct_diff_1db",
        "baseline_mask_db_below_peak",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "The feature file was produced by a previous feature-extraction "
            f"version (missing columns: {missing}). Re-run refined features."
        )

    return df


def load_previous_results():
    if not OUTPUT_FILE.exists():
        return None
    try:
        return pd.read_parquet(OUTPUT_FILE)
    except Exception:
        return None


# ============================================================
# SELECT CONTROLLED DATASETS
# ============================================================

def select_controlled_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep feed-offset experiment + array-7 element-failure experiment.
    Exclude exploratory / mismatched-grid leftovers.
    """
    mask_fo = (
        df["scenario_id"].eq(BASELINE_FEED_OFFSET)
        | df["scenario_id"].str.startswith("FO_DIPOLE_FEED_")
    )
    mask_arr7 = df["scenario_id"].str.startswith("ARR7_")

    result = df.loc[mask_fo | mask_arr7].copy()

    n_fo = int(mask_fo.sum())
    n_arr7 = int(mask_arr7.sum())

    if n_fo != 10:
        raise ValueError(
            f"Expected 10 feed-offset controlled scenarios, found {n_fo}."
        )
    if n_arr7 != 6:
        raise ValueError(
            f"Expected 6 array-7 controlled scenarios, found {n_arr7}."
        )

    return result


def scenario_family(scenario_id: str) -> str:
    if str(scenario_id).startswith("ARR7_"):
        return "array7"
    return "feed_offset"


# ============================================================
# BASELINE-RELATIVE FEATURES
# ============================================================

def add_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Peak-elevation deviation relative to the family baseline peak theta.
    """
    df = df.copy()

    # Per-family baseline peak_theta
    peak_ref = {}
    for fam, baseline_id in (
        ("feed_offset", BASELINE_FEED_OFFSET),
        ("array7", BASELINE_ARRAY7),
    ):
        row = df.loc[df["scenario_id"] == baseline_id]
        if row.empty:
            raise ValueError(f"Baseline '{baseline_id}' not found in controlled set.")
        peak_ref[fam] = float(row.iloc[0]["peak_theta"])

    families = df["scenario_id"].map(scenario_family)
    ref_theta = families.map(peak_ref)

    df["peak_theta_deviation_deg"] = (df["peak_theta"] - ref_theta).abs()

    # Peak gain deviation vs family baseline (audit / optional)
    gain_ref = {}
    for fam, baseline_id in (
        ("feed_offset", BASELINE_FEED_OFFSET),
        ("array7", BASELINE_ARRAY7),
    ):
        row = df.loc[df["scenario_id"] == baseline_id]
        gain_ref[fam] = float(row.iloc[0]["peak_gain"])

    ref_gain = families.map(gain_ref)
    df["peak_gain_deviation_db"] = (df["peak_gain"] - ref_gain).abs()

    df["experiment_family"] = families

    return df


# ============================================================
# ANOMALY SCORE
# ============================================================

def calculate_anomaly_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    mean_component = np.clip(
        df["baseline_mean_absolute_difference"] / SCALE_MEAN_ABS_DB, 0.0, 1.0
    )
    pct_05_component = np.clip(
        df["baseline_pct_diff_0_5db"] / SCALE_PCT_0_5_DB, 0.0, 1.0
    )
    pct_1_component = np.clip(
        df["baseline_pct_diff_1db"] / SCALE_PCT_1_DB, 0.0, 1.0
    )
    tilt_component = np.clip(
        df["peak_theta_deviation_deg"] / SCALE_TILT_DEG, 0.0, 1.0
    )

    df["score_part_mean"] = 100.0 * WEIGHT_MEAN * mean_component
    df["score_part_pct_0_5"] = 100.0 * WEIGHT_PCT_0_5 * pct_05_component
    df["score_part_pct_1"] = 100.0 * WEIGHT_PCT_1 * pct_1_component
    df["score_part_tilt"] = 100.0 * WEIGHT_TILT * tilt_component

    df["anomaly_score_100"] = np.clip(
        df["score_part_mean"]
        + df["score_part_pct_0_5"]
        + df["score_part_pct_1"]
        + df["score_part_tilt"],
        0.0,
        100.0,
    )
    df["anomaly_score"] = df["anomaly_score_100"] / 100.0

    return df


# ============================================================
# HEALTH / SEVERITY
# ============================================================

def determine_health_status(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    anomaly_mask = (
        df["baseline_mean_absolute_difference"] > HEALTH_MEAN_ABS_THRESHOLD_DB
    )
    df["health_status"] = np.where(
        anomaly_mask, "ANOMALY_DETECTED", "HEALTHY"
    )
    return df


def determine_severity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def severity(row):
        if row["health_status"] == "HEALTHY":
            return "NONE"
        score = row["anomaly_score_100"]
        if score < SEVERITY_LOW_BELOW:
            return "LOW"
        if score < SEVERITY_MODERATE_BELOW:
            return "MODERATE"
        return "HIGH"

    df["severity"] = df.apply(severity, axis=1)
    return df


# ============================================================
# FAULT CLASSIFICATION
# ============================================================

def classify_fault(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based labels for the two controlled fault families.

    Feed-offset: type FEED_OFFSET; direction from peak-theta vs 90°.
    Array-7:     type ELEMENT_FAILURE; condition from scenario_id map.
    """
    df = df.copy()

    arr7_condition = {
        "ARR7_BASELINE_ALL_ON": "NOMINAL",
        "ARR7_FAIL_1": "N_OFF_1",
        "ARR7_FAIL_2": "N_OFF_2",
        "ARR7_FAIL_3": "N_OFF_3",
        "ARR7_FAIL_5": "N_OFF_5",
        "ARR7_ONLY_1_ON": "ONLY_ONE_ACTIVE",
    }

    fault_types = []
    directions = []

    for _, row in df.iterrows():
        sid = row["scenario_id"]
        family = scenario_family(sid)

        if row["health_status"] == "HEALTHY":
            fault_types.append("NONE")
            directions.append("NOMINAL")
            continue

        if family == "array7":
            fault_types.append("ELEMENT_FAILURE")
            directions.append(arr7_condition.get(sid, "UNKNOWN"))
        else:
            fault_types.append("FEED_OFFSET")
            theta_shift = float(row["peak_theta"]) - NOMINAL_PEAK_THETA_DEG
            if theta_shift < 0:
                directions.append("LOWER_THAN_REFERENCE")
            elif theta_shift > 0:
                directions.append("HIGHER_THAN_REFERENCE")
            else:
                directions.append("UNKNOWN")

    df["predicted_fault_type"] = fault_types
    df["predicted_condition"] = directions
    return df


# ============================================================
# DETECTION EVIDENCE
# ============================================================

def calculate_confidence(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    confidence_values = []

    for _, row in df.iterrows():
        if row["health_status"] == "HEALTHY":
            confidence_values.append(1.0)
            continue

        indicators = 0
        if row["baseline_mean_absolute_difference"] >= EVIDENCE_MEAN_ABS_DB:
            indicators += 1
        if row["baseline_pct_diff_0_5db"] >= EVIDENCE_PCT_0_5:
            indicators += 1
        if row["baseline_pct_diff_1db"] >= EVIDENCE_PCT_1:
            indicators += 1
        if row["peak_theta_deviation_deg"] >= EVIDENCE_TILT_DEG:
            indicators += 1

        confidence_values.append(indicators / 4.0)

    df["diagnostic_confidence"] = confidence_values
    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["health_correct"] = (
        ((df["known_health"] == "HEALTHY") & (df["health_status"] == "HEALTHY"))
        | (
            (df["known_health"] == "FAULT")
            & (df["health_status"] == "ANOMALY_DETECTED")
        )
    )

    df["fault_type_correct"] = np.where(
        df["known_health"] == "HEALTHY",
        df["predicted_fault_type"] == "NONE",
        df["predicted_fault_type"] == df["known_fault_type"],
    )

    # For array family, known_condition is N_OFF_* / ONLY_ONE_ACTIVE;
    # for feed-offset, LOWER/HIGHER_THAN_REFERENCE.
    df["direction_correct"] = np.where(
        df["known_health"] == "HEALTHY",
        True,
        df["predicted_condition"] == df["known_condition"],
    )

    return df


# ============================================================
# DISPLAY / SAVE
# ============================================================

def display_results(df: pd.DataFrame, previous=None) -> None:
    columns = [
        "scenario_id",
        "experiment_family",
        "feed_setting_percent",
        "health_status",
        "severity",
        "anomaly_score_100",
        "score_part_mean",
        "score_part_pct_0_5",
        "score_part_pct_1",
        "score_part_tilt",
        "predicted_fault_type",
        "predicted_condition",
        "diagnostic_confidence",
        "health_correct",
        "direction_correct",
    ]
    columns = [c for c in columns if c in df.columns]

    print("\n" + "=" * 110)
    print("FINAL ANOMALY DETECTION RESULTS")
    print("=" * 110)
    print(df[columns].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n" + "=" * 110)
    print("VALIDATION (against known simulation labels)")
    print("=" * 110)

    for fam in sorted(df["experiment_family"].unique()):
        sub = df.loc[df["experiment_family"] == fam]
        print(f"\n[{fam}] n={len(sub)}")
        print(f"  Health detection accuracy:     {sub['health_correct'].mean()*100:.1f}%")
        print(f"  Fault classification accuracy: {sub['fault_type_correct'].mean()*100:.1f}%")
        print(f"  Condition/direction accuracy:  {sub['direction_correct'].mean()*100:.1f}%")

    print(f"\n[overall] n={len(df)}")
    print(f"  Health detection accuracy:     {df['health_correct'].mean()*100:.1f}%")
    print(f"  Fault classification accuracy: {df['fault_type_correct'].mean()*100:.1f}%")
    print(f"  Condition/direction accuracy:  {df['direction_correct'].mean()*100:.1f}%")


def save_results(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)
    print("\n" + "=" * 110)
    print("RESULTS SAVED")
    print("=" * 110)
    print(f"Output: {OUTPUT_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 110)
    print("ANTENNA HEALTH MONITORING PIPELINE")
    print("ANOMALY DETECTION + FAULT CLASSIFICATION")
    print("=" * 110)

    print("\nLoading feature dataset...")
    df = load_features()
    previous = load_previous_results()
    print(f"Feature rows available: {len(df)}")

    df = select_controlled_dataset(df)
    print(f"Controlled scenarios: {len(df)}")

    df = add_baseline_features(df)
    df = calculate_anomaly_score(df)
    df = determine_health_status(df)
    df = determine_severity(df)
    df = classify_fault(df)
    df = calculate_confidence(df)
    df = validate_results(df)

    # Stable sort: family, then feed % (FO), then scenario id
    df = df.sort_values(
        by=["experiment_family", "feed_setting_percent", "scenario_id"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    display_results(df, previous)
    save_results(df)

    print("\n" + "=" * 110)
    print("ANOMALY DETECTION STAGE COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()