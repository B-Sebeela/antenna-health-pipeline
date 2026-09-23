
"""
Antenna Health Monitoring - Anomaly Detection

This module:
1. Loads engineered scenario features from Parquet.
2. Excludes development/test scenarios.
3. Uses H01_NOM as the healthy baseline.
4. Calculates a proof-of-concept anomaly score.
5. Classifies scenarios as HEALTHY or ANOMALOUS.
6. Saves the results to Parquet.
7. Stores anomaly results in SQLite.
8. Logs the model pipeline run in SQLite.

NOTE:
This is currently an engineering proof-of-concept anomaly detector.
It is NOT a trained machine-learning model.
The scoring method should be validated once a larger scenario library
has been generated.
"""

# =====================================================================
# IMPORTS
# =====================================================================

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# =====================================================================
# PROJECT IMPORT PATH
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.storage.database import (
    initialise_database,
    start_pipeline_run,
    complete_pipeline_run,
    fail_pipeline_run,
    store_anomaly_score,
)


# =====================================================================
# PROJECT PATHS
# =====================================================================

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "scenario_features_refined.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "anomaly_scores.parquet"
)

DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "antenna_health.db"
)


# =====================================================================
# CONFIGURATION
# =====================================================================

BASELINE_SCENARIO = "H01_NOM"

DETECTOR_NAME = "baseline_deviation_poc"

EXCLUDED_SCENARIOS = {
    "farfield_1",
}

ANOMALY_THRESHOLD = 0.10

DEVIATION_FEATURES = [
    "baseline_mean_absolute_difference",
    "baseline_median_absolute_difference",
    "baseline_std_difference",
    "baseline_max_absolute_difference",
    "baseline_pct_diff_0_5db",
    "baseline_pct_diff_1db",
    "baseline_pct_diff_2db",
    "baseline_pct_diff_3db",
]


# =====================================================================
# LOAD FEATURES
# =====================================================================

def load_features():
    """
    Load the refined feature dataset.
    """

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_FILE}"
        )

    df = pd.read_parquet(FEATURE_FILE)

    if "scenario_id" not in df.columns:
        raise ValueError(
            "Feature dataset does not contain 'scenario_id'."
        )

    return df


# =====================================================================
# VALIDATE FEATURES
# =====================================================================

def validate_features(df):
    """
    Check that all required anomaly-detection features exist.
    """

    missing_features = [
        feature
        for feature in DEVIATION_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "The following required deviation features are missing:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_features
            )
        )


# =====================================================================
# CALCULATE ANOMALY SCORES
# =====================================================================

def calculate_anomaly_scores(df):
    """
    Calculate proof-of-concept anomaly scores.

    Each deviation feature is normalised using the minimum and maximum
    values found among the non-baseline health-monitoring scenarios.

    The final anomaly score is the mean of the normalised deviation
    features.
    """

    result = df.copy()

    # -------------------------------------------------------------
    # Check baseline
    # -------------------------------------------------------------

    baseline_rows = result[
        result["scenario_id"] == BASELINE_SCENARIO
    ]

    if baseline_rows.empty:
        raise ValueError(
            f"Healthy baseline '{BASELINE_SCENARIO}' "
            "was not found in the feature dataset."
        )

    # -------------------------------------------------------------
    # Find non-baseline scenarios
    # -------------------------------------------------------------

    monitoring_rows = result[
        ~result["scenario_id"].isin(EXCLUDED_SCENARIOS)
        & (result["scenario_id"] != BASELINE_SCENARIO)
    ]

    if monitoring_rows.empty:
        raise ValueError(
            "No non-baseline health-monitoring scenarios are available."
        )

    # -------------------------------------------------------------
    # Normalise each deviation feature
    # -------------------------------------------------------------

    normalised_features = []

    for feature in DEVIATION_FEATURES:

        values = monitoring_rows[feature].astype(float)

        minimum = values.min()
        maximum = values.max()

        if np.isclose(maximum, minimum):

            result[f"_normalised_{feature}"] = 0.0

        else:

            result[f"_normalised_{feature}"] = (
                result[feature].astype(float) - minimum
            ) / (maximum - minimum)

        normalised_features.append(
            f"_normalised_{feature}"
        )

    # -------------------------------------------------------------
    # Calculate anomaly score
    # -------------------------------------------------------------

    result["anomaly_score"] = (
        result[normalised_features]
        .mean(axis=1)
        .clip(lower=0.0, upper=1.0)
    )

    # -------------------------------------------------------------
    # Force healthy baseline to zero
    # -------------------------------------------------------------

    result.loc[
        result["scenario_id"] == BASELINE_SCENARIO,
        "anomaly_score"
    ] = 0.0

    # -------------------------------------------------------------
    # Assign health status
    # -------------------------------------------------------------

    result["health_status"] = np.where(
        result["anomaly_score"] >= ANOMALY_THRESHOLD,
        "ANOMALOUS",
        "HEALTHY",
    )

    result.loc[
        result["scenario_id"] == BASELINE_SCENARIO,
        "health_status"
    ] = "HEALTHY"

    return result


# =====================================================================
# SAVE PARQUET RESULTS
# =====================================================================

def save_results(df):
    """
    Save anomaly detection results to Parquet.
    """

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_columns = [
        "scenario_id",
        "anomaly_score",
        "health_status",
    ]

    output_df = df[
        [
            column
            for column in output_columns
            if column in df.columns
        ]
    ].copy()

    output_df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    return output_df


# =====================================================================
# STORE RESULTS IN SQLITE
# =====================================================================

def store_results_in_database(result_df):
    """
    Store anomaly detection results in SQLite.

    The existing database service expects a SQLite connection
    (conn) to be supplied to its pipeline functions.
    """

    # -------------------------------------------------------------
    # Initialise database and obtain connection
    # -------------------------------------------------------------

    conn = initialise_database(
        DATABASE_FILE
    )

    run_id = None

    try:

        # ---------------------------------------------------------
        # Start model pipeline run
        # ---------------------------------------------------------

        run_id = start_pipeline_run(
            conn,
            stage="model",
            scenarios_in=len(result_df),
        )

        # ---------------------------------------------------------
        # Store anomaly results
        # ---------------------------------------------------------

        for _, row in result_df.iterrows():

            scenario_id = row["scenario_id"]

            anomaly_score = float(
                row["anomaly_score"]
            )

            health_status = str(
                row["health_status"]
            )

            is_anomaly = (
                health_status == "ANOMALOUS"
            )

            store_anomaly_score(
                conn,
                run_id=run_id,
                scenario_id=scenario_id,
                anomaly_score=anomaly_score,
                is_anomaly=is_anomaly,
                detector_name=DETECTOR_NAME,
                prediction_label=health_status,
            )

        # ---------------------------------------------------------
        # Complete model pipeline run
        # ---------------------------------------------------------

        complete_pipeline_run(
            conn,
            run_id=run_id,
            scenarios_out=len(result_df),
        )

        return run_id

    except Exception as error:

        if run_id is not None:

            fail_pipeline_run(
                conn,
                run_id=run_id,
                error_message=str(error),
            )

        raise

    finally:

        # ---------------------------------------------------------
        # Close database connection
        # ---------------------------------------------------------

        conn.close()


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)
    print("ANTENNA HEALTH MONITORING - ANOMALY DETECTION")
    print("=" * 70)

    # -------------------------------------------------------------
    # Load feature dataset
    # -------------------------------------------------------------

    print()
    print("Loading feature dataset...")

    df = load_features()

    print(f"Scenarios loaded: {len(df)}")
    print(f"Features available: {len(df.columns) - 1}")

    # -------------------------------------------------------------
    # Validate required features
    # -------------------------------------------------------------

    validate_features(df)

    # -------------------------------------------------------------
    # Exclude development/test scenarios
    # -------------------------------------------------------------

    excluded = df[
        df["scenario_id"].isin(EXCLUDED_SCENARIOS)
    ]["scenario_id"].tolist()

    monitoring_df = df[
        ~df["scenario_id"].isin(EXCLUDED_SCENARIOS)
    ].copy()

    print()
    print(
        f"Excluded development/test scenarios: {len(excluded)}"
    )

    for scenario_id in excluded:
        print(f"  - {scenario_id}")

    print()
    print(
        f"Scenarios used for health monitoring: "
        f"{len(monitoring_df)}"
    )

    # -------------------------------------------------------------
    # Check baseline
    # -------------------------------------------------------------

    if BASELINE_SCENARIO not in monitoring_df[
        "scenario_id"
    ].values:

        raise ValueError(
            f"Healthy baseline '{BASELINE_SCENARIO}' "
            "is not available after filtering."
        )

    print()
    print(
        f"Healthy baseline: {BASELINE_SCENARIO}"
    )

    # -------------------------------------------------------------
    # Display deviation features
    # -------------------------------------------------------------

    print()
    print("Deviation features used:")

    for feature in DEVIATION_FEATURES:
        print(f"  - {feature}")

    # -------------------------------------------------------------
    # Calculate anomaly scores
    # -------------------------------------------------------------

    print()
    print("Calculating anomaly scores...")

    results = calculate_anomaly_scores(
        monitoring_df
    )

    # -------------------------------------------------------------
    # Remove temporary normalisation columns
    # -------------------------------------------------------------

    temporary_columns = [
        column
        for column in results.columns
        if column.startswith("_normalised_")
    ]

    results = results.drop(
        columns=temporary_columns
    )

    # -------------------------------------------------------------
    # Save Parquet
    # -------------------------------------------------------------

    output_df = save_results(
        results
    )

    # -------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("ANOMALY DETECTION RESULTS")
    print("=" * 70)

    print(
        output_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Store results in SQLite
    # -------------------------------------------------------------

    print()
    print("Storing anomaly results in SQLite...")

    run_id = store_results_in_database(
        output_df
    )

    print()
    print(
        f"SQLite model pipeline run completed: #{run_id}"
    )

    # -------------------------------------------------------------
    # Output locations
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("ANOMALY DETECTION COMPLETE")
    print("=" * 70)

    print()
    print("Parquet output:")
    print(f"  {OUTPUT_FILE}")

    print()
    print("SQLite database:")
    print(f"  {DATABASE_FILE}")

    print()
    print(
        "NOTE: This detector is currently a "
        "proof-of-concept baseline-deviation method."
    )

    print(
        "A larger scenario library is required before "
        "the anomaly threshold or ML performance can be validated."
    )


# =====================================================================
# SCRIPT ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()

