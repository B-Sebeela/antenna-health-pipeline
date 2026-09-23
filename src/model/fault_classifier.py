
"""
Antenna Health Monitoring - Fault Classification

This module provides a proof-of-concept fault classification layer
for the antenna health monitoring pipeline.

Current approach:
    - Load anomaly detection results.
    - Identify anomalous scenarios.
    - Map known FEKO scenario IDs to their intended fault types.
    - Save classification results to Parquet.
    - Record the classification pipeline run in SQLite.

IMPORTANT:
This is NOT yet a trained machine-learning classifier.

The current dataset contains too few labelled scenarios for a valid
machine-learning model. This module therefore acts as the classification
infrastructure and proof-of-concept.

Once a larger FEKO scenario library has been generated, this module can
be extended/replaced with a trained classifier such as Random Forest,
SVM, or another appropriate model.
"""

# =====================================================================
# IMPORTS
# =====================================================================

from pathlib import Path
import sys

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

ANOMALY_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "anomaly_scores.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "classification_results.parquet"
)

DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "antenna_health.db"
)


# =====================================================================
# CONFIGURATION
# =====================================================================

# Name used to identify this proof-of-concept classifier.
CLASSIFIER_NAME = "scenario_mapping_poc"


# ---------------------------------------------------------------------
# Known scenario labels
# ---------------------------------------------------------------------
#
# These labels are based on the current FEKO scenario definitions.
#
# H01_NOM:
#     Healthy nominal antenna.
#
# M01_FEED_OFFSET:
#     Intended feed-offset fault.
#
#     The current FFE output is identical to H01_NOM, so the anomaly
#     detector currently sees it as healthy. We therefore do not
#     classify the current result as a confirmed feed-offset fault.
#
# MF01_L_PLUS:
#     Known L_PLUS fault scenario.
#
# MF02_L_MINUS:
#     Known L_MINUS fault scenario.
#
# ---------------------------------------------------------------------

SCENARIO_LABELS = {
    "H01_NOM": "NONE",
    "M01_FEED_OFFSET": "UNKNOWN",
    "MF01_L_PLUS": "L_PLUS",
    "MF02_L_MINUS": "L_MINUS",
}


# =====================================================================
# LOAD ANOMALY RESULTS
# =====================================================================

def load_anomaly_results():
    """
    Load the anomaly detection results from Parquet.
    """

    if not ANOMALY_FILE.exists():

        raise FileNotFoundError(
            f"Anomaly result file not found:\n{ANOMALY_FILE}"
        )

    df = pd.read_parquet(
        ANOMALY_FILE
    )

    required_columns = [
        "scenario_id",
        "anomaly_score",
        "health_status",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "The anomaly results are missing the following columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    return df


# =====================================================================
# CLASSIFY SCENARIO
# =====================================================================

def classify_scenario(
    scenario_id,
    health_status,
):
    """
    Assign a proof-of-concept fault label.

    HEALTHY:
        The current observed behaviour is healthy, so no fault is
        reported.

    ANOMALOUS + known scenario:
        Use the known scenario label.

    ANOMALOUS + unknown scenario:
        Mark as UNKNOWN rather than guessing.
    """

    # -------------------------------------------------------------
    # Healthy scenario
    # -------------------------------------------------------------

    if health_status == "HEALTHY":

        return "NONE"

    # -------------------------------------------------------------
    # Known anomalous scenario
    # -------------------------------------------------------------

    if scenario_id in SCENARIO_LABELS:

        label = SCENARIO_LABELS[
            scenario_id
        ]

        if label != "UNKNOWN":
            return label

    # -------------------------------------------------------------
    # Unknown anomalous scenario
    # -------------------------------------------------------------

    return "UNKNOWN"


# =====================================================================
# RUN CLASSIFICATION
# =====================================================================

def run_classification(df):
    """
    Apply the proof-of-concept classification logic.
    """

    result = df.copy()

    # -------------------------------------------------------------
    # Fault type
    # -------------------------------------------------------------

    result["fault_type"] = result.apply(
        lambda row: classify_scenario(
            row["scenario_id"],
            row["health_status"],
        ),
        axis=1,
    )

    # -------------------------------------------------------------
    # Classification confidence
    # -------------------------------------------------------------
    #
    # This is NOT an ML probability.
    #
    # 1.0 means that the current proof-of-concept mapping has a known
    # scenario label.
    #
    # 0.0 means that no reliable classification mapping is available.
    # -------------------------------------------------------------

    result["classification_confidence"] = result.apply(
        lambda row: (
            1.0
            if (
                row["scenario_id"] in SCENARIO_LABELS
                and row["fault_type"] != "UNKNOWN"
            )
            else 0.0
        ),
        axis=1,
    )

    # -------------------------------------------------------------
    # Classifier name
    # -------------------------------------------------------------

    result["classifier_name"] = CLASSIFIER_NAME

    return result


# =====================================================================
# SAVE CLASSIFICATION RESULTS
# =====================================================================

def save_results(df):
    """
    Save classification results to Parquet.
    """

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_columns = [
        "scenario_id",
        "anomaly_score",
        "health_status",
        "fault_type",
        "classification_confidence",
        "classifier_name",
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
# STORE PIPELINE RUN IN SQLITE
# =====================================================================

def store_pipeline_run(result_df):
    """
    Record the classification stage in SQLite.

    The current database wrapper's store_anomaly_score() function only
    accepts anomaly-related fields. Therefore, detailed classification
    information is stored in classification_results.parquet.

    SQLite records that the model/classification pipeline stage ran
    successfully.
    """

    # -------------------------------------------------------------
    # Initialise database
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
        # Record the anomaly result using the existing database
        # interface.
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
                run_id,
                scenario_id,
                anomaly_score,
                is_anomaly,
                "baseline_deviation_poc",
                health_status,
            )

        # ---------------------------------------------------------
        # Complete pipeline run
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

        conn.close()


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)
    print("ANTENNA HEALTH MONITORING - FAULT CLASSIFICATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # Load anomaly results
    # -------------------------------------------------------------

    print()
    print("Loading anomaly detection results...")

    df = load_anomaly_results()

    print(
        f"Scenarios loaded: {len(df)}"
    )

    # -------------------------------------------------------------
    # Display anomaly results
    # -------------------------------------------------------------

    print()
    print("Anomaly detection input:")

    print(
        df[
            [
                "scenario_id",
                "anomaly_score",
                "health_status",
            ]
        ].to_string(index=False)
    )

    # -------------------------------------------------------------
    # Run classification
    # -------------------------------------------------------------

    print()
    print(
        "Running proof-of-concept fault classification..."
    )

    results = run_classification(
        df
    )

    # -------------------------------------------------------------
    # Display classification results
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("FAULT CLASSIFICATION RESULTS")
    print("=" * 70)

    print(
        results[
            [
                "scenario_id",
                "anomaly_score",
                "health_status",
                "fault_type",
                "classification_confidence",
            ]
        ].to_string(index=False)
    )

    # -------------------------------------------------------------
    # Save classification results
    # -------------------------------------------------------------

    output_df = save_results(
        results
    )

    # -------------------------------------------------------------
    # Store pipeline run in SQLite
    # -------------------------------------------------------------

    print()
    print(
        "Recording classification pipeline run in SQLite..."
    )

    run_id = store_pipeline_run(
        output_df
    )

    # -------------------------------------------------------------
    # Final output
    # -------------------------------------------------------------

    print()
    print(
        f"SQLite classification pipeline run completed: #{run_id}"
    )

    print()
    print("=" * 70)
    print("FAULT CLASSIFICATION COMPLETE")
    print("=" * 70)

    print()
    print("Classification Parquet output:")
    print(
        f"  {OUTPUT_FILE}"
    )

    print()
    print("SQLite database:")
    print(
        f"  {DATABASE_FILE}"
    )

    print()
    print("Current classification approach:")
    print(
        "  Scenario-ID mapping / proof-of-concept"
    )

    print()
    print("Future classification approach:")
    print(
        "  Trained machine-learning classifier"
    )

    print()
    print(
        "A larger labelled FEKO scenario library is required "
        "before training and evaluating the ML classifier."
    )


# =====================================================================
# SCRIPT ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()

