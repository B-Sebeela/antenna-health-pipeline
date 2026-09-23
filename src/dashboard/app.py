
"""
app.py
------

Streamlit dashboard for the Antenna Health Monitoring Pipeline.

Current Milestone
-----------------
Milestone 2 — End-to-End Health Monitoring Prototype

The dashboard visualises information produced by the pipeline:

    FEKO .ffe
        |
        v
    FFE ingestion
        |
        +-------------> SQLite
        |
        +-------------> Far-field Parquet
                              |
                              v
                       Feature Extraction
                              |
                              v
                       Anomaly Detection
                              |
                              v
                     Fault Classification
                              |
                              v
                       Streamlit Dashboard

Current classification is a proof-of-concept scenario mapping.
It is NOT yet a trained machine-learning classifier.

The dashboard does not perform model training or modify the
underlying pipeline.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


# ===========================================================================
# PROJECT PATH
# ===========================================================================

# app.py:
#
# antenna-health-pipeline/
# └── src/
#     └── dashboard/
#         └── app.py
#
# parents[2] = antenna-health-pipeline/

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ===========================================================================
# PROJECT IMPORTS
# ===========================================================================

from src.storage.database import (
    initialise_database,
    fetch_scenarios,
    fetch_recent_runs,
    get_database_summary,
)

from src.storage.parquet import (
    load_parquet,
)

from src.preprocess.visualisation import (
    plot_radiation_pattern,
    plot_far_field_heatmap,
)


# ===========================================================================
# PATHS
# ===========================================================================

DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
)

FEATURE_DIRECTORY = (
    DATA_DIRECTORY
    / "features"
)

PROCESSED_DIRECTORY = (
    DATA_DIRECTORY
    / "processed"
)

OUTPUT_DIRECTORY = (
    DATA_DIRECTORY
    / "outputs"
)

DATABASE_PATH = (
    DATA_DIRECTORY
    / "antenna_health.db"
)

CLASSIFICATION_PATH = (
    FEATURE_DIRECTORY
    / "classification_results.parquet"
)

ANOMALY_PATH = (
    FEATURE_DIRECTORY
    / "anomaly_scores.parquet"
)

REFINED_FEATURE_PATH = (
    FEATURE_DIRECTORY
    / "scenario_features_refined.parquet"
)


# ===========================================================================
# PAGE CONFIGURATION
# ===========================================================================

st.set_page_config(
    page_title="Antenna Health Monitoring",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===========================================================================
# CUSTOM STYLING
# ===========================================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #666666;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# DATABASE CONNECTION
# ===========================================================================

# Do NOT cache the SQLite connection.
#
# Streamlit may rerun the application using a different thread.
# SQLite connections are thread-specific by default.

conn = initialise_database(
    DATABASE_PATH
)


# ===========================================================================
# DATA LOADING FUNCTIONS
# ===========================================================================

def load_database_summary() -> dict:
    """
    Load the current database summary.
    """

    return get_database_summary(
        conn
    )


def load_scenarios() -> list[dict]:
    """
    Load valid scenarios from SQLite.
    """

    return fetch_scenarios(
        conn=conn,
        valid_only=True,
    )


def load_pipeline_runs() -> list[dict]:
    """
    Load recent pipeline activity.
    """

    return fetch_recent_runs(
        conn=conn,
        limit=20,
    )


@st.cache_data
def load_far_field_data(
    parquet_path: str,
) -> pd.DataFrame:
    """
    Load a far-field Parquet dataset.
    """

    return load_parquet(
        parquet_path
    )


@st.cache_data
def load_classification_results() -> pd.DataFrame:
    """
    Load the latest proof-of-concept classification results.
    """

    if not CLASSIFICATION_PATH.exists():
        return pd.DataFrame()

    return pd.read_parquet(
        CLASSIFICATION_PATH
    )


@st.cache_data
def load_anomaly_results() -> pd.DataFrame:
    """
    Load anomaly detection results.
    """

    if not ANOMALY_PATH.exists():
        return pd.DataFrame()

    return pd.read_parquet(
        ANOMALY_PATH
    )


@st.cache_data
def load_refined_features() -> pd.DataFrame:
    """
    Load the refined feature dataset.
    """

    if not REFINED_FEATURE_PATH.exists():
        return pd.DataFrame()

    return pd.read_parquet(
        REFINED_FEATURE_PATH
    )


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def get_scenario_result(
    classification_df: pd.DataFrame,
    scenario_id: str,
) -> pd.Series | None:
    """
    Return the classification result for a scenario.
    """

    if classification_df.empty:
        return None

    matches = classification_df[
        classification_df["scenario_id"]
        == scenario_id
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def get_feature_result(
    feature_df: pd.DataFrame,
    scenario_id: str,
) -> pd.Series | None:
    """
    Return the refined feature row for a scenario.
    """

    if feature_df.empty:
        return None

    matches = feature_df[
        feature_df["scenario_id"]
        == scenario_id
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def format_number(
    value,
    decimals: int = 3,
) -> str:
    """
    Safely format a numerical value.
    """

    if pd.isna(value):
        return "N/A"

    return f"{float(value):.{decimals}f}"


# ===========================================================================
# HEADER
# ===========================================================================

st.markdown(
    """
    <div class="main-title">
        📡 Antenna Health Monitoring Pipeline
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        FEKO simulation data ingestion, processing, health monitoring
        and visualisation
    </div>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# SIDEBAR
# ===========================================================================

st.sidebar.title(
    "Pipeline Controls"
)

st.sidebar.markdown(
    """
    ### Current Milestone

    **Milestone 2 — Health Monitoring Prototype**

    Current pipeline:

    - FEKO `.ffe` ingestion
    - SQLite metadata storage
    - Parquet data storage
    - Feature extraction
    - Baseline-deviation anomaly detection
    - Proof-of-concept fault classification
    - Radiation-pattern visualisation
    """
)

st.sidebar.divider()

if st.sidebar.button(
    "🔄 Refresh Dashboard",
    width="stretch",
):

    st.cache_data.clear()

    st.rerun()


# ===========================================================================
# LOAD DATA
# ===========================================================================

summary = load_database_summary()

scenarios = load_scenarios()

pipeline_runs = load_pipeline_runs()

classification_df = (
    load_classification_results()
)

anomaly_df = (
    load_anomaly_results()
)

feature_df = (
    load_refined_features()
)


# ===========================================================================
# PIPELINE OVERVIEW
# ===========================================================================

st.markdown(
    """
    <div class="section-title">
        Pipeline Overview
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        label="Scenarios",
        value=summary["n_scenarios"],
    )

with col2:

    st.metric(
        label="Valid Scenarios",
        value=summary["n_valid_scenarios"],
    )

with col3:

    st.metric(
        label="Pipeline Runs",
        value=summary["n_runs"],
    )

with col4:

    st.metric(
        label="Stored Scores",
        value=summary["n_scores"],
    )


# ===========================================================================
# PIPELINE STATUS
# ===========================================================================

st.markdown(
    """
    <div class="section-title">
        Pipeline Status
    </div>
    """,
    unsafe_allow_html=True,
)

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:

    st.success(
        "✓ FEKO ingestion available"
    )

with status_col2:

    st.success(
        "✓ SQLite storage available"
    )

with status_col3:

    st.success(
        "✓ Parquet storage available"
    )


status_col4, status_col5, status_col6 = st.columns(3)

with status_col4:

    if not feature_df.empty:

        st.success(
            "✓ Feature extraction available"
        )

    else:

        st.warning(
            "⚠ Feature dataset unavailable"
        )


with status_col5:

    if not anomaly_df.empty:

        st.success(
            "✓ Anomaly detection available"
        )

    else:

        st.warning(
            "⚠ Anomaly results unavailable"
        )


with status_col6:

    if not classification_df.empty:

        st.success(
            "✓ Fault classification available"
        )

    else:

        st.warning(
            "⚠ Classification results unavailable"
        )


# ===========================================================================
# CURRENT ML STATUS
# ===========================================================================

if classification_df.empty:

    st.info(
        """
        **Model status**

        The anomaly detection and classification result files are
        not currently available.
        """
    )

else:

    st.info(
        """
        **Model status**

        Anomaly detection is currently operating against the
        **H01_NOM healthy baseline**.

        Fault classification is currently a **proof-of-concept
        scenario mapping**, not a trained machine-learning model.
        """
    )


# ===========================================================================
# SCENARIO INFORMATION
# ===========================================================================

st.markdown(
    """
    <div class="section-title">
        Scenario Information
    </div>
    """,
    unsafe_allow_html=True,
)

if not scenarios:

    st.warning(
        "No valid scenarios are currently stored in the database."
    )

else:

    # -----------------------------------------------------------------------
    # Scenario selector
    # -----------------------------------------------------------------------

    scenario_ids = [
        scenario["scenario_id"]
        for scenario in scenarios
    ]

    selected_scenario_id = st.selectbox(
        "Select a scenario",
        scenario_ids,
    )

    selected_scenario = next(
        scenario
        for scenario in scenarios
        if scenario["scenario_id"]
        == selected_scenario_id
    )


    # =======================================================================
    # HEALTH MONITORING RESULT
    # =======================================================================

    selected_result = get_scenario_result(
        classification_df,
        selected_scenario_id,
    )

    selected_features = get_feature_result(
        feature_df,
        selected_scenario_id,
    )


    st.markdown(
        "### Health Monitoring Result"
    )


    if selected_result is not None:

        health_status = str(
            selected_result[
                "health_status"
            ]
        )

        anomaly_score = float(
            selected_result[
                "anomaly_score"
            ]
        )

        fault_type = str(
            selected_result[
                "fault_type"
            ]
        )

        confidence = float(
            selected_result[
                "classification_confidence"
            ]
        )


        result_col1, result_col2, result_col3, result_col4 = (
            st.columns(4)
        )


        with result_col1:

            st.metric(
                "Health Status",
                health_status,
            )


        with result_col2:

            st.metric(
                "Anomaly Score",
                f"{anomaly_score:.3f}",
            )


        with result_col3:

            st.metric(
                "Detected Fault",
                fault_type,
            )


        with result_col4:

            st.metric(
                "Classification Confidence",
                f"{confidence:.0%}",
            )


        # ---------------------------------------------------------------
        # Status explanation
        # ---------------------------------------------------------------

        if health_status == "HEALTHY":

            st.success(
                """
                The current far-field data is not considered anomalous
                relative to the H01_NOM healthy baseline.
                """
            )

        else:

            st.warning(
                """
                The current far-field data differs from the healthy
                H01_NOM baseline and has been flagged as anomalous.
                """
            )


    else:

        st.info(
            "No classification result is available for this scenario."
        )


    # =======================================================================
    # SCENARIO METADATA
    # =======================================================================

    st.markdown(
        "### Scenario Metadata"
    )


    metadata_col1, metadata_col2, metadata_col3 = st.columns(3)


    with metadata_col1:

        st.markdown(
            "**Scenario**"
        )

        st.write(
            selected_scenario.get(
                "scenario_id",
                "N/A",
            )
        )

        st.markdown(
            "**Source File**"
        )

        st.write(
            selected_scenario.get(
                "source_file",
                "N/A",
            )
        )


    with metadata_col2:

        frequency_hz = selected_scenario.get(
            "frequency_hz"
        )

        if frequency_hz is not None:

            frequency_mhz = (
                frequency_hz
                / 1_000_000
            )

            st.metric(
                "Frequency",
                f"{frequency_mhz:.0f} MHz",
            )

        else:

            st.metric(
                "Frequency",
                "N/A",
            )


        st.metric(
            "Samples",
            selected_scenario.get(
                "n_samples",
                "N/A",
            ),
        )


    with metadata_col3:

        st.metric(
            "Theta Samples",
            selected_scenario.get(
                "n_theta",
                "N/A",
            ),
        )

        st.metric(
            "Phi Samples",
            selected_scenario.get(
                "n_phi",
                "N/A",
            ),
        )


    # =======================================================================
    # DETAILED METADATA
    # =======================================================================

    with st.expander(
        "View detailed scenario metadata"
    ):

        metadata = {

            "Scenario ID":
                selected_scenario.get(
                    "scenario_id"
                ),

            "Source File":
                selected_scenario.get(
                    "source_file"
                ),

            "Frequency (Hz)":
                selected_scenario.get(
                    "frequency_hz"
                ),

            "Theta Samples":
                selected_scenario.get(
                    "n_theta"
                ),

            "Phi Samples":
                selected_scenario.get(
                    "n_phi"
                ),

            "Total Samples":
                selected_scenario.get(
                    "n_samples"
                ),

            "Coordinate System":
                selected_scenario.get(
                    "coordinate_system"
                ),

            "Result Type":
                selected_scenario.get(
                    "result_type"
                ),

            "Number of Elements":
                selected_scenario.get(
                    "n_elements"
                ),

            "Number of Failures":
                selected_scenario.get(
                    "n_failures"
                ),

            "Parquet Path":
                selected_scenario.get(
                    "parquet_path"
                ),

            "Valid":
                bool(
                    selected_scenario.get(
                        "is_valid",
                        0,
                    )
                ),
        }

        st.json(
            metadata
        )


    # =======================================================================
    # FEATURE SUMMARY
    # =======================================================================

    if selected_features is not None:

        st.markdown(
            "### Feature and Baseline Comparison"
        )

        feature_col1, feature_col2, feature_col3, feature_col4 = (
            st.columns(4)
        )

        with feature_col1:

            st.metric(
                "Mean Absolute Difference",
                (
                    f"{format_number(selected_features.get('baseline_mean_absolute_difference'))}"
                    " dB"
                ),
            )

        with feature_col2:

            st.metric(
                "Median Absolute Difference",
                (
                    f"{format_number(selected_features.get('baseline_median_absolute_difference'))}"
                    " dB"
                ),
            )

        with feature_col3:

            st.metric(
                "Maximum Difference",
                (
                    f"{format_number(selected_features.get('baseline_max_absolute_difference'))}"
                    " dB"
                ),
            )

        with feature_col4:

            st.metric(
                "Samples ≥ 1 dB",
                (
                    f"{format_number(selected_features.get('baseline_pct_diff_1db'), 1)}%"
                ),
            )


        feature_col5, feature_col6, feature_col7, feature_col8 = (
            st.columns(4)
        )

        with feature_col5:

            st.metric(
                "Gain Mean",
                (
                    f"{format_number(selected_features.get('gain_mean'))}"
                    " dB"
                ),
            )

        with feature_col6:

            st.metric(
                "Gain Std. Dev.",
                (
                    f"{format_number(selected_features.get('gain_std'))}"
                    " dB"
                ),
            )

        with feature_col7:

            st.metric(
                "Peak Gain",
                (
                    f"{format_number(selected_features.get('peak_gain'))}"
                    " dBi"
                ),
            )

        with feature_col8:

            st.metric(
                "Samples ≥ 2 dB",
                (
                    f"{format_number(selected_features.get('baseline_pct_diff_2db'), 1)}%"
                ),
            )


        with st.expander(
            "View all extracted features"
        ):

            feature_display = (
                selected_features
                .to_frame("Value")
            )

            feature_display.index.name = (
                "Feature"
            )

            st.dataframe(
                feature_display,
                width="stretch",
            )


    # =======================================================================
    # ANOMALY SCORE COMPARISON
    # =======================================================================

    st.divider()

    st.markdown(
        "### Anomaly Score Comparison"
    )


    if not classification_df.empty:

        score_dataframe = classification_df[
            [
                "scenario_id",
                "anomaly_score",
            ]
        ].copy()

        score_dataframe = (
            score_dataframe
            .set_index("scenario_id")
        )

        st.bar_chart(
            score_dataframe,
            y="anomaly_score",
        )

        st.caption(
            "Higher scores indicate greater deviation from the "
            "H01_NOM healthy baseline under the current "
            "proof-of-concept detector."
        )


    # =======================================================================
    # FAR-FIELD PARQUET
    # =======================================================================

    parquet_path_value = (
        selected_scenario.get(
            "parquet_path"
        )
    )


    if parquet_path_value:

        parquet_path = Path(
            parquet_path_value
        )


        # -------------------------------------------------------------------
        # Convert relative path to project-relative path.
        # -------------------------------------------------------------------

        if not parquet_path.is_absolute():

            parquet_path = (
                PROJECT_ROOT
                / parquet_path
            )


        # -------------------------------------------------------------------
        # Verify Parquet exists.
        # -------------------------------------------------------------------

        if parquet_path.exists():

            st.markdown(
                """
                <div class="section-title">
                    Antenna Visualisation
                </div>
                """,
                unsafe_allow_html=True,
            )


            # ===============================================================
            # LOAD PARQUET
            # ===============================================================

            try:

                dataframe = load_far_field_data(
                    str(
                        parquet_path
                    )
                )

            except Exception as exc:

                st.error(
                    "Could not load the Parquet dataset: "
                    f"{exc}"
                )

                dataframe = None


            if dataframe is not None:

                st.write(
                    f"Loaded **{len(dataframe):,} measurement points** "
                    "from the processed Parquet dataset."
                )


                # ===========================================================
                # DATASET STATISTICS
                # ===========================================================

                data_col1, data_col2, data_col3, data_col4 = (
                    st.columns(4)
                )


                with data_col1:

                    st.metric(
                        "Measurement Points",
                        f"{len(dataframe):,}",
                    )


                with data_col2:

                    st.metric(
                        "Maximum Gain",
                        (
                            f"{dataframe['gain_total'].max():.2f} dBi"
                        ),
                    )


                with data_col3:

                    st.metric(
                        "Mean Gain",
                        (
                            f"{dataframe['gain_total'].mean():.2f} dBi"
                        ),
                    )


                with data_col4:

                    st.metric(
                        "Gain Std. Dev.",
                        (
                            f"{dataframe['gain_total'].std():.2f} dB"
                        ),
                    )


                # ===========================================================
                # PROCESSED DATA PREVIEW
                # ===========================================================

                with st.expander(
                    "View processed Parquet data"
                ):

                    st.dataframe(
                        dataframe.head(100),
                        width="stretch",
                        hide_index=True,
                    )


                # ===========================================================
                # PHI ANGLE SELECTOR
                # ===========================================================

                phi_values = sorted(
                    dataframe[
                        "phi"
                    ].unique()
                )


                selected_phi = st.select_slider(
                    "Phi angle for radiation pattern",
                    options=phi_values,
                    value=phi_values[0],
                )


                # ===========================================================
                # RADIATION PATTERN
                # ===========================================================

                st.markdown(
                    "### Radiation Pattern"
                )


                OUTPUT_DIRECTORY.mkdir(
                    parents=True,
                    exist_ok=True,
                )


                radiation_output = (
                    OUTPUT_DIRECTORY
                    / (
                        "dashboard_radiation_pattern_"
                        f"{selected_scenario_id}.png"
                    )
                )


                try:

                    plot_radiation_pattern(
                        parquet_path=str(
                            parquet_path
                        ),
                        phi_angle=float(
                            selected_phi
                        ),
                        output_path=str(
                            radiation_output
                        ),
                        show=False,
                    )


                    if radiation_output.exists():

                        st.image(
                            str(
                                radiation_output
                            ),
                            caption=(
                                "Radiation pattern — "
                                f"φ = {selected_phi:.1f}°"
                            ),
                            width="stretch",
                        )


                except Exception as exc:

                    st.error(
                        "Could not generate the radiation pattern: "
                        f"{exc}"
                    )


                # ===========================================================
                # FAR-FIELD HEATMAP
                # ===========================================================

                st.markdown(
                    "### Far-Field Heatmap"
                )


                heatmap_output = (
                    OUTPUT_DIRECTORY
                    / (
                        "dashboard_far_field_heatmap_"
                        f"{selected_scenario_id}.png"
                    )
                )


                try:

                    plot_far_field_heatmap(
                        parquet_path=str(
                            parquet_path
                        ),
                        output_path=str(
                            heatmap_output
                        ),
                        show=False,
                    )


                    if heatmap_output.exists():

                        st.image(
                            str(
                                heatmap_output
                            ),
                            caption=(
                                "Far-field gain across "
                                "theta and phi"
                            ),
                            width="stretch",
                        )


                except Exception as exc:

                    st.error(
                        "Could not generate the far-field heatmap: "
                        f"{exc}"
                    )


        else:

            st.warning(
                "The Parquet file linked to this scenario "
                "could not be found:"
            )

            st.code(
                str(
                    parquet_path
                )
            )


    else:

        st.info(
            "No Parquet dataset is linked to this scenario yet."
        )


# ===========================================================================
# PIPELINE ACTIVITY
# ===========================================================================

st.divider()

st.markdown(
    """
    <div class="section-title">
        Pipeline Activity
    </div>
    """,
    unsafe_allow_html=True,
)


if pipeline_runs:

    runs_dataframe = pd.DataFrame(
        pipeline_runs
    )


    preferred_columns = [

        "run_id",

        "stage",

        "status",

        "started_at",

        "finished_at",

        "scenarios_in",

        "scenarios_out",

        "error_message",

        "notes",
    ]


    display_columns = [

        column

        for column in preferred_columns

        if column in runs_dataframe.columns
    ]


    if display_columns:

        st.dataframe(
            runs_dataframe[
                display_columns
            ],
            width="stretch",
            hide_index=True,
        )

    else:

        st.dataframe(
            runs_dataframe,
            width="stretch",
            hide_index=True,
        )


else:

    st.info(
        "No pipeline runs have been recorded yet."
    )


# ===========================================================================
# CURRENT PIPELINE
# ===========================================================================

st.markdown(
    """
    <div class="section-title">
        Current Pipeline Progress
    </div>
    """,
    unsafe_allow_html=True,
)


pipeline_progress = pd.DataFrame(
    {
        "Pipeline Stage": [
            "FEKO .ffe generation",
            "FFE ingestion",
            "SQLite metadata storage",
            "Far-field Parquet storage",
            "Feature extraction",
            "Anomaly detection",
            "Fault classification",
            "Dashboard visualisation",
            "Larger scenario library",
            "Trained ML classifier",
        ],

        "Status": [
            "✓ Working",
            "✓ Working",
            "✓ Working",
            "✓ Working",
            "✓ Working",
            "✓ Working",
            "✓ Proof of concept",
            "✓ Working",
            "Next development stage",
            "Future stage",
        ],
    }
)


st.dataframe(
    pipeline_progress,
    width="stretch",
    hide_index=True,
)


# ===========================================================================
# TECHNICAL NOTES
# ===========================================================================

with st.expander(
    "About the current prototype"
):

    st.markdown(
        """
        ### Current system

        The dashboard is connected to the existing antenna health
        monitoring pipeline.

        The current data flow is:

        **FEKO → FFE ingestion → SQLite/Parquet → feature extraction
        → anomaly detection → fault classification → dashboard**

        ### Healthy baseline

        **H01_NOM** is currently used as the healthy reference scenario.

        ### Anomaly detection

        The current detector measures deviations from the healthy
        baseline using features including:

        - Mean absolute difference
        - Median absolute difference
        - Standard deviation of difference
        - Maximum absolute difference
        - Percentage of samples exceeding 0.5 dB
        - Percentage exceeding 1 dB
        - Percentage exceeding 2 dB
        - Percentage exceeding 3 dB

        ### Fault classification

        Fault classification is currently a proof-of-concept mapping
        of known FEKO scenarios.

        Current known mappings include:

        - H01_NOM → NONE
        - MF01_L_PLUS → L_PLUS
        - MF02_L_MINUS → L_MINUS

        The M01_FEED_OFFSET scenario currently produces the same
        far-field output as H01_NOM, so it is currently classified
        as healthy by the anomaly detector.

        This does not establish that a physical feed-offset fault
        is healthy. A meaningful FEKO representation of that fault
        is still required.

        ### Future development

        Planned development includes:

        - Larger labelled FEKO scenario library
        - Multiple examples of each fault type
        - Machine-learning fault classifier
        - Proper training and test datasets
        - Model evaluation metrics
        - Improved SQLite classification storage
        - Additional antenna fault scenarios
        - S-parameter integration
        - RX induced-voltage measurements
        - Pipeline orchestration
        """
    )


# ===========================================================================
# DATABASE SUMMARY
# ===========================================================================

with st.expander(
    "View database summary"
):

    st.json(
        summary
    )


# ===========================================================================
# FOOTER
# ===========================================================================

st.divider()

st.caption(
    "Antenna Health Monitoring Pipeline — "
    "Stellenbosch University Final-Year Project"
)

st.caption(
    "Current scope: FEKO ingestion → processing → "
    "feature extraction → anomaly detection → "
    "proof-of-concept classification → visualisation"
)

