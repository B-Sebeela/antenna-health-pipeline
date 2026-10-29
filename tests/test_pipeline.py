"""
Regression tests for the controlled Antenna Health Monitoring pipeline.

Covers:
  - Feed-offset family (10 scenarios, baseline H01_DIPOLE_FEED_50)
  - Array element-failure family (6 scenarios, baseline ARR7_BASELINE_ALL_ON)

Run after:
  python -m src.preprocessing.features_refined
  python -m src.model.anomaly_detection
  python -m src.model.store_results

  python -m pytest -v
"""

from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    PROJECT_ROOT / "data" / "features" / "scenario_features_refined.parquet"
)
ANOMALY_RESULTS_PATH = (
    PROJECT_ROOT / "data" / "features" / "anomaly_results.parquet"
)
DB_PATH = PROJECT_ROOT / "data" / "antenna_health.db"

BASELINE_FEED_OFFSET = "H01_DIPOLE_FEED_50"
BASELINE_ARRAY7 = "ARR7_BASELINE_ALL_ON"

# ---------------------------------------------------------------------------
# Expected controlled scenarios
# ---------------------------------------------------------------------------

EXPECTED_FEED_OFFSET = {
    "H01_DIPOLE_FEED_50": {
        "feed": 50,
        "health": "HEALTHY",
        "fault": "NONE",
        "condition": "NOMINAL",
    },
    "FO_DIPOLE_FEED_10": {
        "feed": 10,
        "health": "ANOMALY_DETECTED",
        "fault": "FEED_OFFSET",
        "condition": "LOWER_THAN_REFERENCE",
    },
    "FO_DIPOLE_FEED_20": {
        "feed": 20,
        "health": "ANOMALY_DETECTED",
        "fault": "FEED_OFFSET",
        "condition": "LOWER_THAN_REFERENCE",
    },
    "FO_DIPOLE_FEED_30": {
        "feed": 30,
        "health": "ANOMALY_DETECTED",
        "fault": "FEED_OFFSET",
        "condition": "LOWER_THAN_REFERENCE",
    },
    "FO_DIPOLE_FEED_40": {
        "feed": 40,
        "health": "ANOMALY_DETECTED",
        "fault": "FEED_OFFSET",
        "condition": "LOWER_THAN_REFERENCE",
    },
    "FO_DIPOLE_FEED_45": {
        "feed": 45,
        "health": "ANOMALY_DETECTED",
        "fault": "FEED_OFFSET",
        "condition": "LOWER_THAN_REFERENCE",
    },
    "FO_DIPOLE_FEED_55": {
        "feed": 55,
        "health": "ANOMALY_DETECTED",
        "fault": "FEED_OFFSET",
        "condition": "HIGHER_THAN_REFERENCE",
    },
    "FO_DIPOLE_FEED_60": {
        "feed": 60,
        "health": "ANOMALY_DETECTED",
        "fault": "FEED_OFFSET",
        "condition": "HIGHER_THAN_REFERENCE",
    },
    "FO_DIPOLE_FEED_70": {
        "feed": 70,
        "health": "ANOMALY_DETECTED",
        "fault": "FEED_OFFSET",
        "condition": "HIGHER_THAN_REFERENCE",
    },
    "FO_DIPOLE_FEED_80": {
        "feed": 80,
        "health": "ANOMALY_DETECTED",
        "fault": "FEED_OFFSET",
        "condition": "HIGHER_THAN_REFERENCE",
    },
}

EXPECTED_ARRAY7 = {
    "ARR7_BASELINE_ALL_ON": {
        "health": "HEALTHY",
        "fault": "NONE",
        "condition": "NOMINAL",
    },
    "ARR7_FAIL_1": {
        "health": "ANOMALY_DETECTED",
        "fault": "ELEMENT_FAILURE",
        "condition": "N_OFF_1",
    },
    "ARR7_FAIL_2": {
        "health": "ANOMALY_DETECTED",
        "fault": "ELEMENT_FAILURE",
        "condition": "N_OFF_2",
    },
    "ARR7_FAIL_3": {
        "health": "ANOMALY_DETECTED",
        "fault": "ELEMENT_FAILURE",
        "condition": "N_OFF_3",
    },
    "ARR7_FAIL_5": {
        "health": "ANOMALY_DETECTED",
        "fault": "ELEMENT_FAILURE",
        "condition": "N_OFF_5",
    },
    "ARR7_ONLY_1_ON": {
        "health": "ANOMALY_DETECTED",
        "fault": "ELEMENT_FAILURE",
        "condition": "ONLY_ONE_ACTIVE",
    },
}

EXPECTED_SCENARIOS = {**EXPECTED_FEED_OFFSET, **EXPECTED_ARRAY7}
EXPECTED_COUNT = len(EXPECTED_SCENARIOS)  # 16


def load_results() -> pd.DataFrame:
    assert ANOMALY_RESULTS_PATH.exists(), (
        "anomaly_results.parquet does not exist. "
        "Run features → anomaly_detection → store_results first."
    )
    return pd.read_parquet(ANOMALY_RESULTS_PATH)


def _row(results: pd.DataFrame, scenario_id: str) -> pd.Series:
    match = results.loc[results["scenario_id"] == scenario_id]
    assert len(match) == 1, f"Expected one row for {scenario_id}, found {len(match)}"
    return match.iloc[0]


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------

def test_feature_file_exists():
    assert FEATURES_PATH.exists(), (
        "scenario_features_refined.parquet does not exist."
    )


def test_anomaly_results_file_exists():
    assert ANOMALY_RESULTS_PATH.exists(), (
        "anomaly_results.parquet does not exist."
    )


# ---------------------------------------------------------------------------
# Controlled set membership
# ---------------------------------------------------------------------------

def test_expected_scenarios_present():
    results = load_results()
    actual = set(results["scenario_id"])
    expected = set(EXPECTED_SCENARIOS)
    assert actual == expected, (
        "Anomaly results must contain exactly the controlled scenarios.\n"
        f"Missing: {sorted(expected - actual)}\n"
        f"Unexpected: {sorted(actual - expected)}"
    )


def test_result_count():
    results = load_results()
    assert len(results) == EXPECTED_COUNT


# ---------------------------------------------------------------------------
# Baselines (both families)
# ---------------------------------------------------------------------------

def test_feed_offset_baseline_is_healthy():
    results = load_results()
    baseline = _row(results, BASELINE_FEED_OFFSET)
    assert baseline["health_status"] == "HEALTHY"
    assert baseline["predicted_fault_type"] == "NONE"
    assert baseline["predicted_condition"] == "NOMINAL"


def test_array_baseline_is_healthy():
    results = load_results()
    baseline = _row(results, BASELINE_ARRAY7)
    assert baseline["health_status"] == "HEALTHY"
    assert baseline["predicted_fault_type"] == "NONE"
    assert baseline["predicted_condition"] == "NOMINAL"


def test_feed_offset_baseline_score_is_zero():
    results = load_results()
    baseline = _row(results, BASELINE_FEED_OFFSET)
    assert abs(float(baseline["anomaly_score"])) < 1e-12
    assert abs(float(baseline["anomaly_score_100"])) < 1e-12


def test_array_baseline_score_is_zero():
    results = load_results()
    baseline = _row(results, BASELINE_ARRAY7)
    assert abs(float(baseline["anomaly_score"])) < 1e-12
    assert abs(float(baseline["anomaly_score_100"])) < 1e-12


# ---------------------------------------------------------------------------
# Detection of intentional faults
# ---------------------------------------------------------------------------

def test_all_non_baseline_faults_detected():
    results = load_results()
    baselines = {BASELINE_FEED_OFFSET, BASELINE_ARRAY7}
    faults = results[~results["scenario_id"].isin(baselines)]
    assert len(faults) == EXPECTED_COUNT - 2
    assert (faults["health_status"] == "ANOMALY_DETECTED").all()


# ---------------------------------------------------------------------------
# Fault type and condition (rule-based, per family)
# ---------------------------------------------------------------------------

def test_feed_offset_fault_types():
    results = load_results()
    for scenario_id, expected in EXPECTED_FEED_OFFSET.items():
        if scenario_id == BASELINE_FEED_OFFSET:
            continue
        row = _row(results, scenario_id)
        assert row["predicted_fault_type"] == expected["fault"], scenario_id


def test_array_fault_types():
    results = load_results()
    for scenario_id, expected in EXPECTED_ARRAY7.items():
        if scenario_id == BASELINE_ARRAY7:
            continue
        row = _row(results, scenario_id)
        assert row["predicted_fault_type"] == expected["fault"], scenario_id


def test_condition_labels():
    """FO direction and ARR7 N_OFF / ONLY_ONE_ACTIVE conditions."""
    results = load_results()
    for scenario_id, expected in EXPECTED_SCENARIOS.items():
        row = _row(results, scenario_id)
        assert row["predicted_condition"] == expected["condition"], (
            f"{scenario_id}: got {row['predicted_condition']}, "
            f"expected {expected['condition']}"
        )


def test_feed_settings_on_feed_offset_family():
    results = load_results()
    for scenario_id, expected in EXPECTED_FEED_OFFSET.items():
        row = _row(results, scenario_id)
        assert int(row["feed_setting_percent"]) == expected["feed"], scenario_id


# ---------------------------------------------------------------------------
# Value ranges
# ---------------------------------------------------------------------------

def test_anomaly_scores_are_valid():
    results = load_results()
    assert results["anomaly_score_100"].notna().all()
    assert (results["anomaly_score_100"] >= 0).all()
    assert (results["anomaly_score_100"] <= 100).all()


def test_diagnostic_confidence_is_valid():
    results = load_results()
    assert results["diagnostic_confidence"].notna().all()
    assert (results["diagnostic_confidence"] >= 0).all()
    assert (results["diagnostic_confidence"] <= 1).all()


# ---------------------------------------------------------------------------
# Validation flags (designed labels — not generalisation)
# ---------------------------------------------------------------------------

def test_validation_flags_all_true():
    results = load_results()
    assert results["health_correct"].all()
    assert results["fault_type_correct"].all()
    assert results["direction_correct"].all()


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

def test_sqlite_results_exist():
    assert DB_PATH.exists(), "antenna_health.db does not exist."
    with sqlite3.connect(DB_PATH) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM anomaly_results"
        ).fetchone()[0]
    assert count == EXPECTED_COUNT


def test_sqlite_scenario_ids_match_parquet():
    results = load_results()
    parquet_ids = set(results["scenario_id"])
    with sqlite3.connect(DB_PATH) as conn:
        db_results = pd.read_sql_query(
            "SELECT scenario_id FROM anomaly_results",
            conn,
        )
    database_ids = set(db_results["scenario_id"])
    assert database_ids == parquet_ids