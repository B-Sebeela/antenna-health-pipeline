"""
Antenna Health Monitoring Dashboard – Overview
Multi-user auth + alerts + dual experiment families
  - Feed offset (H01 / FO_*)
  - Array element failure (ARR7_*)
"""

from pathlib import Path
import sys
import sqlite3

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard_auth import (
    require_login,
    sync_alerts_from_results,
    count_open_alerts,
)

DATABASE_FILE = PROJECT_ROOT / "data" / "antenna_health.db"
FEATURE_FILE = PROJECT_ROOT / "data" / "features" / "scenario_features_refined.parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ANOMALY_PARQUET = PROJECT_ROOT / "data" / "features" / "anomaly_results.parquet"

BASELINE_FEED_OFFSET = "H01_DIPOLE_FEED_50"
BASELINE_ARRAY7 = "ARR7_BASELINE_ALL_ON"

HEALTH_MEAN_ABS_THRESHOLD_DB = 0.02
SEVERITY_LOW_BELOW = 15.0
SEVERITY_MODERATE_BELOW = 40.0
NOMINAL_PHI_DEG = 41.0

ARR7_ORDER = [
    "ARR7_BASELINE_ALL_ON",
    "ARR7_FAIL_1",
    "ARR7_FAIL_2",
    "ARR7_FAIL_3",
    "ARR7_FAIL_5",
    "ARR7_ONLY_1_ON",
]

st.set_page_config(
    page_title="Antenna Health Monitoring",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----- Auth gate -----
username, role = require_login()
sync_alerts_from_results()

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }
    .status-banner {
        padding: 0.85rem 1.35rem; border-radius: 0.6rem; margin-bottom: 1.35rem;
        font-weight: 600; font-size: 0.98rem; display: flex; align-items: center; gap: 0.75rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .status-ok   { background: linear-gradient(90deg,#e6f4ea,#f1f8f3); color: #137333; border: 1px solid #a8dab5; }
    .status-warn { background: linear-gradient(90deg,#fef7e0,#fffbf0); color: #b06000; border: 1px solid #fdd663; }
    .status-fail { background: linear-gradient(90deg,#fce8e6,#fef3f2); color: #c5221f; border: 1px solid #f6aea9; }
    div[data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e8eaed; border-radius: 0.6rem;
        padding: 0.85rem 1.1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label { font-size: 0.82rem; color: #5f6368; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 650; }
    .badge { display: inline-block; padding: 0.18rem 0.65rem; border-radius: 999px;
             font-size: 0.72rem; font-weight: 650; letter-spacing: 0.03em; text-transform: uppercase; }
    .badge-none { background: #e6f4ea; color: #137333; }
    .badge-low { background: #e8f0fe; color: #1967d2; }
    .badge-moderate { background: #fef7e0; color: #b06000; }
    .badge-high { background: #fce8e6; color: #c5221f; }
    .health-healthy { color: #137333; font-weight: 700; }
    .health-anomaly { color: #c5221f; font-weight: 700; }
    section[data-testid="stSidebar"] { background: #f8f9fb; border-right: 1px solid #e8eaed; }
    hr { margin: 1.4rem 0; border-color: #e8eaed; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def scenario_family(scenario_id: str) -> str:
    if str(scenario_id).startswith("ARR7_"):
        return "array7"
    return "feed_offset"


def baseline_for(scenario_id: str) -> str:
    return (
        BASELINE_ARRAY7
        if scenario_family(scenario_id) == "array7"
        else BASELINE_FEED_OFFSET
    )


@st.cache_data(show_spinner="Loading results…")
def load_results() -> pd.DataFrame:
    conn = sqlite3.connect(DATABASE_FILE)
    try:
        df = pd.read_sql_query(
            """
            SELECT scenario_id, anomaly_score, anomaly_score_100, health_status, severity,
                   predicted_fault_type, predicted_condition, diagnostic_confidence,
                   known_health, known_fault_type, known_condition, feed_setting_percent,
                   health_correct, fault_type_correct, direction_correct
            FROM anomaly_results
            ORDER BY scenario_id
            """,
            conn,
        )
    finally:
        conn.close()
    return df


@st.cache_data(show_spinner="Loading features…")
def load_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURE_FILE)


@st.cache_data(show_spinner="Loading radiation pattern…")
def load_radiation_data(scenario_id: str) -> pd.DataFrame | None:
    path = PROCESSED_DIR / scenario_id / "farfield.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def health_badge(status: str) -> str:
    if status == "HEALTHY":
        return '<span class="health-healthy">● HEALTHY</span>'
    return '<span class="health-anomaly">● ANOMALY DETECTED</span>'


def severity_badge(severity: str) -> str:
    cls = {
        "NONE": "badge-none",
        "LOW": "badge-low",
        "MODERATE": "badge-moderate",
        "HIGH": "badge-high",
    }.get(severity, "badge-none")
    return f'<span class="badge {cls}">{severity}</span>'


def get_phi_cut(df: pd.DataFrame, target_phi: float = NOMINAL_PHI_DEG) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    closest = min(df["phi"].unique(), key=lambda x: abs(x - target_phi))
    return df.loc[df["phi"] == closest].sort_values("theta").copy()


def make_overlay_figure(baseline_cut, current_cut, scenario_id, baseline_label: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.62, 0.38],
        subplot_titles=(
            f"Radiation Pattern (φ ≈ {NOMINAL_PHI_DEG:.0f}°)",
            "Difference from Healthy Reference (ΔdB)",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=baseline_cut["theta"], y=baseline_cut["gain_total"],
            mode="lines", name=baseline_label,
            line=dict(color="#137333", width=2.5),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=current_cut["theta"], y=current_cut["gain_total"],
            mode="lines", name=scenario_id,
            line=dict(color="#c5221f", width=2.5),
        ),
        row=1, col=1,
    )
    merged = pd.merge(
        baseline_cut[["theta", "gain_total"]],
        current_cut[["theta", "gain_total"]],
        on="theta", suffixes=("_base", "_curr"),
    )
    delta = merged["gain_total_curr"] - merged["gain_total_base"]
    fig.add_trace(
        go.Scatter(
            x=merged["theta"], y=delta, mode="lines", name="Δ Gain",
            line=dict(color="#1967d2", width=2),
            fill="tozeroy", fillcolor="rgba(25, 103, 210, 0.15)",
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#5f6368", row=2, col=1)
    fig.update_layout(
        height=560, margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", template="plotly_white",
    )
    fig.update_xaxes(title_text="θ (deg)", row=2, col=1)
    fig.update_yaxes(title_text="Gain (dBi)", row=1, col=1)
    fig.update_yaxes(title_text="Δ Gain (dB)", row=2, col=1)
    return fig


def make_score_gauge(score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100", "font": {"size": 28}},
            title={"text": "Anomaly Score", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#202124"},
                "steps": [
                    {"range": [0, SEVERITY_LOW_BELOW], "color": "#e8f0fe"},
                    {"range": [SEVERITY_LOW_BELOW, SEVERITY_MODERATE_BELOW], "color": "#fef7e0"},
                    {"range": [SEVERITY_MODERATE_BELOW, 100], "color": "#fce8e6"},
                ],
                "threshold": {
                    "line": {"color": "#c5221f", "width": 3},
                    "thickness": 0.8,
                    "value": SEVERITY_LOW_BELOW,
                },
            },
        )
    )
    fig.update_layout(height=220, margin=dict(l=30, r=30, t=40, b=10))
    return fig


# ============================================================
# LOAD DATA
# ============================================================

try:
    results = load_results()
    features = load_features()
except Exception as exc:
    st.error(f"Unable to load pipeline data:\n\n{exc}")
    st.stop()

score_parts_available = False
try:
    if ANOMALY_PARQUET.exists():
        adf = pd.read_parquet(ANOMALY_PARQUET)
        part_cols = [c for c in adf.columns if c.startswith("score_part_")]
        if part_cols:
            results = results.merge(
                adf[["scenario_id"] + part_cols], on="scenario_id", how="left"
            )
            score_parts_available = True
except Exception:
    pass

# Attach family if missing from SQL
if "experiment_family" not in results.columns:
    results["experiment_family"] = results["scenario_id"].map(scenario_family)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("### 📡 Antenna Health")
st.sidebar.caption("Reference-based FEKO monitoring")
st.sidebar.markdown("---")

st.sidebar.header("Experiment family")
family_choice = st.sidebar.radio(
    "Family",
    ["Feed offset (dipole)", "Array element failure (7-el)", "All"],
    index=0,
)

all_ids = results["scenario_id"].tolist()
if family_choice.startswith("Feed"):
    scenario_options = [s for s in all_ids if scenario_family(s) == "feed_offset"]
elif family_choice.startswith("Array"):
    scenario_options = [s for s in all_ids if scenario_family(s) == "array7"]
else:
    scenario_options = all_ids

if not scenario_options:
    st.sidebar.error("No scenarios for this family.")
    st.stop()

if family_choice.startswith("Feed"):
    default_id = BASELINE_FEED_OFFSET
elif family_choice.startswith("Array"):
    default_id = BASELINE_ARRAY7
else:
    default_id = scenario_options[0]

default_idx = (
    scenario_options.index(default_id) if default_id in scenario_options else 0
)

st.sidebar.header("Scenario Selection")
selected_scenario = st.sidebar.selectbox(
    "Select scenario",
    scenario_options,
    index=default_idx,
)

ref_baseline = baseline_for(selected_scenario)
st.sidebar.markdown("---")
st.sidebar.markdown("**Reference**")
st.sidebar.caption(f"Baseline: `{ref_baseline}`")
st.sidebar.caption(f"Health trigger: mean |Δ| > {HEALTH_MEAN_ABS_THRESHOLD_DB} dB")
st.sidebar.markdown("---")
st.sidebar.caption("Pages: Overview · Alerts · Comparison · Pipeline")


# ============================================================
# SELECTED ROW
# ============================================================

result = results.loc[results["scenario_id"] == selected_scenario].iloc[0]
feature = features.loc[features["scenario_id"] == selected_scenario].iloc[0]
severity = result["severity"]
fam = scenario_family(selected_scenario)


# ============================================================
# ALERT BANNER
# ============================================================

high_open, total_open = count_open_alerts()
if high_open > 0:
    st.error(
        f"🚨 **{high_open} HIGH-severity alert(s) active** "
        f"({total_open} total open/acknowledged). Open the **Alerts** page."
    )
elif total_open > 0:
    st.warning(
        f"⚠ **{total_open} open alert(s)**. Review the **Alerts** page."
    )


# ============================================================
# SYSTEM BANNER (current family filter)
# ============================================================

view = results.loc[results["scenario_id"].isin(scenario_options)]
total = len(view)
healthy_count = int((view["health_status"] == "HEALTHY").sum())
anomaly_count = total - healthy_count
high_count = int((view["severity"] == "HIGH").sum())

if high_count > 0:
    banner_cls, banner_text = (
        "status-fail",
        f"⚠ {high_count} high-severity anomalies in this view ({total} scenarios)",
    )
elif anomaly_count > 0:
    banner_cls, banner_text = (
        "status-warn",
        f"⚠ {anomaly_count} anomalies detected – system monitoring active",
    )
else:
    banner_cls, banner_text = (
        "status-ok",
        "✓ All scenarios in this view within healthy bounds",
    )

st.markdown(
    f'<div class="status-banner {banner_cls}">{banner_text}</div>',
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Scenarios in view", total)
k2.metric("Healthy", healthy_count)
k3.metric("Anomalies", anomaly_count)
k4.metric("High Severity", high_count)

st.divider()


# ============================================================
# DIAGNOSIS
# ============================================================

st.subheader("Selected Scenario Diagnosis")

left, right = st.columns([1.4, 1])
with left:
    st.markdown(f"### {selected_scenario}")
    st.markdown(health_badge(result["health_status"]), unsafe_allow_html=True)
    st.markdown(f"**Severity:** {severity_badge(severity)}", unsafe_allow_html=True)

    feed = result.get("feed_setting_percent")
    if pd.notna(feed):
        st.write(f"**Feed setting:** {float(feed):.0f} %")
    else:
        st.write("**Experiment:** 7-element array element failure")

    st.write(f"**Predicted fault:** {result['predicted_fault_type']}")
    st.write(f"**Condition:** {result['predicted_condition']}")
    st.write(
        f"**Peak gain:** {feature['peak_gain']:.3f} dBi &nbsp;|&nbsp; "
        f"**Peak direction:** θ = {feature['peak_theta']:.1f}°, "
        f"φ = {feature['peak_phi']:.1f}°"
    )
    st.caption(f"Family: `{fam}` · Reference: `{ref_baseline}`")

with right:
    score = float(result["anomaly_score_100"])
    st.plotly_chart(
        make_score_gauge(score),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    conf = float(result["diagnostic_confidence"]) * 100
    st.metric("Detection Evidence Strength", f"{conf:.0f} %")
    st.caption(
        "Fraction of independent deviation indicators that fired "
        "(not a probabilistic confidence)."
    )

if score_parts_available:
    part_cols = [c for c in results.columns if c.startswith("score_part_")]
    if part_cols:
        labels_map = {
            "score_part_mean": "Mean |Δ|",
            "score_part_pct_0_5": "% area > 0.5 dB",
            "score_part_pct_1": "% area > 1 dB",
            "score_part_tilt": "Peak-tilt",
        }
        part_labels = [labels_map.get(c, c) for c in part_cols]
        part_values = [float(result[c]) for c in part_cols]
        st.markdown("**Score drivers**")
        fig_parts = go.Figure(
            go.Bar(
                x=part_values, y=part_labels, orientation="h",
                marker=dict(color=["#1967d2", "#1a73e8", "#8ab4f8", "#a8c7fa"]),
                text=[f"{v:.1f}" for v in part_values],
                textposition="outside", cliponaxis=False,
            )
        )
        fig_parts.update_layout(
            height=180, margin=dict(l=10, r=40, t=10, b=10),
            xaxis_title="Points (of 100)", yaxis=dict(autorange="reversed"),
            template="plotly_white", showlegend=False,
        )
        st.plotly_chart(
            fig_parts, use_container_width=True, config={"displayModeBar": False}
        )
        parts = dict(zip(part_cols, part_values))
        dominant = max(parts, key=parts.get)
        dominant_val = parts[dominant]
        total_score = float(result["anomaly_score_100"])
        if total_score > 0 and dominant_val > 0:
            pct = 100.0 * dominant_val / total_score
            st.info(
                f"**Primary driver:** {labels_map.get(dominant, dominant)} "
                f"({dominant_val:.1f} pts · {pct:.0f}% of score)."
            )
        elif total_score == 0:
            st.success("No significant deviation from the healthy reference.")

st.divider()


# ============================================================
# BASELINE COMPARISON
# ============================================================

st.subheader("Baseline Comparison")
b1, b2, b3, b4 = st.columns(4)
b1.metric(
    "Mean Absolute Difference",
    f"{feature['baseline_mean_absolute_difference']:.4f} dB",
)
b2.metric(
    "Maximum Difference",
    f"{feature['baseline_max_absolute_difference']:.4f} dB",
)
b3.metric("Points > 0.5 dB", f"{feature['baseline_pct_diff_0_5db']:.2f} %")
b4.metric("Points > 1 dB", f"{feature['baseline_pct_diff_1db']:.2f} %")

st.divider()


# ============================================================
# RADIATION PATTERN EVIDENCE
# ============================================================

st.subheader("Radiation Pattern Evidence")

baseline_rad = load_radiation_data(ref_baseline)
current_rad = load_radiation_data(selected_scenario)

if baseline_rad is None or current_rad is None:
    st.warning("Radiation-pattern data not found for one or both scenarios.")
else:
    required = {"theta", "phi", "gain_total"}
    if required.issubset(baseline_rad.columns) and required.issubset(current_rad.columns):
        baseline_cut = get_phi_cut(baseline_rad)
        current_cut = get_phi_cut(current_rad)
        baseline_label = (
            "Healthy array (all on)"
            if fam == "array7"
            else "Healthy (50 % feed)"
        )
        fig = make_overlay_figure(
            baseline_cut, current_cut, selected_scenario, baseline_label
        )
        st.plotly_chart(fig, use_container_width=True)

        merged = pd.merge(
            baseline_cut[["theta", "gain_total"]],
            current_cut[["theta", "gain_total"]],
            on="theta", suffixes=("_base", "_curr"),
        )
        if not merged.empty:
            delta = merged["gain_total_curr"] - merged["gain_total_base"]
            abs_delta = delta.abs()
            peak_idx = abs_delta.idxmax()
            h1, h2, h3 = st.columns(3)
            h1.metric("Max |Δ| at", f"θ ≈ {float(merged.loc[peak_idx, 'theta']):.0f}°")
            h2.metric("Peak Δ gain", f"{float(delta.loc[peak_idx]):+.2f} dB")
            h3.metric("Mean |Δ| on cut", f"{float(abs_delta.mean()):.3f} dB")
        st.caption(
            f"2-D cut at φ closest to {NOMINAL_PHI_DEG:.0f}°. "
            f"Reference: `{ref_baseline}`. "
            "Top: absolute patterns. Bottom: difference vs healthy reference."
        )
    else:
        st.warning("Required radiation-pattern columns are missing.")

st.divider()


# ============================================================
# SCORE TREND (family-specific)
# ============================================================

st.subheader("Anomaly Score Across Controlled Cases")

family_results = results.loc[
    results["scenario_id"].map(scenario_family) == fam
].copy()

fig_trend = go.Figure()

if fam == "feed_offset":
    chart_df = family_results.sort_values("feed_setting_percent")
    fig_trend.add_trace(
        go.Scatter(
            x=chart_df["feed_setting_percent"],
            y=chart_df["anomaly_score_100"],
            mode="lines+markers",
            name="Anomaly Score",
            line=dict(color="#1967d2", width=2.5),
            marker=dict(size=8),
        )
    )
    sel = chart_df.loc[chart_df["scenario_id"] == selected_scenario]
    if not sel.empty:
        fig_trend.add_trace(
            go.Scatter(
                x=sel["feed_setting_percent"],
                y=sel["anomaly_score_100"],
                mode="markers",
                name="Selected",
                marker=dict(size=14, color="#c5221f", symbol="diamond"),
            )
        )
    x_title = "Feed Setting (%)"
else:
    order_map = {s: i for i, s in enumerate(ARR7_ORDER)}
    chart_df = family_results.copy()
    chart_df["order"] = chart_df["scenario_id"].map(order_map)
    chart_df = chart_df.sort_values("order")
    fig_trend.add_trace(
        go.Scatter(
            x=chart_df["scenario_id"],
            y=chart_df["anomaly_score_100"],
            mode="lines+markers",
            name="Anomaly Score",
            line=dict(color="#1967d2", width=2.5),
            marker=dict(size=8),
        )
    )
    sel = chart_df.loc[chart_df["scenario_id"] == selected_scenario]
    if not sel.empty:
        fig_trend.add_trace(
            go.Scatter(
                x=sel["scenario_id"],
                y=sel["anomaly_score_100"],
                mode="markers",
                name="Selected",
                marker=dict(size=14, color="#c5221f", symbol="diamond"),
            )
        )
    x_title = "Array scenario"

fig_trend.add_hline(
    y=SEVERITY_LOW_BELOW, line_dash="dash", line_color="#1967d2",
    annotation_text="LOW / MODERATE", annotation_position="bottom right",
)
fig_trend.add_hline(
    y=SEVERITY_MODERATE_BELOW, line_dash="dash", line_color="#c5221f",
    annotation_text="MODERATE / HIGH", annotation_position="bottom right",
)
fig_trend.update_layout(
    height=340, margin=dict(l=50, r=30, t=20, b=80),
    xaxis_title=x_title, yaxis_title="Anomaly Score (0–100)",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_trend, use_container_width=True)
if fam == "array7":
    st.caption(
        "Array failures saturate the provisional 0–100 score scales (large mean |Δ|). "
        "Use peak gain / mean |Δ| in the feature table to rank severity within this family."
    )
else:
    st.caption(
        "Higher values indicate greater deviation from the healthy 50 % feed reference. "
        "Dashed lines mark provisional severity boundaries."
    )

st.divider()


# ============================================================
# RESULTS TABLE (current view)
# ============================================================

st.subheader("Scenario Results")

display = view[
    [
        "scenario_id",
        "feed_setting_percent",
        "health_status",
        "severity",
        "anomaly_score_100",
        "predicted_fault_type",
        "predicted_condition",
        "diagnostic_confidence",
    ]
].copy()
display = display.rename(
    columns={
        "scenario_id": "Scenario",
        "feed_setting_percent": "Feed (%)",
        "health_status": "Health",
        "severity": "Severity",
        "anomaly_score_100": "Anomaly Score",
        "predicted_fault_type": "Fault Type",
        "predicted_condition": "Condition",
        "diagnostic_confidence": "Evidence",
    }
)
display["Anomaly Score"] = display["Anomaly Score"].round(2)
display["Evidence"] = (
    (display["Evidence"] * 100).round(0).astype(int).astype(str) + " %"
)


def colour_severity(val):
    return {
        "NONE": "background-color: #e6f4ea; color: #137333",
        "LOW": "background-color: #e8f0fe; color: #1967d2",
        "MODERATE": "background-color: #fef7e0; color: #b06000",
        "HIGH": "background-color: #fce8e6; color: #c5221f",
    }.get(val, "")


def colour_health(val):
    if val == "HEALTHY":
        return "background-color: #e6f4ea; color: #137333; font-weight: 600"
    return "background-color: #fce8e6; color: #c5221f; font-weight: 600"


styled = (
    display.style.map(colour_severity, subset=["Severity"])
    .map(colour_health, subset=["Health"])
)
st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()


# ============================================================
# VALIDATION (current view)
# ============================================================

st.subheader("Pipeline Validation (current view)")
v1, v2, v3 = st.columns(3)
v1.metric("Health Detection", f"{view['health_correct'].mean()*100:.1f} %")
v2.metric("Fault Classification", f"{view['fault_type_correct'].mean()*100:.1f} %")
v3.metric("Condition / Direction", f"{view['direction_correct'].mean()*100:.1f} %")
st.info(
    "Validation metrics are calculated against known labels of the controlled "
    "FEKO simulation datasets (feed-offset and/or array element failure). "
    "They demonstrate performance on these scenarios and should not be interpreted "
    "as generalisation to unseen antenna designs or real hardware."
)

st.divider()
st.caption(
    "Antenna Health Monitoring Pipeline | "
    "FEKO → Ingestion → Parquet → Features → "
    "Anomaly Detection → Classification → SQLite → Dashboard"
)