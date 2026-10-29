"""
Scenario Comparison – Antenna Health Monitoring System
Supports feed-offset and 7-element array families.
"""

from pathlib import Path
import sys
import sqlite3

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard_auth import require_login, sync_alerts_from_results

PROJECT_ROOT = ROOT
DATABASE_FILE = PROJECT_ROOT / "data" / "antenna_health.db"
FEATURE_FILE = PROJECT_ROOT / "data" / "features" / "scenario_features_refined.parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ANOMALY_PARQUET = PROJECT_ROOT / "data" / "features" / "anomaly_results.parquet"

BASELINE_FEED_OFFSET = "H01_DIPOLE_FEED_50"
BASELINE_ARRAY7 = "ARR7_BASELINE_ALL_ON"
NOMINAL_PHI_DEG = 41.0

ARR7_ORDER = [
    "ARR7_BASELINE_ALL_ON",
    "ARR7_FAIL_1",
    "ARR7_FAIL_2",
    "ARR7_FAIL_3",
    "ARR7_FAIL_5",
    "ARR7_ONLY_1_ON",
]

st.set_page_config(page_title="Scenario Comparison", page_icon="📊", layout="wide")

username, role = require_login()
sync_alerts_from_results()

st.title("📊 Scenario Comparison")
st.caption("Side-by-side diagnosis and radiation-pattern evidence for two configurations")


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
            SELECT scenario_id, anomaly_score_100, health_status, severity,
                   predicted_fault_type, predicted_condition, diagnostic_confidence,
                   feed_setting_percent
            FROM anomaly_results
            ORDER BY scenario_id
            """,
            conn,
        )
    finally:
        conn.close()

    if ANOMALY_PARQUET.exists():
        adf = pd.read_parquet(ANOMALY_PARQUET)
        part_cols = [c for c in adf.columns if c.startswith("score_part_")]
        if part_cols:
            df = df.merge(adf[["scenario_id"] + part_cols], on="scenario_id", how="left")
    return df


@st.cache_data(show_spinner="Loading features…")
def load_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURE_FILE)


@st.cache_data(show_spinner="Loading pattern…")
def load_radiation(scenario_id: str) -> pd.DataFrame | None:
    path = PROCESSED_DIR / scenario_id / "farfield.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def get_phi_cut(df: pd.DataFrame, target_phi: float = NOMINAL_PHI_DEG) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    closest = min(df["phi"].unique(), key=lambda x: abs(x - target_phi))
    return df.loc[df["phi"] == closest].sort_values("theta").copy()


results = load_results()
features = load_features()

# ----- Family filter -----
family_choice = st.radio(
    "Experiment family",
    ["Feed offset (dipole)", "Array element failure (7-el)"],
    horizontal=True,
    index=0,
)

if family_choice.startswith("Feed"):
    options = [
        s for s in results["scenario_id"].tolist()
        if scenario_family(s) == "feed_offset"
    ]
    default_a, default_b = "FO_DIPOLE_FEED_10", "FO_DIPOLE_FEED_80"
else:
    options = [
        s for s in results["scenario_id"].tolist()
        if scenario_family(s) == "array7"
    ]
    # Prefer ordered list intersection
    options = [s for s in ARR7_ORDER if s in options] or options
    default_a, default_b = "ARR7_BASELINE_ALL_ON", "ARR7_FAIL_5"


def _idx(name: str, fallback: int) -> int:
    try:
        return options.index(name)
    except ValueError:
        return fallback


if len(options) < 2:
    st.error("Need at least two scenarios in this family to compare.")
    st.stop()

col_a, col_b = st.columns(2)
with col_a:
    scen_a = st.selectbox(
        "Scenario A",
        options,
        index=_idx(default_a, 0),
        key="cmp_a",
    )
with col_b:
    scen_b = st.selectbox(
        "Scenario B",
        options,
        index=_idx(default_b, min(1, len(options) - 1)),
        key="cmp_b",
    )

if scen_a == scen_b:
    st.warning("Select two different scenarios to compare.")
    st.stop()

if scenario_family(scen_a) != scenario_family(scen_b):
    st.error("Scenarios must belong to the same experiment family.")
    st.stop()

ref_baseline = baseline_for(scen_a)
fam = scenario_family(scen_a)

ra = results.loc[results["scenario_id"] == scen_a].iloc[0]
rb = results.loc[results["scenario_id"] == scen_b].iloc[0]
fa = features.loc[features["scenario_id"] == scen_a].iloc[0]
fb = features.loc[features["scenario_id"] == scen_b].iloc[0]

st.caption(f"Family: `{fam}` · Pattern reference: `{ref_baseline}`")
st.divider()
st.subheader("Diagnosis comparison")

left, right = st.columns(2)


def render_card(col, row, feat, title):
    with col:
        st.markdown(f"### {title}")
        st.markdown(f"**{row['scenario_id']}**")
        health = row["health_status"]
        colour = "#137333" if health == "HEALTHY" else "#c5221f"
        st.markdown(
            f"<span style='color:{colour};font-weight:700'>● {health}</span>",
            unsafe_allow_html=True,
        )
        st.write(f"**Severity:** {row['severity']}")
        feed = row.get("feed_setting_percent")
        if pd.notna(feed):
            st.write(f"**Feed:** {float(feed):.0f} %")
        st.write(f"**Fault:** {row['predicted_fault_type']}")
        st.write(f"**Condition:** {row['predicted_condition']}")
        st.metric("Anomaly score", f"{row['anomaly_score_100']:.2f} / 100")
        st.metric("Evidence strength", f"{100 * row['diagnostic_confidence']:.0f} %")
        st.write(
            f"Peak gain: **{feat['peak_gain']:.3f} dBi**  ·  "
            f"θ = {feat['peak_theta']:.1f}°, φ = {feat['peak_phi']:.1f}°"
        )
        st.write(
            f"Mean |Δ|: **{feat['baseline_mean_absolute_difference']:.4f} dB**  ·  "
            f"Max |Δ|: **{feat['baseline_max_absolute_difference']:.4f} dB**"
        )


render_card(left, ra, fa, "Scenario A")
render_card(right, rb, fb, "Scenario B")

part_cols = [c for c in results.columns if c.startswith("score_part_")]
if part_cols:
    st.markdown("**Score component comparison**")
    labels_map = {
        "score_part_mean": "Mean |Δ|",
        "score_part_pct_0_5": "% > 0.5 dB",
        "score_part_pct_1": "% > 1 dB",
        "score_part_tilt": "Tilt",
    }
    fig_cmp = go.Figure()
    fig_cmp.add_trace(
        go.Bar(
            name=scen_a,
            x=[labels_map.get(c, c) for c in part_cols],
            y=[float(ra[c]) for c in part_cols],
            marker_color="#1967d2",
        )
    )
    fig_cmp.add_trace(
        go.Bar(
            name=scen_b,
            x=[labels_map.get(c, c) for c in part_cols],
            y=[float(rb[c]) for c in part_cols],
            marker_color="#c5221f",
        )
    )
    fig_cmp.update_layout(
        barmode="group",
        height=280,
        margin=dict(l=40, r=20, t=20, b=40),
        yaxis_title="Points",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})

st.divider()
st.subheader("Radiation pattern comparison")

rad_a = load_radiation(scen_a)
rad_b = load_radiation(scen_b)
rad_base = load_radiation(ref_baseline)

if rad_a is None or rad_b is None:
    st.warning("Radiation data missing for one or both scenarios.")
else:
    cut_a = get_phi_cut(rad_a)
    cut_b = get_phi_cut(rad_b)
    cut_base = get_phi_cut(rad_base) if rad_base is not None else None

    healthy_name = (
        "Healthy array (all on)" if fam == "array7" else "Healthy (50% feed)"
    )

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=(
            f"Radiation patterns (φ ≈ {NOMINAL_PHI_DEG:.0f}°)",
            "Difference from healthy reference (ΔdB)",
        ),
    )

    if cut_base is not None and not cut_base.empty:
        fig.add_trace(
            go.Scatter(
                x=cut_base["theta"], y=cut_base["gain_total"],
                mode="lines", name=healthy_name,
                line=dict(color="#137333", width=2, dash="dash"),
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=cut_a["theta"], y=cut_a["gain_total"],
            mode="lines", name=scen_a,
            line=dict(color="#1967d2", width=2.5),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cut_b["theta"], y=cut_b["gain_total"],
            mode="lines", name=scen_b,
            line=dict(color="#c5221f", width=2.5),
        ),
        row=1, col=1,
    )

    if cut_base is not None and not cut_base.empty:
        for cut, name, colour in [
            (cut_a, f"Δ {scen_a}", "#1967d2"),
            (cut_b, f"Δ {scen_b}", "#c5221f"),
        ]:
            merged = pd.merge(
                cut_base[["theta", "gain_total"]],
                cut[["theta", "gain_total"]],
                on="theta",
                suffixes=("_base", "_curr"),
            )
            delta = merged["gain_total_curr"] - merged["gain_total_base"]
            fig.add_trace(
                go.Scatter(
                    x=merged["theta"], y=delta, mode="lines", name=name,
                    line=dict(color=colour, width=2),
                ),
                row=2, col=1,
            )
        fig.add_hline(y=0, line_dash="dot", line_color="#5f6368", row=2, col=1)

    fig.update_layout(
        height=620,
        margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white",
    )
    fig.update_xaxes(title_text="θ (deg)", row=2, col=1)
    fig.update_yaxes(title_text="Gain (dBi)", row=1, col=1)
    fig.update_yaxes(title_text="Δ Gain (dB)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Top: absolute patterns (reference dashed: `{ref_baseline}`). "
        "Bottom: difference of each scenario from the family healthy reference."
    )

st.divider()
st.caption(
    "Antenna Health Monitoring Pipeline | "
    "FEKO → Ingestion → Parquet → Features → "
    "Anomaly Detection → Classification → SQLite → Dashboard"
)