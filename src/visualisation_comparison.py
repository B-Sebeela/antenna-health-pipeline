"""
Scenario comparison visualisations for the Antenna Health Monitoring Pipeline.

This module compares processed FEKO far-field data between antenna
scenarios.

Current comparisons:
    1. Overlayed 2D radiation patterns at a selected phi angle.
    2. Radiation-pattern difference relative to a nominal scenario.

Architecture:

    processed/<scenario>/farfield.parquet
                    |
                    v
        visualisation_comparison.py
                    |
          +---------+---------+
          |                   |
          v                   v
    Pattern overlay      Difference plot
"""


from pathlib import Path
from typing import Optional

import matplotlib

# Use a non-GUI backend so plots can be generated
# from PowerShell/automation without Tkinter.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scenario_data(
    processed_directory: str | Path,
    scenario_id: str,
) -> pd.DataFrame:
    """
    Load processed far-field data for one scenario.

    Parameters
    ----------
    processed_directory : str or Path
        Root processed-data directory.

    scenario_id : str
        Scenario identifier, e.g. H01_NOM.

    Returns
    -------
    pd.DataFrame
        Scenario far-field data.
    """

    processed_directory = Path(processed_directory)

    parquet_path = (
        processed_directory
        / scenario_id
        / "farfield.parquet"
    )

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Processed Parquet file not found for scenario "
            f"'{scenario_id}': {parquet_path}"
        )

    df = pd.read_parquet(parquet_path)

    required_columns = {
        "theta",
        "phi",
        "gain_total",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Scenario '{scenario_id}' is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return df


# ---------------------------------------------------------------------------
# 2D scenario comparison
# ---------------------------------------------------------------------------

def plot_scenario_comparison(
    processed_directory: str | Path,
    scenario_ids: list[str],
    phi_angle: float = 0.0,
    output_path: Optional[str | Path] = None,
    show: bool = True,
):
    """
    Plot multiple antenna radiation patterns on the same graph.

    Parameters
    ----------
    processed_directory : str or Path
        Root processed-data directory.

    scenario_ids : list[str]
        Scenario identifiers to compare.

    phi_angle : float, optional
        Phi angle to compare.

    output_path : str or Path, optional
        Path for the generated image.

    show : bool, optional
        Whether to display the plot.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for scenario_id in scenario_ids:

        df = load_scenario_data(
            processed_directory,
            scenario_id,
        )

        # Find the closest available phi value.
        available_phi = df["phi"].unique()

        closest_phi = min(
            available_phi,
            key=lambda value: abs(value - phi_angle),
        )

        pattern = df[
            df["phi"] == closest_phi
        ].copy()

        pattern = pattern.sort_values("theta")

        if pattern.empty:
            raise ValueError(
                f"No data found for scenario "
                f"'{scenario_id}' at phi={phi_angle}."
            )

        ax.plot(
            pattern["theta"],
            pattern["gain_total"],
            linewidth=1.5,
            label=scenario_id,
        )

    ax.set_title(
        f"Antenna Radiation Pattern Comparison "
        f"(φ ≈ {closest_phi:.1f}°)"
    )

    ax.set_xlabel(
        "Theta (degrees)"
    )

    ax.set_ylabel(
        "Total Gain (dBi)"
    )

    ax.grid(True)
    ax.legend()

    ax.set_xlim(
        0,
        180,
    )

    fig.tight_layout()

    if output_path is not None:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Difference from nominal
# ---------------------------------------------------------------------------

def calculate_pattern_difference(
    nominal_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    phi_angle: float = 0.0,
) -> pd.DataFrame:
    """
    Calculate gain difference between a scenario and the nominal antenna.

    Difference is:

        scenario gain - nominal gain

    Parameters
    ----------
    nominal_df : pd.DataFrame
        Nominal antenna data.

    scenario_df : pd.DataFrame
        Scenario being compared against nominal.

    phi_angle : float
        Phi angle to compare.

    Returns
    -------
    pd.DataFrame
        Theta and gain-difference values.
    """

    nominal_phi = min(
        nominal_df["phi"].unique(),
        key=lambda value: abs(value - phi_angle),
    )

    scenario_phi = min(
        scenario_df["phi"].unique(),
        key=lambda value: abs(value - phi_angle),
    )

    nominal = nominal_df[
        nominal_df["phi"] == nominal_phi
    ][
        ["theta", "gain_total"]
    ].copy()

    scenario = scenario_df[
        scenario_df["phi"] == scenario_phi
    ][
        ["theta", "gain_total"]
    ].copy()

    nominal = nominal.rename(
        columns={
            "gain_total": "nominal_gain"
        }
    )

    scenario = scenario.rename(
        columns={
            "gain_total": "scenario_gain"
        }
    )

    comparison = pd.merge(
        nominal,
        scenario,
        on="theta",
        how="inner",
    )

    comparison["gain_difference_db"] = (
        comparison["scenario_gain"]
        - comparison["nominal_gain"]
    )

    return comparison


def plot_pattern_difference(
    processed_directory: str | Path,
    nominal_scenario: str,
    scenario_id: str,
    phi_angle: float = 0.0,
    output_path: Optional[str | Path] = None,
    show: bool = True,
):
    """
    Plot radiation-pattern difference relative to nominal.

    Positive values mean the scenario has higher gain than nominal.

    Negative values mean the scenario has lower gain than nominal.
    """

    nominal_df = load_scenario_data(
        processed_directory,
        nominal_scenario,
    )

    scenario_df = load_scenario_data(
        processed_directory,
        scenario_id,
    )

    comparison = calculate_pattern_difference(
        nominal_df=nominal_df,
        scenario_df=scenario_df,
        phi_angle=phi_angle,
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.plot(
        comparison["theta"],
        comparison["gain_difference_db"],
        linewidth=1.5,
    )

    ax.axhline(
        0,
        linewidth=1,
        linestyle="--",
    )

    ax.set_title(
        f"Radiation Pattern Difference: "
        f"{scenario_id} vs {nominal_scenario} "
        f"(φ ≈ {phi_angle:.1f}°)"
    )

    ax.set_xlabel(
        "Theta (degrees)"
    )

    ax.set_ylabel(
        "Gain Difference (dB)"
    )

    ax.grid(True)

    ax.set_xlim(
        comparison["theta"].min(),
        comparison["theta"].max(),
    )

    fig.tight_layout()

    if output_path is not None:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Generate comparison plots
# ---------------------------------------------------------------------------

def generate_scenario_comparison(
    processed_directory: str | Path = "data/processed",
    output_directory: str | Path = "data/outputs/comparison",
    phi_angle: float = 0.0,
):
    """
    Generate comparison plots for the current antenna scenarios.

    The nominal scenario is H01_NOM.
    """

    processed_directory = Path(
        processed_directory
    )

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    scenarios = [
        "H01_NOM",
        "MF01_L_PLUS",
        "MF02_L_MINUS",
        "M01_FEED_OFFSET",
    ]

    # ---------------------------------------------------------------
    # Overlay comparison
    # ---------------------------------------------------------------

    overlay_path = (
        output_directory
        / f"scenario_comparison_phi_{phi_angle:.0f}.png"
    )

    plot_scenario_comparison(
        processed_directory=processed_directory,
        scenario_ids=scenarios,
        phi_angle=phi_angle,
        output_path=overlay_path,
        show=False,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Individual difference plots
    # ---------------------------------------------------------------

    difference_paths = {}

    for scenario_id in scenarios:

        if scenario_id == "H01_NOM":
            continue

        difference_path = (
            output_directory
            / f"{scenario_id}_vs_H01_NOM_phi_{phi_angle:.0f}.png"
        )

        plot_pattern_difference(
            processed_directory=processed_directory,
            nominal_scenario="H01_NOM",
            scenario_id=scenario_id,
            phi_angle=phi_angle,
            output_path=difference_path,
            show=False,
        )

        plt.close()

        difference_paths[scenario_id] = difference_path

    return {
        "overlay": overlay_path,
        "differences": difference_paths,
    }


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    outputs = generate_scenario_comparison()

    print()
    print("=" * 70)
    print("SCENARIO COMPARISON COMPLETE")
    print("=" * 70)

    print()
    print(f"Overlay plot:")
    print(f"  {outputs['overlay']}")

    print()
    print("Difference plots:")

    for scenario_id, path in outputs["differences"].items():
        print(f"  {scenario_id}: {path}")

    print()
    print("=" * 70)

