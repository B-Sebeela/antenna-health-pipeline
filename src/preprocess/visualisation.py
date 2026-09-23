"""
Visualisation utilities for the Antenna Health Monitoring Pipeline.

This module reads processed antenna data from Parquet and produces
engineering-oriented visualisations.

Current visualisations:
    1. 2D radiation pattern:
       Gain Total vs Theta at a selected Phi angle.

    2. Full far-field heatmap:
       Gain Total across the complete Theta/Phi grid.

Architecture:

    FEKO .ffe
        |
        v
    ffe_parser.py
        |
        v
    parquet.py
        |
        v
    processed/*.parquet
        |
        v
    visualisation.py
        |
        +--> 2D radiation pattern
        |
        +--> Full far-field heatmap
"""

from pathlib import Path
from typing import Optional

import matplotlib

# Use a non-GUI backend so the pipeline can generate plots
# without depending on Tkinter/Tcl.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_processed_data(
    parquet_path: str | Path,
) -> pd.DataFrame:
    """
    Load processed antenna data from a Parquet file.

    Parameters
    ----------
    parquet_path : str or Path
        Path to the processed Parquet file.

    Returns
    -------
    pd.DataFrame
        Processed antenna measurement data.

    Raises
    ------
    FileNotFoundError
        If the Parquet file does not exist.
    ValueError
        If required columns are missing.
    """

    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {parquet_path}"
        )

    df = pd.read_parquet(parquet_path)

    required_columns = {
        "theta",
        "phi",
        "gain_total",
        "magnitude_norm_db",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Parquet file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return df


# ---------------------------------------------------------------------------
# 2D radiation pattern
# ---------------------------------------------------------------------------

def plot_radiation_pattern(
    parquet_path: str | Path,
    phi_angle: float = 0.0,
    output_path: Optional[str | Path] = None,
    show: bool = True,
):
    """
    Plot the total antenna gain against theta for a selected phi angle.

    This produces a conventional 2D antenna radiation pattern.

    Parameters
    ----------
    parquet_path : str or Path
        Path to the processed Parquet file.

    phi_angle : float, optional
        Phi angle to plot, in degrees. Default is 0 degrees.

    output_path : str or Path, optional
        If provided, save the figure to this path.

    show : bool, optional
        Whether to display the plot. Default is True.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """

    df = load_processed_data(parquet_path)

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
            f"No data found for phi = {phi_angle} degrees."
        )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        pattern["theta"],
        pattern["gain_total"],
        marker="o",
        markersize=3,
        linewidth=1.5,
    )

    ax.set_title(
        f"Antenna Radiation Pattern "
        f"(φ = {closest_phi:.1f}°)"
    )

    ax.set_xlabel("Theta (degrees)")
    ax.set_ylabel("Total Gain (dBi)")

    ax.grid(True)

    ax.set_xlim(
        pattern["theta"].min(),
        pattern["theta"].max(),
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
# Full far-field heatmap
# ---------------------------------------------------------------------------

def plot_far_field_heatmap(
    parquet_path: str | Path,
    output_path: Optional[str | Path] = None,
    show: bool = True,
):
    """
    Plot the complete theta/phi gain dataset as a heatmap.

    Parameters
    ----------
    parquet_path : str or Path
        Path to the processed Parquet file.

    output_path : str or Path, optional
        If provided, save the figure to this path.

    show : bool, optional
        Whether to display the plot. Default is True.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """

    df = load_processed_data(parquet_path)

    # Convert the long-form Parquet data into a 2D matrix.
    gain_grid = df.pivot(
        index="theta",
        columns="phi",
        values="gain_total",
    )

    # Sort both dimensions.
    gain_grid = gain_grid.sort_index()
    gain_grid = gain_grid.sort_index(axis=1)

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    image = ax.imshow(
        gain_grid.values,
        aspect="auto",
        origin="lower",
        extent=[
            gain_grid.columns.min(),
            gain_grid.columns.max(),
            gain_grid.index.min(),
            gain_grid.index.max(),
        ],
    )

    ax.set_title(
        "Antenna Far-Field Gain"
    )

    ax.set_xlabel(
        "Phi (degrees)"
    )

    ax.set_ylabel(
        "Theta (degrees)"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Total Gain (dBi)",
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
# Generate all visualisations
# ---------------------------------------------------------------------------

def generate_all_plots(
    parquet_path: str | Path,
    output_directory: str | Path = "data/outputs",
    phi_angle: float = 0.0,
    show: bool = True,
):
    """
    Generate all standard antenna visualisations.

    Parameters
    ----------
    parquet_path : str or Path
        Path to the processed Parquet file.

    output_directory : str or Path, optional
        Directory where plots should be saved.

    phi_angle : float, optional
        Phi angle for the 2D radiation pattern.

    show : bool, optional
        Whether to display the plots.

    Returns
    -------
    dict
        Paths to generated visualisation files.
    """

    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    radiation_path = (
        output_directory /
        f"radiation_pattern_phi_{phi_angle:.0f}.png"
    )

    heatmap_path = (
        output_directory /
        "far_field_heatmap.png"
    )

    plot_radiation_pattern(
        parquet_path=parquet_path,
        phi_angle=phi_angle,
        output_path=radiation_path,
        show=show,
    )

    # Close the first figure when generating both without displaying.
    if not show:
        plt.close()

    plot_far_field_heatmap(
        parquet_path=parquet_path,
        output_path=heatmap_path,
        show=show,
    )

    if not show:
        plt.close()

    return {
        "radiation_pattern": radiation_path,
        "far_field_heatmap": heatmap_path,
    }