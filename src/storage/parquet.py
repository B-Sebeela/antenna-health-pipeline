"""
parquet.py
----------

Parquet storage utilities for the Antenna Health Monitoring Pipeline.

Two types of datasets are stored:

1. Far-field data
   ----------------
   Generated from FEKO .ffe files.

       .ffe
         |
         v
       ffe_parser.py
         |
         v
       farfield.parquet


2. S-parameter data
   -----------------
   Generated from FEKO / Touchstone .sNp files.

       .s7p
         |
         v
       sparam_parser.py
         |
         v
       sparameters.parquet


SQLite stores scenario metadata and paths to these datasets.

Important
---------
Far-field and S-parameter data are stored in separate Parquet files
because they have fundamentally different structures.

Far-field:
    rows = theta/phi angular samples

S-parameters:
    rows = frequency × port_from × port_to
"""


from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


# ===========================================================================
# FAR-FIELD COLUMN MAPPING
# ===========================================================================

# These names correspond to the output contract of ffe_parser.py.

PARQUET_COLUMNS = {

    "theta": "theta_deg",

    "phi": "phi_deg",

    "e_theta_real": "re_etheta",

    "e_theta_imag": "im_etheta",

    "e_phi_real": "re_ephi",

    "e_phi_imag": "im_ephi",

    "gain_theta": "gain_theta_dbi",

    "gain_phi": "gain_phi_dbi",

    "gain_total": "gain_total_dbi",

    "magnitude_norm_db": "magnitude_norm_db",
}


# ===========================================================================
# S-PARAMETER COLUMNS
# ===========================================================================

# These columns correspond directly to the DataFrame produced by
# sparam_parser.py.

SPARAMETER_COLUMNS = [

    "frequency_hz",

    "port_from",

    "port_to",

    "s_parameter",

    "magnitude",

    "phase_deg",

    "real",

    "imag",

    "magnitude_db",

    "return_loss_db",

    "reflected_power_fraction",

    "reference_impedance_ohm",

    "input_impedance_real_ohm",

    "input_impedance_imag_ohm",

    "input_impedance_magnitude_ohm",
]


# ===========================================================================
# FAR-FIELD DATAFRAME
# ===========================================================================

def record_to_dataframe(
    record: Dict[str, Any],
) -> pd.DataFrame:
    """
    Convert a parsed FFE record into a pandas DataFrame.

    Each row represents one theta/phi measurement point.

    Parameters
    ----------
    record : dict
        Structured record returned by ffe_parser.parse_ffe().

    Returns
    -------
    pd.DataFrame
        Far-field measurement and feature data.
    """

    # ----------------------------------------------------------------------
    # Check required parser fields.
    # ----------------------------------------------------------------------

    missing = [

        parser_field

        for parser_field in PARQUET_COLUMNS.values()

        if parser_field not in record

    ]

    if missing:

        raise ValueError(
            "Record is missing required fields: "
            f"{missing}"
        )

    # ----------------------------------------------------------------------
    # Build clean DataFrame column names.
    # ----------------------------------------------------------------------

    data = {

        parquet_name: record[parser_name]

        for parquet_name, parser_name
        in PARQUET_COLUMNS.items()

    }

    # ----------------------------------------------------------------------
    # Verify array lengths.
    # ----------------------------------------------------------------------

    lengths = {

        column: len(values)

        for column, values
        in data.items()

    }

    unique_lengths = set(
        lengths.values()
    )

    if len(unique_lengths) != 1:

        raise ValueError(
            "Measurement arrays have inconsistent lengths: "
            f"{lengths}"
        )

    return pd.DataFrame(
        data
    )


# ===========================================================================
# SAVE FAR-FIELD DATA
# ===========================================================================

def save_record_to_parquet(
    record: Dict[str, Any],
    output_path: str | Path,
) -> Path:
    """
    Convert a parsed FFE record to a Parquet file.

    Parameters
    ----------
    record : dict
        Parsed FFE record.

    output_path : str | Path
        Destination Parquet file.

    Returns
    -------
    Path
        Path to created Parquet file.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = record_to_dataframe(
        record
    )

    df.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
    )

    return output_path


# ===========================================================================
# LOAD FAR-FIELD DATA
# ===========================================================================

def load_parquet(
    parquet_path: str | Path,
) -> pd.DataFrame:
    """
    Load a far-field Parquet dataset.
    """

    parquet_path = Path(
        parquet_path
    )

    if not parquet_path.exists():

        raise FileNotFoundError(
            f"Parquet file not found: "
            f"{parquet_path}"
        )

    return pd.read_parquet(
        parquet_path,
        engine="pyarrow",
    )


# ===========================================================================
# FAR-FIELD SUMMARY
# ===========================================================================

def parquet_summary(
    parquet_path: str | Path,
) -> Dict[str, Any]:
    """
    Return basic information about a far-field Parquet dataset.
    """

    df = load_parquet(
        parquet_path
    )

    return {

        "path":
            str(parquet_path),

        "rows":
            len(df),

        "columns":
            list(df.columns),

        "theta_min":
            float(df["theta"].min()),

        "theta_max":
            float(df["theta"].max()),

        "phi_min":
            float(df["phi"].min()),

        "phi_max":
            float(df["phi"].max()),
    }


# ===========================================================================
# S-PARAMETER DATAFRAME
# ===========================================================================

def validate_sparameter_dataframe_for_parquet(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate the S-parameter DataFrame before writing it to Parquet.

    This function intentionally performs storage-level validation only.
    The detailed electrical validation remains in sparam_parser.py.
    """

    missing = set(
        SPARAMETER_COLUMNS
    ) - set(
        dataframe.columns
    )

    if missing:

        raise ValueError(
            "S-parameter DataFrame is missing required columns: "
            f"{sorted(missing)}"
        )

    if dataframe.empty:

        raise ValueError(
            "Cannot store an empty S-parameter DataFrame."
        )

    if dataframe["frequency_hz"].isna().any():

        raise ValueError(
            "S-parameter frequency contains NaN values."
        )

    if (
        dataframe["port_from"] <= 0
    ).any():

        raise ValueError(
            "S-parameter port_from contains invalid values."
        )

    if (
        dataframe["port_to"] <= 0
    ).any():

        raise ValueError(
            "S-parameter port_to contains invalid values."
        )


# ===========================================================================
# SAVE S-PARAMETERS
# ===========================================================================

def save_sparameter_to_parquet(
    dataframe: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Save parsed S-parameter data to Parquet.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame produced by sparam_parser.py.

    output_path : str | Path
        Destination Parquet file.

    Returns
    -------
    Path
        Path to created Parquet file.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Validate before storage.
    validate_sparameter_dataframe_for_parquet(
        dataframe
    )

    # Keep a predictable column order.
    dataframe = dataframe[
        SPARAMETER_COLUMNS
    ].copy()

    dataframe.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
    )

    return output_path


# ===========================================================================
# LOAD S-PARAMETERS
# ===========================================================================

def load_sparameter_parquet(
    parquet_path: str | Path,
) -> pd.DataFrame:
    """
    Load an S-parameter Parquet dataset.

    Parameters
    ----------
    parquet_path : str | Path
        Path to S-parameter Parquet file.

    Returns
    -------
    pd.DataFrame
        S-parameter dataset.
    """

    parquet_path = Path(
        parquet_path
    )

    if not parquet_path.exists():

        raise FileNotFoundError(
            f"S-parameter Parquet file not found: "
            f"{parquet_path}"
        )

    dataframe = pd.read_parquet(
        parquet_path,
        engine="pyarrow",
    )

    validate_sparameter_dataframe_for_parquet(
        dataframe
    )

    return dataframe


# ===========================================================================
# S-PARAMETER SUMMARY
# ===========================================================================

def sparameter_parquet_summary(
    parquet_path: str | Path,
) -> Dict[str, Any]:
    """
    Return summary information about an S-parameter Parquet dataset.

    Parameters
    ----------
    parquet_path : str | Path
        Path to S-parameter Parquet file.

    Returns
    -------
    dict
        Dataset summary.
    """

    df = load_sparameter_parquet(
        parquet_path
    )

    return {

        "path":
            str(parquet_path),

        "rows":
            len(df),

        "columns":
            list(df.columns),

        "num_ports":
            int(
                max(
                    df["port_from"].max(),
                    df["port_to"].max(),
                )
            ),

        "num_frequency_points":
            int(
                df["frequency_hz"].nunique()
            ),

        "frequency_min_hz":
            float(
                df["frequency_hz"].min()
            ),

        "frequency_max_hz":
            float(
                df["frequency_hz"].max()
            ),

        "reference_impedance_ohm":
            float(
                df["reference_impedance_ohm"].iloc[0]
            ),
    }


# ===========================================================================
# CONVENIENCE S-PARAMETER FILTERS
# ===========================================================================

def get_sparameter_reflections(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract S11, S22, ..., SNN from an S-parameter DataFrame.

    These are the reflection parameters for each antenna port.
    """

    validate_sparameter_dataframe_for_parquet(
        dataframe
    )

    return dataframe[
        dataframe["port_from"]
        ==
        dataframe["port_to"]
    ].copy().reset_index(
        drop=True
    )


def get_sparameter_coupling(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract off-diagonal S-parameters.

    These represent coupling between different antenna ports/elements.
    """

    validate_sparameter_dataframe_for_parquet(
        dataframe
    )

    return dataframe[
        dataframe["port_from"]
        !=
        dataframe["port_to"]
    ].copy().reset_index(
        drop=True
    )


def get_sparameter_for_port(
    dataframe: pd.DataFrame,
    port: int,
) -> pd.DataFrame:
    """
    Extract the reflection parameter Sii for a specific port.

    Example
    -------
    port=2

        returns S22
    """

    if port <= 0:

        raise ValueError(
            f"Port number must be positive. "
            f"Got {port}"
        )

    validate_sparameter_dataframe_for_parquet(
        dataframe
    )

    return dataframe[
        (dataframe["port_from"] == port)
        &
        (dataframe["port_to"] == port)
    ].copy().reset_index(
        drop=True
    )