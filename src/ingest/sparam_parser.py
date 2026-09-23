"""
sparam_parser.py

Parser for FEKO / Touchstone S-parameter files.

Supported formats:
    - MA : Magnitude / Angle
    - RI : Real / Imaginary
    - DB : dB / Angle

Typical FEKO header:
    # Hz S MA R 5.0000000E+01

For a 7-port antenna array, the file contains:
    7 x 7 = 49 S-parameters per frequency point.

Example output columns:
    frequency_hz
    port_from
    port_to
    s_parameter
    magnitude
    phase_deg
    real
    imag
    magnitude_db
    return_loss_db
    reflected_power_fraction
    reference_impedance_ohm

Author:
    Antenna Health Monitoring Pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TouchstoneHeader:
    """Stores the information contained in a Touchstone option line."""

    frequency_unit: str
    parameter_type: str
    data_format: str
    reference_impedance_ohm: float


@dataclass
class SParameterResult:
    """Container returned by the parser."""

    dataframe: pd.DataFrame
    num_ports: int
    num_frequency_points: int
    header: TouchstoneHeader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FREQUENCY_MULTIPLIERS = {
    "HZ": 1.0,
    "KHZ": 1e3,
    "MHZ": 1e6,
    "GHZ": 1e9,
}

SUPPORTED_FORMATS = {"MA", "RI", "DB"}


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

def parse_touchstone_header(line: str) -> TouchstoneHeader:
    """
    Parse a Touchstone option line.

    Expected format:

        # Hz S MA R 50

    Meaning:

        Hz  -> frequency unit
        S   -> S-parameter
        MA  -> magnitude / angle
        R   -> reference impedance
        50  -> reference impedance value

    Parameters
    ----------
    line : str
        Touchstone option line.

    Returns
    -------
    TouchstoneHeader
        Parsed header information.

    Raises
    ------
    ValueError
        If the header is invalid or unsupported.
    """

    line = line.strip()

    if not line.startswith("#"):
        raise ValueError(
            f"Invalid Touchstone option line. Expected '#', got: {line}"
        )

    tokens = line[1:].split()

    if len(tokens) < 5:
        raise ValueError(
            f"Invalid Touchstone header: {line}"
        )

    frequency_unit = tokens[0].upper()
    parameter_type = tokens[1].upper()
    data_format = tokens[2].upper()

    if frequency_unit not in FREQUENCY_MULTIPLIERS:
        raise ValueError(
            f"Unsupported frequency unit: {frequency_unit}"
        )

    if parameter_type != "S":
        raise ValueError(
            f"Unsupported network parameter type: {parameter_type}. "
            "This parser expects S-parameters."
        )

    if data_format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported S-parameter format: {data_format}. "
            f"Supported formats: {SUPPORTED_FORMATS}"
        )

    try:
        reference_index = tokens.index("R")
    except ValueError:
        try:
            reference_index = tokens.index("r")
        except ValueError:
            raise ValueError(
                f"Touchstone header does not contain reference impedance: {line}"
            )

    if reference_index + 1 >= len(tokens):
        raise ValueError(
            f"Reference impedance value missing from header: {line}"
        )

    try:
        reference_impedance = float(tokens[reference_index + 1])
    except ValueError as exc:
        raise ValueError(
            f"Invalid reference impedance in header: {line}"
        ) from exc

    return TouchstoneHeader(
        frequency_unit=frequency_unit,
        parameter_type=parameter_type,
        data_format=data_format,
        reference_impedance_ohm=reference_impedance,
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def detect_num_ports(file_path: Path) -> int:
    """
    Determine the number of ports from a Touchstone filename.

    Examples
    --------
    antenna.s1p -> 1
    antenna.s2p -> 2
    antenna.s7p -> 7
    antenna.snp -> requires explicit detection from data

    Parameters
    ----------
    file_path : Path
        Path to the Touchstone file.

    Returns
    -------
    int
        Number of ports.
    """

    suffix = file_path.suffix.lower()

    # Standard Touchstone naming:
    # .s1p, .s2p, .s3p, ... .sNp
    if suffix.startswith(".s") and suffix.endswith("p"):
        port_text = suffix[2:-1]

        if port_text.isdigit():
            num_ports = int(port_text)

            if num_ports <= 0:
                raise ValueError(
                    f"Invalid number of ports in filename: {file_path.name}"
                )

            return num_ports

    raise ValueError(
        f"Could not determine number of ports from filename: "
        f"{file_path.name}. "
        "Rename the file to .s7p for a 7-port file, or provide "
        "num_ports explicitly to parse_sparameter_file()."
    )


def convert_frequency_to_hz(
    frequency: float,
    frequency_unit: str,
) -> float:
    """Convert a Touchstone frequency value to Hz."""

    unit = frequency_unit.upper()

    if unit not in FREQUENCY_MULTIPLIERS:
        raise ValueError(
            f"Unsupported frequency unit: {frequency_unit}"
        )

    return frequency * FREQUENCY_MULTIPLIERS[unit]


def s_parameter_name(port_from: int, port_to: int) -> str:
    """
    Create an S-parameter label.

    Example:
        port_from=1, port_to=2 -> S12
    """

    return f"S{port_from}{port_to}"


# ---------------------------------------------------------------------------
# S-parameter conversion
# ---------------------------------------------------------------------------

def convert_pair_to_complex(
    value_1: float,
    value_2: float,
    data_format: str,
) -> complex:
    """
    Convert a Touchstone value pair to a complex number.

    MA:
        value_1 = magnitude
        value_2 = angle in degrees

    RI:
        value_1 = real
        value_2 = imaginary

    DB:
        value_1 = magnitude in dB
        value_2 = angle in degrees
    """

    data_format = data_format.upper()

    if data_format == "MA":
        magnitude = value_1
        phase_rad = np.deg2rad(value_2)

        return magnitude * np.exp(1j * phase_rad)

    if data_format == "RI":
        return complex(value_1, value_2)

    if data_format == "DB":
        magnitude = 10 ** (value_1 / 20.0)
        phase_rad = np.deg2rad(value_2)

        return magnitude * np.exp(1j * phase_rad)

    raise ValueError(
        f"Unsupported S-parameter format: {data_format}"
    )


def calculate_input_impedance(
    reflection_coefficient: complex,
    reference_impedance_ohm: float,
) -> complex:
    """
    Calculate input impedance from a reflection coefficient.

    Formula:
        Zin = Z0 * (1 + Gamma) / (1 - Gamma)

    Parameters
    ----------
    reflection_coefficient : complex
        Complex reflection coefficient Sii.

    reference_impedance_ohm : float
        Reference impedance Z0 in ohms.

    Returns
    -------
    complex
        Complex input impedance in ohms.

        Real part  -> resistance (R)
        Imaginary  -> reactance (X)
    """

    if reference_impedance_ohm <= 0:
        raise ValueError(
            "Reference impedance must be greater than zero."
        )

    denominator = 1.0 - reflection_coefficient

    # Avoid division by zero if Gamma approaches 1.
    if abs(denominator) < 1e-12:
        return complex(np.inf, np.inf)

    return (
        reference_impedance_ohm
        * (1.0 + reflection_coefficient)
        / denominator
    )



# ---------------------------------------------------------------------------
# Data reading
# ---------------------------------------------------------------------------

def _remove_comments_and_blank_lines(
    lines: List[str],
) -> List[str]:
    """
    Remove blank lines and comment lines.

    Touchstone comments begin with '!'.

    Returns
    -------
    list[str]
        Cleaned lines.
    """

    cleaned = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("!"):
            continue

        cleaned.append(stripped)

    return cleaned


def _find_option_line(
    lines: List[str],
) -> Tuple[int, TouchstoneHeader]:
    """
    Find and parse the Touchstone option line.
    """

    for index, line in enumerate(lines):
        if line.startswith("#"):
            return index, parse_touchstone_header(line)

    raise ValueError(
        "No Touchstone option line beginning with '#' was found."
    )


def _parse_numeric_tokens(
    lines: List[str],
    option_line_index: int,
) -> List[float]:
    """
    Convert all data tokens after the option line into floats.

    Returns
    -------
    list[float]
        Flat list of numeric values.
    """

    numeric_values = []

    for line in lines[option_line_index + 1:]:
        # Remove inline comments.
        if "!" in line:
            line = line.split("!", 1)[0].strip()

        if not line:
            continue

        tokens = line.split()

        for token in tokens:
            try:
                numeric_values.append(float(token))
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse numeric token '{token}' "
                    f"from line: {line}"
                ) from exc

    return numeric_values


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_sparameter_file(
    file_path: str | Path,
    num_ports: int | None = None,
) -> SParameterResult:
    """
    Parse a Touchstone S-parameter file.

    Parameters
    ----------
    file_path : str or Path
        Path to .sNp / Touchstone file.

    num_ports : int, optional
        Number of ports.

        If omitted, the parser attempts to determine the number of ports
        from the filename, e.g. .s7p -> 7.

    Returns
    -------
    SParameterResult
        Parsed S-parameter data and metadata.

    Notes
    -----
    A single frequency point for an N-port network contains:

        1 frequency value
        +
        2 * N^2 S-parameter values

    For a 7-port network:

        1 + 2 * 49 = 99 numeric values

    The FEKO export may wrap the matrix over multiple physical lines.
    This parser therefore treats the numerical data as a continuous stream
    rather than assuming one S-parameter row per physical line.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"S-parameter file not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Provided path is not a file: {file_path}"
        )

    if num_ports is None:
        num_ports = detect_num_ports(file_path)

    if num_ports <= 0:
        raise ValueError(
            f"Number of ports must be positive. Got: {num_ports}"
        )

    with file_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:
        raw_lines = file.readlines()

    lines = _remove_comments_and_blank_lines(raw_lines)

    if not lines:
        raise ValueError(
            f"S-parameter file is empty: {file_path}"
        )

    option_line_index, header = _find_option_line(lines)

    numeric_values = _parse_numeric_tokens(
        lines,
        option_line_index,
    )

    values_per_frequency = 1 + (2 * num_ports * num_ports)

    if len(numeric_values) % values_per_frequency != 0:
        raise ValueError(
            "The number of numeric values in the Touchstone file is "
            "inconsistent with the detected number of ports.\n"
            f"Number of ports: {num_ports}\n"
            f"Expected values per frequency: {values_per_frequency}\n"
            f"Actual numeric values: {len(numeric_values)}\n"
            f"Remainder: {len(numeric_values) % values_per_frequency}\n\n"
            "Check that the number of ports is correct."
        )

    num_frequency_points = (
        len(numeric_values) // values_per_frequency
    )

    rows = []

    index = 0

    for _ in range(num_frequency_points):

        frequency_raw = numeric_values[index]
        index += 1

        frequency_hz = convert_frequency_to_hz(
            frequency_raw,
            header.frequency_unit,
        )

        # Touchstone matrix is stored as:
        #
        # S11 S12 S13 ...
        # S21 S22 S23 ...
        # ...
        #
        # Each S parameter consists of two values:
        # magnitude/angle, real/imaginary, or dB/angle.

        for port_from in range(1, num_ports + 1):

            for port_to in range(1, num_ports + 1):

                value_1 = numeric_values[index]
                value_2 = numeric_values[index + 1]

                index += 2

                complex_value = convert_pair_to_complex(
                    value_1,
                    value_2,
                    header.data_format,
                )

                # Calculate input impedance only for reflection parameters Sii.
                # Off-diagonal coupling parameters do not represent input impedance.
                if port_from == port_to:
                    input_impedance = calculate_input_impedance(
                        reflection_coefficient=complex_value,
                        reference_impedance_ohm=header.reference_impedance_ohm,
                    )

                    input_impedance_real = input_impedance.real
                    input_impedance_imag = input_impedance.imag
                    input_impedance_magnitude = abs(input_impedance)

                else:
                    input_impedance_real = np.nan
                    input_impedance_imag = np.nan
                    input_impedance_magnitude = np.nan

                magnitude = abs(complex_value)

                # Protect against log10(0).
                if magnitude > 0:
                    magnitude_db = 20.0 * np.log10(magnitude)
                    return_loss_db = -magnitude_db
                else:
                    magnitude_db = -np.inf
                    return_loss_db = np.inf

                phase_deg = np.angle(
                    complex_value,
                    deg=True,
                )

                reflected_power_fraction = magnitude ** 2

                rows.append(
                    {
                        "frequency_hz": frequency_hz,
                        "port_from": port_from,
                        "port_to": port_to,
                        "s_parameter": s_parameter_name(
                            port_from,
                            port_to,
                        ),
                        "magnitude": magnitude,
                        "phase_deg": phase_deg,
                        "real": complex_value.real,
                        "imag": complex_value.imag,
                        "magnitude_db": magnitude_db,
                        "return_loss_db": return_loss_db,
                        "reflected_power_fraction": (
                            reflected_power_fraction
                        ),
                        "reference_impedance_ohm": (
                            header.reference_impedance_ohm
                        ),
                        "input_impedance_real_ohm": input_impedance_real,
                        "input_impedance_imag_ohm": input_impedance_imag,
                        "input_impedance_magnitude_ohm": input_impedance_magnitude,
                        
                    }
                )

    dataframe = pd.DataFrame(rows)

    return SParameterResult(
        dataframe=dataframe,
        num_ports=num_ports,
        num_frequency_points=num_frequency_points,
        header=header,
    )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def parse_sparameter_dataframe(
    file_path: str | Path,
    num_ports: int | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper returning only the DataFrame.
    """

    result = parse_sparameter_file(
        file_path=file_path,
        num_ports=num_ports,
    )

    return result.dataframe


def get_reflection_parameters(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract diagonal S-parameters.

    For an N-port network:

        S11
        S22
        ...
        SNN

    These represent the reflection behaviour of each port.
    """

    required_columns = {
        "port_from",
        "port_to",
    }

    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    reflection_df = dataframe[
        dataframe["port_from"] == dataframe["port_to"]
    ].copy()

    return reflection_df.reset_index(drop=True)


def get_coupling_parameters(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract off-diagonal S-parameters.

    These represent coupling between different ports/elements.
    """

    required_columns = {
        "port_from",
        "port_to",
    }

    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    coupling_df = dataframe[
        dataframe["port_from"] != dataframe["port_to"]
    ].copy()

    return coupling_df.reset_index(drop=True)


def get_port_parameter(
    dataframe: pd.DataFrame,
    port: int,
) -> pd.DataFrame:
    """
    Extract the reflection parameter for a specific port.

    Example:
        port=2 -> S22
    """

    if port <= 0:
        raise ValueError(
            f"Port number must be positive. Got {port}"
        )

    return dataframe[
        (dataframe["port_from"] == port)
        & (dataframe["port_to"] == port)
    ].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_sparameter_dataframe(
    dataframe: pd.DataFrame,
    expected_num_ports: int | None = None,
) -> None:
    """
    Validate the parsed S-parameter DataFrame.

    Raises
    ------
    ValueError
        If validation fails.
    """

    required_columns = {
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
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError(
            "S-parameter DataFrame is empty."
        )

    if dataframe["frequency_hz"].isna().any():
        raise ValueError(
            "Frequency contains NaN values."
        )

    if (dataframe["frequency_hz"] <= 0).any():
        raise ValueError(
            "Frequency contains zero or negative values."
        )

    if (dataframe["port_from"] <= 0).any():
        raise ValueError(
            "port_from contains invalid port numbers."
        )

    if (dataframe["port_to"] <= 0).any():
        raise ValueError(
            "port_to contains invalid port numbers."
        )

    if expected_num_ports is not None:

        if expected_num_ports <= 0:
            raise ValueError(
                "expected_num_ports must be positive."
            )

        max_port = max(
            dataframe["port_from"].max(),
            dataframe["port_to"].max(),
        )

        if max_port != expected_num_ports:
            raise ValueError(
                f"Expected {expected_num_ports} ports, "
                f"but highest detected port is {max_port}."
            )

        expected_parameters = (
            expected_num_ports
            * expected_num_ports
        )

        parameters_per_frequency = dataframe.groupby(
            "frequency_hz"
        ).size()

        if not (
            parameters_per_frequency
            == expected_parameters
        ).all():
            raise ValueError(
                "Each frequency point must contain exactly "
                f"{expected_parameters} S-parameters."
            )

    # The magnitude of a passive reflection coefficient should normally
    # be <= 1. We do not make this a hard error because active networks,
    # numerical effects, or unusual simulation conditions can produce
    # values greater than 1.
    #
    # Instead, this is only a warning printed to the console.
    if (dataframe["magnitude"] > 1.0 + 1e-6).any():
        print(
            "WARNING: Some S-parameter magnitudes are greater than 1. "
            "This may be valid for an active network, but should be "
            "checked for a passive antenna."
        )


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Simple command-line interface.

    Example:

        python ingest/sparam_parser.py data/raw/H01_NOM/H01_NOM.s7p
    """

    import argparse

    parser = argparse.ArgumentParser(
        description="Parse a FEKO/Touchstone S-parameter file."
    )

    parser.add_argument(
        "file",
        type=str,
        help="Path to the .sNp Touchstone file.",
    )

    parser.add_argument(
        "--ports",
        type=int,
        default=None,
        help=(
            "Number of ports. "
            "If omitted, it is inferred from the filename."
        ),
    )

    args = parser.parse_args()

    result = parse_sparameter_file(
        file_path=args.file,
        num_ports=args.ports,
    )

    validate_sparameter_dataframe(
        result.dataframe,
        expected_num_ports=result.num_ports,
    )

    df = result.dataframe

    print()
    print("=" * 60)
    print("S-PARAMETER PARSER")
    print("=" * 60)

    print(f"File: {args.file}")
    print(f"Ports: {result.num_ports}")
    print(
        f"Frequency points: "
        f"{result.num_frequency_points}"
    )
    print(
        f"Reference impedance: "
        f"{result.header.reference_impedance_ohm:.3f} Ω"
    )
    print(
        f"Format: "
        f"{result.header.data_format}"
    )
    print(
        f"Frequency unit: "
        f"{result.header.frequency_unit}"
    )

    print()
    print("DataFrame shape:")
    print(df.shape)

    print()
    print("Columns:")
    print(list(df.columns))

    print()
    print("Reflection parameters:")
    reflection = get_reflection_parameters(df)

    print(
        reflection[
            [
                "frequency_hz",
                "s_parameter",
                "magnitude",
                "phase_deg",
                "magnitude_db",
                "return_loss_db",
            ]
        ].to_string(index=False)
    )

    print()
    print("Input impedance:")
    print(
        reflection[
            [
                "frequency_hz",
                "s_parameter",
                "input_impedance_real_ohm",
                "input_impedance_imag_ohm",
                "input_impedance_magnitude_ohm",
            ]
        ].to_string(index=False)
    )

    print()
    print("First 10 rows:")
    print(df.head(10).to_string(index=False))

    print()
    print("TEST / VALIDATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

