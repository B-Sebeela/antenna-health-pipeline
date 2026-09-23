"""
ffe_parser.py
-------------
Parses POSTFEKO far-field export files (.ffe) into structured Python
dictionaries ready for preprocessing and storage.

File format (observed from CADFEKO/POSTFEKO 2022):
  ## comment lines  -> file-level metadata
  #  comment lines  -> request-level metadata
  data rows         -> whitespace-separated floats (9 columns)

Columns (0-indexed):
  0  Theta [deg]
  1  Phi   [deg]
  2  Re(Etheta)
  3  Im(Etheta)
  4  Re(Ephi)
  5  Im(Ephi)
  6  Gain(Theta)  [dBi]
  7  Gain(Phi)    [dBi]
  8  Gain(Total)  [dBi]

Usage:
    from src.ingest.ffe_parser import parse_ffe, parse_ffe_batch

    record = parse_ffe("path/to/file.ffe")
    records = parse_ffe_batch("data/raw/", failed_elements=[2, 5])
"""

from __future__ import annotations

import os
import re
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_ffe(
    filepath: str | Path,
    failed_elements: Optional[list[int]] = None,
    n_elements: Optional[int] = None,
    scenario_id: Optional[str] = None,
) -> dict:
    """
    Parse a single POSTFEKO .ffe file into a structured record.

    Parameters
    ----------
    filepath : str | Path
        Path to the .ffe file.
    failed_elements : list[int] | None
        Which element indices (0-based) are OFF in this simulation.
        Pass an empty list [] for the nominal (all-on) scenario.
        Pass None if unknown (the label_vector will be None).
    n_elements : int | None
        Total number of elements in the array. Required if
        failed_elements is provided.
    scenario_id : str | None
        Optional human-readable ID (e.g. "scenario_003"). If None,
        derived from the filename stem.

    Returns
    -------
    dict with keys:
        scenario_id       : str
        source_file       : str  (absolute path)
        ingested_at       : str  (ISO-8601 UTC)
        frequency_hz      : float
        n_theta           : int
        n_phi             : int
        n_samples         : int  (n_theta * n_phi)
        coordinate_system : str
        result_type       : str
        failed_elements   : list[int] | None
        n_elements        : int | None
        n_failures        : int | None
        label_vector      : list[int] | None  (1 = failed, 0 = working)
        theta_deg         : np.ndarray  shape (n_samples,)
        phi_deg           : np.ndarray  shape (n_samples,)
        re_etheta         : np.ndarray  shape (n_samples,)
        im_etheta         : np.ndarray  shape (n_samples,)
        re_ephi           : np.ndarray  shape (n_samples,)
        im_ephi           : np.ndarray  shape (n_samples,)
        gain_theta_dbi    : np.ndarray  shape (n_samples,)
        gain_phi_dbi      : np.ndarray  shape (n_samples,)
        gain_total_dbi    : np.ndarray  shape (n_samples,)
        magnitude_norm_db : np.ndarray  shape (n_samples,)  <- preprocessed
        parse_warnings    : list[str]
    """
    filepath = Path(filepath).resolve()
    if not filepath.exists():
        raise FileNotFoundError(f"FFE file not found: {filepath}")

    warnings: list[str] = []
    meta: dict = {}
    data_rows: list[list[float]] = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip()

            # Skip blank lines and the column-header line (starts with #" )
            if not line or line.startswith('#"') or line.startswith('*'):
                continue

            # File-level metadata (##)
            if line.startswith("##"):
                _parse_meta_line(line[2:], meta)
                continue

            # Request-level metadata (#)
            if line.startswith("#"):
                _parse_meta_line(line[1:], meta)
                continue

            # Data row
            try:
                values = [float(v) for v in line.split()]
                if len(values) == 9:
                    data_rows.append(values)
                elif len(values) > 0:
                    warnings.append(
                        f"Skipped row with {len(values)} columns: {line[:60]}"
                    )
            except ValueError:
                warnings.append(f"Could not parse data row: {line[:60]}")

    if not data_rows:
        raise ValueError(f"No valid data rows found in {filepath}")

    arr = np.array(data_rows, dtype=np.float64)

    # ---- Build label vector ------------------------------------------------
    label_vector = None
    n_failures = None
    if failed_elements is not None and n_elements is not None:
        if any(e >= n_elements or e < 0 for e in failed_elements):
            raise ValueError(
                f"failed_elements {failed_elements} out of range "
                f"for n_elements={n_elements}"
            )
        label_vector = [1 if i in failed_elements else 0 for i in range(n_elements)]
        n_failures = len(failed_elements)
    elif failed_elements is not None and n_elements is None:
        warnings.append(
            "failed_elements provided but n_elements is None — "
            "label_vector will be None"
        )

    # ---- Preprocessing: magnitude → normalise → dB  ------------------------
    # Matches the method described in De Lange (2020) Chapter 3, Section 3.1:
    #   1. Compute total |E| from real/imag components
    #   2. Normalise by maximum magnitude
    #   3. Convert to dB scale (20 * log10)
    re_et = arr[:, 2]
    im_et = arr[:, 3]
    re_ep = arr[:, 4]
    im_ep = arr[:, 5]

    magnitude = np.sqrt(re_et**2 + im_et**2 + re_ep**2 + im_ep**2)
    max_mag = magnitude.max()

    if max_mag < 1e-30:
        warnings.append("Maximum magnitude is near zero — pattern may be all-off")
        magnitude_norm_db = np.full_like(magnitude, -200.0)
    else:
        magnitude_norm = magnitude / max_mag
        # Clamp to avoid log10(0)
        magnitude_norm = np.clip(magnitude_norm, 1e-30, None)
        magnitude_norm_db = 20.0 * np.log10(magnitude_norm)

    # ---- Assemble record ---------------------------------------------------
    sid = scenario_id or filepath.stem
    n_theta = int(meta.get("No. of Theta Samples", arr[:, 0].size))
    n_phi   = int(meta.get("No. of Phi Samples",   arr[:, 1].size))

    record = {
        "scenario_id":       sid,
        "source_file":       str(filepath),
        "ingested_at":       datetime.now(timezone.utc).isoformat(),
        "frequency_hz":      float(meta.get("Frequency", 0.0)),
        "n_theta":           n_theta,
        "n_phi":             n_phi,
        "n_samples":         len(data_rows),
        "coordinate_system": str(meta.get("Coordinate System", "Spherical")),
        "result_type":       str(meta.get("Result Type", "Unknown")),
        "failed_elements":   failed_elements,
        "n_elements":        n_elements,
        "n_failures":        n_failures,
        "label_vector":      label_vector,
        # Raw columns
        "theta_deg":         arr[:, 0],
        "phi_deg":           arr[:, 1],
        "re_etheta":         arr[:, 2],
        "im_etheta":         arr[:, 3],
        "re_ephi":           arr[:, 4],
        "im_ephi":           arr[:, 5],
        "gain_theta_dbi":    arr[:, 6],
        "gain_phi_dbi":      arr[:, 7],
        "gain_total_dbi":    arr[:, 8],
        # Preprocessed feature vector (thesis method)
        "magnitude_norm_db": magnitude_norm_db,
        "parse_warnings":    warnings,
    }

    if warnings:
        for w in warnings:
            logger.warning("[%s] %s", sid, w)

    logger.info(
        "Parsed %s → %d samples | freq=%.3e Hz | failures=%s",
        filepath.name, len(data_rows), record["frequency_hz"], failed_elements
    )
    return record


def parse_ffe_batch(
    directory: str | Path,
    scenario_map: Optional[dict[str, dict]] = None,
    glob_pattern: str = "*.ffe",
) -> list[dict]:
    """
    Parse all .ffe files in a directory.

    Parameters
    ----------
    directory : str | Path
        Directory containing .ffe files.
    scenario_map : dict[str, dict] | None
        Optional mapping from filename stem → kwargs for parse_ffe.
        Keys must match the stem of the .ffe filename (no extension).
        Example:
            {
                "array_nominal":   {"failed_elements": [], "n_elements": 8},
                "array_fail_0_2":  {"failed_elements": [0, 2], "n_elements": 8},
            }
        If None, all files are parsed without label information.
    glob_pattern : str
        Glob pattern for file discovery (default "*.ffe").

    Returns
    -------
    list of record dicts (same structure as parse_ffe output).
    """
    directory = Path(directory)
    ffe_files = sorted(directory.glob(glob_pattern))

    if not ffe_files:
        logger.warning("No .ffe files found in %s", directory)
        return []

    records = []
    for ffe_path in ffe_files:
        stem = ffe_path.stem
        kwargs = (scenario_map or {}).get(stem, {})
        try:
            record = parse_ffe(ffe_path, scenario_id=stem, **kwargs)
            records.append(record)
        except Exception as exc:
            logger.error("Failed to parse %s: %s", ffe_path.name, exc)

    logger.info(
        "Batch parse complete: %d/%d files successful",
        len(records), len(ffe_files)
    )
    return records


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_meta_line(line: str, meta: dict) -> None:
    """
    Extract key-value pairs from metadata comment lines.

    Handles both:
      "Key: Value"        →  meta["Key"] = "Value"
      "File Type: X"      →  meta["File Type"] = "X"
    """
    if ":" not in line:
        return
    key, _, value = line.partition(":")
    key   = key.strip().strip('"')
    value = value.strip().strip('"')

    # Attempt numeric conversion where sensible
    if key in ("Frequency", "No. of Theta Samples", "No. of Phi Samples",
               "No. of Header Lines", "File Format"):
        try:
            meta[key] = float(value)
            return
        except ValueError:
            pass

    meta[key] = value