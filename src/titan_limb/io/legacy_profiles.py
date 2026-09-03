"""Convert saved sorted profiles into a compact Polars table."""

import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from titan_limb.io.atomic import atomic_write_parquet
from titan_limb.io.legacy import PICKLE_PATTERN, parse_wavelength_key
from titan_limb.processing.bands import channel_for_band
from titan_limb.processing.profiles import PIXEL_INDEX_COLUMNS


class LegacyProcessing(TypedDict, total=False):
    sorted: bool
    filtered: bool | None


class LegacyProfileMeta(TypedDict):
    actual_angle: float
    processing: LegacyProcessing


class LegacyProfile(TypedDict):
    pixel_indices: NDArray[np.int64]
    pixel_distances: NDArray[np.float64]
    emission_angles: NDArray[np.float64]
    brightness_values: NDArray[np.float64]
    meta: LegacyProfileMeta


PROFILE_SCHEMA = {
    "cube_id": pl.String,
    "band": pl.Int64,
    "wavelength_um": pl.Float64,
    "channel": pl.String,
    "slant_degrees": pl.Int64,
    "actual_angle_degrees": pl.Float64,
    "sorted": pl.Boolean,
    "filtered": pl.Boolean,
    "pixel_rows": pl.List(pl.Int64),
    "pixel_columns": pl.List(pl.Int64),
    "pixel_distances": pl.List(pl.Float64),
    "emission_angles": pl.List(pl.Float64),
    "brightness": pl.List(pl.Float64),
}


@dataclass(frozen=True)
class ProfileConversionReport:
    files: int
    rows: int
    points: int


def _validated_arrays(profile: LegacyProfile) -> tuple[NDArray, ...]:
    indices = np.asarray(profile["pixel_indices"])
    distances = np.asarray(profile["pixel_distances"])
    emission = np.asarray(profile["emission_angles"])
    brightness = np.asarray(profile["brightness_values"])
    lengths = (len(indices), len(distances), len(emission), len(brightness))
    if len(set(lengths)) != 1:
        raise ValueError("legacy profile arrays have different lengths")
    if indices.ndim != PIXEL_INDEX_COLUMNS or indices.shape[1] != PIXEL_INDEX_COLUMNS:
        raise ValueError("legacy pixel indices are not row-column pairs")
    if np.any(np.diff(emission) < 0):
        raise ValueError("legacy profile is not sorted by emission angle")
    return indices, distances, emission, brightness


def read_profile_pickle(path: Path) -> pl.DataFrame:
    """Read one saved profile pickle into a schema-checked table."""
    if path.suffix != ".pkl":
        raise ValueError("legacy profile input must be a .pkl file")
    with path.open("rb") as source:
        cube = cast(Mapping[str, Mapping[int, LegacyProfile]], pickle.load(source))
    rows: list[dict] = []
    for wavelength_key, profiles in cube.items():
        parsed = parse_wavelength_key(wavelength_key)
        if parsed is None:
            continue
        wavelength_um, band = parsed
        for slant, profile in profiles.items():
            indices, distances, emission, brightness = _validated_arrays(profile)
            processing = profile["meta"]["processing"]
            rows.append(
                {
                    "cube_id": path.stem,
                    "band": band,
                    "wavelength_um": wavelength_um,
                    "channel": channel_for_band(band).value,
                    "slant_degrees": int(slant),
                    "actual_angle_degrees": float(profile["meta"]["actual_angle"]),
                    "sorted": processing.get("sorted", False),
                    "filtered": processing.get("filtered"),
                    "pixel_rows": indices[:, 0].tolist(),
                    "pixel_columns": indices[:, 1].tolist(),
                    "pixel_distances": distances.tolist(),
                    "emission_angles": emission.tolist(),
                    "brightness": brightness.tolist(),
                }
            )
    return pl.DataFrame(rows, schema=PROFILE_SCHEMA).sort("band", "slant_degrees")


def write_profile_directory(source_dir: Path, output: Path) -> ProfileConversionReport:
    paths = sorted(source_dir.glob(PICKLE_PATTERN))
    if not paths:
        raise FileNotFoundError(f"no profile pickle files found in {source_dir}")
    frames = [read_profile_pickle(path) for path in paths]
    table = pl.concat(frames, rechunk=False)
    atomic_write_parquet(table, output)
    points = table.select(pl.col("emission_angles").list.len().sum()).item()
    return ProfileConversionReport(files=len(paths), rows=table.height, points=points)
