"""Explicit limb-darkening to limb-brightening crossing detection."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

from titan_limb.config_bands import BandPolicy
from titan_limb.models.core import Hemisphere, QualityStatus

FloatArray = NDArray[np.float64]
SAMPLE_POINTS = 3000
SMOOTHING_SIGMA = 4.0
MINIMUM_WAVELENGTH_UM = 0.6
MINIMUM_CROSSING_POINTS = 2
TRANSITION_SCHEMA = {
    "cube_id": pl.String,
    "hemisphere": pl.String,
    "crossing_index": pl.Int64,
    "crossing_count": pl.Int64,
    "wavelength_um": pl.Float64,
    "left_u_sum": pl.Float64,
    "right_u_sum": pl.Float64,
}


@dataclass(frozen=True)
class Crossing:
    wavelength_um: float
    left_u_sum: float
    right_u_sum: float


def detect_sampled_crossings(
    wavelength_um: FloatArray,
    u_sum: FloatArray,
    *,
    minimum_wavelength_um: float = MINIMUM_WAVELENGTH_UM,
) -> tuple[Crossing, ...]:
    """Preserve the old smoothed, sampled crossing rule without averaging."""
    if wavelength_um.ndim != 1 or len(wavelength_um) != len(u_sum):
        raise ValueError("wavelength and u-sum values must be paired vectors")
    finite = np.isfinite(wavelength_um) & np.isfinite(u_sum)
    wavelength = wavelength_um[finite]
    values = u_sum[finite]
    if len(wavelength) < MINIMUM_CROSSING_POINTS:
        return ()
    order = np.argsort(wavelength, kind="stable")
    wavelength = wavelength[order]
    values = values[order]
    if np.any(np.diff(wavelength) <= 0):
        raise ValueError("wavelength values must be distinct")
    smoothed = gaussian_filter1d(values, sigma=SMOOTHING_SIGMA)
    grid = np.linspace(wavelength[0], wavelength[-1], SAMPLE_POINTS)
    sampled = PchipInterpolator(wavelength, smoothed)(grid)
    crossing_indices = np.flatnonzero(np.diff(np.sign(sampled)))
    return tuple(
        Crossing(float(grid[index]), float(sampled[index]), float(sampled[index + 1]))
        for index in crossing_indices
        if grid[index] > minimum_wavelength_um
    )


def build_transition_table(
    fits: pl.DataFrame,
    quality: pl.DataFrame,
    band_policy: BandPolicy,
) -> pl.DataFrame:
    """Detect crossings only where north and south fits are both eligible."""
    eligible = (
        fits.join(quality, on=["cube_id", "band", "hemisphere"], how="inner")
        .filter(pl.col("quality_status") == QualityStatus.ELIGIBLE.value)
        .filter(~pl.col("band").is_in(sorted(band_policy.excluded_bands)))
        .select("cube_id", "band", "wavelength_um", "hemisphere", "u_sum")
    )
    north = eligible.filter(pl.col("hemisphere") == Hemisphere.NORTH.value).drop(
        "hemisphere"
    )
    south = eligible.filter(pl.col("hemisphere") == Hemisphere.SOUTH.value).drop(
        "hemisphere"
    )
    paired = north.join(
        south,
        on=["cube_id", "band", "wavelength_um"],
        suffix="_south",
    ).sort("cube_id", "wavelength_um")
    rows: list[dict[str, str | int | float | None]] = []
    for cube_id in paired.get_column("cube_id").unique(maintain_order=True):
        cube = paired.filter(pl.col("cube_id") == cube_id)
        wavelengths = cube.get_column("wavelength_um").to_numpy()
        for hemisphere, column in (
            (Hemisphere.NORTH, "u_sum"),
            (Hemisphere.SOUTH, "u_sum_south"),
        ):
            crossings = detect_sampled_crossings(
                wavelengths, cube.get_column(column).to_numpy()
            )
            if not crossings:
                rows.append(
                    {
                        "cube_id": cube_id,
                        "hemisphere": hemisphere.value,
                        "crossing_index": None,
                        "crossing_count": 0,
                        "wavelength_um": None,
                        "left_u_sum": None,
                        "right_u_sum": None,
                    }
                )
            for index, crossing in enumerate(crossings):
                rows.append(
                    {
                        "cube_id": cube_id,
                        "hemisphere": hemisphere.value,
                        "crossing_index": index,
                        "crossing_count": len(crossings),
                        "wavelength_um": crossing.wavelength_um,
                        "left_u_sum": crossing.left_u_sum,
                        "right_u_sum": crossing.right_u_sum,
                    }
                )
    return pl.DataFrame(rows, schema=TRANSITION_SCHEMA)


def write_transition_parquet(
    fits_path: Path,
    quality_path: Path,
    output: Path,
    band_policy: BandPolicy,
) -> pl.DataFrame:
    result = build_transition_table(
        pl.read_parquet(fits_path), pl.read_parquet(quality_path), band_policy
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output, compression="zstd", statistics=True)
    return result
