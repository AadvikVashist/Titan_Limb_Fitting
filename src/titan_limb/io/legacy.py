"""Read-only conversion of saved selected-fit pickle files."""

import pickle
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from titan_limb.io.atomic import atomic_write_parquet
from titan_limb.models.core import (
    FitFailureReason,
    FitStatus,
    Hemisphere,
    SmoothingMethod,
)
from titan_limb.models.fit import LegacyFitRecord
from titan_limb.processing.bands import channel_for_band

WAVELENGTH_KEY = re.compile(r"^(?P<wavelength>[0-9]+(?:\.[0-9]+)?)µm_(?P<band>[0-9]+)$")
PICKLE_PATTERN = "C*.pkl"


class LegacyParameters(TypedDict):
    I_0: float
    u1: float
    u2: float


class LegacyOptimalFit(TypedDict, total=False):
    fit_params: LegacyParameters
    covariance_matrix: NDArray[np.float64]
    r2: float
    sigma: float
    window: int


class LegacyQuadraticFit(TypedDict, total=False):
    optimal_fit: LegacyOptimalFit


class LegacyFits(TypedDict, total=False):
    quadratic: LegacyQuadraticFit


class LegacySideData(TypedDict, total=False):
    angle: int
    emission_angles: Sequence[float]
    fit: LegacyFits


@dataclass(frozen=True)
class FitContext:
    cube_id: str
    wavelength_um: float
    band: int
    hemisphere: Hemisphere


def parse_wavelength_key(key: str) -> tuple[float, int] | None:
    match = WAVELENGTH_KEY.fullmatch(key)
    if match is None:
        return None
    return float(match.group("wavelength")), int(match.group("band"))


def smoothing_method(optimal_fit: LegacyOptimalFit) -> SmoothingMethod:
    if "sigma" in optimal_fit:
        return SmoothingMethod.GAUSSIAN
    if "window" in optimal_fit:
        return SmoothingMethod.MOVING_AVERAGE
    return SmoothingMethod.INTERPOLATED


def failed_record(
    context: FitContext,
    side_data: LegacySideData,
    reason: FitFailureReason,
) -> LegacyFitRecord:
    return LegacyFitRecord(
        cube_id=context.cube_id,
        band=context.band,
        wavelength_um=context.wavelength_um,
        channel=channel_for_band(context.band),
        hemisphere=context.hemisphere,
        slant_angle_degrees=side_data.get("angle", -1),
        profile_points=len(side_data.get("emission_angles", ())),
        status=FitStatus.FAILED,
        failure_reason=reason,
    )


def extract_side_record(
    context: FitContext,
    side_data: LegacySideData,
) -> LegacyFitRecord:
    fits = side_data.get("fit")
    quadratic_fit = fits.get("quadratic") if fits is not None else None
    optimal_fit = (
        quadratic_fit.get("optimal_fit") if quadratic_fit is not None else None
    )
    if not optimal_fit:
        return failed_record(
            context,
            side_data,
            FitFailureReason.MISSING_OPTIMAL_FIT,
        )
    parameters = optimal_fit.get("fit_params")
    if parameters is None:
        return failed_record(
            context,
            side_data,
            FitFailureReason.MISSING_PARAMETERS,
        )
    covariance = optimal_fit["covariance_matrix"]
    u1 = float(parameters["u1"])
    u2 = float(parameters["u2"])
    return LegacyFitRecord(
        cube_id=context.cube_id,
        band=context.band,
        wavelength_um=context.wavelength_um,
        channel=channel_for_band(context.band),
        hemisphere=context.hemisphere,
        slant_angle_degrees=side_data["angle"],
        profile_points=len(side_data.get("emission_angles", ())),
        status=FitStatus.SUCCEEDED,
        smoothing_method=smoothing_method(optimal_fit),
        intensity_center=float(parameters["I_0"]),
        u1=u1,
        u2=u2,
        u_sum=u1 + u2,
        r_squared=float(optimal_fit["r2"]),
        covariance=tuple(float(value) for value in covariance.reshape(-1)),
    )


def read_selected_fit_pickle(path: Path) -> tuple[LegacyFitRecord, ...]:
    if path.suffix != ".pkl":
        raise ValueError("legacy selected-fit input must be a .pkl file")
    with path.open("rb") as source:
        cube_data = cast(dict[str, dict[str, LegacySideData]], pickle.load(source))
    records: list[LegacyFitRecord] = []
    for wavelength_key, wave_data in cube_data.items():
        parsed = parse_wavelength_key(wavelength_key)
        if parsed is None:
            continue
        wavelength_um, band = parsed
        for hemisphere, legacy_key in (
            (Hemisphere.NORTH, "north_side"),
            (Hemisphere.SOUTH, "south_side"),
        ):
            records.append(
                extract_side_record(
                    FitContext(path.stem, wavelength_um, band, hemisphere),
                    wave_data[legacy_key],
                )
            )
    return tuple(records)


def read_selected_fit_directory(source_dir: Path) -> tuple[LegacyFitRecord, ...]:
    paths = sorted(source_dir.glob(PICKLE_PATTERN))
    if not paths:
        raise FileNotFoundError(f"no selected-fit pickle files found in {source_dir}")
    return tuple(record for path in paths for record in read_selected_fit_pickle(path))


def records_to_frame(records: Sequence[LegacyFitRecord]) -> pl.DataFrame:
    rows = [record.model_dump(mode="json") for record in records]
    return pl.from_dicts(rows, infer_schema_length=None).sort(
        "cube_id", "band", "hemisphere"
    )


def write_selected_fit_parquet(
    records: Sequence[LegacyFitRecord], output: Path
) -> None:
    atomic_write_parquet(records_to_frame(records), output)
