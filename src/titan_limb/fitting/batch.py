"""Fit typed selected-profile tables without legacy objects."""

from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import polars as pl

from titan_limb.fitting.optimizer import PARAMETER_COUNT, fit_quadratic_profile
from titan_limb.io.atomic import atomic_write_parquet
from titan_limb.io.legacy import records_to_frame
from titan_limb.models.core import (
    Channel,
    FitFailureReason,
    FitStatus,
    Hemisphere,
)
from titan_limb.models.fit import LegacyFitRecord


class ProfileRow(TypedDict):
    cube_id: str
    band: int
    wavelength_um: float
    channel: str
    hemisphere: str
    slant_angle_degrees: int
    emission_angles: list[float]
    brightness: list[float]


def _failed_profile(row: ProfileRow, reason: FitFailureReason) -> LegacyFitRecord:
    return LegacyFitRecord(
        cube_id=str(row["cube_id"]),
        band=int(row["band"]),
        wavelength_um=float(row["wavelength_um"]),
        channel=Channel(str(row["channel"])),
        hemisphere=Hemisphere(str(row["hemisphere"])),
        slant_angle_degrees=int(row["slant_angle_degrees"]),
        profile_points=len(row["emission_angles"]),
        status=FitStatus.FAILED,
        failure_reason=reason,
    )


def fit_profile_table(profiles: pl.DataFrame) -> pl.DataFrame:
    records: list[LegacyFitRecord] = []
    for raw_row in profiles.iter_rows(named=True):
        row = cast(ProfileRow, raw_row)
        emission = np.asarray(row["emission_angles"], dtype=np.float64)
        brightness = np.asarray(row["brightness"], dtype=np.float64)
        if len(emission) < PARAMETER_COUNT:
            records.append(
                _failed_profile(row, FitFailureReason.TOO_FEW_PROFILE_POINTS)
            )
            continue
        try:
            optimal = fit_quadratic_profile(emission, brightness).optimal
        except (RuntimeError, ValueError, FloatingPointError):
            records.append(_failed_profile(row, FitFailureReason.OPTIMIZATION_FAILED))
            continue
        records.append(
            LegacyFitRecord(
                cube_id=str(row["cube_id"]),
                band=int(row["band"]),
                wavelength_um=float(row["wavelength_um"]),
                channel=Channel(str(row["channel"])),
                hemisphere=Hemisphere(str(row["hemisphere"])),
                slant_angle_degrees=int(row["slant_angle_degrees"]),
                profile_points=len(emission),
                status=FitStatus.SUCCEEDED,
                smoothing_method=optimal.method,
                intensity_center=optimal.intensity_center,
                u1=optimal.u1,
                u2=optimal.u2,
                u_sum=optimal.u1 + optimal.u2,
                r_squared=optimal.r_squared,
                covariance=optimal.covariance,
            )
        )
    return records_to_frame(records)


def fit_profile_parquet(source: Path, output: Path) -> pl.DataFrame:
    result = fit_profile_table(pl.read_parquet(source))
    atomic_write_parquet(result, output)
    return result
