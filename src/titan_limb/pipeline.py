"""Resumable raw-cube processing pipeline."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from titan_limb.fitting.batch import fit_profile_table
from titan_limb.io.vims import find_cube_pair, load_cube_pair
from titan_limb.processing.cube_profiles import build_selected_profiles
from titan_limb.processing.fit_filter import (
    DEFAULT_MINIMUM_EMISSION_DEGREES,
    filter_profiles_by_emission,
)


@dataclass(frozen=True)
class RawPipelineReport:
    requested_cubes: int
    completed_cubes: int
    resumed_cubes: int
    profile_rows: int
    fit_rows: int
    failed_fits: int
    minimum_emission_degrees: float


def _write_frame(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd", statistics=True)


def build_raw_dataset(
    cubes_dir: Path,
    cube_ids: tuple[str, ...],
    output_dir: Path,
    *,
    minimum_emission_degrees: float = DEFAULT_MINIMUM_EMISSION_DEGREES,
    resume: bool = True,
) -> RawPipelineReport:
    """Build selected profiles and quadratic fits from VIMS cube pairs."""
    profile_dir = output_dir / "profiles"
    sorted_profile_dir = output_dir / "sorted-profiles"
    fit_dir = output_dir / "fits"
    resumed = 0
    for cube_id in cube_ids:
        profile_path = profile_dir / f"{cube_id}.parquet"
        sorted_profile_path = sorted_profile_dir / f"{cube_id}.parquet"
        fit_path = fit_dir / f"{cube_id}.parquet"
        if (
            resume
            and sorted_profile_path.is_file()
            and profile_path.is_file()
            and fit_path.is_file()
        ):
            resumed += 1
            continue
        visible, infrared = load_cube_pair(find_cube_pair(cubes_dir, cube_id))
        sorted_profiles = build_selected_profiles(visible, infrared)
        profiles = filter_profiles_by_emission(
            sorted_profiles,
            minimum_emission_degrees,
        )
        _write_frame(sorted_profiles, sorted_profile_path)
        _write_frame(profiles, profile_path)
        _write_frame(fit_profile_table(profiles), fit_path)

    profile_paths = [profile_dir / f"{cube_id}.parquet" for cube_id in cube_ids]
    sorted_profile_paths = [
        sorted_profile_dir / f"{cube_id}.parquet" for cube_id in cube_ids
    ]
    fit_paths = [fit_dir / f"{cube_id}.parquet" for cube_id in cube_ids]
    profiles = pl.concat([pl.read_parquet(path) for path in profile_paths])
    sorted_profiles = pl.concat(
        [pl.read_parquet(path) for path in sorted_profile_paths]
    )
    fits = pl.concat([pl.read_parquet(path) for path in fit_paths])
    _write_frame(profiles, output_dir / "profiles.parquet")
    _write_frame(sorted_profiles, output_dir / "sorted-profiles.parquet")
    _write_frame(fits, output_dir / "fits.parquet")
    report = RawPipelineReport(
        requested_cubes=len(cube_ids),
        completed_cubes=len(fit_paths),
        resumed_cubes=resumed,
        profile_rows=profiles.height,
        fit_rows=fits.height,
        failed_fits=fits.filter(pl.col("status") != "succeeded").height,
        minimum_emission_degrees=minimum_emission_degrees,
    )
    (output_dir / "report.json").write_text(json.dumps(asdict(report), indent=2) + "\n")
    return report
