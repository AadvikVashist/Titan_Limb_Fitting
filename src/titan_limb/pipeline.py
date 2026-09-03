"""Resumable raw-cube processing pipeline."""

from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from titan_limb.fitting.batch import fit_profile_table
from titan_limb.io.atomic import atomic_write_json, atomic_write_parquet
from titan_limb.io.vims import find_cube_pair, load_cube_pair
from titan_limb.processing.cube_profiles import build_selected_profiles
from titan_limb.processing.fit_filter import (
    DEFAULT_MINIMUM_EMISSION_DEGREES,
    filter_profiles_by_emission,
)
from titan_limb.provenance import RunDefinition, RunRecorder, receipt_allows_resume
from titan_limb.rejections import RejectionKind, RejectionLedger


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
    atomic_write_parquet(frame, path)


def build_raw_dataset(
    cubes_dir: Path,
    cube_ids: tuple[str, ...],
    output_dir: Path,
    *,
    minimum_emission_degrees: float = DEFAULT_MINIMUM_EMISSION_DEGREES,
    resume: bool = True,
) -> RawPipelineReport:
    """Build selected profiles and quadratic fits from VIMS cube pairs."""
    project_dir = Path.cwd().resolve()
    profile_dir = output_dir / "profiles"
    sorted_profile_dir = output_dir / "sorted-profiles"
    fit_dir = output_dir / "fits"
    receipt_dir = output_dir / "receipts"
    rejection_dir = output_dir / "rejections"
    pairs = tuple(find_cube_pair(cubes_dir, cube_id) for cube_id in cube_ids)
    combined_outputs = (
        output_dir / "profiles.parquet",
        output_dir / "sorted-profiles.parquet",
        output_dir / "fits.parquet",
        output_dir / "report.json",
    )
    run_recorder = RunRecorder(
        RunDefinition(
            command="data.build-raw",
            receipt_path=output_dir / "run-receipt.json",
            project_dir=project_dir,
            settings={
                "cubes_dir": str(cubes_dir.resolve()),
                "output_dir": str(output_dir.resolve()),
            },
            parameters={
                "cube_ids": list(cube_ids),
                "minimum_emission_degrees": minimum_emission_degrees,
                "resume": resume,
            },
            inputs=tuple(
                path for pair in pairs for path in (pair.visible, pair.infrared)
            ),
            outputs=combined_outputs,
            output_schema_versions={"raw_pipeline": 1},
        )
    )
    resumed = 0
    with run_recorder:
        for pair in pairs:
            cube_id = pair.cube_id
            profile_path = profile_dir / f"{cube_id}.parquet"
            sorted_profile_path = sorted_profile_dir / f"{cube_id}.parquet"
            fit_path = fit_dir / f"{cube_id}.parquet"
            rejection_path = rejection_dir / f"{cube_id}.json"
            cube_outputs = (
                sorted_profile_path,
                profile_path,
                fit_path,
                rejection_path,
            )
            cube_recorder = RunRecorder(
                RunDefinition(
                    command="data.build-raw.cube",
                    receipt_path=receipt_dir / f"{cube_id}.json",
                    project_dir=project_dir,
                    settings={
                        "cubes_dir": str(cubes_dir.resolve()),
                        "output_dir": str(output_dir.resolve()),
                    },
                    parameters={
                        "cube_id": cube_id,
                        "minimum_emission_degrees": minimum_emission_degrees,
                    },
                    inputs=(pair.visible, pair.infrared),
                    outputs=cube_outputs,
                    output_schema_versions={"raw_cube": 1, "rejection_ledger": 1},
                    rejection_ledger=rejection_path,
                )
            )
            if resume and receipt_allows_resume(
                cube_recorder.receipt_path,
                cube_recorder.input_fingerprint,
                cube_outputs,
            ):
                resumed += 1
                continue
            with cube_recorder:
                visible, infrared = load_cube_pair(pair)
                sorted_profiles = build_selected_profiles(visible, infrared)
                profiles = filter_profiles_by_emission(
                    sorted_profiles,
                    minimum_emission_degrees,
                )
                fits = fit_profile_table(profiles)
                ledger = RejectionLedger()
                for row in fits.filter(pl.col("status") != "succeeded").iter_rows(
                    named=True
                ):
                    identifier = f"{row['cube_id']}:{row['band']}:{row['hemisphere']}"
                    ledger = ledger.with_rejection(
                        RejectionKind.FIT,
                        identifier,
                        "quadratic_fit",
                        str(row["failure_reason"]),
                    )
                cube_recorder.rejection_count = len(ledger.records)
                _write_frame(sorted_profiles, sorted_profile_path)
                _write_frame(profiles, profile_path)
                _write_frame(fits, fit_path)
                ledger.write(rejection_path)

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
        atomic_write_json(output_dir / "report.json", asdict(report))
    return report
