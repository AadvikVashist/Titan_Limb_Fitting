"""Compare raw-cube results with the preserved selected-fit data."""

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from titan_limb.io.atomic import atomic_write_json, atomic_write_parquet
from titan_limb.io.legacy import (
    parse_wavelength_key,
    read_selected_fit_directory,
    records_to_frame,
)
from titan_limb.validation.gates import (
    ValidationGateReport,
    equal_check,
    maximum_check,
    minimum_check,
    write_gate_report,
)


@dataclass(frozen=True)
class RawValidationSummary:
    profile_input_rows: int
    fit_input_rows: int
    legacy_profile_rows: int
    legacy_fit_rows: int
    rows: int
    exact_profiles: int
    changed_profiles: int
    equal_point_counts: int
    equal_fit_status: int
    both_succeeded: int
    maximum_absolute_u1_drift: float | None
    maximum_absolute_u2_drift: float | None
    median_absolute_u1_drift: float | None
    median_absolute_u2_drift: float | None


def _profile_rows(source_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(source_dir.glob("C*.pkl")):
        with path.open("rb") as source:
            cube = cast(dict[str, dict[str, dict[str, Any]]], pickle.load(source))
        for key, band_data in cube.items():
            parsed = parse_wavelength_key(key)
            if parsed is None:
                continue
            _, band = parsed
            for hemisphere, side in (
                ("north", "north_side"),
                ("south", "south_side"),
            ):
                profile = band_data[side]
                pixels = np.asarray(profile["pixel_indices"], dtype=np.int64)
                rows.append(
                    {
                        "cube_id": path.stem,
                        "band": band,
                        "hemisphere": hemisphere,
                        "legacy_pixel_rows": pixels[:, 0].tolist(),
                        "legacy_pixel_columns": pixels[:, 1].tolist(),
                        "legacy_emission_angles": np.asarray(
                            profile["emission_angles"], dtype=np.float64
                        ).tolist(),
                        "legacy_brightness": np.asarray(
                            profile["brightness_values"], dtype=np.float64
                        ).tolist(),
                    }
                )
    return rows


def validate_raw_results(
    profiles: pl.DataFrame,
    fits: pl.DataFrame,
    legacy_dir: Path,
) -> tuple[pl.DataFrame, RawValidationSummary]:
    legacy_profiles = pl.from_dicts(_profile_rows(legacy_dir))
    profile_check = profiles.join(
        legacy_profiles, on=["cube_id", "band", "hemisphere"], validate="1:1"
    ).with_columns(
        (
            (pl.col("pixel_rows") == pl.col("legacy_pixel_rows"))
            & (pl.col("pixel_columns") == pl.col("legacy_pixel_columns"))
            & (pl.col("emission_angles") == pl.col("legacy_emission_angles"))
            & (pl.col("brightness") == pl.col("legacy_brightness"))
        ).alias("exact_profile"),
        (
            pl.col("emission_angles").list.len()
            == pl.col("legacy_emission_angles").list.len()
        ).alias("equal_point_count"),
    )
    legacy_fits = records_to_frame(read_selected_fit_directory(legacy_dir)).select(
        "cube_id",
        "band",
        "hemisphere",
        pl.col("status").alias("legacy_status"),
        pl.col("u1").alias("legacy_u1"),
        pl.col("u2").alias("legacy_u2"),
    )
    fit_check = fits.select("cube_id", "band", "hemisphere", "status", "u1", "u2").join(
        legacy_fits, on=["cube_id", "band", "hemisphere"], validate="1:1"
    )
    checks = profile_check.select(
        "cube_id", "band", "hemisphere", "exact_profile", "equal_point_count"
    ).join(fit_check, on=["cube_id", "band", "hemisphere"], validate="1:1")
    checks = checks.with_columns(
        (pl.col("status") == pl.col("legacy_status")).alias("equal_fit_status"),
        (
            (pl.col("status") == "succeeded") & (pl.col("legacy_status") == "succeeded")
        ).alias("both_succeeded"),
        (pl.col("u1") - pl.col("legacy_u1")).abs().alias("absolute_u1_drift"),
        (pl.col("u2") - pl.col("legacy_u2")).abs().alias("absolute_u2_drift"),
    ).sort("cube_id", "band", "hemisphere")
    successful = checks.filter(pl.col("both_succeeded"))
    exact = checks.get_column("exact_profile").sum()
    u1_drift = successful.get_column("absolute_u1_drift").to_numpy()
    u2_drift = successful.get_column("absolute_u2_drift").to_numpy()
    summary = RawValidationSummary(
        profile_input_rows=profiles.height,
        fit_input_rows=fits.height,
        legacy_profile_rows=legacy_profiles.height,
        legacy_fit_rows=legacy_fits.height,
        rows=checks.height,
        exact_profiles=int(exact),
        changed_profiles=checks.height - int(exact),
        equal_point_counts=int(checks.get_column("equal_point_count").sum()),
        equal_fit_status=int(checks.get_column("equal_fit_status").sum()),
        both_succeeded=successful.height,
        maximum_absolute_u1_drift=float(np.max(u1_drift)) if u1_drift.size else None,
        maximum_absolute_u2_drift=float(np.max(u2_drift)) if u2_drift.size else None,
        median_absolute_u1_drift=float(np.median(u1_drift)) if u1_drift.size else None,
        median_absolute_u2_drift=float(np.median(u2_drift)) if u2_drift.size else None,
    )
    return checks, summary


def write_raw_validation(
    profiles_path: Path,
    fits_path: Path,
    legacy_dir: Path,
    output_dir: Path,
) -> RawValidationSummary:
    checks, summary = validate_raw_results(
        pl.read_parquet(profiles_path), pl.read_parquet(fits_path), legacy_dir
    )
    atomic_write_parquet(checks, output_dir / "raw-validation.parquet")
    atomic_write_json(output_dir / "raw-validation.json", asdict(summary))
    return summary


def write_raw_validation_gate(
    summary: RawValidationSummary,
    output: Path,
    *,
    maximum_changed_profiles: int | None = None,
    maximum_u1_drift: float | None = None,
    maximum_u2_drift: float | None = None,
) -> ValidationGateReport:
    """Require complete joins and apply only explicitly supplied drift bounds."""
    checks = [
        equal_check(
            "all_profile_inputs_joined",
            summary.rows,
            summary.profile_input_rows,
            "each rebuilt profile must have one validation row",
        ),
        equal_check(
            "all_legacy_profiles_joined",
            summary.rows,
            summary.legacy_profile_rows,
            "each legacy profile must have one validation row",
        ),
        equal_check(
            "all_fit_inputs_joined",
            summary.rows,
            summary.fit_input_rows,
            "each rebuilt fit must have one validation row",
        ),
        equal_check(
            "all_legacy_fits_joined",
            summary.rows,
            summary.legacy_fit_rows,
            "each legacy fit must have one validation row",
        ),
        minimum_check(
            "successful_fit_pairs",
            summary.both_succeeded,
            1,
            "at least one successful fit pair is required to measure drift",
        ),
    ]
    if maximum_changed_profiles is not None:
        checks.append(
            maximum_check(
                "changed_profiles",
                summary.changed_profiles,
                maximum_changed_profiles,
                "changed profile count exceeds the declared bound",
            )
        )
    if maximum_u1_drift is not None:
        observed = summary.maximum_absolute_u1_drift
        checks.append(
            maximum_check(
                "maximum_absolute_u1_drift",
                observed,
                maximum_u1_drift,
                "u1 drift exceeds the declared bound",
            )
        )
    if maximum_u2_drift is not None:
        observed = summary.maximum_absolute_u2_drift
        checks.append(
            maximum_check(
                "maximum_absolute_u2_drift",
                observed,
                maximum_u2_drift,
                "u2 drift exceeds the declared bound",
            )
        )
    return write_gate_report(tuple(checks), output)
