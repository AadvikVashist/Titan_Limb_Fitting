"""Tests for raw pipeline resume and report behavior."""

from pathlib import Path

import polars as pl

from titan_limb.pipeline import build_raw_dataset
from titan_limb.processing.cube_profiles import SELECTED_PROFILE_SCHEMA


def test_build_raw_dataset_resumes_completed_cube(tmp_path: Path) -> None:
    output = tmp_path / "raw"
    profile = pl.DataFrame(schema=SELECTED_PROFILE_SCHEMA)
    fit = pl.DataFrame({"status": ["succeeded"]})
    (output / "profiles").mkdir(parents=True)
    (output / "sorted-profiles").mkdir(parents=True)
    (output / "fits").mkdir(parents=True)
    profile.write_parquet(output / "profiles" / "C1.parquet")
    profile.write_parquet(output / "sorted-profiles" / "C1.parquet")
    fit.write_parquet(output / "fits" / "C1.parquet")

    report = build_raw_dataset(tmp_path, ("C1",), output)

    assert report.resumed_cubes == 1
    assert report.completed_cubes == 1
    assert report.fit_rows == 1
    assert (output / "report.json").is_file()
