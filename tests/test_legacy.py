"""Tests for legacy selected-fit conversion."""

import pickle
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from titan_limb.io.legacy import (
    LegacyOptimalFit,
    LegacySideData,
    parse_wavelength_key,
    read_selected_fit_directory,
    read_selected_fit_pickle,
    records_to_frame,
    write_selected_fit_parquet,
)
from titan_limb.models.core import FitFailureReason, FitStatus, Hemisphere


def legacy_side(angle: int, optimal_fit: LegacyOptimalFit) -> LegacySideData:
    return {
        "angle": angle,
        "emission_angles": [20.0, 40.0, 60.0],
        "fit": {"quadratic": {"optimal_fit": optimal_fit}},
    }


def write_legacy_pickle(path: Path) -> None:
    covariance = np.identity(3)
    north = legacy_side(
        60,
        {
            "fit_params": {"I_0": 0.5, "u1": 0.2, "u2": -0.1},
            "covariance_matrix": covariance,
            "r2": 0.95,
            "sigma": 20,
        },
    )
    south = legacy_side(120, {})
    with path.open("wb") as output:
        pickle.dump(
            {
                "meta": {},
                "0.35054µm_1": {
                    "north_side": north,
                    "south_side": south,
                },
            },
            output,
        )


def test_parse_wavelength_key() -> None:
    assert parse_wavelength_key("0.35054µm_1") == (0.35054, 1)
    assert parse_wavelength_key("meta") is None


def test_read_selected_fit_pickle_preserves_success_and_failure(tmp_path: Path) -> None:
    source = tmp_path / "C_TEST.pkl"
    write_legacy_pickle(source)

    north, south = read_selected_fit_pickle(source)

    assert north.hemisphere is Hemisphere.NORTH
    assert north.status is FitStatus.SUCCEEDED
    assert north.u_sum == pytest.approx(0.1)
    assert north.r_squared == pytest.approx(0.95)
    assert north.profile_points == 3
    assert south.hemisphere is Hemisphere.SOUTH
    assert south.status is FitStatus.FAILED
    assert south.failure_reason is FitFailureReason.MISSING_OPTIMAL_FIT


def test_legacy_reader_rejects_wrong_extension(tmp_path: Path) -> None:
    source = tmp_path / "selected.pickle"
    source.write_bytes(b"unused")

    with pytest.raises(ValueError):
        read_selected_fit_pickle(source)


def test_directory_reader_requires_matching_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_selected_fit_directory(tmp_path)


def test_records_write_to_sorted_parquet(tmp_path: Path) -> None:
    source_dir = tmp_path / "selected"
    source_dir.mkdir()
    source = source_dir / "C_TEST.pkl"
    write_legacy_pickle(source)
    records = read_selected_fit_directory(source_dir)
    output = tmp_path / "fits.parquet"

    frame = records_to_frame(records)
    write_selected_fit_parquet(records, output)
    saved = pl.read_parquet(output)

    assert frame.shape == (2, 16)
    assert saved.equals(frame)
    assert saved.get_column("hemisphere").to_list() == ["north", "south"]


def test_frame_infers_failure_reason_after_initial_successes(tmp_path: Path) -> None:
    source = tmp_path / "C_TEST.pkl"
    write_legacy_pickle(source)
    north, south = read_selected_fit_pickle(source)

    frame = records_to_frame((*([north] * 101), south))

    assert frame.get_column("failure_reason").drop_nulls().to_list() == [
        "missing_optimal_fit"
    ]
