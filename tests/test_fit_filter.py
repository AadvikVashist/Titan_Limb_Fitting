"""Tests for the pre-fit limb cutoff."""

import polars as pl

from titan_limb.processing.cube_profiles import SELECTED_PROFILE_SCHEMA
from titan_limb.processing.fit_filter import filter_profiles_by_emission


def test_filter_profiles_uses_strict_emission_cutoff() -> None:
    row = {
        "cube_id": "C1",
        "band": 1,
        "wavelength_um": 0.5,
        "channel": "visible",
        "hemisphere": "north",
        "slant_angle_degrees": 60,
        "actual_angle_degrees": 60.0,
        "north_orientation_degrees": 0.0,
        "illumination_degrees": 120.0,
        "center_row": 1.0,
        "center_column": 1.0,
        "filtered": True,
        "fit_filtered": False,
        "minimum_fit_emission_degrees": None,
        "pixel_rows": [0, 1, 2, 3],
        "pixel_columns": [0, 0, 0, 0],
        "pixel_distances": [0.0, 1.0, 2.0, 3.0],
        "emission_angles": [10.0, 25.0, 30.0, 40.0],
        "brightness": [1.0, 2.0, 3.0, 4.0],
    }
    profiles = pl.DataFrame([row], schema=SELECTED_PROFILE_SCHEMA)
    cutoff = 25.0

    result = filter_profiles_by_emission(profiles, cutoff)
    row = result.row(0, named=True)

    assert all(value > cutoff for value in row["emission_angles"])
    assert row["fit_filtered"] is True
    assert row["minimum_fit_emission_degrees"] == cutoff
