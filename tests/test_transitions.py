"""Tests for explicit transition detection."""

import numpy as np
import polars as pl
import pytest

from titan_limb.analysis.transitions import (
    build_transition_table,
    detect_sampled_crossings,
)
from titan_limb.config_bands import BandPolicy


def test_detector_keeps_multiple_crossings() -> None:
    wavelength = np.linspace(0.7, 2.0, 60)
    values = np.sin((wavelength - 0.7) * 12)

    crossings = detect_sampled_crossings(wavelength, values)

    assert len(crossings) > 1
    assert [crossing.wavelength_um for crossing in crossings] == sorted(
        crossing.wavelength_um for crossing in crossings
    )


def test_detector_handles_too_little_data_and_bad_input() -> None:
    assert detect_sampled_crossings(np.array([1.0]), np.array([1.0])) == ()
    with pytest.raises(ValueError, match="paired vectors"):
        detect_sampled_crossings(np.ones((2, 2)), np.ones(4))
    with pytest.raises(ValueError, match="distinct"):
        detect_sampled_crossings(np.array([1.0, 1.0]), np.array([-1.0, 1.0]))


def test_transition_table_requires_paired_eligible_rows() -> None:
    fits = pl.DataFrame(
        {
            "cube_id": ["C1", "C1", "C1", "C1", "C1", "C1"],
            "band": [1, 1, 2, 2, 3, 3],
            "wavelength_um": [0.7, 0.7, 1.0, 1.0, 1.3, 1.3],
            "hemisphere": ["north", "south"] * 3,
            "u_sum": [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0],
        }
    )
    quality = fits.select("cube_id", "band", "hemisphere").with_columns(
        pl.lit("eligible").alias("quality_status")
    )

    result = build_transition_table(fits, quality, BandPolicy())

    assert result.get_column("hemisphere").to_list() == ["north", "south"]
    assert result.get_column("crossing_count").to_list() == [1, 1]
