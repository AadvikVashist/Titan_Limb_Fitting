"""Tests for paired north-south analysis."""

import polars as pl

from titan_limb.analysis.asymmetry import build_asymmetry_table
from titan_limb.config_bands import BandPolicy


def test_asymmetry_uses_only_paired_eligible_rows() -> None:
    fits = pl.DataFrame(
        {
            "cube_id": ["C1"] * 4,
            "band": [1, 1, 2, 2],
            "wavelength_um": [0.7, 0.7, 0.8, 0.8],
            "channel": ["visible"] * 4,
            "hemisphere": ["north", "south"] * 2,
            "u1": [0.3, 0.1, 0.4, 0.2],
            "u2": [0.2, 0.1, 0.3, 0.2],
            "u_sum": [0.5, 0.2, 0.7, 0.4],
            "r_squared": [0.9] * 4,
        }
    )
    quality = fits.select("cube_id", "band", "hemisphere").with_columns(
        pl.Series("quality_status", ["eligible", "eligible", "eligible", "review"])
    )

    result = build_asymmetry_table(fits, quality, BandPolicy())

    assert result.height == 1
    assert result.row(0, named=True)["u_sum_difference"] == 0.3
