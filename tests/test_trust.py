"""Tests for band trust checks."""

from datetime import UTC, datetime

import polars as pl

from titan_limb.analysis.trust import build_band_trust_table


def test_band_trust_keeps_coverage_and_time_check() -> None:
    fits = pl.DataFrame(
        {
            "cube_id": ["C1", "C2"],
            "band": [1, 1],
            "hemisphere": ["north", "north"],
            "wavelength_um": [0.5, 0.5],
            "channel": ["visible", "visible"],
            "r_squared": [0.8, 0.9],
            "u_sum": [0.1, 0.2],
        }
    )
    quality = fits.select("cube_id", "band", "hemisphere").with_columns(
        pl.lit("eligible").alias("quality_status")
    )
    observations = pl.DataFrame(
        {
            "cube_id": ["C1", "C2"],
            "selection_label": ["a", "b"],
            "mid_time": [
                datetime(2005, 1, 1, tzinfo=UTC),
                datetime(2006, 1, 1, tzinfo=UTC),
            ],
            "decimal_year": [2005.0, 2006.0],
            "flyby": ["T1", "T2"],
        }
    )

    result = build_band_trust_table(fits, quality, observations)

    assert result.item(0, "eligible_fraction") == 1.0
    assert result.item(0, "trusted") is True
    assert result.item(0, "time_correlation") == 1.0
