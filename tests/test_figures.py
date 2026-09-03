"""Tests for global analysis figures."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from titan_limb.config_seasons import SeasonPolicy
from titan_limb.plotting.figures import (
    plot_asymmetry_spectrum,
    plot_seasonal_timeline,
    plot_transition_timeline,
    summarize_asymmetry_by_wavelength,
)


def test_plot_transition_timeline_writes_image(tmp_path: Path) -> None:
    data = pl.DataFrame(
        {
            "mid_time": [
                datetime(2005, 1, 1, tzinfo=UTC),
                datetime(2005, 1, 1, tzinfo=UTC),
            ],
            "hemisphere": ["north", "south"],
            "crossing_index": [0, 0],
            "crossing_count": [1, 1],
            "wavelength_um": [1.2, 1.3],
        }
    )
    output = tmp_path / "transition.png"

    plot_transition_timeline(data, output)

    assert output.stat().st_size > 0


def test_asymmetry_summary_and_plot(tmp_path: Path) -> None:
    data = pl.DataFrame(
        {
            "channel": ["visible", "visible", "infrared", "infrared"],
            "band": [1, 1, 2, 2],
            "wavelength_um": [0.5, 0.5, 1.5, 1.5],
            "u_sum_difference": [-0.2, 0.2, 0.1, 0.3],
        }
    )
    output = tmp_path / "asymmetry.png"

    summary = plot_asymmetry_spectrum(data, output)

    assert output.stat().st_size > 0
    assert summary.filter(pl.col("channel") == "visible").get_column(
        "median_difference"
    )[0] == pytest.approx(0.0)
    assert summary.get_column("observation_count").to_list() == [2, 2]


def test_empty_figure_inputs_are_rejected(tmp_path: Path) -> None:
    transitions = pl.DataFrame(
        schema={
            "mid_time": pl.Datetime(time_zone="UTC"),
            "hemisphere": pl.String,
            "crossing_index": pl.Int64,
            "crossing_count": pl.Int64,
            "wavelength_um": pl.Float64,
        }
    )
    asymmetry = pl.DataFrame(
        schema={
            "channel": pl.String,
            "band": pl.Int64,
            "wavelength_um": pl.Float64,
            "u_sum_difference": pl.Float64,
        }
    )

    with pytest.raises(ValueError, match="no crossings"):
        plot_transition_timeline(transitions, tmp_path / "one.png")
    with pytest.raises(ValueError, match="no paired rows"):
        plot_asymmetry_spectrum(asymmetry, tmp_path / "two.png")


def test_asymmetry_summary_is_separate_from_plotting() -> None:
    result = summarize_asymmetry_by_wavelength(
        pl.DataFrame(
            {
                "channel": ["visible"],
                "band": [1],
                "wavelength_um": [0.5],
                "u_sum_difference": [0.25],
            }
        )
    )
    assert result.get_column("median_difference").to_list() == [0.25]


def test_plot_seasonal_timeline_writes_image(tmp_path: Path) -> None:
    policy = SeasonPolicy(
        northern_vernal_equinox=datetime(2009, 8, 11, tzinfo=UTC),
        northern_summer_solstice=datetime(2017, 5, 24, tzinfo=UTC),
    )
    seasons = ["northern winter", "northern spring", "northern summer"]
    dates = [
        datetime(2008, 1, 1, tzinfo=UTC),
        datetime(2012, 1, 1, tzinfo=UTC),
        datetime(2017, 6, 1, tzinfo=UTC),
    ]
    cube_rows = [
        {
            "channel": channel,
            "mid_time": date,
            "median_u_sum_difference": value,
        }
        for channel in ("visible", "infrared")
        for date, value in zip(dates, (-0.1, 0.1, 0.2), strict=True)
    ]
    summary_rows = [
        {
            "channel": channel,
            "northern_season": season,
            "median_u_sum_difference": 0.0,
            "bootstrap_lower": -0.1,
            "bootstrap_upper": 0.1,
            "interval_available": season != "northern summer",
        }
        for season in seasons
        for channel in ("visible", "infrared")
    ]
    output = tmp_path / "seasons.png"

    plot_seasonal_timeline(
        pl.from_dicts(cube_rows), pl.from_dicts(summary_rows), output, policy
    )

    assert output.stat().st_size > 0
