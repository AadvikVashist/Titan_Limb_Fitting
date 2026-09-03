"""Tests for observation-level seasonal summaries."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from titan_limb.analysis.seasons import (
    build_seasonal_cube_table,
    summarize_seasonal_groups,
    write_seasonal_parquet,
)
from titan_limb.config_seasons import SeasonPolicy


def policy() -> SeasonPolicy:
    return SeasonPolicy(
        northern_vernal_equinox=datetime(2009, 8, 11, tzinfo=UTC),
        northern_summer_solstice=datetime(2017, 5, 24, tzinfo=UTC),
        minimum_group_observations=3,
        bootstrap_resamples=200,
        random_seed=7,
    )


def asymmetry_rows() -> pl.DataFrame:
    records: list[dict[str, object]] = []
    dates = [
        datetime(2008, 1, 1, tzinfo=UTC),
        datetime(2008, 2, 1, tzinfo=UTC),
        datetime(2008, 3, 1, tzinfo=UTC),
        datetime(2012, 1, 1, tzinfo=UTC),
        datetime(2018, 1, 1, tzinfo=UTC),
    ]
    for cube_index, mid_time in enumerate(dates):
        for band, difference in ((1, float(cube_index)), (2, float(cube_index + 2))):
            records.append(
                {
                    "cube_id": f"C{cube_index}",
                    "channel": "visible",
                    "selection_label": f"T{cube_index}",
                    "mid_time": mid_time,
                    "decimal_year": float(mid_time.year),
                    "flyby": f"T{cube_index}",
                    "band": band,
                    "north_u_sum": difference / 2,
                    "south_u_sum": -difference / 2,
                    "u_sum_difference": difference,
                }
            )
    return pl.from_dicts(records)


def test_cube_table_reduces_bands_and_assigns_seasons() -> None:
    result = build_seasonal_cube_table(asymmetry_rows(), policy())

    assert result.height == 5
    assert result.get_column("band_count").to_list() == [2, 2, 2, 2, 2]
    assert result.get_column("northern_season").to_list() == [
        "northern winter",
        "northern winter",
        "northern winter",
        "northern spring",
        "northern summer",
    ]
    assert result.get_column("median_u_sum_difference").to_list() == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]


def test_season_summary_is_seeded_and_withholds_small_group_interval() -> None:
    cube_table = build_seasonal_cube_table(asymmetry_rows(), policy())

    first = summarize_seasonal_groups(cube_table, policy())
    second = summarize_seasonal_groups(cube_table, policy())
    winter = first.filter(
        (pl.col("northern_season") == "northern winter")
        & (pl.col("channel") == "visible")
    ).row(0, named=True)
    spring = first.filter(
        (pl.col("northern_season") == "northern spring")
        & (pl.col("channel") == "visible")
    ).row(0, named=True)

    assert first.equals(second)
    assert winter["interval_available"] is True
    assert winter["bootstrap_lower"] is not None
    assert spring["interval_available"] is False
    assert spring["bootstrap_lower"] is None


def test_write_seasonal_parquet(tmp_path: Path) -> None:
    source = tmp_path / "asymmetry.parquet"
    cube_output = tmp_path / "cubes.parquet"
    summary_output = tmp_path / "summary.parquet"
    asymmetry_rows().write_parquet(source)

    cubes, summary = write_seasonal_parquet(
        source, cube_output, summary_output, policy()
    )

    assert cubes.height == 5
    assert summary.height == 6
    assert pl.read_parquet(cube_output).equals(cubes)
    assert pl.read_parquet(summary_output).equals(summary)
