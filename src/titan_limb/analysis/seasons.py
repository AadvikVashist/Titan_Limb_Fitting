"""Observation-level summaries for Titan's northern seasons."""

from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from titan_limb.config_seasons import SeasonPolicy

FloatArray = NDArray[np.float64]
SEASON_ORDER = ["northern winter", "northern spring", "northern summer"]
SOUTHERN_SEASON = {
    "northern winter": "southern summer",
    "northern spring": "southern autumn",
    "northern summer": "southern winter",
}


def build_seasonal_cube_table(
    asymmetry: pl.DataFrame, policy: SeasonPolicy
) -> pl.DataFrame:
    """Reduce each cube and channel to one spectral median before comparison."""
    cube_table = (
        asymmetry.group_by(
            "cube_id",
            "channel",
            "selection_label",
            "mid_time",
            "decimal_year",
            "flyby",
        )
        .agg(
            pl.len().alias("band_count"),
            pl.col("north_u_sum").median().alias("median_north_u_sum"),
            pl.col("south_u_sum").median().alias("median_south_u_sum"),
            pl.col("u_sum_difference").median().alias("median_u_sum_difference"),
            pl.col("u_sum_difference")
            .quantile(0.25, interpolation="linear")
            .alias("lower_band_quartile"),
            pl.col("u_sum_difference")
            .quantile(0.75, interpolation="linear")
            .alias("upper_band_quartile"),
        )
        .with_columns(
            pl.when(pl.col("mid_time") < policy.northern_vernal_equinox)
            .then(pl.lit(SEASON_ORDER[0]))
            .when(pl.col("mid_time") < policy.northern_summer_solstice)
            .then(pl.lit(SEASON_ORDER[1]))
            .otherwise(pl.lit(SEASON_ORDER[2]))
            .alias("northern_season")
        )
        .with_columns(
            pl.col("northern_season")
            .replace_strict(SOUTHERN_SEASON)
            .alias("southern_season")
        )
        .sort("mid_time", "channel")
    )
    return cube_table


def _bootstrap_median_interval(
    values: FloatArray, policy: SeasonPolicy, rng: np.random.Generator
) -> tuple[float | None, float | None]:
    if len(values) < policy.minimum_group_observations:
        return None, None
    draws = rng.choice(
        values,
        size=(policy.bootstrap_resamples, len(values)),
        replace=True,
    )
    medians = np.median(draws, axis=1)
    tail = (1 - policy.confidence_level) / 2
    return float(np.quantile(medians, tail)), float(np.quantile(medians, 1 - tail))


def summarize_seasonal_groups(
    cube_table: pl.DataFrame, policy: SeasonPolicy
) -> pl.DataFrame:
    """Summarize cube-level values and add intervals only for adequate groups."""
    rng = np.random.default_rng(policy.random_seed)
    rows: list[dict[str, str | int | float | bool | None]] = []
    for season in SEASON_ORDER:
        for channel in ("visible", "infrared"):
            group = cube_table.filter(
                (pl.col("northern_season") == season) & (pl.col("channel") == channel)
            )
            values = group.get_column("median_u_sum_difference").to_numpy()
            lower, upper = _bootstrap_median_interval(values, policy, rng)
            rows.append(
                {
                    "northern_season": season,
                    "southern_season": SOUTHERN_SEASON[season],
                    "channel": channel,
                    "observation_count": len(values),
                    "median_u_sum_difference": (
                        float(np.median(values)) if len(values) else None
                    ),
                    "confidence_level": policy.confidence_level,
                    "bootstrap_lower": lower,
                    "bootstrap_upper": upper,
                    "interval_available": lower is not None,
                }
            )
    return pl.from_dicts(rows, infer_schema_length=None)


def write_seasonal_parquet(
    asymmetry_path: Path,
    cube_output: Path,
    summary_output: Path,
    policy: SeasonPolicy,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    cube_table = build_seasonal_cube_table(pl.read_parquet(asymmetry_path), policy)
    summary = summarize_seasonal_groups(cube_table, policy)
    cube_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    cube_table.write_parquet(cube_output, compression="zstd", statistics=True)
    summary.write_parquet(summary_output, compression="zstd", statistics=True)
    return cube_table, summary
