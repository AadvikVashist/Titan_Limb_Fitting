"""Paired north-south limb-coefficient differences."""

from pathlib import Path

import polars as pl

from titan_limb.config_bands import BandPolicy
from titan_limb.models.core import Hemisphere, QualityStatus

FIT_KEYS = ["cube_id", "band", "hemisphere"]
PAIR_KEYS = ["cube_id", "band", "wavelength_um", "channel"]


def build_asymmetry_table(
    fits: pl.DataFrame,
    quality: pl.DataFrame,
    band_policy: BandPolicy,
) -> pl.DataFrame:
    """Pair eligible hemispheres and calculate signed north-minus-south values."""
    eligible = (
        fits.join(quality, on=FIT_KEYS, how="inner")
        .filter(pl.col("quality_status") == QualityStatus.ELIGIBLE.value)
        .filter(~pl.col("band").is_in(sorted(band_policy.excluded_bands)))
        .select(*PAIR_KEYS, "hemisphere", "u1", "u2", "u_sum", "r_squared")
    )
    north = eligible.filter(pl.col("hemisphere") == Hemisphere.NORTH.value).drop(
        "hemisphere"
    )
    south = eligible.filter(pl.col("hemisphere") == Hemisphere.SOUTH.value).drop(
        "hemisphere"
    )
    return (
        north.join(south, on=PAIR_KEYS, suffix="_south")
        .rename(
            {
                "u1": "north_u1",
                "u2": "north_u2",
                "u_sum": "north_u_sum",
                "r_squared": "north_r_squared",
                "u1_south": "south_u1",
                "u2_south": "south_u2",
                "u_sum_south": "south_u_sum",
                "r_squared_south": "south_r_squared",
            }
        )
        .with_columns(
            (pl.col("north_u1") - pl.col("south_u1")).alias("u1_difference"),
            (pl.col("north_u2") - pl.col("south_u2")).alias("u2_difference"),
            (pl.col("north_u_sum") - pl.col("south_u_sum")).alias("u_sum_difference"),
        )
        .sort("cube_id", "band")
    )


def write_asymmetry_parquet(
    fits_path: Path,
    quality_path: Path,
    output: Path,
    band_policy: BandPolicy,
) -> pl.DataFrame:
    result = build_asymmetry_table(
        pl.read_parquet(fits_path), pl.read_parquet(quality_path), band_policy
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output, compression="zstd", statistics=True)
    return result
