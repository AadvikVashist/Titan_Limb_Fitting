"""Band-level checks for fit coverage, error, and time drift."""

from pathlib import Path

import polars as pl

MINIMUM_TRUSTED_FRACTION = 0.8


def build_band_trust_table(
    fits: pl.DataFrame, quality: pl.DataFrame, observations: pl.DataFrame
) -> pl.DataFrame:
    """Summarize fit health for each band without hiding failed rows."""
    metadata_columns = [
        column
        for column in ("cube_id", "decimal_year", "phase_degrees", "distance_km")
        if column in observations.columns
    ]
    joined = fits.join(quality, on=["cube_id", "band", "hemisphere"]).join(
        observations.select(metadata_columns),
        on="cube_id",
        how="left",
        validate="m:1",
    )
    return (
        joined.group_by("band", "channel")
        .agg(
            pl.col("wavelength_um").median().alias("wavelength_um"),
            pl.len().alias("fit_count"),
            (pl.col("quality_status") == "eligible").mean().alias("eligible_fraction"),
            pl.col("r_squared").median().alias("median_r_squared"),
            pl.col("r_squared").quantile(0.1).alias("tenth_percentile_r_squared"),
            pl.col("u_sum").std().alias("u_sum_standard_deviation"),
            pl.corr("decimal_year", "u_sum").alias("time_correlation"),
            pl.corr("phase_degrees", "u_sum").alias("phase_correlation")
            if "phase_degrees" in joined.columns
            else pl.lit(None, dtype=pl.Float64).alias("phase_correlation"),
            pl.corr("distance_km", "u_sum").alias("distance_correlation")
            if "distance_km" in joined.columns
            else pl.lit(None, dtype=pl.Float64).alias("distance_correlation"),
        )
        .with_columns(
            (
                (pl.col("eligible_fraction") >= MINIMUM_TRUSTED_FRACTION)
                & (pl.col("median_r_squared") >= 0)
            ).alias("trusted")
        )
        .sort("band")
    )


def write_band_trust_parquet(
    fits_path: Path,
    quality_path: Path,
    observations_path: Path,
    output: Path,
) -> pl.DataFrame:
    result = build_band_trust_table(
        pl.read_parquet(fits_path),
        pl.read_parquet(quality_path),
        pl.read_parquet(observations_path),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output, compression="zstd", statistics=True)
    return result
