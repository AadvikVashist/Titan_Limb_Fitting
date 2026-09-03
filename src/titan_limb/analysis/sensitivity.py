"""Sensitivity runs for fit cutoffs and quality rules."""

from pathlib import Path

import polars as pl

from titan_limb.analysis.asymmetry import build_asymmetry_table
from titan_limb.analysis.seasons import (
    build_seasonal_cube_table,
    summarize_seasonal_groups,
)
from titan_limb.analysis.transitions import build_transition_table
from titan_limb.config_bands import BandPolicy
from titan_limb.config_seasons import SeasonPolicy
from titan_limb.fitting.batch import fit_profile_table
from titan_limb.fitting.quality import FitQualityPolicy, audit_fit_table
from titan_limb.io.atomic import atomic_write_csv, atomic_write_parquet
from titan_limb.io.observations import attach_observation_metadata
from titan_limb.processing.fit_filter import filter_profiles_by_emission

EMISSION_CUTOFFS = (20.0, 25.0, 30.0)
R_SQUARED_LIMITS = (None, 0.25, 0.5)
COEFFICIENT_LIMITS = (None, 10.0)


def build_sensitivity_table(
    sorted_profiles: pl.DataFrame,
    observations: pl.DataFrame,
    band_policy: BandPolicy,
    season_policy: SeasonPolicy,
) -> pl.DataFrame:
    """Re-fit a small declared grid and report each season result."""
    rows: list[dict[str, str | int | float | None]] = []
    for emission_cutoff in EMISSION_CUTOFFS:
        fits = fit_profile_table(
            filter_profiles_by_emission(sorted_profiles, emission_cutoff)
        )
        for r_squared in R_SQUARED_LIMITS:
            for coefficient_limit in COEFFICIENT_LIMITS:
                quality = audit_fit_table(
                    fits,
                    FitQualityPolicy(
                        minimum_r_squared=r_squared,
                        maximum_absolute_coefficient=coefficient_limit,
                    ),
                )
                asymmetry = attach_observation_metadata(
                    build_asymmetry_table(fits, quality, band_policy), observations
                )
                seasons = summarize_seasonal_groups(
                    build_seasonal_cube_table(asymmetry, season_policy), season_policy
                )
                transitions = build_transition_table(fits, quality, band_policy)
                eligible = quality.filter(pl.col("quality_status") == "eligible").height
                multi = (
                    transitions.filter(pl.col("crossing_count") > 1)
                    .select("cube_id", "hemisphere")
                    .unique()
                    .height
                )
                rows.extend(
                    {
                        "minimum_emission_degrees": emission_cutoff,
                        "minimum_r_squared": r_squared,
                        "maximum_absolute_coefficient": coefficient_limit,
                        "eligible_fits": eligible,
                        "asymmetry_rows": asymmetry.height,
                        "multi_crossing_series": multi,
                        **seasonal_row,
                    }
                    for seasonal_row in seasons.iter_rows(named=True)
                )
    return pl.from_dicts(rows, infer_schema_length=None)


def write_sensitivity_table(
    sorted_profiles_path: Path,
    observations_path: Path,
    output: Path,
    band_policy: BandPolicy,
    season_policy: SeasonPolicy,
) -> pl.DataFrame:
    result = build_sensitivity_table(
        pl.read_parquet(sorted_profiles_path),
        pl.read_parquet(observations_path),
        band_policy,
        season_policy,
    )
    atomic_write_parquet(result, output)
    atomic_write_csv(result, output.with_suffix(".csv"))
    return result
