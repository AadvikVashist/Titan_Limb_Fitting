"""Prepare sorted limb profiles for fitting."""

from typing import Any

import numpy as np
import polars as pl

DEFAULT_MINIMUM_EMISSION_DEGREES = 25.0
PROFILE_COLUMNS = (
    "pixel_rows",
    "pixel_columns",
    "pixel_distances",
    "emission_angles",
    "brightness",
)


def filter_profiles_by_emission(
    profiles: pl.DataFrame,
    minimum_emission_degrees: float = DEFAULT_MINIMUM_EMISSION_DEGREES,
) -> pl.DataFrame:
    """Keep points beyond the old 25-degree inner-limb cutoff."""
    rows: list[dict[str, Any]] = []
    for source_row in profiles.iter_rows(named=True):
        row = dict(source_row)
        emission = np.asarray(row["emission_angles"], dtype=np.float64)
        keep = emission > minimum_emission_degrees
        row["fit_filtered"] = bool(np.any(~keep))
        row["minimum_fit_emission_degrees"] = minimum_emission_degrees
        for column in PROFILE_COLUMNS:
            row[column] = np.asarray(row[column])[keep].tolist()
        rows.append(row)
    return pl.DataFrame(rows, schema=profiles.schema).sort("band", "hemisphere")
