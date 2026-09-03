"""Fit-quality rules expressed as a Polars table transform."""

from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from titan_limb.io.atomic import atomic_write_parquet
from titan_limb.models.core import (
    FitQualityReason,
    FitStatus,
    QualityStatus,
)

FIT_PARAMETER_COUNT = 3
COVARIANCE_VALUE_COUNT = FIT_PARAMETER_COUNT**2
IDENTITY_COLUMNS = ["cube_id", "band", "hemisphere"]
REQUIRED_COLUMNS = {
    *IDENTITY_COLUMNS,
    "profile_points",
    "status",
    "intensity_center",
    "u1",
    "u2",
    "r_squared",
    "covariance",
}


class FitQualityPolicy(BaseModel):
    """Rules that can change which otherwise valid fits need review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_profile_points: int = Field(default=6, ge=FIT_PARAMETER_COUNT)
    minimum_r_squared: float | None = Field(default=None, ge=-1, le=1)
    maximum_absolute_coefficient: float | None = Field(default=None, gt=0)


def _reason(condition: pl.Expr, reason: FitQualityReason) -> pl.Expr:
    return pl.when(condition).then(pl.lit(reason.value)).otherwise(pl.lit(None))


def audit_fit_table(table: pl.DataFrame, policy: FitQualityPolicy) -> pl.DataFrame:
    """Return one quality row for each fit without dropping source rows."""
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"fit table is missing columns: {', '.join(sorted(missing))}")
    failed = pl.col("status") != FitStatus.SUCCEEDED.value
    succeeded = ~failed
    too_few = succeeded & (pl.col("profile_points") < policy.minimum_profile_points)
    values_finite = pl.all_horizontal(
        pl.col("intensity_center", "u1", "u2", "r_squared").is_finite()
    ).fill_null(False)
    covariance_valid = (
        (pl.col("covariance").list.len() == COVARIANCE_VALUE_COUNT)
        & pl.col("covariance").list.eval(pl.element().is_finite()).list.all()
    ).fill_null(False)
    non_finite = succeeded & ~values_finite
    invalid_covariance = succeeded & ~covariance_valid
    negative_r_squared = succeeded & (pl.col("r_squared") < 0).fill_null(False)
    below_policy = (
        (pl.col("r_squared") < policy.minimum_r_squared).fill_null(False)
        if policy.minimum_r_squared is not None
        else pl.lit(False)
    )
    outside_policy = (
        (
            (pl.col("u1").abs() > policy.maximum_absolute_coefficient)
            | (pl.col("u2").abs() > policy.maximum_absolute_coefficient)
        ).fill_null(False)
        if policy.maximum_absolute_coefficient is not None
        else pl.lit(False)
    )
    ineligible = failed | too_few | non_finite | invalid_covariance
    review = negative_r_squared | below_policy | outside_policy
    return table.select(
        *IDENTITY_COLUMNS,
        pl.when(ineligible)
        .then(pl.lit(QualityStatus.INELIGIBLE.value))
        .when(review)
        .then(pl.lit(QualityStatus.REVIEW.value))
        .otherwise(pl.lit(QualityStatus.ELIGIBLE.value))
        .alias("quality_status"),
        pl.concat_list(
            _reason(failed, FitQualityReason.FIT_FAILED),
            _reason(too_few, FitQualityReason.TOO_FEW_POINTS),
            _reason(non_finite, FitQualityReason.NON_FINITE_VALUE),
            _reason(invalid_covariance, FitQualityReason.INVALID_COVARIANCE),
            _reason(negative_r_squared, FitQualityReason.NEGATIVE_R_SQUARED),
            _reason(below_policy, FitQualityReason.R_SQUARED_BELOW_POLICY),
            _reason(outside_policy, FitQualityReason.COEFFICIENT_OUTSIDE_POLICY),
        )
        .list.drop_nulls()
        .alias("quality_reasons"),
    )


def audit_fit_parquet(
    source: Path, output: Path, policy: FitQualityPolicy
) -> pl.DataFrame:
    result = audit_fit_table(pl.read_parquet(source), policy)
    atomic_write_parquet(result, output)
    return result
