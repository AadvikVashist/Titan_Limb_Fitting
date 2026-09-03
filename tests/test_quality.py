"""Tests for fit-quality rules."""

import math

import polars as pl
import pytest

from titan_limb.fitting.quality import FitQualityPolicy, audit_fit_table


def fit_table() -> pl.DataFrame:
    rows = []
    for band, values in enumerate(
        [
            ("succeeded", 10, 0.8, 0.2, -0.1),
            ("failed", 10, None, None, None),
            ("succeeded", 4, 0.8, 0.2, -0.1),
            ("succeeded", 10, -0.2, 0.2, -0.1),
            ("succeeded", 10, 0.8, 20.0, -0.1),
            ("succeeded", 10, 0.8, math.nan, -0.1),
        ],
        start=1,
    ):
        status, points, r_squared, u1, u2 = values
        rows.append(
            {
                "cube_id": "C_TEST",
                "band": band,
                "hemisphere": "north",
                "profile_points": points,
                "status": status,
                "intensity_center": 1.0 if status == "succeeded" else None,
                "u1": u1,
                "u2": u2,
                "r_squared": r_squared,
                "covariance": [1.0] * 9 if status == "succeeded" else None,
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None)


def test_audit_separates_hard_failures_and_review_flags() -> None:
    result = audit_fit_table(
        fit_table(), FitQualityPolicy(maximum_absolute_coefficient=10)
    )

    assert result.get_column("quality_status").to_list() == [
        "eligible",
        "ineligible",
        "ineligible",
        "review",
        "review",
        "ineligible",
    ]
    assert result.row(1, named=True)["quality_reasons"] == ["fit_failed"]
    assert result.row(3, named=True)["quality_reasons"] == ["negative_r_squared"]
    assert result.row(4, named=True)["quality_reasons"] == [
        "coefficient_outside_policy"
    ]


def test_audit_requires_contract_columns() -> None:
    with pytest.raises(ValueError, match="covariance"):
        audit_fit_table(pl.DataFrame({"cube_id": ["C_TEST"]}), FitQualityPolicy())
