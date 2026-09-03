"""Finite scientific states used across the Titan limb analysis."""

from enum import StrEnum


class Channel(StrEnum):
    VISIBLE = "visible"
    INFRARED = "infrared"


class LimbLaw(StrEnum):
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    SQUARE_ROOT = "square_root"


class SmoothingMethod(StrEnum):
    INTERPOLATED = "interpolated"
    GAUSSIAN = "gaussian"
    MOVING_AVERAGE = "moving_average"


class FitStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Hemisphere(StrEnum):
    NORTH = "north"
    SOUTH = "south"


class FitFailureReason(StrEnum):
    MISSING_OPTIMAL_FIT = "missing_optimal_fit"
    MISSING_PARAMETERS = "missing_parameters"
    TOO_FEW_PROFILE_POINTS = "too_few_profile_points"
    OPTIMIZATION_FAILED = "optimization_failed"


class QualityStatus(StrEnum):
    ELIGIBLE = "eligible"
    REVIEW = "review"
    INELIGIBLE = "ineligible"


class FitQualityReason(StrEnum):
    FIT_FAILED = "fit_failed"
    TOO_FEW_POINTS = "too_few_points"
    NON_FINITE_VALUE = "non_finite_value"
    INVALID_COVARIANCE = "invalid_covariance"
    NEGATIVE_R_SQUARED = "negative_r_squared"
    R_SQUARED_BELOW_POLICY = "r_squared_below_policy"
    COEFFICIENT_OUTSIDE_POLICY = "coefficient_outside_policy"
