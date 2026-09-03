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
