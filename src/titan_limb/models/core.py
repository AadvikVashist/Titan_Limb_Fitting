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
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"
