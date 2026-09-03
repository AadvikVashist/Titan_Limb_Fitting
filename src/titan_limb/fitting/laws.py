"""Pure limb-law functions using emission-angle cosine as mu."""

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def emission_angle_to_mu(emission_angle_degrees: FloatArray) -> FloatArray:
    return np.cos(np.deg2rad(emission_angle_degrees))


def linear(mu: FloatArray, intensity_center: float, u: float) -> FloatArray:
    return intensity_center * (1 - u * (1 - mu))


def quadratic(
    mu: FloatArray, intensity_center: float, u1: float, u2: float
) -> FloatArray:
    return intensity_center * (1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2)


def square_root(
    mu: FloatArray, intensity_center: float, u1: float, u2: float
) -> FloatArray:
    return intensity_center * (1 - u1 * (1 - mu) - u2 * (1 - np.sqrt(mu)))
