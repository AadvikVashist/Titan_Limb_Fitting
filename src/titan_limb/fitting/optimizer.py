"""Quadratic limb fitting with explicit interpolation and smoothing choices."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from titan_limb.fitting.laws import emission_angle_to_mu, quadratic
from titan_limb.models.core import SmoothingMethod

FloatArray = NDArray[np.float64]
INTERPOLATED_POINTS = 200
LEGACY_SMOOTHING_SIZE = 20
PARAMETER_COUNT = 3


@dataclass(frozen=True)
class QuadraticCandidate:
    method: SmoothingMethod
    intensity_center: float
    u1: float
    u2: float
    covariance: tuple[float, ...]
    r_squared: float


@dataclass(frozen=True)
class QuadraticFitSet:
    standard: QuadraticCandidate
    gaussian: QuadraticCandidate
    moving_average: QuadraticCandidate
    optimal: QuadraticCandidate


def legacy_moving_average(values: FloatArray, window: int) -> FloatArray:
    """Preserve the moving average used to create the saved fits."""
    if window < 1:
        raise ValueError("window must be positive")
    reverse = np.mean(values[: len(values) // 2]) > np.mean(
        values[len(values) // 2 : -1]
    )
    source = values[::-1] if reverse else values
    result: list[float] = []
    current: list[float] = []
    for value in source:
        current.append(float(value))
        if len(current) >= window:
            current.pop(0)
        result.append(float(np.mean(current)))
    output = np.asarray(result, dtype=np.float64)
    return output[::-1] if reverse else output


def interpolate_profile(
    emission_angles: FloatArray, brightness: FloatArray
) -> tuple[FloatArray, FloatArray]:
    if (
        len(emission_angles) != len(brightness)
        or len(emission_angles) < PARAMETER_COUNT
    ):
        raise ValueError("a quadratic fit needs at least three paired profile points")
    mu = emission_angle_to_mu(emission_angles)
    order = np.argsort(mu, kind="stable")
    sorted_mu = mu[order]
    sorted_brightness = brightness[order]
    if np.any(np.diff(sorted_mu) <= 0):
        raise ValueError("profile emission angles must map to distinct mu values")
    fit_mu = np.linspace(sorted_mu[0], sorted_mu[-1], INTERPOLATED_POINTS)
    fit_brightness = PchipInterpolator(sorted_mu, sorted_brightness)(fit_mu)
    return fit_mu, np.asarray(fit_brightness, dtype=np.float64)


def _fit_candidate(
    method: SmoothingMethod,
    fit_mu: FloatArray,
    fit_brightness: FloatArray,
    source_mu: FloatArray,
    source_brightness: FloatArray,
) -> QuadraticCandidate:
    parameters, covariance = curve_fit(
        quadratic,
        fit_mu,
        fit_brightness,
        p0=[1.0, 0.5, 0.5],
        bounds=([0.0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
    )
    prediction = quadratic(source_mu, *parameters)
    return QuadraticCandidate(
        method=method,
        intensity_center=float(parameters[0]),
        u1=float(parameters[1]),
        u2=float(parameters[2]),
        covariance=tuple(float(value) for value in covariance.reshape(-1)),
        r_squared=float(r2_score(source_brightness, prediction)),
    )


def fit_quadratic_profile(
    emission_angles: FloatArray, brightness: FloatArray
) -> QuadraticFitSet:
    """Fit the three saved smoothing variants and choose the best R²."""
    fit_mu, interpolated = interpolate_profile(emission_angles, brightness)
    source_mu = emission_angle_to_mu(emission_angles)
    standard = _fit_candidate(
        SmoothingMethod.INTERPOLATED,
        fit_mu,
        interpolated,
        source_mu,
        brightness,
    )
    gaussian = _fit_candidate(
        SmoothingMethod.GAUSSIAN,
        fit_mu,
        gaussian_filter1d(interpolated, sigma=LEGACY_SMOOTHING_SIZE),
        source_mu,
        brightness,
    )
    moving_average = _fit_candidate(
        SmoothingMethod.MOVING_AVERAGE,
        fit_mu,
        legacy_moving_average(interpolated, LEGACY_SMOOTHING_SIZE),
        source_mu,
        brightness,
    )
    candidates = (standard, gaussian, moving_average)
    optimal = max(candidates, key=lambda candidate: candidate.r_squared)
    return QuadraticFitSet(standard, gaussian, moving_average, optimal)
