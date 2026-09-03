"""Pure orientation and transect-selection rules for VIMS cubes."""

from dataclasses import dataclass

import numpy as np

from titan_limb.processing.geometry import (
    IMAGE_DIMENSIONS,
    FloatArray,
    ImageCenter,
    find_image_center,
)

EXTREME_ANGLE_SAMPLES = 20
ANGLE_WRAP_SPREAD_DEGREES = 10.0
HALF_TURN_DEGREES = 180.0
FULL_TURN_DEGREES = 360.0
QUARTER_TURN_DEGREES = 90.0


@dataclass(frozen=True)
class TransectGeometry:
    center: ImageCenter
    north_orientation_degrees: float
    illumination_degrees: float
    north_slant_degrees: int
    south_slant_degrees: int


def _unwrap_angles(angles: FloatArray) -> FloatArray:
    if np.std(angles) <= ANGLE_WRAP_SPREAD_DEGREES:
        return angles
    return np.where(angles < 0, angles + FULL_TURN_DEGREES, angles)


def find_north_orientation(latitude: FloatArray, center: ImageCenter) -> float:
    """Match the saved latitude-extrema estimate without plots or file access."""
    if latitude.ndim != IMAGE_DIMENSIONS:
        raise ValueError("latitude must be a two-dimensional image")
    rows, columns = np.indices(latitude.shape)
    angles = np.trunc(
        np.degrees(
            np.arctan2(
                columns - center.pixel[1],
                center.pixel[0] - rows,
            )
        )
    ).astype(np.int64)
    actual_angles = np.unique(angles)
    extreme_latitudes = np.asarray(
        [
            latitude[angles == angle][np.argmax(np.abs(latitude[angles == angle]))]
            for angle in actual_angles
        ],
        dtype=np.float64,
    )
    order = np.argsort(extreme_latitudes, kind="stable")
    minimum_angles = _unwrap_angles(
        actual_angles[order[:EXTREME_ANGLE_SAMPLES]].astype(np.float64)
    )
    maximum_angles = _unwrap_angles(
        actual_angles[order[-EXTREME_ANGLE_SAMPLES:]].astype(np.float64)
    )
    minimum = float(np.mean(minimum_angles)) % FULL_TURN_DEGREES
    maximum = float(np.mean(maximum_angles)) % FULL_TURN_DEGREES
    positive_weight = float(np.count_nonzero(angles > 0) / angles.size)
    negative_weight = float(np.count_nonzero(angles < 0) / angles.size)
    if minimum > maximum:
        orientation = (
            minimum * negative_weight + maximum * positive_weight - QUARTER_TURN_DEGREES
        )
    else:
        orientation = (
            minimum * negative_weight + maximum * positive_weight + QUARTER_TURN_DEGREES
        )
    if orientation > HALF_TURN_DEGREES:
        orientation -= FULL_TURN_DEGREES
    return orientation


def illumination_angle(
    incidence: FloatArray,
    center: ImageCenter,
    north_orientation_degrees: float,
) -> float:
    """Return the lowest-incidence direction relative to north."""
    incidence_center = find_image_center(incidence).subpixel
    absolute = np.degrees(
        np.arctan2(
            incidence_center[1] - center.pixel[1],
            center.pixel[0] - incidence_center[0],
        )
    )
    if absolute < 0:
        absolute += FULL_TURN_DEGREES
    relative = float(absolute - north_orientation_degrees)
    if relative > FULL_TURN_DEGREES:
        relative -= FULL_TURN_DEGREES
    return relative


def select_hemisphere_slants(illumination_degrees: float) -> tuple[int, int]:
    """Keep the legacy lit-side choice between the two diagonal slant pairs."""
    if illumination_degrees <= HALF_TURN_DEGREES:
        return 60, 120
    return 300, 240


def derive_transect_geometry(
    emission: FloatArray,
    latitude: FloatArray,
    incidence: FloatArray,
) -> TransectGeometry:
    center = find_image_center(emission)
    orientation = find_north_orientation(latitude, center)
    illumination = illumination_angle(incidence, center, orientation)
    north, south = select_hemisphere_slants(illumination)
    return TransectGeometry(center, orientation, illumination, north, south)
