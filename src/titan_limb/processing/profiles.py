"""Extract and filter radial brightness profiles without file or plot access."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from titan_limb.processing.geometry import FloatArray, IndexArray

PIXEL_INDEX_COLUMNS = 2


@dataclass(frozen=True)
class RadialProfile:
    pixel_indices: IndexArray
    pixel_distances: FloatArray
    emission_angles: FloatArray
    brightness: FloatArray

    def __post_init__(self) -> None:
        row_count = len(self.pixel_indices)
        lengths = (
            row_count,
            len(self.pixel_distances),
            len(self.emission_angles),
            len(self.brightness),
        )
        if len(set(lengths)) != 1:
            raise ValueError("profile arrays must have equal lengths")


def extract_profile(
    image: NDArray[np.floating],
    emission_angles: FloatArray,
    distances: FloatArray,
    indices: IndexArray,
) -> RadialProfile:
    """Read brightness, emission angle, and distance at selected pixels."""
    if image.shape != emission_angles.shape or image.shape != distances.shape:
        raise ValueError("image, emission angles, and distances must share a shape")
    if indices.ndim != PIXEL_INDEX_COLUMNS or indices.shape[1] != PIXEL_INDEX_COLUMNS:
        raise ValueError("pixel indices must have row-column pairs")
    if np.any(indices < 0) or np.any(indices >= np.asarray(image.shape)):
        raise IndexError("profile pixel lies outside the image")
    rows, columns = indices.T
    return RadialProfile(
        pixel_indices=indices.copy(),
        pixel_distances=np.asarray(distances[rows, columns], dtype=np.float64),
        emission_angles=np.asarray(emission_angles[rows, columns], dtype=np.float64),
        brightness=np.asarray(image[rows, columns], dtype=np.float64),
    )


def sort_and_filter_profile(
    profile: RadialProfile, *, maximum_emission_degrees: float = 89.0
) -> RadialProfile:
    """Match the saved profile order and emission-angle cutoff."""
    order = np.argsort(profile.emission_angles, kind="stable")
    keep = profile.emission_angles[order] <= maximum_emission_degrees
    selected = order[keep]
    return RadialProfile(
        pixel_indices=profile.pixel_indices[selected],
        pixel_distances=profile.pixel_distances[selected],
        emission_angles=profile.emission_angles[selected],
        brightness=profile.brightness[selected],
    )
