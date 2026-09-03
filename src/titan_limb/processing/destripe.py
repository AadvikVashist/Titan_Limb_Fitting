"""Visible-channel column-bias correction."""

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

MINIMUM_BACKGROUND_PIXELS = 3
MINIMUM_EDGE_GAP = 4
DILATION_SCALE_PIXELS = 15
MEAN_WEIGHT = 0.3
MEDIAN_WEIGHT = 0.7
IMAGE_DIMENSIONS = 2
MINIMUM_INTERPOLATION_COLUMNS = 2


def expanded_surface_mask(ground: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Expand Titan's ground mask by the legacy image-size rule."""
    if ground.ndim != IMAGE_DIMENSIONS:
        raise ValueError("ground mask must be a two-dimensional image")
    iterations = int(np.rint(np.mean(ground.shape) / DILATION_SCALE_PIXELS))
    iterations = max(iterations, 1)
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(ground.astype(np.uint8), kernel, iterations=iterations).astype(
        np.bool_
    )


def vertical_edge_gap(mask: NDArray[np.bool_]) -> int:
    """Return the combined clear space above and below the expanded surface."""
    rows = np.flatnonzero(np.any(mask, axis=1))
    if not len(rows):
        raise ValueError("surface mask has no selected pixels")
    return int(rows[0] + mask.shape[0] - rows[-1] - 1)


def destripe_visible(
    image: NDArray[np.floating], ground: NDArray[np.bool_]
) -> NDArray[np.floating]:
    """Subtract the robust background level from each image column."""
    if image.ndim != IMAGE_DIMENSIONS or image.shape != ground.shape:
        raise ValueError(
            "image and ground mask must be matching two-dimensional arrays"
        )
    surface = expanded_surface_mask(ground)
    if vertical_edge_gap(surface) < MINIMUM_EDGE_GAP:
        return image.copy()
    background = np.where(surface, np.nan, image)
    column_levels = np.full(image.shape[1], np.nan, dtype=np.float64)
    for column in range(image.shape[1]):
        values = background[:, column]
        values = values[np.isfinite(values)]
        if len(values) >= MINIMUM_BACKGROUND_PIXELS:
            column_levels[column] = (
                np.mean(values) * MEAN_WEIGHT + np.median(values) * MEDIAN_WEIGHT
            )
    known = np.flatnonzero(np.isfinite(column_levels))
    if len(known) < MINIMUM_INTERPOLATION_COLUMNS:
        raise ValueError("not enough background columns for destriping")
    if len(known) != len(column_levels):
        interpolation = interp1d(known, column_levels[known], kind="linear")
        column_levels = interpolation(np.arange(len(column_levels)))
    corrected = image - column_levels
    return corrected.astype(image.dtype, copy=False)
