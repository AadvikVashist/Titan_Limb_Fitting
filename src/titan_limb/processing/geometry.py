"""Pure image geometry used to build radial Titan profiles."""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int64]
IMAGE_DIMENSIONS = 2
MINIMUM_CENTER_PIXELS = 6


@dataclass(frozen=True)
class ImageCenter:
    subpixel: tuple[float, float]
    pixel: tuple[int, int]


def find_image_center(emission_angles: FloatArray) -> ImageCenter:
    """Match the legacy center estimate from the lowest emission pixels."""
    if (
        emission_angles.ndim != IMAGE_DIMENSIONS
        or emission_angles.size < MINIMUM_CENTER_PIXELS
    ):
        raise ValueError("emission angles must be a two-dimensional image")
    flat = emission_angles.reshape(-1)
    nearby_flat = np.argpartition(flat, 5)[:4]
    nearby = np.array(np.unravel_index(nearby_flat, emission_angles.shape))
    lowest = np.array(np.unravel_index(np.argmin(flat), emission_angles.shape))
    subpixel = np.mean((lowest, np.mean(nearby, axis=1)), axis=0)
    pixel = np.rint(subpixel).astype(np.int64)
    return ImageCenter(
        subpixel=(float(subpixel[0]), float(subpixel[1])),
        pixel=(int(pixel[0]), int(pixel[1])),
    )


def distance_from_center(shape: tuple[int, int], center: tuple[int, int]) -> FloatArray:
    """Return pixel distance from a row-column center."""
    rows, columns = np.indices(shape)
    return np.hypot(columns - center[1], rows - center[0])


def radial_line_indices(
    shape: tuple[int, int],
    center: tuple[float, float],
    angle_degrees: float,
    *,
    scale: int = 5,
    thickness: int = 2,
) -> IndexArray:
    """Match the legacy upscaled radial-line pixel selection."""
    if scale < 1 or thickness < 1:
        raise ValueError("scale and thickness must be positive")
    upscaled_shape = (shape[0] * scale, shape[1] * scale)
    center_scaled = np.rint(np.asarray(center)) * scale + (scale - 1) / 2
    start_row, start_column = center_scaled
    distance = np.hypot(
        upscaled_shape[0] - start_row,
        upscaled_shape[1] - start_column,
    )
    angle = np.deg2rad(angle_degrees)
    end_column = start_column + np.sin(angle) * distance
    end_row = start_row - np.cos(angle) * distance
    mask = np.zeros(upscaled_shape, dtype=np.uint8)
    cv2.line(
        mask,
        (int(start_column), int(start_row)),
        (int(end_column), int(end_row)),
        color=1,
        thickness=thickness,
    )
    upscaled_indices = np.argwhere(mask == 1)
    indices = np.rint(upscaled_indices / scale)
    upper = np.asarray(shape) - 1
    return np.unique(np.clip(indices, 0, upper), axis=0).astype(np.int64)
