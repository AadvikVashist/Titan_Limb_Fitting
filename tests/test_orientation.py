"""Tests for pure cube-orientation rules."""

import numpy as np

from titan_limb.processing.geometry import ImageCenter
from titan_limb.processing.orientation import (
    illumination_angle,
    select_hemisphere_slants,
)


def test_select_hemisphere_slants() -> None:
    assert select_hemisphere_slants(180.0) == (60, 120)
    assert select_hemisphere_slants(180.1) == (300, 240)


def test_illumination_angle_uses_image_geometry() -> None:
    incidence = np.full((5, 5), 10.0)
    incidence[1:3, 2:4] = [[0.0, 1.0], [2.0, 3.0]]
    center = ImageCenter(subpixel=(2.0, 2.0), pixel=(2, 2))

    result = illumination_angle(incidence, center, 0.0)

    assert 0 <= result <= 90
