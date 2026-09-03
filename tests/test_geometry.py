"""Tests for radial profile geometry."""

import numpy as np
import pytest

from titan_limb.processing.geometry import (
    distance_from_center,
    find_image_center,
    radial_line_indices,
)


def test_find_image_center_uses_lowest_emission_pixels() -> None:
    emission = np.full((4, 4), 90.0)
    emission[1:3, 1:3] = [[1.0, 2.0], [3.0, 4.0]]

    center = find_image_center(emission)

    assert center.subpixel == pytest.approx((1.25, 1.25))
    assert center.pixel == (1, 1)


def test_find_image_center_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        find_image_center(np.ones(5))


def test_distance_from_center() -> None:
    distances = distance_from_center((3, 3), (1, 1))

    assert distances[1, 1] == 0
    assert distances[0, 0] == pytest.approx(np.sqrt(2))


def test_radial_line_indices_stay_in_bounds() -> None:
    indices = radial_line_indices((5, 7), (2.0, 3.0), 90)

    assert indices.shape[1] == 2
    assert np.all(indices >= 0)
    assert np.all(indices < np.array([5, 7]))


@pytest.mark.parametrize(("scale", "thickness"), [(0, 2), (5, 0)])
def test_radial_line_indices_rejects_invalid_settings(
    scale: int, thickness: int
) -> None:
    with pytest.raises(ValueError):
        radial_line_indices((5, 5), (2.0, 2.0), 0, scale=scale, thickness=thickness)
