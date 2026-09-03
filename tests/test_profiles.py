"""Tests for pure radial-profile processing."""

import numpy as np
import pytest

from titan_limb.processing.profiles import (
    RadialProfile,
    extract_profile,
    sort_and_filter_profile,
)


def test_extract_sort_and_filter_profile() -> None:
    image = np.arange(9, dtype=np.float64).reshape(3, 3)
    emission = np.array([[90, 80, 70], [60, 50, 40], [30, 20, 10]], dtype=float)
    distances = np.arange(9, dtype=np.float64).reshape(3, 3) / 2
    indices = np.array([[0, 0], [2, 2], [1, 1]], dtype=np.int64)

    profile = extract_profile(image, emission, distances, indices)
    result = sort_and_filter_profile(profile)

    assert result.pixel_indices.tolist() == [[2, 2], [1, 1]]
    assert result.emission_angles.tolist() == [10.0, 50.0]
    assert result.brightness.tolist() == [8.0, 4.0]


def test_profile_requires_equal_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        RadialProfile(
            pixel_indices=np.array([[0, 0]], dtype=np.int64),
            pixel_distances=np.array([], dtype=float),
            emission_angles=np.array([1.0]),
            brightness=np.array([1.0]),
        )


def test_extract_profile_validates_shapes_and_indices() -> None:
    image = np.ones((2, 2))
    indices = np.array([[0, 0]], dtype=np.int64)
    with pytest.raises(ValueError, match="share a shape"):
        extract_profile(image, np.ones((3, 3)), image, indices)
    with pytest.raises(ValueError, match="row-column pairs"):
        extract_profile(image, image, image, np.array([0, 0], dtype=np.int64))
    with pytest.raises(IndexError, match="outside"):
        extract_profile(image, image, image, np.array([[2, 0]], dtype=np.int64))
