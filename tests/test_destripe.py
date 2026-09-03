"""Tests for visible-channel destriping."""

import numpy as np
import pytest

from titan_limb.processing.destripe import (
    destripe_visible,
    expanded_surface_mask,
    vertical_edge_gap,
)


def test_expanded_surface_mask() -> None:
    ground = np.zeros((15, 15), dtype=bool)
    ground[7, 7] = True

    expanded = expanded_surface_mask(ground)

    assert expanded.sum() == 9


def test_vertical_edge_gap() -> None:
    mask = np.zeros((10, 4), dtype=bool)
    mask[2:8] = True
    assert vertical_edge_gap(mask) == 4
    with pytest.raises(ValueError, match="no selected pixels"):
        vertical_edge_gap(np.zeros((3, 3), dtype=bool))


def test_destripe_visible_removes_column_bias() -> None:
    column_bias = np.arange(8, dtype=np.float32)
    image = np.tile(column_bias, (8, 1))
    ground = np.zeros((8, 8), dtype=bool)
    ground[3:5, 3:5] = True

    result = destripe_visible(image, ground)

    np.testing.assert_allclose(result, 0)
    assert result.dtype == image.dtype


def test_destripe_visible_returns_copy_when_surface_reaches_edges() -> None:
    image = np.ones((8, 8), dtype=np.float32)
    ground = np.ones((8, 8), dtype=bool)
    result = destripe_visible(image, ground)
    np.testing.assert_array_equal(result, image)
    assert result is not image


def test_destripe_visible_validates_input() -> None:
    with pytest.raises(ValueError, match="matching"):
        destripe_visible(np.ones((2, 2)), np.ones((3, 3), dtype=bool))
