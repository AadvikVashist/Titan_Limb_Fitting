"""Tests for explicit VIMS band numbering."""

import pytest

from titan_limb.models.core import Channel
from titan_limb.processing.bands import (
    band_to_index,
    channel_for_band,
    index_to_band,
)


@pytest.mark.parametrize(
    ("band", "index"),
    [(1, 0), (96, 95), (97, 96), (352, 351)],
)
def test_band_index_round_trip(band: int, index: int) -> None:
    assert band_to_index(band) == index
    assert index_to_band(index) == band


@pytest.mark.parametrize("band", [0, 353])
def test_band_to_index_rejects_invalid_band(band: int) -> None:
    with pytest.raises(ValueError):
        band_to_index(band)


@pytest.mark.parametrize("index", [-1, 352])
def test_index_to_band_rejects_invalid_index(index: int) -> None:
    with pytest.raises(ValueError):
        index_to_band(index)


def test_channel_boundary() -> None:
    assert channel_for_band(96) is Channel.VISIBLE
    assert channel_for_band(97) is Channel.INFRARED
