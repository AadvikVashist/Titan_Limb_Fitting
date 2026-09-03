"""Tests for direct cube-to-profile processing."""

from dataclasses import dataclass

import numpy as np
import polars as pl

from titan_limb.processing.cube_profiles import build_selected_profiles


@dataclass
class FakeCube:
    img_id: str
    bands: np.ndarray
    wvlns: np.ndarray
    eme: np.ndarray
    inc: np.ndarray
    lat: np.ndarray
    ground: np.ndarray
    image: np.ndarray

    def __getitem__(self, band: int) -> np.ndarray:
        return self.image + band / 1000


def fake_cube(band: int, wavelength: float) -> FakeCube:
    rows, columns = np.indices((15, 15))
    radius = np.hypot(rows - 7, columns - 7)
    return FakeCube(
        img_id="C1",
        bands=np.array([band]),
        wvlns=np.array([wavelength]),
        eme=radius * 10,
        inc=np.hypot(rows - 4, columns - 9),
        lat=(7 - rows).astype(float),
        ground=radius <= 5,
        image=(rows + columns).astype(float),
    )


def test_build_selected_profiles_returns_two_sides_per_band() -> None:
    visible = fake_cube(1, 0.5)
    infrared = fake_cube(97, 1.0)

    result = build_selected_profiles(
        visible, infrared, legacy_ir_incidence_source=False
    )

    assert result.height == 4
    assert result.get_column("band").to_list() == [1, 1, 97, 97]
    assert result.get_column("hemisphere").to_list() == [
        "north",
        "south",
        "north",
        "south",
    ]
    minimum_points = result.select(pl.col("emission_angles").list.len().min()).item()
    assert isinstance(minimum_points, int)
    assert minimum_points >= 3
