"""Reference checks against one preserved selected-fit pickle."""

import os
import pickle
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from titan_limb.io.legacy import read_selected_fit_pickle
from titan_limb.io.vims import find_cube_pair, load_cube_pair
from titan_limb.models.core import Hemisphere
from titan_limb.processing.geometry import find_image_center, radial_line_indices

DATA_ENV = "TITAN_LEGACY_SELECTED_DIR"
REFERENCE_CUBE = "C1477456872_1"
REFERENCE_BAND = 1
NORTH_INTENSITY_CENTER = 0.056382679454333354
NORTH_U1 = 0.12333728538048683
NORTH_U2 = 0.06264528024855065
NORTH_R_SQUARED = 0.8300854193263463
SOUTH_INTENSITY_CENTER = 0.06403271289245795
SOUTH_U1 = 0.31494449405482283
SOUTH_U2 = -0.28774366440771537
SOUTH_R_SQUARED = 0.2744249400699411


def legacy_selected_dir() -> Path:
    source_dir_value = os.environ.get(DATA_ENV)
    if source_dir_value is None:
        pytest.skip(f"set {DATA_ENV} to run preserved-data checks")
    return Path(source_dir_value)


@pytest.mark.real_data
def test_reference_cube_first_band_matches_saved_fit() -> None:
    records = read_selected_fit_pickle(legacy_selected_dir() / f"{REFERENCE_CUBE}.pkl")
    by_side = {
        record.hemisphere: record for record in records if record.band == REFERENCE_BAND
    }
    north = by_side[Hemisphere.NORTH]
    south = by_side[Hemisphere.SOUTH]

    assert north.intensity_center == pytest.approx(NORTH_INTENSITY_CENTER)
    assert north.u1 == pytest.approx(NORTH_U1)
    assert north.u2 == pytest.approx(NORTH_U2)
    assert north.r_squared == pytest.approx(NORTH_R_SQUARED)
    assert south.intensity_center == pytest.approx(SOUTH_INTENSITY_CENTER)
    assert south.u1 == pytest.approx(SOUTH_U1)
    assert south.u2 == pytest.approx(SOUTH_U2)
    assert south.r_squared == pytest.approx(SOUTH_R_SQUARED)


@pytest.mark.real_data
def test_reference_cube_load_and_first_radial_line() -> None:
    data_dir = legacy_selected_dir().parent
    pair = find_cube_pair(data_dir / "original_cubes", REFERENCE_CUBE)
    visible, infrared = load_cube_pair(pair)

    assert len(visible.bands) == 96
    assert len(infrared.bands) == 256
    assert visible[1].shape == (54, 54)
    assert infrared[97].shape == (54, 54)
    assert visible.wvlns[[0, -1]].tolist() == pytest.approx([0.35054, 1.04598])
    assert infrared.wvlns[[0, -1]].tolist() == pytest.approx([0.88421, 5.12342])

    center = find_image_center(visible.eme)
    assert center.subpixel == pytest.approx((24.875, 28.0))

    analysis_path = data_dir / "cube_analysis" / f"{REFERENCE_CUBE}.pkl"
    with analysis_path.open("rb") as source:
        analysis = cast(
            dict[str, dict[int, dict[str, np.ndarray]]], pickle.load(source)
        )
    expected = analysis["0.35054µm_1"][0]["pixel_indices"]
    actual = radial_line_indices(visible.eme.shape, center.subpixel, 176.09482167352536)
    expected_pixels = {tuple(pixel) for pixel in expected.tolist()}
    actual_pixels = {tuple(pixel) for pixel in actual.tolist()}
    assert len(expected_pixels) == 42
    assert len(actual_pixels) == 43
    assert expected_pixels - actual_pixels == set()
    assert actual_pixels - expected_pixels == {(38, 30)}
