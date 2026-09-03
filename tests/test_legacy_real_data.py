"""Reference checks against one preserved selected-fit pickle."""

import os
import pickle
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from titan_limb.fitting.optimizer import fit_quadratic_profile
from titan_limb.io.legacy import read_selected_fit_pickle
from titan_limb.io.vims import find_cube_pair, load_cube_pair
from titan_limb.models.core import Hemisphere, SmoothingMethod
from titan_limb.processing.destripe import destripe_visible
from titan_limb.processing.geometry import find_image_center, radial_line_indices
from titan_limb.processing.orientation import (
    find_north_orientation,
    illumination_angle,
)

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
FIT_REFERENCE_CUBES = ("C1477456872_1", "C1649210035_1", "C1875658704_1")
FIT_REFERENCE_BANDS = (1, 100, 200, 352)


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
    visible_north = find_north_orientation(visible.lat, center)
    assert visible_north == pytest.approx(176.09482167352536)
    assert illumination_angle(visible.inc, center, visible_north) == pytest.approx(
        138.90517832647464
    )

    infrared_center = find_image_center(infrared.eme)
    infrared_north = find_north_orientation(infrared.lat, infrared_center)
    assert infrared_center.subpixel == pytest.approx((26.25, 26.75))
    assert infrared_north == pytest.approx(177.22685185185185)

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


@pytest.mark.real_data
def test_reference_visible_cube_destriping() -> None:
    data_dir = legacy_selected_dir().parent
    pair = find_cube_pair(data_dir / "original_cubes", REFERENCE_CUBE)
    visible, _ = load_cube_pair(pair)
    analysis_path = data_dir / "cube_analysis" / f"{REFERENCE_CUBE}.pkl"
    with analysis_path.open("rb") as source:
        analysis = pickle.load(source)
    expected_bands = analysis["meta"]["cube_vis"]["bands"]

    for band in range(1, 97):
        actual = destripe_visible(visible[band], visible.ground)
        np.testing.assert_array_equal(actual, expected_bands[band - 1])


@pytest.mark.real_data
def test_reference_first_band_quadratic_fits() -> None:
    path = legacy_selected_dir() / f"{REFERENCE_CUBE}.pkl"
    with path.open("rb") as source:
        band = pickle.load(source)["0.35054µm_1"]

    for side_name in ("north_side", "south_side"):
        side = band[side_name]
        actual = fit_quadratic_profile(
            np.asarray(side["emission_angles"]),
            np.asarray(side["brightness_values"]),
        ).optimal
        expected = side["fit"]["quadratic"]["optimal_fit"]
        parameters = expected["fit_params"]
        assert actual.intensity_center == pytest.approx(parameters["I_0"], abs=3e-8)
        assert actual.u1 == pytest.approx(parameters["u1"], abs=3e-8)
        assert actual.u2 == pytest.approx(parameters["u2"], abs=3e-8)
        assert actual.r_squared == pytest.approx(expected["r2"], abs=5e-9)
        np.testing.assert_allclose(
            np.asarray(actual.covariance).reshape(3, 3),
            expected["covariance_matrix"],
            rtol=0,
            atol=7e-12,
        )


@pytest.mark.real_data
def test_reference_fits_across_time_and_channels() -> None:
    for cube_id in FIT_REFERENCE_CUBES:
        with (legacy_selected_dir() / f"{cube_id}.pkl").open("rb") as source:
            cube = pickle.load(source)
        by_band = {
            int(key.rsplit("_", 1)[1]): value
            for key, value in cube.items()
            if "µm_" in key
        }
        for band_number in FIT_REFERENCE_BANDS:
            for side_name in ("north_side", "south_side"):
                side = by_band[band_number][side_name]
                actual = fit_quadratic_profile(
                    np.asarray(side["emission_angles"]),
                    np.asarray(side["brightness_values"]),
                ).optimal
                expected = side["fit"]["quadratic"]["optimal_fit"]
                parameters = expected["fit_params"]
                expected_method = (
                    SmoothingMethod.GAUSSIAN
                    if "sigma" in expected
                    else SmoothingMethod.MOVING_AVERAGE
                    if "window" in expected
                    else SmoothingMethod.INTERPOLATED
                )
                assert actual.method is expected_method
                assert actual.intensity_center == pytest.approx(
                    parameters["I_0"], abs=2e-7
                )
                assert actual.u1 == pytest.approx(parameters["u1"], abs=2e-7)
                assert actual.u2 == pytest.approx(parameters["u2"], abs=2e-7)
                assert actual.r_squared == pytest.approx(expected["r2"], abs=5e-8)
                np.testing.assert_allclose(
                    np.asarray(actual.covariance),
                    np.asarray(expected["covariance_matrix"]).reshape(-1),
                    rtol=0,
                    atol=1e-7,
                )
