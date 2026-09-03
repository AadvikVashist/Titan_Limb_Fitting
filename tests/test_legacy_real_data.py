"""Reference checks against one preserved selected-fit pickle."""

import os
from pathlib import Path

import pytest

from titan_limb.io.legacy import read_selected_fit_pickle
from titan_limb.models.core import Hemisphere

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


@pytest.mark.real_data
def test_reference_cube_first_band_matches_saved_fit() -> None:
    source_dir_value = os.environ.get(DATA_ENV)
    if source_dir_value is None:
        pytest.skip(f"set {DATA_ENV} to run preserved-data checks")
    records = read_selected_fit_pickle(Path(source_dir_value) / f"{REFERENCE_CUBE}.pkl")
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
