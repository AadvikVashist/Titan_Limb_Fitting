"""Tests for typed profile-table fitting."""

import numpy as np
import polars as pl
import pytest

from titan_limb.fitting.batch import fit_profile_table
from titan_limb.fitting.laws import quadratic


def test_fit_profile_table_handles_success_and_short_profile() -> None:
    emission = np.linspace(5, 80, 12)
    mu = np.cos(np.deg2rad(emission))
    brightness = quadratic(mu, 2.0, 0.3, -0.1)
    profiles = pl.DataFrame(
        {
            "cube_id": ["C1", "C1"],
            "band": [1, 2],
            "wavelength_um": [0.4, 0.5],
            "channel": ["visible", "visible"],
            "hemisphere": ["north", "north"],
            "slant_angle_degrees": [60, 60],
            "emission_angles": [emission.tolist(), [10.0, 20.0]],
            "brightness": [brightness.tolist(), [1.0, 0.9]],
        }
    )

    result = fit_profile_table(profiles)

    assert result.get_column("status").to_list() == ["succeeded", "failed"]
    assert result.get_column("failure_reason").to_list() == [
        None,
        "too_few_profile_points",
    ]
    assert result.get_column("u1")[0] == pytest.approx(0.3, abs=1e-4)
