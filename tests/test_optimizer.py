"""Tests for the quadratic fit process."""

import numpy as np
import pytest

from titan_limb.fitting.laws import emission_angle_to_mu, quadratic
from titan_limb.fitting.optimizer import (
    fit_quadratic_profile,
    interpolate_profile,
    legacy_moving_average,
)
from titan_limb.models.core import SmoothingMethod


def test_quadratic_fit_recovers_clean_profile() -> None:
    angles = np.linspace(10, 80, 30)
    brightness = quadratic(emission_angle_to_mu(angles), 0.8, 0.2, -0.1)

    result = fit_quadratic_profile(angles, brightness)

    assert result.optimal.method is SmoothingMethod.INTERPOLATED
    assert result.optimal.intensity_center == pytest.approx(0.8)
    assert result.optimal.u1 == pytest.approx(0.2, abs=3e-6)
    assert result.optimal.u2 == pytest.approx(-0.1, abs=3e-6)
    assert result.optimal.r_squared == pytest.approx(1.0)


def test_legacy_moving_average_preserves_saved_window_rule() -> None:
    result = legacy_moving_average(np.array([1.0, 2.0, 3.0, 4.0]), 3)
    np.testing.assert_allclose(result, [1.0, 1.5, 2.5, 3.5])
    with pytest.raises(ValueError, match="positive"):
        legacy_moving_average(np.ones(3), 0)


def test_interpolation_validates_profile() -> None:
    with pytest.raises(ValueError, match="at least three"):
        interpolate_profile(np.ones(2), np.ones(2))
    with pytest.raises(ValueError, match="distinct"):
        interpolate_profile(np.ones(3), np.ones(3))
