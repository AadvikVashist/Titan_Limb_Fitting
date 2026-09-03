"""Reference tests for the legacy limb-law equations."""

import numpy as np
from numpy.testing import assert_allclose

from titan_limb.fitting.laws import (
    emission_angle_to_mu,
    linear,
    quadratic,
    square_root,
)


def test_emission_angle_to_mu_reference_values() -> None:
    angles = np.array([0.0, 60.0, 90.0])

    assert_allclose(emission_angle_to_mu(angles), [1.0, 0.5, 0.0], atol=1e-15)


def test_laws_equal_center_intensity_at_mu_one() -> None:
    mu = np.array([1.0])

    assert_allclose(linear(mu, 2.0, 0.4), [2.0])
    assert_allclose(quadratic(mu, 2.0, 0.4, 0.2), [2.0])
    assert_allclose(square_root(mu, 2.0, 0.4, 0.2), [2.0])


def test_quadratic_limb_value_depends_on_u1_plus_u2() -> None:
    mu = np.array([0.0])

    assert_allclose(quadratic(mu, 2.0, 0.4, 0.2), [0.8])


def test_reference_values_match_legacy_equations() -> None:
    mu = np.array([0.25, 0.5, 0.75])

    assert_allclose(linear(mu, 1.2, -0.3), [1.47, 1.38, 1.29])
    assert_allclose(quadratic(mu, 1.2, 0.2, -0.1), [1.0875, 1.11, 1.1475])
    assert_allclose(
        square_root(mu, 1.2, 0.2, -0.1),
        [1.08, 1.1151471862576143, 1.1560769515458673],
    )
