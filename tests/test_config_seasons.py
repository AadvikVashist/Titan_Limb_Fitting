"""Tests for season and uncertainty config."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from titan_limb.config_seasons import SeasonPolicy, load_season_policy


def test_default_season_policy() -> None:
    policy = load_season_policy(Path("configs/seasons.toml"))

    assert policy.northern_vernal_equinox == datetime(2009, 8, 11, tzinfo=UTC)
    assert policy.northern_summer_solstice == datetime(2017, 5, 24, tzinfo=UTC)
    assert policy.bootstrap_resamples == 5000


def test_season_policy_rejects_reversed_dates() -> None:
    with pytest.raises(ValueError, match="equinox must occur before solstice"):
        SeasonPolicy(
            northern_vernal_equinox=datetime(2018, 1, 1, tzinfo=UTC),
            northern_summer_solstice=datetime(2017, 1, 1, tzinfo=UTC),
        )
