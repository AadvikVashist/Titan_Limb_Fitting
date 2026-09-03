"""Tests for one-based band policy."""

from pathlib import Path

import pytest

from titan_limb.config_bands import BandPolicy, load_band_policy


def test_default_band_policy_matches_legacy_list() -> None:
    policy = load_band_policy(Path("configs/bands.toml"))

    assert len(policy.excluded_bands) == 211
    assert 55 in policy.excluded_bands
    assert 352 in policy.excluded_bands
    assert 54 not in policy.excluded_bands


def test_band_policy_rejects_bad_ranges() -> None:
    with pytest.raises(ValueError, match="invalid one-based"):
        BandPolicy(excluded_ranges=((20, 10),))
