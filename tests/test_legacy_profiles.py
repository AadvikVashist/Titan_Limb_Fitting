"""Tests for saved profile conversion."""

import pickle
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from titan_limb.io.legacy_profiles import read_profile_pickle, write_profile_directory


def write_profile(path: Path) -> None:
    profile = {
        "pixel_indices": np.array([[1, 2], [3, 4]]),
        "pixel_distances": np.array([1.0, 2.0]),
        "emission_angles": np.array([10.0, 20.0]),
        "brightness_values": np.array([0.1, 0.2]),
        "meta": {
            "actual_angle": 45.5,
            "processing": {"sorted": True, "filtered": True},
        },
    }
    with path.open("wb") as output:
        pickle.dump({"meta": {}, "0.35054µm_1": {0: profile}}, output)


def test_read_profile_pickle(tmp_path: Path) -> None:
    source = tmp_path / "C_TEST.pkl"
    write_profile(source)

    table = read_profile_pickle(source)

    assert table.shape == (1, 13)
    assert table.row(0, named=True)["pixel_rows"] == [1, 3]
    assert table.row(0, named=True)["channel"] == "visible"


def test_write_profile_directory(tmp_path: Path) -> None:
    source_dir = tmp_path / "profiles"
    source_dir.mkdir()
    write_profile(source_dir / "C_TEST.pkl")
    output = tmp_path / "profiles.parquet"

    report = write_profile_directory(source_dir, output)

    assert report.files == 1
    assert report.rows == 1
    assert report.points == 2
    assert pl.read_parquet(output).height == 1


def test_profile_reader_rejects_bad_arrays(tmp_path: Path) -> None:
    source = tmp_path / "C_TEST.pkl"
    write_profile(source)
    with source.open("rb") as input_file:
        data = pickle.load(input_file)
    data["0.35054µm_1"][0]["emission_angles"] = np.array([20.0, 10.0])
    with source.open("wb") as output:
        pickle.dump(data, output)

    with pytest.raises(ValueError, match="not sorted"):
        read_profile_pickle(source)
