"""Tests for the PyVIMS input boundary."""

from pathlib import Path

import pytest

from titan_limb.io.vims import find_cube_pair


def test_find_cube_pair(tmp_path: Path) -> None:
    cube_id = "C_TEST"
    directory = tmp_path / cube_id
    directory.mkdir()
    visible = directory / f"{cube_id}_vis.cub"
    infrared = directory / f"{cube_id}_ir.cub"
    visible.touch()
    infrared.touch()

    pair = find_cube_pair(tmp_path, cube_id)

    assert pair.visible == visible
    assert pair.infrared == infrared


def test_find_cube_pair_lists_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"C_TEST_vis\.cub, C_TEST_ir\.cub"):
        find_cube_pair(tmp_path, "C_TEST")
