"""Tests for the selected-cube boundary."""

from pathlib import Path

from titan_limb.io.selection import read_selected_cube_ids


def test_read_selected_cube_ids_normalizes_prefix(tmp_path: Path) -> None:
    source = tmp_path / "selection.json"
    source.write_text('{"selected": {"a": "2_1", "b": "C1_1"}}')

    assert read_selected_cube_ids(source) == ("C1_1", "C2_1")
