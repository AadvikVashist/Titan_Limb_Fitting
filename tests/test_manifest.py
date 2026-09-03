"""Tests for deterministic data inventory and validation."""

from pathlib import Path

import pytest

from titan_limb.manifest import (
    ValidationStatus,
    create_manifest,
    read_manifest,
    validate_manifest,
    write_manifest,
)


def test_manifest_is_sorted_and_round_trips(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "z.txt").write_text("last", encoding="utf-8")
    (data_dir / "a.txt").write_text("first", encoding="utf-8")
    output = tmp_path / "manifest.json"

    manifest = create_manifest(data_dir)
    write_manifest(manifest, output)

    assert [entry.path for entry in manifest.entries] == ["a.txt", "z.txt"]
    assert read_manifest(output) == manifest


def test_validation_reports_changed_missing_and_unexpected_files(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    changed = data_dir / "changed.txt"
    missing = data_dir / "missing.txt"
    changed.write_text("before", encoding="utf-8")
    missing.write_text("present", encoding="utf-8")
    manifest = create_manifest(data_dir)

    changed.write_text("after", encoding="utf-8")
    missing.unlink()
    (data_dir / "new.txt").write_text("new", encoding="utf-8")
    result = validate_manifest(data_dir, manifest)

    assert result.status is ValidationStatus.INVALID
    assert result.changed_paths == ("changed.txt",)
    assert result.missing_paths == ("missing.txt",)
    assert result.unexpected_paths == ("new.txt",)


def test_validation_accepts_unchanged_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "cube.cub").write_bytes(b"cassini")
    manifest = create_manifest(data_dir)

    result = validate_manifest(data_dir, manifest)

    assert result.status is ValidationStatus.VALID


def test_manifest_rejects_a_file_path(tmp_path: Path) -> None:
    source = tmp_path / "not-a-directory"
    source.write_text("data", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        create_manifest(source)
