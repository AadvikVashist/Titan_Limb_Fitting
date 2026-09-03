"""Tests for raw pipeline resume and report behavior."""

from pathlib import Path

import polars as pl
import pytest

from titan_limb.pipeline import build_raw_dataset
from titan_limb.processing.cube_profiles import SELECTED_PROFILE_SCHEMA


def test_build_raw_dataset_resumes_only_with_matching_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "raw"
    cube_dir = tmp_path / "C1"
    cube_dir.mkdir()
    (cube_dir / "C1_vis.cub").write_bytes(b"visible")
    (cube_dir / "C1_ir.cub").write_bytes(b"infrared")
    profile = pl.DataFrame(schema=SELECTED_PROFILE_SCHEMA)
    fit = pl.DataFrame({"status": ["succeeded"]})
    calls = 0

    def load_pair(_: object) -> tuple[object, object]:
        nonlocal calls
        calls += 1
        return object(), object()

    monkeypatch.setattr("titan_limb.pipeline.load_cube_pair", load_pair)
    monkeypatch.setattr(
        "titan_limb.pipeline.build_selected_profiles", lambda *_: profile
    )
    monkeypatch.setattr(
        "titan_limb.pipeline.filter_profiles_by_emission", lambda frame, _: frame
    )
    monkeypatch.setattr("titan_limb.pipeline.fit_profile_table", lambda _: fit)

    first = build_raw_dataset(tmp_path, ("C1",), output)
    second = build_raw_dataset(tmp_path, ("C1",), output)

    assert first.resumed_cubes == 0
    assert second.resumed_cubes == 1
    assert calls == 1
    assert (output / "receipts" / "C1.json").is_file()
    assert (output / "rejections" / "C1.json").is_file()


def test_build_raw_dataset_recomputes_after_input_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "raw"
    cube_dir = tmp_path / "C1"
    cube_dir.mkdir()
    visible_path = cube_dir / "C1_vis.cub"
    visible_path.write_bytes(b"visible")
    (cube_dir / "C1_ir.cub").write_bytes(b"infrared")
    profile = pl.DataFrame(schema=SELECTED_PROFILE_SCHEMA)
    fit = pl.DataFrame({"status": ["succeeded"]})
    calls = 0

    def load_pair(_: object) -> tuple[object, object]:
        nonlocal calls
        calls += 1
        return object(), object()

    monkeypatch.setattr("titan_limb.pipeline.load_cube_pair", load_pair)
    monkeypatch.setattr(
        "titan_limb.pipeline.build_selected_profiles", lambda *_: profile
    )
    monkeypatch.setattr(
        "titan_limb.pipeline.filter_profiles_by_emission", lambda frame, _: frame
    )
    monkeypatch.setattr("titan_limb.pipeline.fit_profile_table", lambda _: fit)

    build_raw_dataset(tmp_path, ("C1",), output)
    visible_path.write_bytes(b"changed")
    report = build_raw_dataset(tmp_path, ("C1",), output)

    assert report.resumed_cubes == 0
    assert calls == 2
