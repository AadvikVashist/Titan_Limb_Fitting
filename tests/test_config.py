"""Tests for stable project configuration."""

from pathlib import Path

import pytest

from titan_limb.config import ARTIFACT_DIR_ENV, DATA_DIR_ENV, load_settings


def test_load_settings_resolves_paths_from_config_location(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "test.toml"
    config_path.write_text(
        'data_dir = "science-data"\nartifact_dir = "build"\n', encoding="utf-8"
    )

    settings = load_settings(config_path)

    assert settings.data_dir == tmp_path / "science-data"
    assert settings.artifact_dir == tmp_path / "build"


def test_environment_paths_override_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "test.toml"
    config_path.write_text(
        'data_dir = "file-data"\nartifact_dir = "file-build"\n', encoding="utf-8"
    )
    environment_data = tmp_path / "environment-data"
    environment_artifacts = tmp_path / "environment-artifacts"
    monkeypatch.setenv(DATA_DIR_ENV, str(environment_data))
    monkeypatch.setenv(ARTIFACT_DIR_ENV, str(environment_artifacts))

    settings = load_settings(config_path)

    assert settings.data_dir == environment_data
    assert settings.artifact_dir == environment_artifacts
