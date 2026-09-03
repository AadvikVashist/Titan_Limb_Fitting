"""Tests for stable project configuration."""

from pathlib import Path

import pytest

from titan_limb.config import (
    ARTIFACT_DIR_ENV,
    CONFIG_DIR_ENV,
    DATA_DIR_ENV,
    PACKAGE_CONFIG_DIR,
    load_settings,
)


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
    assert settings.config_dir == config_dir


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


def test_packaged_defaults_resolve_from_invocation_directory(tmp_path: Path) -> None:
    settings = load_settings(working_dir=tmp_path)

    assert settings.project_dir == tmp_path
    assert settings.data_dir == tmp_path / "data"
    assert settings.artifact_dir == tmp_path / "artifacts"
    assert settings.config_dir == PACKAGE_CONFIG_DIR.resolve()
    assert settings.config_path(None, "bands.toml").is_file()


def test_config_environment_and_explicit_path_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "policies"
    config_dir.mkdir()
    explicit = tmp_path / "custom.toml"
    explicit.write_text("value = true\n", encoding="utf-8")
    monkeypatch.setenv(CONFIG_DIR_ENV, str(config_dir))

    settings = load_settings(working_dir=tmp_path)

    assert settings.config_path(None, "bands.toml") == config_dir / "bands.toml"
    assert settings.config_path(explicit, "bands.toml") == explicit
