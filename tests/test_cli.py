"""Tests for the public command-line interface."""

from pathlib import Path

from typer.testing import CliRunner

from titan_limb.cli import app

runner = CliRunner()


def test_status_prints_resolved_paths(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "test.toml"
    config_path.write_text(
        'data_dir = "data"\nartifact_dir = "artifacts"\n', encoding="utf-8"
    )

    result = runner.invoke(app, ["status", "--config", str(config_path)])

    assert result.exit_code == 0
    assert f"data_dir={tmp_path / 'data'}" in result.stdout
    assert f"artifact_dir={tmp_path / 'artifacts'}" in result.stdout


def test_manifest_and_validation_commands(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "cube.cub").write_bytes(b"cassini")
    manifest_path = tmp_path / "manifest.json"

    manifest_result = runner.invoke(
        app,
        [
            "data",
            "manifest",
            "--data-dir",
            str(data_dir),
            "--output",
            str(manifest_path),
        ],
    )
    validation_result = runner.invoke(
        app,
        [
            "data",
            "validate",
            "--data-dir",
            str(data_dir),
            "--manifest",
            str(manifest_path),
        ],
    )

    assert manifest_result.exit_code == 0
    assert "files=1" in manifest_result.stdout
    assert validation_result.exit_code == 0
    assert '"status": "valid"' in validation_result.stdout


def test_validation_command_fails_for_changed_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source = data_dir / "cube.cub"
    source.write_bytes(b"before")
    manifest_path = tmp_path / "manifest.json"
    runner.invoke(
        app,
        [
            "data",
            "manifest",
            "--data-dir",
            str(data_dir),
            "--output",
            str(manifest_path),
        ],
    )
    source.write_bytes(b"after")

    result = runner.invoke(
        app,
        [
            "data",
            "validate",
            "--data-dir",
            str(data_dir),
            "--manifest",
            str(manifest_path),
        ],
    )

    assert result.exit_code == 1
    assert '"status": "invalid"' in result.stdout
