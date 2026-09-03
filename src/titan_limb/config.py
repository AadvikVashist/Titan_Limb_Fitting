"""Typed project paths with file, environment, and CLI overrides."""

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict

PACKAGE_CONFIG_DIR = Path(__file__).with_name("resources")
DEFAULT_CONFIG_PATH = PACKAGE_CONFIG_DIR / "default.toml"
DATA_DIR_ENV = "TITAN_DATA_DIR"
ARTIFACT_DIR_ENV = "TITAN_ARTIFACT_DIR"
CONFIG_DIR_ENV = "TITAN_CONFIG_DIR"


class FileSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")


class ProjectSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_dir: Path
    data_dir: Path
    artifact_dir: Path
    config_dir: Path
    source_config: Path

    def data_path(self, explicit: Path | None, *parts: str) -> Path:
        return _resolved_override(explicit, self.data_dir.joinpath(*parts))

    def artifact_path(self, explicit: Path | None, *parts: str) -> Path:
        return _resolved_override(explicit, self.artifact_dir.joinpath(*parts))

    def config_path(self, explicit: Path | None, name: str) -> Path:
        return _resolved_override(explicit, self.config_dir / name)

    def project_path(self, explicit: Path | None, *parts: str) -> Path:
        return _resolved_override(explicit, self.project_dir.joinpath(*parts))

    def receipt_settings(self) -> dict[str, str]:
        return {
            "project_dir": str(self.project_dir),
            "data_dir": str(self.data_dir),
            "artifact_dir": str(self.artifact_dir),
            "config_dir": str(self.config_dir),
            "source_config": str(self.source_config),
        }


def _resolved_override(explicit: Path | None, default: Path) -> Path:
    return (explicit if explicit is not None else default).resolve()


def _resolved_from(value: Path, base_dir: Path) -> Path:
    return (value if value.is_absolute() else base_dir / value).resolve()


def load_settings(
    config_path: Path | None = None,
    *,
    working_dir: Path | None = None,
) -> ProjectSettings:
    """Load one path set for a CLI invocation.

    Packaged defaults resolve data and artifacts from the current directory.
    An explicit file inside a ``configs`` directory keeps the prior project-root
    behavior. Environment paths override file values.
    """
    invocation_dir = (working_dir or Path.cwd()).resolve()
    explicit_config = config_path is not None
    source_config = (config_path or DEFAULT_CONFIG_PATH).resolve()
    values = tomllib.loads(source_config.read_text(encoding="utf-8"))
    file_settings = FileSettings.model_validate(values)
    project_dir = (
        source_config.parent.parent
        if explicit_config and source_config.parent.name == "configs"
        else source_config.parent
        if explicit_config
        else invocation_dir
    )
    data_environment = os.environ.get(DATA_DIR_ENV)
    artifact_environment = os.environ.get(ARTIFACT_DIR_ENV)
    config_environment = os.environ.get(CONFIG_DIR_ENV)
    data_dir = _resolved_from(
        Path(data_environment) if data_environment else file_settings.data_dir,
        invocation_dir if data_environment else project_dir,
    )
    artifact_dir = _resolved_from(
        Path(artifact_environment)
        if artifact_environment
        else file_settings.artifact_dir,
        invocation_dir if artifact_environment else project_dir,
    )
    config_dir = _resolved_from(
        Path(config_environment) if config_environment else source_config.parent,
        invocation_dir,
    )
    return ProjectSettings(
        project_dir=project_dir,
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        config_dir=config_dir,
        source_config=source_config,
    )
