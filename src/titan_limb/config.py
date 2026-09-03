"""Typed project configuration with file and environment overrides."""

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict

DEFAULT_CONFIG_PATH = Path("configs/default.toml")
DATA_DIR_ENV = "TITAN_DATA_DIR"
ARTIFACT_DIR_ENV = "TITAN_ARTIFACT_DIR"


class ProjectSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")

    def resolved(self, base_dir: Path) -> "ProjectSettings":
        data_dir = self.data_dir
        artifact_dir = self.artifact_dir
        if not data_dir.is_absolute():
            data_dir = base_dir / data_dir
        if not artifact_dir.is_absolute():
            artifact_dir = base_dir / artifact_dir
        return self.model_copy(
            update={
                "data_dir": data_dir.resolve(),
                "artifact_dir": artifact_dir.resolve(),
            }
        )


def load_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> ProjectSettings:
    config_path = config_path.resolve()
    values = tomllib.loads(config_path.read_text(encoding="utf-8"))
    file_settings = ProjectSettings.model_validate(values)
    data_dir = Path(os.environ.get(DATA_DIR_ENV, file_settings.data_dir))
    artifact_dir = Path(os.environ.get(ARTIFACT_DIR_ENV, file_settings.artifact_dir))
    return ProjectSettings(data_dir=data_dir, artifact_dir=artifact_dir).resolved(
        config_path.parent.parent
    )
