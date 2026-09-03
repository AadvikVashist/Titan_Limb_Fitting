"""Typed season boundaries and uncertainty policy."""

import tomllib
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

DEFAULT_SEASON_CONFIG = Path("configs/seasons.toml")
MINIMUM_GROUP_SIZE = 2
MINIMUM_BOOTSTRAP_RESAMPLES = 100


class SeasonPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    northern_vernal_equinox: datetime
    northern_summer_solstice: datetime
    minimum_group_observations: int = 5
    bootstrap_resamples: int = 5000
    confidence_level: float = 0.95
    random_seed: int = 1729

    @model_validator(mode="after")
    def validate_policy(self) -> "SeasonPolicy":
        if self.northern_vernal_equinox.tzinfo is None:
            raise ValueError("season boundaries must include a time zone")
        if self.northern_summer_solstice.tzinfo is None:
            raise ValueError("season boundaries must include a time zone")
        if self.northern_vernal_equinox >= self.northern_summer_solstice:
            raise ValueError("equinox must occur before solstice")
        if self.minimum_group_observations < MINIMUM_GROUP_SIZE:
            raise ValueError("minimum group observations must be at least two")
        if self.bootstrap_resamples < MINIMUM_BOOTSTRAP_RESAMPLES:
            raise ValueError("bootstrap resamples must be at least 100")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence level must be between zero and one")
        return self


def load_season_policy(path: Path = DEFAULT_SEASON_CONFIG) -> SeasonPolicy:
    values = tomllib.loads(path.read_text(encoding="utf-8"))
    return SeasonPolicy.model_validate(values)
