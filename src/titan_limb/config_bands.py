"""Typed one-based band policy."""

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from titan_limb.config import PACKAGE_CONFIG_DIR

DEFAULT_BAND_CONFIG = PACKAGE_CONFIG_DIR / "bands.toml"
FIRST_BAND = 1
LAST_BAND = 352


class BandPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    excluded_ranges: tuple[tuple[int, int], ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_ranges(self) -> "BandPolicy":
        for start, end in self.excluded_ranges:
            if start < FIRST_BAND or end > LAST_BAND or start > end:
                raise ValueError(f"invalid one-based band range: {start}-{end}")
        return self

    @property
    def excluded_bands(self) -> frozenset[int]:
        return frozenset(
            band
            for start, end in self.excluded_ranges
            for band in range(start, end + 1)
        )


def load_band_policy(path: Path = DEFAULT_BAND_CONFIG) -> BandPolicy:
    values = tomllib.loads(path.read_text(encoding="utf-8"))
    return BandPolicy.model_validate(values)
