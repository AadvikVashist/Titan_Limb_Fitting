"""Typed metadata for one selected Cassini/VIMS observation."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ObservationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cube_id: str
    source_name: str
    selection_label: str
    target: str
    mid_time: datetime
    decimal_year: float
    samples: int
    lines: int
    visible_sampling: str | None
    infrared_sampling: str | None
    exposure: str | None
    observation_sequence: str | None
    sequence: str | None
    revolution: str | None
    orbit: str | None
    mission: str | None
    flyby: str | None
    distance_km: float | None
    mean_resolution: str | None
    sub_spacecraft_point: str | None
    sub_solar_point: str | None
    incidence_range: str | None
    emergence_range: str | None
    phase_degrees: float | None
    limb_visible: bool | None
