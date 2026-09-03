"""Read the selected observation set from the preserved Nantes table."""

import csv
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import polars as pl

from titan_limb.io.atomic import atomic_write_parquet
from titan_limb.models.observation import ObservationRecord

TIME_FORMAT = "%d/%m/%Y at %H:%M:%S"
SAMPLING_CHANNEL_COUNT = 2
OBSERVATION_JOIN_COLUMNS = [
    "cube_id",
    "selection_label",
    "mid_time",
    "decimal_year",
    "flyby",
]


def decimal_year(value: datetime) -> float:
    start = datetime(value.year, 1, 1, tzinfo=UTC)
    end = datetime(value.year + 1, 1, 1, tzinfo=UTC)
    return value.year + (value - start).total_seconds() / (end - start).total_seconds()


def _optional_text(value: str) -> str | None:
    stripped = value.strip()
    return None if not stripped or stripped.lower() == "n/a" else stripped


def _optional_float(value: str) -> float | None:
    text = _optional_text(value)
    return None if text is None else float(text)


def _distance_km(value: str) -> float | None:
    text = _optional_text(value)
    return None if text is None else float(text.removesuffix(" km").replace(",", ""))


def _limb_visible(value: str) -> bool | None:
    text = _optional_text(value)
    return None if text is None else text.lower() == "yes"


def _sampling(value: str) -> tuple[str | None, str | None]:
    parts = [part.strip() for part in value.split("|")]
    if len(parts) != SAMPLING_CHANNEL_COUNT:
        raise ValueError(f"invalid VIS/IR sampling value: {value}")
    return _optional_text(parts[0]), _optional_text(parts[1])


def read_selected_observations(
    nantes_csv: Path, selection_json: Path
) -> tuple[ObservationRecord, ...]:
    selection_data = cast(
        Mapping[str, Mapping[str, str]], json.loads(selection_json.read_text())
    )
    selected = selection_data["selected"]
    label_by_name = {cube_name: label for label, cube_name in selected.items()}
    with nantes_csv.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    matching = [row for row in rows if row["Name"] in label_by_name]
    found = {row["Name"] for row in matching}
    missing = set(label_by_name) - found
    if missing:
        raise ValueError(
            f"selected cubes missing from Nantes table: {', '.join(sorted(missing))}"
        )
    records: list[ObservationRecord] = []
    for row in matching:
        source_name = row["Name"]
        mid_time = datetime.strptime(row["Image mid-time"], TIME_FORMAT).replace(
            tzinfo=UTC
        )
        samples, lines = (int(value) for value in row["Samples Lines"].split("x"))
        visible_sampling, infrared_sampling = _sampling(row["Sampling Mode (VIS | IR)"])
        records.append(
            ObservationRecord(
                cube_id=f"C{source_name}",
                source_name=source_name,
                selection_label=label_by_name[source_name],
                target=row["Target"],
                mid_time=mid_time,
                decimal_year=decimal_year(mid_time),
                samples=samples,
                lines=lines,
                visible_sampling=visible_sampling,
                infrared_sampling=infrared_sampling,
                exposure=_optional_text(row["Exposure (VIS | IR)"]),
                observation_sequence=_optional_text(row["Observation Sequence"]),
                sequence=_optional_text(row["Sequence"]),
                revolution=_optional_text(row["Revolution"]),
                orbit=_optional_text(row["Orbit"]),
                mission=_optional_text(row["Mission"]),
                flyby=_optional_text(row["Flyby"]),
                distance_km=_distance_km(row["Distance"]),
                mean_resolution=_optional_text(row["Mean resolution"]),
                sub_spacecraft_point=_optional_text(row["Sub-Spacecraft point"]),
                sub_solar_point=_optional_text(row["Sub-Solar point"]),
                incidence_range=_optional_text(row["Incidence (min | max)"]),
                emergence_range=_optional_text(row["Emergence (min | max)"]),
                phase_degrees=_optional_float(row["Phase"]),
                limb_visible=_limb_visible(row["Limb visible"]),
            )
        )
    return tuple(sorted(records, key=lambda record: record.mid_time))


def observations_to_frame(records: tuple[ObservationRecord, ...]) -> pl.DataFrame:
    return pl.from_dicts(
        [record.model_dump() for record in records], infer_schema_length=None
    ).sort("mid_time")


def attach_observation_metadata(
    table: pl.DataFrame, observations: pl.DataFrame
) -> pl.DataFrame:
    """Add stable time and selection fields to a cube-level result table."""
    metadata = observations.select(OBSERVATION_JOIN_COLUMNS)
    duplicate_ids = (
        metadata.group_by("cube_id")
        .len()
        .filter(pl.col("len") > 1)
        .get_column("cube_id")
        .to_list()
    )
    if duplicate_ids:
        raise ValueError(
            f"duplicate observation cube IDs: {', '.join(sorted(duplicate_ids))}"
        )
    missing_ids = sorted(
        set(table.get_column("cube_id").unique())
        - set(metadata.get_column("cube_id").unique())
    )
    if missing_ids:
        raise ValueError(
            f"result cubes missing observation metadata: {', '.join(missing_ids)}"
        )
    return table.join(metadata, on="cube_id", how="left", validate="m:1")


def write_observations_parquet(
    records: tuple[ObservationRecord, ...], output: Path
) -> None:
    atomic_write_parquet(observations_to_frame(records), output)
