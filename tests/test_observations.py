"""Tests for selected observation metadata."""

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from titan_limb.io.observations import (
    attach_observation_metadata,
    decimal_year,
    observations_to_frame,
    read_selected_observations,
)

HEADERS = [
    "Name",
    "Target",
    "Image mid-time",
    "Samples Lines",
    "Sampling Mode (VIS | IR)",
    "Exposure (VIS | IR)",
    "Observation Sequence",
    "Sequence",
    "Revolution",
    "Orbit",
    "Mission",
    "Flyby",
    "Distance",
    "Mean resolution",
    "Sub-Spacecraft point",
    "Sub-Solar point",
    "Incidence (min | max)",
    "Emergence (min | max)",
    "Phase",
    "Limb visible",
]


def write_sources(tmp_path: Path) -> tuple[Path, Path]:
    csv_path = tmp_path / "nantes.csv"
    csv_path.write_text(
        ",".join(HEADERS)
        + "\n"
        + ",".join(
            [
                "123_1",
                "Titan",
                "01/07/2012 at 00:00:00",
                "10x20",
                "NORMAL | HI-RES",
                "80 ms | 40 ms",
                "OBS",
                "S1",
                "1",
                "Orbit",
                "Prime",
                "T1",
                '"1,234 km"',
                "10 km",
                "1 N | 2 E",
                "3 N | 4 E",
                "5 | 6",
                "7 | 8",
                "9",
                "Yes",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    selection_path = tmp_path / "selected.json"
    selection_path.write_text(json.dumps({"selected": {"T1": "123_1"}}))
    return csv_path, selection_path


def test_read_selected_observation(tmp_path: Path) -> None:
    csv_path, selection_path = write_sources(tmp_path)

    record = read_selected_observations(csv_path, selection_path)[0]

    assert record.cube_id == "C123_1"
    assert record.selection_label == "T1"
    assert record.samples == 10
    assert record.lines == 20
    assert record.distance_km == 1234
    assert record.phase_degrees == 9
    assert record.limb_visible is True


def test_decimal_year_handles_leap_year() -> None:
    value = datetime(2020, 7, 2, tzinfo=UTC)
    assert decimal_year(value) == pytest.approx(2020.5)


def test_selected_observation_requires_matching_row(tmp_path: Path) -> None:
    csv_path, selection_path = write_sources(tmp_path)
    selection_path.write_text(json.dumps({"selected": {"T2": "missing"}}))

    with pytest.raises(ValueError, match="missing"):
        read_selected_observations(csv_path, selection_path)


def test_attach_observation_metadata(tmp_path: Path) -> None:
    csv_path, selection_path = write_sources(tmp_path)
    observations = observations_to_frame(
        read_selected_observations(csv_path, selection_path)
    )
    result = attach_observation_metadata(
        pl.DataFrame({"cube_id": ["C123_1"], "value": [2.0]}), observations
    )

    assert result.get_column("selection_label").to_list() == ["T1"]
    assert result.get_column("decimal_year")[0] == pytest.approx(2012.4973)


def test_attach_observation_metadata_requires_matching_cube(tmp_path: Path) -> None:
    csv_path, selection_path = write_sources(tmp_path)
    observations = observations_to_frame(
        read_selected_observations(csv_path, selection_path)
    )

    with pytest.raises(ValueError, match="missing observation metadata"):
        attach_observation_metadata(pl.DataFrame({"cube_id": ["C999_1"]}), observations)
