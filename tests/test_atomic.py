"""Tests for crash-safe artifact writers."""

from pathlib import Path

import polars as pl

from titan_limb.io.atomic import (
    atomic_write_csv,
    atomic_write_parquet,
    atomic_write_text,
)


def test_atomic_writers_replace_complete_files(tmp_path: Path) -> None:
    text_path = tmp_path / "result.json"
    parquet_path = tmp_path / "result.parquet"
    csv_path = tmp_path / "result.csv"
    frame = pl.DataFrame({"value": [1, 2]})

    atomic_write_text(text_path, "complete\n")
    atomic_write_parquet(frame, parquet_path)
    atomic_write_csv(frame, csv_path)

    assert text_path.read_text(encoding="utf-8") == "complete\n"
    assert pl.read_parquet(parquet_path).equals(frame)
    assert pl.read_csv(csv_path).equals(frame)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "result.csv",
        "result.json",
        "result.parquet",
    ]
