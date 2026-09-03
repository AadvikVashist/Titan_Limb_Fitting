"""Atomic writers for project artifacts."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import polars as pl


def atomic_write_text(path: Path, text: str) -> None:
    """Replace a text file only after its full content has been written."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}."
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    """Write indented JSON through an atomic replacement."""
    atomic_write_text(path, json.dumps(value, indent=2) + "\n")


def atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    """Replace a Parquet file only after Polars completes the write."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=path.suffix,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(frame: pl.DataFrame, path: Path) -> None:
    """Replace a CSV file only after Polars completes the write."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=path.suffix,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.write_csv(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
