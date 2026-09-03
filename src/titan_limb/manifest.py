"""Deterministic data manifests for preservation and integrity checks."""

import hashlib
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from titan_limb.io.atomic import atomic_write_text

HASH_CHUNK_BYTES = 8 * 1024 * 1024
MANIFEST_SCHEMA_VERSION = 1


class ValidationStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    size_bytes: int
    modified_ns: int
    sha256: str


class DataManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = MANIFEST_SCHEMA_VERSION
    root: str
    generated_at: datetime
    entries: tuple[ManifestEntry, ...]


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ValidationStatus
    checked_files: int
    missing_paths: tuple[str, ...]
    changed_paths: tuple[str, ...]
    unexpected_paths: tuple[str, ...]


def iter_files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(root: Path) -> DataManifest:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    entries = tuple(
        ManifestEntry(
            path=path.relative_to(root).as_posix(),
            size_bytes=path.stat().st_size,
            modified_ns=path.stat().st_mtime_ns,
            sha256=sha256_file(path),
        )
        for path in iter_files(root)
    )
    return DataManifest(
        root=root.name,
        generated_at=datetime.now(UTC),
        entries=entries,
    )


def write_manifest(manifest: DataManifest, output: Path) -> None:
    payload = manifest.model_dump_json(indent=2)
    atomic_write_text(output, payload + "\n")


def read_manifest(path: Path) -> DataManifest:
    return DataManifest.model_validate_json(path.read_text(encoding="utf-8"))


def validate_manifest(root: Path, manifest: DataManifest) -> ValidationResult:
    root = root.resolve()
    expected = {entry.path: entry for entry in manifest.entries}
    actual_paths = {
        path.relative_to(root).as_posix(): path for path in iter_files(root)
    }
    missing_paths = tuple(sorted(expected.keys() - actual_paths.keys()))
    unexpected_paths = tuple(sorted(actual_paths.keys() - expected.keys()))
    changed_paths = tuple(
        relative_path
        for relative_path in sorted(expected.keys() & actual_paths.keys())
        if actual_paths[relative_path].stat().st_size
        != expected[relative_path].size_bytes
        or sha256_file(actual_paths[relative_path]) != expected[relative_path].sha256
    )
    status = (
        ValidationStatus.VALID
        if not missing_paths and not unexpected_paths and not changed_paths
        else ValidationStatus.INVALID
    )
    return ValidationResult(
        status=status,
        checked_files=len(expected),
        missing_paths=missing_paths,
        changed_paths=changed_paths,
        unexpected_paths=unexpected_paths,
    )
