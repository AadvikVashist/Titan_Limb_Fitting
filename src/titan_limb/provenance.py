"""Small, typed run receipts for traceable project artifacts."""

import hashlib
import json
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Any, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from titan_limb.io.atomic import atomic_write_text
from titan_limb.manifest import sha256_file

RUN_RECEIPT_SCHEMA_VERSION = 1


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class FileReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    exists: bool
    size_bytes: int | None = None
    sha256: str | None = None


class RunReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = RUN_RECEIPT_SCHEMA_VERSION
    run_id: str
    command: str
    started_at: datetime
    finished_at: datetime
    status: RunStatus
    code_revision: str | None
    dependency_lock: FileReference | None
    settings: dict[str, str]
    parameters: dict[str, Any]
    input_fingerprint: str
    inputs: tuple[FileReference, ...]
    outputs: tuple[FileReference, ...]
    output_schema_versions: dict[str, int]
    rejection_ledger: str | None = None
    rejection_count: int = 0
    error: str | None = None


class RunDefinition(BaseModel):
    """Static inputs needed to record one run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    command: str
    receipt_path: Path
    project_dir: Path
    settings: dict[str, str]
    parameters: dict[str, Any]
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]
    output_schema_versions: dict[str, int] = Field(default_factory=dict)
    rejection_ledger: Path | None = None
    rejection_count: int = 0


def file_reference(path: Path, *, hash_content: bool = True) -> FileReference:
    resolved = path.resolve()
    if not resolved.is_file():
        return FileReference(path=str(resolved), exists=resolved.exists())
    return FileReference(
        path=str(resolved),
        exists=True,
        size_bytes=resolved.stat().st_size,
        sha256=sha256_file(resolved) if hash_content else None,
    )


def code_revision(project_dir: Path) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    result = subprocess.run(
        [git, "-C", str(project_dir), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def run_fingerprint(
    inputs: Sequence[FileReference],
    parameters: Mapping[str, Any],
    revision: str | None,
    dependency_lock: FileReference | None,
) -> str:
    payload = {
        "inputs": [reference.model_dump(mode="json") for reference in inputs],
        "parameters": dict(parameters),
        "code_revision": revision,
        "dependency_lock": (
            dependency_lock.model_dump(mode="json") if dependency_lock else None
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class RunRecorder:
    """Write one receipt when an artifact-producing block ends."""

    def __init__(self, definition: RunDefinition) -> None:
        self.command = definition.command
        self.receipt_path = definition.receipt_path
        self.project_dir = definition.project_dir
        self.settings = definition.settings
        self.parameters = definition.parameters
        self.input_references = tuple(
            file_reference(path) for path in definition.inputs
        )
        self.output_paths = definition.outputs
        self.output_schema_versions = definition.output_schema_versions
        self.rejection_ledger = definition.rejection_ledger
        self.rejection_count = definition.rejection_count
        self.started_at = datetime.now(UTC)
        self.revision = code_revision(self.project_dir)
        lock_path = self.project_dir / "uv.lock"
        self.dependency_lock = (
            file_reference(lock_path) if lock_path.is_file() else None
        )
        self.input_fingerprint = run_fingerprint(
            self.input_references,
            self.parameters,
            self.revision,
            self.dependency_lock,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del traceback
        status = RunStatus.SUCCEEDED if exception is None else RunStatus.FAILED
        error = None
        if exception is not None:
            name = exception_type.__name__ if exception_type else "Error"
            error = f"{name}: {exception}"
        receipt = RunReceipt(
            run_id=str(uuid4()),
            command=self.command,
            started_at=self.started_at,
            finished_at=datetime.now(UTC),
            status=status,
            code_revision=self.revision,
            dependency_lock=self.dependency_lock,
            settings=self.settings,
            parameters=self.parameters,
            input_fingerprint=self.input_fingerprint,
            inputs=self.input_references,
            outputs=tuple(file_reference(path) for path in self.output_paths),
            output_schema_versions=self.output_schema_versions,
            rejection_ledger=(
                str(self.rejection_ledger.resolve()) if self.rejection_ledger else None
            ),
            rejection_count=self.rejection_count,
            error=error,
        )
        atomic_write_text(self.receipt_path, receipt.model_dump_json(indent=2) + "\n")
        return False


def receipt_allows_resume(
    receipt_path: Path,
    expected_fingerprint: str,
    outputs: Sequence[Path],
) -> bool:
    if not receipt_path.is_file():
        return False
    receipt = RunReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
    if receipt.status is not RunStatus.SUCCEEDED:
        return False
    if receipt.input_fingerprint != expected_fingerprint:
        return False
    expected_outputs = tuple(file_reference(path) for path in outputs)
    return receipt.outputs == expected_outputs
