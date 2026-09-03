"""Typed records for inputs and results that a run does not accept."""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from titan_limb.io.atomic import atomic_write_text

REJECTION_LEDGER_SCHEMA_VERSION = 1


class RejectionKind(StrEnum):
    INPUT = "input"
    CUBE = "cube"
    BAND = "band"
    PROFILE = "profile"
    FIT = "fit"


class RejectionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: RejectionKind
    identifier: str
    stage: str
    reason: str
    detail: str | None = None


class RejectionLedger(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = REJECTION_LEDGER_SCHEMA_VERSION
    records: tuple[RejectionRecord, ...] = ()

    def with_rejection(
        self,
        kind: RejectionKind,
        identifier: str,
        stage: str,
        reason: str,
        detail: str | None = None,
    ) -> "RejectionLedger":
        record = RejectionRecord(
            kind=kind,
            identifier=identifier,
            stage=stage,
            reason=reason,
            detail=detail,
        )
        return self.model_copy(update={"records": (*self.records, record)})

    def write(self, path: Path) -> None:
        atomic_write_text(path, self.model_dump_json(indent=2) + "\n")
