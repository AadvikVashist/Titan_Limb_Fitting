"""Machine-readable checks that can fail a command."""

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from titan_limb.io.atomic import atomic_write_text

VALIDATION_GATE_SCHEMA_VERSION = 1


class GateStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class GateCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    status: GateStatus
    observed: Any
    expected: Any
    reason: str


class ValidationGateReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = VALIDATION_GATE_SCHEMA_VERSION
    generated_at: datetime
    status: GateStatus
    checks: tuple[GateCheck, ...]

    @property
    def passed(self) -> bool:
        return self.status is GateStatus.PASSED


class ValidationGateError(RuntimeError):
    """Raised after a failed gate report has been saved."""


def equal_check(name: str, observed: Any, expected: Any, reason: str) -> GateCheck:
    status = GateStatus.PASSED if observed == expected else GateStatus.FAILED
    return GateCheck(
        name=name,
        status=status,
        observed=observed,
        expected=expected,
        reason=reason,
    )


def maximum_check(
    name: str,
    observed: float | None,
    maximum: float,
    reason: str,
) -> GateCheck:
    status = (
        GateStatus.PASSED
        if observed is not None and observed <= maximum
        else GateStatus.FAILED
    )
    return GateCheck(
        name=name,
        status=status,
        observed=observed,
        expected={"maximum": maximum},
        reason=reason,
    )


def minimum_check(name: str, observed: float, minimum: float, reason: str) -> GateCheck:
    status = GateStatus.PASSED if observed >= minimum else GateStatus.FAILED
    return GateCheck(
        name=name,
        status=status,
        observed=observed,
        expected={"minimum": minimum},
        reason=reason,
    )


def write_gate_report(
    checks: tuple[GateCheck, ...],
    output: Path,
) -> ValidationGateReport:
    status = (
        GateStatus.PASSED
        if all(check.status is GateStatus.PASSED for check in checks)
        else GateStatus.FAILED
    )
    report = ValidationGateReport(
        generated_at=datetime.now(UTC),
        status=status,
        checks=checks,
    )
    atomic_write_text(output, report.model_dump_json(indent=2) + "\n")
    return report


def require_gate(report: ValidationGateReport) -> None:
    if not report.passed:
        failed = ", ".join(
            check.name for check in report.checks if check.status is GateStatus.FAILED
        )
        raise ValidationGateError(f"validation gate failed: {failed}")
