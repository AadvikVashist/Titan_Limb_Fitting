"""Tests for saved validation gates."""

from pathlib import Path

import pytest

from titan_limb.validation.gates import (
    GateStatus,
    ValidationGateError,
    equal_check,
    maximum_check,
    require_gate,
    write_gate_report,
)


def test_failed_gate_is_saved_before_it_raises(tmp_path: Path) -> None:
    output = tmp_path / "gate.json"
    report = write_gate_report(
        (
            equal_check("rows", 3, 3, "row sets must match"),
            maximum_check("drift", 0.2, 0.1, "drift must stay in bounds"),
        ),
        output,
    )

    assert report.status is GateStatus.FAILED
    assert output.is_file()
    with pytest.raises(ValidationGateError, match="drift"):
        require_gate(report)


def test_missing_measurement_fails_maximum_check() -> None:
    check = maximum_check("drift", None, 0.1, "drift must be measured")

    assert check.status is GateStatus.FAILED
    assert check.observed is None
