"""Tests for structural and declared raw-result gates."""

from pathlib import Path

from titan_limb.validation.gates import GateStatus
from titan_limb.validation.raw import RawValidationSummary, write_raw_validation_gate


def summary(*, rows: int = 4) -> RawValidationSummary:
    return RawValidationSummary(
        profile_input_rows=4,
        fit_input_rows=4,
        legacy_profile_rows=4,
        legacy_fit_rows=4,
        rows=rows,
        exact_profiles=3,
        changed_profiles=1,
        equal_point_counts=4,
        equal_fit_status=4,
        both_succeeded=4,
        maximum_absolute_u1_drift=0.02,
        maximum_absolute_u2_drift=0.03,
        median_absolute_u1_drift=0.01,
        median_absolute_u2_drift=0.01,
    )


def test_raw_gate_passes_complete_rows_and_declared_bounds(tmp_path: Path) -> None:
    report = write_raw_validation_gate(
        summary(),
        tmp_path / "gate.json",
        maximum_changed_profiles=1,
        maximum_u1_drift=0.02,
        maximum_u2_drift=0.03,
    )

    assert report.status is GateStatus.PASSED


def test_raw_gate_fails_missing_rows_and_excess_drift(tmp_path: Path) -> None:
    report = write_raw_validation_gate(
        summary(rows=3),
        tmp_path / "gate.json",
        maximum_u1_drift=0.01,
    )

    assert report.status is GateStatus.FAILED
    failed = {
        check.name for check in report.checks if check.status is GateStatus.FAILED
    }
    assert "all_profile_inputs_joined" in failed
    assert "maximum_absolute_u1_drift" in failed
