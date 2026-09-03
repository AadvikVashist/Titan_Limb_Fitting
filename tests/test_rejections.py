"""Tests for typed rejection ledgers."""

from pathlib import Path

from titan_limb.rejections import RejectionKind, RejectionLedger


def test_rejection_ledger_keeps_reason_and_identity(tmp_path: Path) -> None:
    ledger = RejectionLedger().with_rejection(
        RejectionKind.PROFILE,
        "C1:15:north",
        "profile_filter",
        "too_few_points",
        "observed=3 minimum=6",
    )
    output = tmp_path / "rejections.json"

    ledger.write(output)
    saved = RejectionLedger.model_validate_json(output.read_text(encoding="utf-8"))

    assert saved == ledger
    assert saved.records[0].reason == "too_few_points"
