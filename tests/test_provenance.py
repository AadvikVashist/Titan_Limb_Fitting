"""Tests for typed run receipts and resume checks."""

from pathlib import Path

import pytest

from titan_limb.provenance import (
    RunDefinition,
    RunReceipt,
    RunRecorder,
    RunStatus,
    receipt_allows_resume,
)


def recorder(tmp_path: Path) -> RunRecorder:
    source = tmp_path / "input.txt"
    source.write_text("source", encoding="utf-8")
    return RunRecorder(
        RunDefinition(
            command="test.run",
            receipt_path=tmp_path / "receipt.json",
            project_dir=tmp_path,
            settings={"artifact_dir": str(tmp_path)},
            parameters={"cutoff": 25.0},
            inputs=(source,),
            outputs=(tmp_path / "output.txt",),
            output_schema_versions={"test": 1},
        )
    )


def test_successful_receipt_allows_only_exact_resume(tmp_path: Path) -> None:
    run = recorder(tmp_path)
    with run:
        (tmp_path / "output.txt").write_text("result", encoding="utf-8")

    receipt = RunReceipt.model_validate_json(
        (tmp_path / "receipt.json").read_text(encoding="utf-8")
    )
    assert receipt.status is RunStatus.SUCCEEDED
    assert receipt.outputs[0].sha256 is not None
    assert receipt_allows_resume(
        tmp_path / "receipt.json",
        run.input_fingerprint,
        (tmp_path / "output.txt",),
    )

    (tmp_path / "output.txt").write_text("changed", encoding="utf-8")
    assert not receipt_allows_resume(
        tmp_path / "receipt.json",
        run.input_fingerprint,
        (tmp_path / "output.txt",),
    )


def test_failed_run_is_saved_and_exception_is_not_hidden(tmp_path: Path) -> None:
    run = recorder(tmp_path)
    with pytest.raises(ValueError, match="broken"), run:
        raise ValueError("broken")

    receipt = RunReceipt.model_validate_json(
        (tmp_path / "receipt.json").read_text(encoding="utf-8")
    )
    assert receipt.status is RunStatus.FAILED
    assert receipt.error == "ValueError: broken"
