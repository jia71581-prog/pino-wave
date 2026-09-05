from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from grouped_ufno_mionet_v3.training.checkpoint import CHECKPOINT_FORMAT
from grouped_ufno_mionet_v3.training.pilot import validate_pilot_prerequisite


FAMILIES = ("uniform", "layered", "marmousi")


def _write_prerequisite(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "manifest_digest": "manifest",
            "config_digest": "gate-config",
            "global_step": 750,
            "model_state": {},
        },
        checkpoint,
    )
    report = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "manifest_digest": "manifest",
        "config_digest": "gate-config",
        "step": 750,
        "decision": {"passed": True, "failures": []},
        "metrics": {
            "record_count_by_family": {family: 3 for family in FAMILIES},
            "aggregate_query_relative_l2": 0.08,
            "aggregate_dense_relative_l2": 0.07,
            "family_query_relative_l2": {family: 0.08 for family in FAMILIES},
            "family_dense_relative_l2": {family: 0.07 for family in FAMILIES},
            "family_late_relative_l2": {family: 0.09 for family in FAMILIES},
            "missing_gradient_groups": [],
        },
        "diagnostics": {
            "records": [
                {"sample_id": f"{family}_{index}", "medium_type": family}
                for family in FAMILIES
                for index in range(3)
            ]
        },
    }
    report_path = tmp_path / "passed_report.json"
    report_path.write_text(json.dumps(report), encoding="utf8")
    return report_path, checkpoint, report


def test_pilot_prerequisite_binds_passed_report_and_checkpoint(tmp_path: Path):
    report, checkpoint, _ = _write_prerequisite(tmp_path)
    identity = validate_pilot_prerequisite(
        report,
        checkpoint,
        expected_manifest_digest="manifest",
    )
    assert identity.gate_step == 750
    assert identity.gate_config_digest == "gate-config"
    assert identity.parent_checkpoint_sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert identity.prerequisite_report_sha256 == hashlib.sha256(report.read_bytes()).hexdigest()
    assert identity.parent_checkpoint_path == str(checkpoint.resolve())
    assert identity.prerequisite_report_path == str(report.resolve())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report["decision"].update(passed=False), "passed"),
        (
            lambda report: report["metrics"]["record_count_by_family"].update(layered=2),
            "three",
        ),
        (
            lambda report: report["metrics"].update(aggregate_query_relative_l2=0.10),
            "threshold",
        ),
        (
            lambda report: report["metrics"]["family_late_relative_l2"].update(layered=0.10),
            "threshold",
        ),
        (
            lambda report: report["metrics"].update(missing_gradient_groups=["query"]),
            "gradient",
        ),
        (
            lambda report: report["diagnostics"]["records"].append(
                {"sample_id": "bad", "medium_type": "anomaly"}
            ),
            "anomaly",
        ),
    ],
)
def test_pilot_prerequisite_rejects_failed_or_weakened_gate(
    tmp_path: Path, mutation, message: str
):
    report_path, checkpoint, original = _write_prerequisite(tmp_path)
    report = copy.deepcopy(original)
    mutation(report)
    report_path.write_text(json.dumps(report), encoding="utf8")
    with pytest.raises((ValueError, RuntimeError), match=message):
        validate_pilot_prerequisite(
            report_path,
            checkpoint,
            expected_manifest_digest="manifest",
        )


def test_pilot_prerequisite_rejects_identity_and_file_mismatches(tmp_path: Path):
    report, checkpoint, _ = _write_prerequisite(tmp_path)
    with pytest.raises(ValueError, match="manifest"):
        validate_pilot_prerequisite(report, checkpoint, expected_manifest_digest="other")
    with pytest.raises(ValueError, match="checkpoint"):
        validate_pilot_prerequisite(report, tmp_path / "missing.pt", expected_manifest_digest="manifest")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["global_step"] = 749
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="step"):
        validate_pilot_prerequisite(report, checkpoint, expected_manifest_digest="manifest")
