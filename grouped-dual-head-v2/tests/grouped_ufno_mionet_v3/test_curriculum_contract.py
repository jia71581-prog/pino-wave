from __future__ import annotations

from pathlib import Path

import pytest
import torch

from grouped_ufno_mionet_v3.training.checkpoint import CHECKPOINT_FORMAT
from grouped_ufno_mionet_v3.training.curriculum import (
    accept_stage_checkpoint,
    curriculum_run_digest,
    validate_curriculum_parent,
)


def _parent_files(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "epoch": 20,
            "global_step": 3740,
            "manifest_digest": "manifest",
            "config_digest": "pilot-run",
            "metrics": {"validation_score": 2.0},
            "model_state": {},
        },
        checkpoint,
    )
    terminal = {
        "status": "complete",
        "epochs": 20,
        "global_step": 3740,
        "best_validation_score": 2.0,
        "run_digest": "pilot-run",
    }
    best_validation = {
        "epoch": 20,
        "global_step": 3740,
        "manifest_digest": "manifest",
        "run_digest": "pilot-run",
        "checkpoint": str(checkpoint),
        "validation": {"score": 2.0},
    }
    return terminal, best_validation, checkpoint


def test_curriculum_parent_requires_completed_pilot_and_bound_best_checkpoint(tmp_path: Path):
    terminal, best_validation, checkpoint = _parent_files(tmp_path)
    identity = validate_curriculum_parent(
        terminal,
        best_validation,
        checkpoint,
        expected_manifest_digest="manifest",
        expected_run_digest="pilot-run",
    )
    assert identity.parent_epoch == 20
    assert identity.parent_global_step == 3740
    assert identity.parent_validation_score == 2.0
    assert len(identity.parent_checkpoint_sha256) == 64
    assert curriculum_run_digest("curriculum-config", identity) == curriculum_run_digest(
        "curriculum-config", identity
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda terminal, validation, checkpoint: terminal.update(status="failed"), "complete"),
        (lambda terminal, validation, checkpoint: validation.update(manifest_digest="wrong"), "manifest"),
        (lambda terminal, validation, checkpoint: validation.update(run_digest="wrong"), "run"),
        (lambda terminal, validation, checkpoint: validation.update(global_step=3739), "step"),
        (lambda terminal, validation, checkpoint: checkpoint.write_bytes(b"corrupt"), "checkpoint"),
    ],
)
def test_curriculum_parent_rejects_unsealed_or_mismatched_evidence(tmp_path, mutation, message):
    terminal, best_validation, checkpoint = _parent_files(tmp_path)
    mutation(terminal, best_validation, checkpoint)
    with pytest.raises((ValueError, RuntimeError, OSError), match=message):
        validate_curriculum_parent(
            terminal,
            best_validation,
            checkpoint,
            expected_manifest_digest="manifest",
            expected_run_digest="pilot-run",
        )


def test_stage_acceptance_allows_target_improvement_with_bounded_replay_change():
    decision = accept_stage_checkpoint(
        parent_family_scores={"uniform": 0.50, "layered": 0.70, "marmousi": 0.55},
        candidate_family_scores={"uniform": 0.54, "layered": 0.65, "marmousi": 0.56},
        target_family="layered",
        learned_families=("uniform",),
        aggregate_parent=2.30,
        aggregate_candidate=2.35,
    )
    assert decision.accepted
    assert decision.failures == ()


@pytest.mark.parametrize(
    ("candidate", "aggregate", "message"),
    [
        ({"uniform": 0.50, "layered": 0.71, "marmousi": 0.55}, 2.30, "target"),
        ({"uniform": 0.56, "layered": 0.65, "marmousi": 0.55}, 2.30, "forget"),
        ({"uniform": 0.50, "layered": 0.65, "marmousi": 0.55}, 2.50, "aggregate"),
    ],
)
def test_stage_acceptance_rejects_no_improvement_forgetting_or_aggregate_regression(
    candidate, aggregate, message
):
    decision = accept_stage_checkpoint(
        parent_family_scores={"uniform": 0.50, "layered": 0.70, "marmousi": 0.55},
        candidate_family_scores=candidate,
        target_family="layered",
        learned_families=("uniform",),
        aggregate_parent=2.30,
        aggregate_candidate=aggregate,
    )
    assert not decision.accepted
    assert any(message in failure for failure in decision.failures)
