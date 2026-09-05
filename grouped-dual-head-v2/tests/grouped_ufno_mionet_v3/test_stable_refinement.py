from __future__ import annotations

import pytest
import torch

from scripts.train_grouped_v3_stable_refinement import (
    accept_refinement_checkpoint,
    reduce_optimizer_learning_rate,
    refinement_control_values,
    select_matching_validation,
)


def test_refinement_accepts_aggregate_improvement_without_family_regression():
    decision = accept_refinement_checkpoint(
        anchor_family_scores={"uniform": 2.0, "layered": 2.5, "marmousi": 2.0},
        candidate_family_scores={"uniform": 1.99, "layered": 2.52, "marmousi": 1.95},
        anchor_aggregate=2.20,
        candidate_aggregate=2.17,
        family_regression_tolerance=0.02,
    )
    assert decision.accepted
    assert decision.failures == ()


def test_refinement_rejects_non_improvement_and_family_forgetting():
    decision = accept_refinement_checkpoint(
        anchor_family_scores={"uniform": 2.0, "layered": 2.5, "marmousi": 2.0},
        candidate_family_scores={"uniform": 2.05, "layered": 2.4, "marmousi": 1.9},
        anchor_aggregate=2.20,
        candidate_aggregate=2.21,
        family_regression_tolerance=0.02,
    )
    assert not decision.accepted
    assert "balanced validation score did not improve" in decision.failures
    assert "family regression tolerance exceeded for uniform" in decision.failures


def test_refinement_rejects_dual_head_consistency_regression():
    decision = accept_refinement_checkpoint(
        anchor_family_scores={"uniform": 2.0, "layered": 2.5, "marmousi": 2.0},
        candidate_family_scores={"uniform": 1.9, "layered": 2.4, "marmousi": 1.9},
        anchor_aggregate=2.20,
        candidate_aggregate=2.10,
        family_regression_tolerance=0.02,
        anchor_head_consistency=0.10,
        candidate_head_consistency=0.11,
        head_consistency_regression_tolerance=0.02,
    )
    assert not decision.accepted
    assert "dual-head consistency regression tolerance exceeded" in decision.failures


def test_selected_curriculum_checkpoint_must_match_one_validation_report(tmp_path):
    selected = tmp_path / "marmousi" / "checkpoints" / "epoch.pt"
    selected.parent.mkdir(parents=True)
    selected.touch()
    reports = [
        {"checkpoint": str(tmp_path / "layered" / "checkpoints" / "epoch.pt")},
        {"checkpoint": str(selected)},
    ]
    assert select_matching_validation(selected, reports) is reports[1]


def test_reduce_optimizer_learning_rate_preserves_optimizer_state():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-3)
    parameter.square().backward()
    optimizer.step()
    state_id = id(optimizer.state[parameter])

    reduced = reduce_optimizer_learning_rate(
        optimizer,
        factor=0.5,
        minimum=1.0e-5,
    )

    assert reduced == pytest.approx(5.0e-4)
    assert id(optimizer.state[parameter]) == state_id
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5.0e-4)


def test_continuous_mode_is_explicitly_configured():
    values = refinement_control_values(
        {
            "training_control": {
                "mode": "continuous",
                "lr_patience_epochs": 3,
                "early_stopping_patience_epochs": 0,
            }
        }
    )
    assert values == ("continuous", 3, 0)


def test_missing_control_config_preserves_legacy_behavior():
    assert refinement_control_values({}) == ("legacy_guarded", 1, 0)
