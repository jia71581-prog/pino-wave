from __future__ import annotations

import torch

from scripts.train_grouped_v3_curriculum import (
    accumulate_curriculum_update,
    family_validation_scores,
    rejection_recovery,
    select_stage_output,
)


def test_accumulation_performs_one_effective_batch_update():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    closures = [lambda: parameter.square(), lambda: parameter.square(), lambda: parameter.square()]
    report = accumulate_curriculum_update(
        optimizer=optimizer,
        parameters=(parameter,),
        required_gradient_groups={"all": (parameter,)},
        closures=closures,
        record_counts=(4, 4, 4),
        loss_scales=(1.0, 1.0, 1.0),
        effective_records=12,
        gradient_clip=10.0,
    )
    assert report["microbatch_count"] == 3
    assert report["effective_record_count"] == 12
    assert report["missing_gradient_groups"] == []
    torch.testing.assert_close(parameter, torch.tensor(0.8))


def test_marmousi_ten_record_loss_scale_matches_effective_batch_weight():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    report = accumulate_curriculum_update(
        optimizer=optimizer,
        parameters=(parameter,),
        required_gradient_groups={"all": (parameter,)},
        closures=(lambda: parameter.square(),),
        record_counts=(10,),
        loss_scales=(1.2,),
        effective_records=12,
        gradient_clip=10.0,
    )
    assert report["weighted_loss"] == 1.0
    torch.testing.assert_close(parameter, torch.tensor(0.8))


def test_family_validation_scores_sum_exact_and_midpoint_dense_and_query():
    validation = {
        mode: {
            "family_dense_relative_l2": {"uniform": 1.0, "layered": 2.0, "marmousi": 3.0},
            "family_query_relative_l2": {"uniform": 0.5, "layered": 1.0, "marmousi": 1.5},
        }
        for mode in ("exact", "interpolated")
    }
    assert family_validation_scores(validation) == {
        "uniform": 3.0,
        "layered": 6.0,
        "marmousi": 9.0,
    }


def test_stage_transition_uses_accepted_best_or_unchanged_parent():
    assert select_stage_output("parent.pt", "best.pt") == "best.pt"
    assert select_stage_output("parent.pt", None) == "parent.pt"


def test_rejection_recovery_rolls_back_to_anchor_and_halves_learning_rate():
    checkpoint, learning_rate = rejection_recovery(
        stage_parent="parent.pt",
        accepted_checkpoint="accepted.pt",
        learning_rate=2.0e-5,
        factor=0.5,
        minimum=2.5e-6,
    )
    assert checkpoint == "accepted.pt"
    assert learning_rate == 1.0e-5

    checkpoint, learning_rate = rejection_recovery(
        stage_parent="parent.pt",
        accepted_checkpoint=None,
        learning_rate=3.0e-6,
        factor=0.5,
        minimum=2.5e-6,
    )
    assert checkpoint == "parent.pt"
    assert learning_rate == 2.5e-6
