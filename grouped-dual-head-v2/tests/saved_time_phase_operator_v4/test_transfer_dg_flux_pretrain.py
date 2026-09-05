from __future__ import annotations

import pytest

from scripts.summarize_transfer_dg_flux_pretrain import summarize
from scripts.train_transfer_dg_flux_pretrain import (
    SLOT_WEIGHTS,
    _model,
    dg_rank_for_arm,
    weighted_multiphase_score,
)


def test_only_conceptual_arm_change_is_dg_flux_rank():
    assert dg_rank_for_arm("control") == 0
    assert dg_rank_for_arm("dg_flux") == 16
    with pytest.raises(ValueError):
        dg_rank_for_arm("halo")


def test_frozen_multiphase_score():
    rows = [{"aggregate": 1.0}, {"aggregate": 2.0}, {"aggregate": 3.0}]
    assert SLOT_WEIGHTS == (0.60, 0.25, 0.15)
    assert weighted_multiphase_score(rows) == pytest.approx(1.55)


def test_only_outer_time_checkpointing_is_enabled():
    model = _model(dg_rank=16, device="cpu")
    assert model.activation_checkpointing is True
    assert model.step_stack.activation_checkpointing is False


def test_summary_requires_strict_gain_over_control_and_parent():
    rows = [
        {"arm": "dg_flux", "seed": 372, "best_score": 0.19, "parent_score": 0.21},
        {"arm": "dg_flux", "seed": 733, "best_score": 0.20, "parent_score": 0.22},
        {"arm": "control", "seed": 372, "best_score": 0.20, "parent_score": 0.21},
        {"arm": "control", "seed": 733, "best_score": 0.21, "parent_score": 0.22},
    ]
    assert summarize(rows)["decision"] == "accepted_train_only"
    rows[0]["best_score"] = 0.23
    assert summarize(rows)["decision"] == "rejected"
