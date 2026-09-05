from __future__ import annotations

import pytest

from scripts.summarize_transfer_dg_a1 import summarize
from scripts.train_transfer_dg_a1_boundary_halo import (
    SLOT_WEIGHTS,
    halo_radius_for_arm,
    weighted_multiphase_score,
)


def test_transfer_dg_a1_arm_is_only_boundary_halo_radius():
    assert halo_radius_for_arm("control") == 0
    assert halo_radius_for_arm("halo") == 20
    with pytest.raises(ValueError):
        halo_radius_for_arm("other")


def test_weighted_multiphase_score_uses_frozen_slot_weights():
    phases = [{"aggregate": 1.0}, {"aggregate": 2.0}, {"aggregate": 3.0}]
    assert SLOT_WEIGHTS == (0.60, 0.25, 0.15)
    assert weighted_multiphase_score(phases) == pytest.approx(1.55)


def test_paired_summary_requires_halo_to_beat_control_and_parent():
    rows = [
        {"arm": "halo", "seed": 372, "best_score": 0.19, "parent_score": 0.21},
        {"arm": "halo", "seed": 733, "best_score": 0.20, "parent_score": 0.22},
        {"arm": "control", "seed": 372, "best_score": 0.20, "parent_score": 0.21},
        {"arm": "control", "seed": 733, "best_score": 0.21, "parent_score": 0.22},
    ]
    report = summarize(rows)
    assert report["decision"] == "accepted_train_only"
    rows[0]["best_score"] = 0.23
    assert summarize(rows)["decision"] == "rejected"
