import numpy as np
import pytest

from saved_time_phase_operator_v4.adaptive_pretraining import (
    initial_controller_state,
    probe_selection_sha256,
    stable_probe_positions,
    update_controller,
)


FAMILIES = ("uniform", "layered", "anomaly", "marmousi")
CONFIG = {
    "ema_beta": 0.8,
    "gain": 0.08,
    "max_step_ratio": 1.10,
    "min_multiplier": 0.5,
    "max_multiplier": 2.0,
    "worst_weight": 0.5,
}


def test_probe_is_deterministic_balanced_and_bound():
    records = [
        (0, index, f"{family}-{index}", family)
        for family in FAMILIES
        for index in range(12)
    ]
    first = stable_probe_positions(records, FAMILIES, per_family=3, namespace="unit")
    second = stable_probe_positions(records, FAMILIES, per_family=3, namespace="unit")
    assert first == second
    assert all(len(first[family]) == 3 for family in FAMILIES)
    assert len(probe_selection_sha256(records, first)) == 64


def test_controller_upweights_hard_cells_with_bounded_action():
    state = initial_controller_state(4, 8)
    energy = np.ones((4, 8), dtype=np.float64)
    energy[3, 7] = 100.0
    updated, event = update_controller({"energy": energy.tolist()}, state, CONFIG, epoch=22)
    weights = np.asarray(updated["weights"])
    action = np.asarray(event["action"])
    assert weights[3, 7] > 1.0
    assert weights[0, 0] < 1.0
    assert action.min() >= 1.0 / CONFIG["max_step_ratio"]
    assert action.max() <= CONFIG["max_step_ratio"]
    assert weights.min() >= CONFIG["min_multiplier"]
    assert weights.max() <= CONFIG["max_multiplier"]


def test_controller_reward_is_improvement_and_state_is_resumable():
    first, _ = update_controller(
        {"energy": np.full((4, 8), 4.0).tolist()}, None, CONFIG, epoch=22
    )
    second, event = update_controller(
        {"energy": np.full((4, 8), 1.0).tolist()}, first, CONFIG, epoch=23
    )
    assert event["reward"] > 0.0
    assert second["evaluations"] == 2
    assert second["last_epoch"] == 23


def test_controller_rejects_nonfinite_feedback():
    bad = np.ones((4, 8), dtype=np.float64)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        update_controller({"energy": bad.tolist()}, None, CONFIG, epoch=22)
