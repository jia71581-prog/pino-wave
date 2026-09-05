from __future__ import annotations

import numpy as np

from scripts.build_b2_train_generalization_manifest import select_group_disjoint
from scripts.evaluate_b2_v3_train_generalization import judge_lanes
from scripts.evaluate_b2_v3_causal_selector import choose_by_prefix, judge_selector
from scripts.train_b2_group_disjoint import calibration_gate


def test_group_disjoint_selection_is_deterministic_and_unique():
    candidates = np.arange(8)
    sample_ids = np.asarray([f"train_uniform_{i:05d}" for i in candidates])
    group_ids = np.asarray(["a", "a", "b", "c", "c", "d", "e", "f"])
    first = select_group_disjoint(
        candidates, sample_ids, group_ids, count=4, rng=np.random.default_rng(17)
    )
    second = select_group_disjoint(
        candidates, sample_ids, group_ids, count=4, rng=np.random.default_rng(17)
    )
    assert first == second
    assert len({group_ids[index] for index in first}) == 4


def _lane(name, *, candidate, baseline, nonworse, family_gain=True):
    return {
        "lane": name,
        "aggregate": candidate,
        "baseline_aggregate": baseline,
        "nonworse_count": nonworse,
        "per_family": {
            family: {
                "candidate": candidate if family_gain else baseline + 0.001,
                "baseline": baseline,
            }
            for family in ("uniform", "layered", "marmousi")
        },
    }


def test_generalization_gate_requires_both_seeds_and_all_subgates():
    gate = {"minimum_nonworse_records": 27, "minimum_absolute_gain_vs_anchor": 0.01}
    passing = [
        _lane("s372", candidate=0.10, baseline=0.12, nonworse=28),
        _lane("s733", candidate=0.105, baseline=0.12, nonworse=27),
    ]
    assert judge_lanes(passing, gate)["passed"]
    passing[1] = _lane(
        "s733", candidate=0.105, baseline=0.12, nonworse=27, family_gain=False
    )
    assert not judge_lanes(passing, gate)["passed"]


def test_causal_selector_uses_prefix_argmin_only():
    names, choice = choose_by_prefix({
        "anchor": np.asarray([0.1, 0.3, 0.2]),
        "seed372": np.asarray([0.2, 0.1, 0.3]),
        "seed733": np.asarray([0.3, 0.2, 0.1]),
    })
    assert names == ["anchor", "seed372", "seed733"]
    assert choice.tolist() == [0, 1, 2]


def test_selector_gate_requires_gain_family_and_nonworse():
    summary = {
        "aggregate": 0.10,
        "baseline_aggregate": 0.12,
        "nonworse_count": 28,
        "per_family": {
            family: {"candidate": 0.10, "baseline": 0.12}
            for family in ("uniform", "layered", "marmousi")
        },
    }
    gate = {"minimum_absolute_gain_vs_anchor": 0.005, "minimum_nonworse_records": 27}
    assert judge_selector(summary, gate)["passed"]
    summary["per_family"]["uniform"]["candidate"] = 0.121
    assert not judge_selector(summary, gate)["passed"]


def test_group_disjoint_calibration_gate_requires_both_seeds():
    gate = {"minimum_absolute_gain_vs_anchor": 0.01, "minimum_nonworse_records": 54}
    lane = {
        "lane": "seed372",
        "aggregate": 0.10,
        "baseline": 0.12,
        "nonworse": 56,
        "per_family": {family: 0.10 for family in ("uniform", "layered", "marmousi")},
        "baseline_per_family": {
            family: 0.12 for family in ("uniform", "layered", "marmousi")
        },
    }
    second = {**lane, "lane": "seed733"}
    assert calibration_gate([lane, second], gate)["passed"]
    second = {**second, "nonworse": 53}
    assert not calibration_gate([lane, second], gate)["passed"]
