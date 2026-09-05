"""V16 runner unit tests: gate arithmetic, allowlist, head contract."""
import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.train_r16_dscp_v16 as v16


def test_wide_head_parameter_count():
    head = v16.Wide128Head()
    assert sum(p.numel() for p in head.parameters()) == v16.EXPECTED_PARAMETERS


def test_wide_head_zero_init_output_is_zero():
    head = v16.Wide128Head()
    out = head(torch.randn(1, 29, 8, 8))
    assert out.shape == (1, 16, 8, 8)
    assert float(out.abs().max()) == 0.0


def test_truth_allowlist_refuses_off_list_and_non_train():
    class R:
        split = "train"
        sample_id = "train_uniform_99999"
    with pytest.raises(v16.TruthScopeRefusal):
        v16.load_truth_allowlisted(R(), frozenset({"train_uniform_00321"}))

    class V:
        split = "validation"
        sample_id = "validation_uniform_00001"
    with pytest.raises(v16.TruthScopeRefusal):
        v16.load_truth_allowlisted(V(), frozenset({"validation_uniform_00001"}))


def _score(sid, fam, loss, cand, par):
    return {"sample_id": sid, "family": fam, "loss": loss,
            "aggregate_rel_l2": cand, "parent_rel_l2": par,
            "gain_iii": (par - cand) / abs(par), "nonworse": cand <= par}


def test_smoke_gates_thresholds_and_pass_logic():
    plan = v16.STAGE_PLAN["smoke"]
    assert plan["loss_reduction_min"] == 0.10
    assert plan["oracle_fraction"] == 0.35
    initial = {"a": 1.0, "b": 1.0, "c": 1.0}
    scores = [_score("a", "uniform", 0.85, 0.07, 0.076),
              _score("b", "layered", 0.45, 0.090, 0.105),
              _score("c", "marmousi", 0.75, 0.54, 0.60)]
    oracle = {k: {"acceptance_convention_gain": g}
              for k, g in (("a", 0.124), ("b", 0.210), ("c", 0.133))}
    gates = v16.smoke_gates(plan, initial, scores, oracle, 6 * 1024**3, 200_000)
    assert gates["loss_per_record"]["passed"]
    assert gates["oracle_gain"]["passed"]
    assert gates["nonworse"]["passed"]
    assert gates["vram"]["passed"] and gates["checkpoint"]["passed"]
    # one record below the loss floor fails the per-record gate
    gates2 = v16.smoke_gates(plan, initial, [
        _score("a", "uniform", 0.95, 0.07, 0.076), scores[1], scores[2],
    ], oracle, 6 * 1024**3, 200_000)
    assert not gates2["loss_per_record"]["passed"]


def test_eval_gates_family_and_nonworse():
    plan = v16.STAGE_PLAN["pilot"]
    scores = []
    for fam in ("uniform", "layered", "marmousi"):
        for i in range(8):
            scores.append(_score(f"{fam}{i}", fam, 0.5, 0.90, 1.00))
    gates = v16.eval_gates(plan, scores, 6 * 1024**3, 200_000)
    assert gates["joint_improvement"]["passed"]
    assert gates["per_family_improvement"]["passed"]
    assert gates["nonworse_records"]["passed"]
    # two worse records break the 23/24 gate
    scores[0] = _score("u0", "uniform", 0.5, 1.10, 1.00)
    scores[1] = _score("u1", "uniform", 0.5, 1.10, 1.00)
    gates = v16.eval_gates(plan, scores, 6 * 1024**3, 200_000)
    assert not gates["nonworse_records"]["passed"]


def test_vram_gate_refuses_zero_measurement():
    plan = v16.STAGE_PLAN["pilot"]
    scores = [_score("a", "uniform", 0.5, 0.9, 1.0)]
    gates = v16.eval_gates(plan, scores, 0, 200_000)
    assert not gates["vram"]["passed"]


def test_stage_plan_matches_frozen_preregistration():
    prereg = json.loads((ROOT / "results/r16_dscp_v16_preregistration_20260826.json").read_text())
    stages = prereg["stages"]
    assert stages["smoke"]["updates"] == v16.STAGE_PLAN["smoke"]["updates"] == 3072
    assert stages["smoke"]["gates"]["loss_per_record"]["threshold"] == 0.10
    assert stages["smoke"]["gates"]["oracle_gain"]["fraction"] == 0.35
    assert stages["pilot"]["gates"]["joint_improvement"]["threshold"] == 0.01
    assert stages["pilot"]["gates"]["per_family_improvement"]["threshold"] == 0.005
    assert stages["pilot"]["gates"]["nonworse_records"]["threshold"] == 23
    assert stages["long"]["max_epochs"] == v16.STAGE_PLAN["long"]["max_epochs"] == 20
    assert stages["long"]["patience"] == v16.STAGE_PLAN["long"]["patience"] == 4
    assert stages["long"]["wall_s_max"] == v16.STAGE_PLAN["long"]["wall_s"] == 7200


def test_role_sample_ids_counts():
    assert len(v16.role_sample_ids("smoke")) == 3
    assert len(v16.role_sample_ids("pilot_fit")) == 24
    assert len(v16.role_sample_ids("pilot_confirm")) == 24
    assert len(v16.role_sample_ids("long_fit")) == 192
    assert len(v16.role_sample_ids("long_calibration")) == 24
    with pytest.raises(v16.V16Refusal):
        v16.role_sample_ids("no_such_role")
