from dataclasses import dataclass

import pytest
import torch
from torch import nn

from saved_time_phase_operator_v4.lbfgs import (
    FixedClosureBatch,
    effective_batch_records,
    freeze_for_band_adapter_readout_lbfgs,
    freeze_for_dense_lbfgs,
    refinement_gate,
)


class TinyOperator(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.dense_decoder = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))


class TinyBandExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(4, 4)
        self.output = nn.Linear(4, 1)


class TinyBandOperator(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.dense_decoder = nn.Module()
        self.dense_decoder.band_limited_adapter = nn.Module()
        self.dense_decoder.band_limited_adapter.experts = nn.ModuleList(
            [TinyBandExpert() for _ in range(3)]
        )


@dataclass(frozen=True)
class Batch:
    sample_id: tuple[str, ...]


def test_freeze_policy_selects_only_dense_decoder_parameters():
    model = TinyOperator()
    selected = freeze_for_dense_lbfgs(model)

    assert selected
    assert {name for name, value in model.named_parameters() if value.requires_grad} == {
        "dense_decoder.0.weight", "dense_decoder.0.bias",
        "dense_decoder.1.weight", "dense_decoder.1.bias",
    }
    assert sum(value.numel() for value in selected) == sum(
        value.numel() for value in model.dense_decoder.parameters()
    )


def test_band_adapter_lbfgs_selects_only_three_linear_readout_heads():
    model = TinyBandOperator()

    selected = freeze_for_band_adapter_readout_lbfgs(model)

    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert trainable == {
        f"dense_decoder.band_limited_adapter.experts.{index}.output.{kind}"
        for index in range(3)
        for kind in ("weight", "bias")
    }
    assert sum(value.numel() for value in selected) == 3 * (4 + 1)


def test_effective_batch_and_fixed_closure_identity_are_strict():
    assert effective_batch_records(macro_records=12, accumulated_macros=4) == 48
    cached = FixedClosureBatch((Batch(("a", "b")), Batch(("c", "d"))))
    cached.verify((Batch(("a", "b")), Batch(("c", "d"))))
    with pytest.raises(ValueError, match="changed"):
        cached.verify((Batch(("a", "b")), Batch(("c", "x"))))


def test_refinement_gate_requires_aggregate_gain_and_family_safety():
    parent_family = {"uniform": 0.5, "layered": 0.6, "marmousi": 0.7}
    assert refinement_gate(
        parent_score=0.56, candidate_score=0.54,
        parent_family=parent_family,
        candidate_family={"uniform": 0.49, "layered": 0.58, "marmousi": 0.68},
    )["passed"]
    result = refinement_gate(
        parent_score=0.56, candidate_score=0.54,
        parent_family=parent_family,
        candidate_family={"uniform": 0.49, "layered": 0.58, "marmousi": 0.73},
    )
    assert not result["passed"]
    assert result["family_safe"] is False


def test_refinement_gate_rejects_gain_above_absolute_accuracy_target():
    result = refinement_gate(
        parent_score=0.20,
        candidate_score=0.15,
        parent_family={"uniform": 0.15, "layered": 0.20, "marmousi": 0.25},
        candidate_family={"uniform": 0.10, "layered": 0.15, "marmousi": 0.20},
        maximum_candidate_score=0.05,
        maximum_candidate_family_score=0.08,
    )
    assert result["relative_improvement"] > 0.02
    assert result["passed"] is False
    assert result["absolute_aggregate_accuracy"] is False
    assert result["absolute_family_accuracy"] is False


def test_refinement_gate_accepts_absolute_accuracy_target():
    result = refinement_gate(
        parent_score=0.09,
        candidate_score=0.045,
        parent_family={"uniform": 0.07, "layered": 0.09, "marmousi": 0.11},
        candidate_family={"uniform": 0.04, "layered": 0.06, "marmousi": 0.075},
        maximum_candidate_score=0.05,
        maximum_candidate_family_score=0.08,
    )
    assert result["passed"] is True
    assert result["absolute_aggregate_accuracy"] is True
    assert result["absolute_family_accuracy"] is True


@pytest.mark.parametrize("macro,accumulated", [(0, 4), (12, 0)])
def test_effective_batch_rejects_nonpositive_counts(macro, accumulated):
    with pytest.raises(ValueError, match="positive"):
        effective_batch_records(macro_records=macro, accumulated_macros=accumulated)
