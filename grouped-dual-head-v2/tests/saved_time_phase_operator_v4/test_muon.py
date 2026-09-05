import copy

import pytest
import torch
from torch import nn

from saved_time_phase_operator_v4.muon import (
    build_module_muon_adamw,
    build_staged_muon_adamw,
    parameter_uses_muon,
)
from saved_time_phase_operator_v4.instance_adaptation.defect_correction import (
    CausalErrorBasisGenerator,
)


class TinyHybridOperator(nn.Module):
    def __init__(self):
        super().__init__()
        self.medium_encoder = nn.Linear(3, 4)
        self.source_encoder = nn.Linear(3, 4)
        self.coordinate_encoder = nn.Linear(3, 4)
        self.travel_branch = nn.Linear(3, 4)
        self.fusion = nn.Linear(4, 4)
        self.dense_decoder = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))


def _build(model):
    return build_staged_muon_adamw(
        model,
        dense_lr=1.0e-3,
        geometry_lr=2.0e-4,
        backbone_lr=1.0e-4,
        weight_decay=1.0e-6,
        muon_lr_scale=10.0,
        adamw_implementation="single_tensor",
    )


def test_hybrid_routes_hidden_matrices_but_keeps_output_head_on_adamw():
    model = TinyHybridOperator()
    optimizer = _build(model)
    names_by_parameter = {id(parameter): name for name, parameter in model.named_parameters()}
    muon_names = {
        names_by_parameter[id(parameter)]
        for group in optimizer.muon.param_groups
        for parameter in group["params"]
    }
    adamw_names = {
        names_by_parameter[id(parameter)]
        for group in optimizer.adamw.param_groups
        for parameter in group["params"]
    }

    assert "dense_decoder.0.weight" in muon_names
    assert "dense_decoder.1.weight" in adamw_names
    assert "dense_decoder.1.bias" in adamw_names
    assert muon_names.isdisjoint(adamw_names)
    assert muon_names | adamw_names == set(names_by_parameter.values())
    assert parameter_uses_muon("dense_decoder.0.weight", model.dense_decoder[0].weight)
    assert not parameter_uses_muon(
        "dense_decoder.output.weight", nn.Parameter(torch.ones(1, 4))
    )


def test_hybrid_state_round_trip_reproduces_the_next_step():
    torch.manual_seed(7)
    first = TinyHybridOperator()
    second = copy.deepcopy(first)
    first_optimizer = _build(first)
    second_optimizer = _build(second)

    for parameter in first.parameters():
        parameter.grad = torch.randn_like(parameter)
    first_optimizer.step()
    second.load_state_dict(first.state_dict())
    # Checkpoints serialize optimizer tensors; deepcopy models that durable
    # boundary rather than sharing live momentum buffers in-process.
    second_optimizer.load_state_dict(copy.deepcopy(first_optimizer.state_dict()))

    torch.manual_seed(11)
    gradients = [torch.randn_like(parameter) for parameter in first.parameters()]
    for parameter, gradient in zip(first.parameters(), gradients, strict=True):
        parameter.grad = gradient.clone()
    for parameter, gradient in zip(second.parameters(), gradients, strict=True):
        parameter.grad = gradient.clone()
    first_optimizer.step()
    second_optimizer.step()

    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(left, right)
    assert [group["group_name"] for group in second_optimizer.param_groups] == [
        group["group_name"] for group in first_optimizer.param_groups
    ]


def test_hybrid_rejects_nonpositive_muon_lr_scale():
    with pytest.raises(ValueError, match="learning-rate scale"):
        build_staged_muon_adamw(
            TinyHybridOperator(),
            dense_lr=1.0e-3,
            geometry_lr=2.0e-4,
            backbone_lr=1.0e-4,
            weight_decay=0.0,
            muon_lr_scale=0.0,
        )


def test_cpadc_hybrid_keeps_physical_scale_heads_on_adamw():
    generator = CausalErrorBasisGenerator(
        rank=8, phase_rank=2, width=16, ramp_steps=4
    )
    optimizer = build_module_muon_adamw(
        generator,
        adamw_lr=5.0e-5,
        muon_lr_scale=20.0,
        weight_decay=1.0e-6,
        adamw_betas=(0.9, 0.99),
    )
    names = {id(parameter): name for name, parameter in generator.named_parameters()}
    muon_names = {
        names[id(parameter)]
        for group in optimizer.muon.param_groups
        for parameter in group["params"]
    }
    adamw_names = {
        names[id(parameter)]
        for group in optimizer.adamw.param_groups
        for parameter in group["params"]
    }
    assert "spatial_encoder.3.weight" in muon_names
    assert "temporal_encoder.2.weight" in muon_names
    assert "spatial_encoder.6.weight" in adamw_names
    assert "temporal_encoder.4.weight" in adamw_names
    assert "trust_head.2.weight" in adamw_names
    assert muon_names.isdisjoint(adamw_names)
    assert muon_names | adamw_names == set(names.values())
