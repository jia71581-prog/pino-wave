import pytest
import torch
from torch import nn

from saved_time_phase_operator_v4.experts import (
    FamilyRoutedResidualExperts,
    VelocityFamilyRouter,
)


class ConstantExpert(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, shared, time_features):
        records, times = time_features.shape[:2]
        return shared.new_full((records, times, *shared.shape[-2:]), self.value)


def test_velocity_router_starts_from_a_confident_structural_family_prior():
    router = VelocityFamilyRouter(width=8)
    uniform = torch.full((1, 9, 9), 2000.0)
    layered = torch.full((1, 9, 9), 3000.0)
    layered[:, :3] = 2000.0
    marmousi = torch.full((1, 9, 9), 2500.0)
    marmousi[:, :3] = 1500.0
    marmousi[:, :, ::2] += 150.0
    velocity = torch.stack((uniform, layered, marmousi))

    logits = router(velocity, torch.zeros(3, 8, 9, 9))
    probabilities = logits.softmax(dim=-1)

    assert torch.equal(probabilities.argmax(dim=-1), torch.tensor([0, 1, 2]))
    assert torch.all(probabilities.diagonal() > 0.95)
    assert torch.count_nonzero(router.network[-1].weight) == 0
    assert torch.count_nonzero(router.network[-1].bias) == 0


def test_family_experts_are_parent_identity_and_velocity_only():
    module = FamilyRoutedResidualExperts(width=8, rank=4, family_count=3)
    velocity = torch.stack(
        (torch.ones(1, 9, 9), torch.full((1, 9, 9), 2.0))
    )
    medium_features = torch.randn(2, 8, 9, 9)
    shared = torch.randn(6, 8, 9, 9)
    time_features = torch.randn(3, 2, 5)
    mapping = torch.tensor([0, 1, 0])

    result = module(
        velocity, medium_features, shared, time_features, mapping
    )

    assert result.correction.shape == (3, 2, 9, 9)
    assert result.router_logits.shape == (2, 3)
    assert torch.count_nonzero(result.correction) == 0
    result.correction.sum().backward()
    assert all(expert.output.weight.grad is not None for expert in module.experts)


def test_family_experts_reject_invalid_record_mapping():
    module = FamilyRoutedResidualExperts(width=8, rank=4, family_count=3)

    with pytest.raises(ValueError, match="record_to_medium"):
        module(
            torch.ones(1, 1, 9, 9),
            torch.randn(1, 8, 9, 9),
            torch.randn(2, 8, 9, 9),
            torch.randn(1, 2, 5),
            torch.tensor([1]),
        )


def test_teacher_routing_selects_the_registered_expert_without_changing_router():
    module = FamilyRoutedResidualExperts(width=8, rank=4, family_count=3)
    module.experts = nn.ModuleList(ConstantExpert(index + 1) for index in range(3))
    velocity = torch.stack(
        (torch.ones(1, 9, 9), torch.full((1, 9, 9), 2.0))
    )
    medium_features = torch.randn(2, 8, 9, 9)
    shared = torch.randn(6, 8, 9, 9)
    time_features = torch.randn(3, 2, 5)
    mapping = torch.tensor([0, 1, 0])

    result = module(
        velocity,
        medium_features,
        shared,
        time_features,
        mapping,
        route_override=torch.tensor([0, 2]),
    )

    assert torch.equal(result.correction[:, 0, 0, 0], torch.tensor([1.0, 3.0, 1.0]))
    assert result.router_logits.shape == (2, 3)
    assert result.router_probabilities.shape == (2, 3)


def test_teacher_routing_rejects_invalid_medium_targets():
    module = FamilyRoutedResidualExperts(width=8, rank=4, family_count=3)

    with pytest.raises(ValueError, match="route override"):
        module(
            torch.ones(1, 1, 9, 9),
            torch.randn(1, 8, 9, 9),
            torch.randn(2, 8, 9, 9),
            torch.randn(1, 2, 5),
            torch.tensor([0]),
            route_override=torch.tensor([3]),
        )
