from __future__ import annotations

import torch

from saved_time_phase_operator_v4.instance_adaptation.b2_v6_encoding import (
    ParentLatentConditioner,
    enriched_physical_conditioning,
)


def test_enriched_conditioning_has_bounded_twenty_channel_contract():
    base = torch.zeros(2, 7, 21, 21)
    base[:, 4] = 0.03
    velocity = torch.full((2, 1, 21, 21), 2000.0)
    result = enriched_physical_conditioning(
        base, velocity, torch.tensor([10.0, 30.0]), torch.tensor([0.15, 0.05])
    )
    assert result.shape == (2, 20, 21, 21)
    assert torch.isfinite(result).all()
    assert result[:, 7:].abs().max() <= 4.0
    assert not torch.equal(result[0, 9:11], result[1, 9:11])
    assert result[0, 15, 10, 0] == 1.0
    assert result[0, 16, 10, -1] == 1.0
    assert result[0, 18, 0, 10] == 1.0


def test_uniform_medium_has_zero_interior_multiscale_contrast():
    base = torch.zeros(1, 7, 41, 41)
    velocity = torch.full((1, 1, 41, 41), 2500.0)
    result = enriched_physical_conditioning(
        base, velocity, torch.tensor([20.0]), torch.tensor([0.075])
    )
    assert result[:, 12:15, 10:-10, 10:-10].abs().max() < 1.0e-6


def test_parent_latent_conditioner_projects_and_preserves_base_channels():
    module = ParentLatentConditioner(parent_width=8, projected_width=3)
    base = torch.randn(2, 7, 12, 14)
    medium = torch.randn(2, 8, 12, 14)
    source_map = torch.randn(2, 8, 12, 14)
    source_hidden = torch.randn(2, 8)
    result = module(base, medium, source_map, source_hidden)
    assert result.shape == (2, 16, 12, 14)
    assert torch.equal(result[:, :7], base)
    result.square().mean().backward()
    assert all(parameter.grad is not None for parameter in module.parameters())
