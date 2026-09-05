from __future__ import annotations

import torch

from saved_time_phase_operator_v4.dg_interface import (
    DGInterfaceFluxResidual2d,
    acoustic_interface_reflection_maps,
)


def _conditioning(velocity_mps: torch.Tensor) -> torch.Tensor:
    value = torch.zeros(velocity_mps.shape[0], 7, *velocity_mps.shape[-2:])
    value[:, 0] = (velocity_mps - 4500.0) / 2500.0
    return value


def test_uniform_medium_has_exactly_zero_reflection_maps():
    velocity = torch.full((2, 11, 13), 3000.0)
    maps = acoustic_interface_reflection_maps(
        _conditioning(velocity), cpml_margin=2
    )
    assert maps.shape == (2, 2, 11, 13)
    assert torch.count_nonzero(maps) == 0


def test_layer_interface_is_signed_and_cpml_faces_are_excluded():
    velocity = torch.full((1, 11, 13), 2000.0)
    velocity[:, 5:] = 4000.0
    maps = acoustic_interface_reflection_maps(
        _conditioning(velocity), cpml_margin=2
    )
    expected = (4000.0 - 2000.0) / (4000.0 + 2000.0)
    torch.testing.assert_close(
        maps[:, 1, 4, 2:-2], torch.full((1, 9), expected)
    )
    assert torch.count_nonzero(maps[:, 1, :, :2]) == 0
    assert torch.count_nonzero(maps[:, 1, :, -2:]) == 0
    assert torch.count_nonzero(maps[:, 1, -3:]) == 0
    assert torch.count_nonzero(maps[:, 0]) == 0


def test_saved_physical_domain_keeps_boundary_adjacent_material_faces():
    velocity = torch.full((1, 11, 13), 2000.0)
    velocity[:, :, 1:] = 4000.0
    velocity[:, -1, :] = 5000.0
    maps = acoustic_interface_reflection_maps(_conditioning(velocity))

    left_expected = (4000.0 - 2000.0) / (4000.0 + 2000.0)
    bottom_expected = (5000.0 - 4000.0) / (5000.0 + 4000.0)
    torch.testing.assert_close(
        maps[:, 0, :-1, 0], torch.full((1, 10), left_expected)
    )
    assert maps[:, 0, -1, 0].item() == 0.0
    torch.testing.assert_close(
        maps[:, 1, -2, 1:], torch.full((1, 12), bottom_expected)
    )


def test_flux_block_is_zero_gated_conservative_and_trainable():
    torch.manual_seed(211)
    block = DGInterfaceFluxResidual2d(6, rank=4)
    value = torch.randn(2, 6, 11, 13)
    velocity = torch.full((2, 11, 13), 2500.0)
    velocity[:, :, 6:] = 4500.0
    maps = acoustic_interface_reflection_maps(
        _conditioning(velocity), cpml_margin=1
    )

    assert torch.count_nonzero(block(value, maps)) == 0
    block(value, maps).square().sum().backward()
    assert block.scale.grad is not None
    block.zero_grad(set_to_none=True)
    block.scale.data.fill_(0.2)
    output = block(value, maps)
    output.square().mean().backward()

    assert output.abs().max() > 0.0
    torch.testing.assert_close(
        output.sum(dim=(-2, -1)),
        torch.zeros_like(output.sum(dim=(-2, -1))),
        atol=2.0e-6,
        rtol=0.0,
    )
    assert block.channel_in.weight.grad is not None
    assert torch.count_nonzero(block.channel_in.weight.grad) > 0
