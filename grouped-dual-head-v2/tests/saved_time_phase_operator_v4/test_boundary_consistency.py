from __future__ import annotations

import torch
from torch import nn

from saved_time_phase_operator_v4.boundary_consistency import (
    HardFreeSurfaceWrapper,
    crop_pressure_halo,
    free_surface_violation,
    odd_top_three_side_halo,
    project_pressure_free_surface,
)


def test_hard_projection_never_increases_error_for_zero_pressure_surface():
    prediction = torch.randn(2, 6, 1, 9, 8)
    target = torch.randn_like(prediction)
    target[..., 0, :] = 0.0
    projected = project_pressure_free_surface(prediction)
    assert torch.equal(projected[..., 1:, :], prediction[..., 1:, :])
    assert torch.count_nonzero(projected[..., 0, :]) == 0
    assert (projected - target).norm() <= (prediction - target).norm()


def test_generator_matched_halo_is_odd_at_top_and_zero_on_three_sides():
    field = torch.randn(1, 1, 9, 8)
    halo = odd_top_three_side_halo(field, radius=3)
    assert halo.shape[-2:] == (15, 14)
    cropped = crop_pressure_halo(halo, radius=3)
    assert cropped[..., 0, :].abs().max() == 0.0
    assert torch.equal(cropped[..., 1:, :], field[..., 1:, :])
    assert torch.equal(halo[..., :3, 3:-3], -torch.flip(cropped[..., 1:4, :], dims=(-2,)))
    assert torch.count_nonzero(halo[..., :, :3]) == 0
    assert torch.count_nonzero(halo[..., :, -3:]) == 0
    assert torch.count_nonzero(halo[..., -3:, :]) == 0


class _IdentityParent(nn.Module):
    def forward(self, value):
        return value

    def forward_anchored(self, value):
        return value


def test_wrapper_projects_free_and_anchored_outputs():
    wrapper = HardFreeSurfaceWrapper(_IdentityParent())
    value = torch.ones(1, 4, 1, 7, 6)
    for output in (wrapper(value), wrapper.forward_anchored(value)):
        assert output[..., 0, :].abs().max() == 0.0
        assert torch.equal(output[..., 1:, :], value[..., 1:, :])
        violation = free_surface_violation(output)
        assert violation["maximum_absolute"] == 0.0
        assert violation["rms"] == 0.0
