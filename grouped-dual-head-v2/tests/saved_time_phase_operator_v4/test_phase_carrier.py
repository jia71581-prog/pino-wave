from __future__ import annotations

import math

import torch

from saved_time_phase_operator_v4.phase_carrier import (
    rotate_complex_pairs,
    travel_phase_carrier,
)


def test_phase_carrier_has_unit_modulus_and_expected_quarter_turn():
    travel = torch.tensor([[[0.0, 0.25]]])
    carrier = travel_phase_carrier(travel, torch.tensor([1.0]))
    torch.testing.assert_close(carrier.square().sum(dim=1), torch.ones(1, 1, 2))
    torch.testing.assert_close(carrier[0, :, 0, 0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(
        carrier[0, :, 0, 1], torch.tensor([0.0, -1.0]), atol=1.0e-6, rtol=0.0
    )


def test_complex_pair_rotation_matches_complex_multiplication():
    value = torch.tensor([[[[2.0]], [[3.0]], [[-1.0]], [[4.0]]]])
    carrier = torch.tensor([[[[0.0]], [[-1.0]]]])
    rotated = rotate_complex_pairs(value, carrier)
    expected = torch.tensor([[[[3.0]], [[-2.0]], [[4.0]], [[1.0]]]])
    torch.testing.assert_close(rotated, expected)


def test_rotation_preserves_pair_energy_and_gradients():
    torch.manual_seed(4)
    value = torch.randn(2, 10, 7, 9, requires_grad=True)
    carrier = travel_phase_carrier(torch.rand(2, 7, 9), torch.tensor([13.0, 29.0]))
    rotated = rotate_complex_pairs(value, carrier)
    torch.testing.assert_close(
        rotated.square().sum(dim=1), value.square().sum(dim=1), atol=2.0e-5, rtol=2.0e-5
    )
    rotated.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
