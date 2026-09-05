from __future__ import annotations

import torch

from saved_time_phase_operator_v4.wfp import (
    BackgroundFrequencyOperator,
    ComplexFrequencyWindowConv2d,
)


def test_frequency_window_has_full_support_and_preserves_shape():
    torch.manual_seed(7)
    layer = ComplexFrequencyWindowConv2d(8, rank=4, radius=2)
    value = torch.randn(2, 8, 33, 35, requires_grad=True)
    output = layer(value)
    assert output.shape == value.shape
    output.square().mean().backward()
    assert layer.weight_real.grad is not None
    assert torch.count_nonzero(layer.weight_real.grad) > 0


def test_background_operator_has_separate_physical_and_cpml_heads():
    model = BackgroundFrequencyOperator(
        medium_channels=12,
        source_channels=5,
        width=16,
        rank=8,
        depth=2,
        radii=(1, 2),
        physical_shape=(21, 21),
        cpml_layers=4,
    )
    medium = torch.randn(2, 12, 25, 29)
    source = torch.randn(2, 5, 25, 29)
    scalars = torch.randn(2, 5)
    physical, cpml = model(medium, source, scalars)
    assert physical.shape == (2, 2, 21, 21)
    assert cpml.shape == (2, 10, 25, 29)
    assert torch.count_nonzero(physical[..., 0, :]) == 0
    assert torch.count_nonzero(cpml[:, :2, 0, :]) == 0
    assert torch.count_nonzero(cpml[:, :2, -1, :]) == 0
    assert torch.count_nonzero(cpml[:, :2, :, 0]) == 0
    assert torch.count_nonzero(cpml[:, :2, :, -1]) == 0


def test_fno_control_uses_identical_input_output_contract():
    model = BackgroundFrequencyOperator(
        medium_channels=12,
        source_channels=5,
        width=16,
        rank=8,
        depth=2,
        arm="fno",
        radii=(1, 2),
        fno_modes=8,
        physical_shape=(21, 21),
        cpml_layers=4,
    )
    physical, cpml = model(
        torch.randn(1, 12, 25, 29),
        torch.randn(1, 5, 25, 29),
        torch.randn(1, 5),
    )
    assert physical.shape == (1, 2, 21, 21)
    assert cpml.shape == (1, 10, 25, 29)
