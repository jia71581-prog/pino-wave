from __future__ import annotations

import torch

from saved_time_phase_operator_v4.instance_adaptation.b2_fe_weak_adapter import (
    apply_decoder_channel_scales,
)
from saved_time_phase_operator_v4.instance_adaptation.fe_weak_residual import (
    q1_acoustic_weak_residual,
    q1_element_matrices,
)


def test_q1_matrices_are_symmetric_and_mass_is_positive():
    mass, stiffness = q1_element_matrices(10.0, 10.0)
    assert torch.allclose(mass, mass.T)
    assert torch.allclose(stiffness, stiffness.T)
    assert torch.linalg.eigvalsh(mass).min() > 0.0
    assert torch.linalg.eigvalsh(stiffness).min() > -1.0e-12


def test_constant_wavefield_has_zero_interior_weak_residual():
    pressure = torch.ones(1, 8, 21, 21, dtype=torch.float64)
    velocity = torch.full((1, 21, 21), 2000.0, dtype=torch.float64)
    residual = q1_acoustic_weak_residual(
        pressure,
        velocity,
        dt_s=0.0025,
        dx_m=10.0,
        dz_m=10.0,
        coarsen=1,
        source_off_frame=2,
        cpml_margin_fine=2,
    )
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1.0e-12)


def test_linear_spatial_field_cancels_at_interior_q1_nodes():
    z = torch.arange(21, dtype=torch.float64)[:, None]
    x = torch.arange(21, dtype=torch.float64)[None, :]
    frame = 2.0 * x + 3.0 * z
    pressure = frame[None, None].expand(1, 8, 21, 21).clone()
    velocity = torch.full((1, 21, 21), 2000.0, dtype=torch.float64)
    residual = q1_acoustic_weak_residual(
        pressure,
        velocity,
        dt_s=0.0025,
        dx_m=10.0,
        dz_m=10.0,
        coarsen=1,
        source_off_frame=2,
        cpml_margin_fine=2,
    )
    assert residual.abs().max() < 1.0e-10


def test_zero_coefficients_are_exact_parent_rollback():
    parent = torch.randn(1, 5, 1, 8, 8)
    responses = torch.randn(1, 5, 4, 8, 8)
    candidate = apply_decoder_channel_scales(
        parent, responses, torch.zeros(4)
    )
    assert torch.equal(candidate, parent)
