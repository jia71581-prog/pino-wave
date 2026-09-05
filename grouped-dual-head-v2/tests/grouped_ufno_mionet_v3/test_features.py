from __future__ import annotations

import math

import torch

from grouped_ufno_mionet_v3.model.features import (
    PhaseAlignedCoordinateEncoder,
    SineLayer,
    build_phase_features,
)
from grouped_ufno_mionet_v3.model.travel_time import RayTravelTime


def _inputs():
    coords = torch.tensor(
        [[[100.0, 200.0, 0.05], [300.0, 400.0, 0.35], [500.0, 600.0, 0.7]]]
    )
    source = torch.tensor([[100.0, 200.0, 10.0, 0.1, 1.0]])
    seconds = torch.tensor([[0.0, 0.2, 0.4]])
    travel = RayTravelTime(
        seconds=seconds,
        distance_m=torch.tensor([[0.0, math.sqrt(80000.0), math.sqrt(320000.0)]]),
        path_velocity_mps=torch.full((1, 3), 2000.0),
        endpoint_velocity_mps=torch.full((1, 3), 2000.0),
        mean_slowness_s_per_m=torch.full((1, 3), 1.0 / 2000.0),
    )
    return coords, source, travel


def test_phase_features_include_tau_causality_phase_and_bounded_gabor():
    coords, source, travel = _inputs()
    bundle = build_phase_features(
        coords,
        source,
        travel,
        domain_x_m=2000.0,
        domain_z_m=2000.0,
        domain_t_s=1.0,
        fourier_bands=3,
        gabor_scales_s=(0.02, 0.05),
    )
    expected_tau = coords[..., 2] - source[:, None, 3] - travel.seconds
    torch.testing.assert_close(bundle.tau_s, expected_tau)
    torch.testing.assert_close(bundle.phase_rad, 2.0 * math.pi * 10.0 * expected_tau)
    assert torch.all(bundle.causal_feature >= 0) and torch.all(bundle.causal_feature <= 1)
    assert bundle.causal_feature[0, 0] < 0.5
    assert bundle.causal_feature[0, 2] > 0.5
    assert bundle.gabor.abs().max() <= 1.0 + 1.0e-6
    assert torch.isfinite(bundle.raw).all()


def test_sine_layer_uses_siren_initialization_bounds():
    torch.manual_seed(11)
    first = SineLayer(12, 16, first=True, omega0=30.0)
    hidden = SineLayer(16, 16, first=False, omega0=30.0)
    assert first.linear.weight.abs().max() <= 1.0 / 12.0 + 1.0e-7
    hidden_bound = math.sqrt(6.0 / 16.0) / 30.0
    assert hidden.linear.weight.abs().max() <= hidden_bound + 1.0e-7


def test_coordinate_encoder_returns_rank_features_and_raw_residual_gradient():
    coords, source, travel = _inputs()
    coords = coords.clone().requires_grad_()
    encoder = PhaseAlignedCoordinateEncoder(
        width=32,
        rank=12,
        fourier_bands=3,
        gabor_scales_s=(0.02, 0.05),
    )
    rank, bundle = encoder(
        coords,
        source,
        travel,
        domain_x_m=2000.0,
        domain_z_m=2000.0,
        domain_t_s=1.0,
    )
    assert rank.shape == (1, 3, 12)
    assert bundle.raw.shape[:2] == (1, 3)
    rank.square().mean().backward()
    assert coords.grad is not None and coords.grad.abs().sum() > 0
    assert encoder.raw_projection.weight.grad is not None
    assert encoder.periodic_out.weight.grad is not None
