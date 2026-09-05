from __future__ import annotations

import math

import pytest
import torch

from saved_time_phase_operator_v4.instance_adaptation.transfer_dg_adapt import (
    TransferDGConfig,
    TransferDGLinearSystem,
    build_transfer_dg_linear_system,
    complex_to_pairs,
    dg_normal_flux_jumps,
    hard_free_surface_pairs,
    helmholtz_volume_residual,
    irfft_observed_frames,
    pairs_to_complex,
    solve_transfer_dg_system,
    validate_two_onset_frames,
)


def test_complex_pair_roundtrip_and_free_surface_are_exact():
    torch.manual_seed(401)
    value = torch.randn(3, 2, 11, 13)
    recovered = complex_to_pairs(pairs_to_complex(value))
    torch.testing.assert_close(recovered, value)
    projected = hard_free_surface_pairs(value)
    assert torch.count_nonzero(projected[..., 0, :]) == 0
    assert torch.equal(projected[..., 1:, :], value[..., 1:, :])


def test_discrete_plane_wave_satisfies_matched_helmholtz_operator():
    height = width = 41
    dx = dz = 1.0
    velocity = torch.full((height, width), 2.0, dtype=torch.float64)
    discrete_phase = 0.31
    x = torch.arange(width, dtype=torch.float64)
    pressure = torch.exp(1j * discrete_phase * x)[None, None].expand(1, height, -1)
    discrete_wave_number = 2.0 * math.sin(discrete_phase / 2.0) / dx
    frequency = torch.tensor(
        [float(velocity[0, 0]) * discrete_wave_number / (2.0 * math.pi)],
        dtype=torch.float64,
    )
    residual = helmholtz_volume_residual(
        complex_to_pairs(pressure),
        velocity,
        frequency,
        dx_m=dx,
        dz_m=dz,
    )
    assert residual.abs().max() < 1.0e-10


def test_affine_field_has_zero_internal_one_sided_flux_jump():
    z = torch.arange(41, dtype=torch.float64)[:, None]
    x = torch.arange(41, dtype=torch.float64)[None, :]
    pressure = (2.0 * x + 3.0j * z)[None]
    velocity = torch.full((41, 41), 2500.0, dtype=torch.float64)
    flux_x, flux_z, weight_x, weight_z = dg_normal_flux_jumps(
        complex_to_pairs(pressure),
        velocity,
        dx_m=1.0,
        dz_m=1.0,
        element_intervals=20,
    )
    assert flux_x.abs().max() < 1.0e-12
    assert flux_z.abs().max() < 1.0e-12
    torch.testing.assert_close(weight_x, torch.full_like(weight_x, 0.05))
    torch.testing.assert_close(weight_z, torch.full_like(weight_z, 0.05))


def test_online_contract_accepts_only_two_adjacent_onset_frames():
    observed = torch.zeros(2, 21, 21)
    assert validate_two_onset_frames(observed, (3, 4), spatial_shape=(21, 21)) == (3, 4)
    with pytest.raises(ValueError, match="exactly two adjacent"):
        validate_two_onset_frames(observed, (3, 5), spatial_shape=(21, 21))
    with pytest.raises(ValueError, match="exactly two adjacent"):
        validate_two_onset_frames(torch.zeros(3, 21, 21), (2, 3, 4), spatial_shape=(21, 21))


def test_two_frame_linear_basis_solve_improves_objective_without_future_truth():
    torch.manual_seed(409)
    frequencies, height, width, rank, time_count = 4, 21, 21, 3, 17
    parent = hard_free_surface_pairs(torch.randn(frequencies, 2, height, width) * 0.1)
    basis = hard_free_surface_pairs(torch.randn(rank, frequencies, 2, height, width) * 0.02)
    true_coefficients = torch.tensor([0.20, -0.15, 0.10])
    target_coefficients = parent + torch.einsum(
        "k,kfczx->fczx", true_coefficients, basis
    )
    observed_indices = (2, 3)
    observed = irfft_observed_frames(
        target_coefficients,
        time_count=time_count,
        observed_indices=observed_indices,
    )
    config = TransferDGConfig(
        dx_m=10.0,
        dz_m=10.0,
        element_intervals=10,
        observed_weight=1.0,
        volume_weight=0.0,
        flux_weight=0.0,
        ridge_weight=1.0e-8,
        trust_ratio=1.0,
    )
    system = build_transfer_dg_linear_system(
        parent,
        basis,
        torch.full((height, width), 2500.0),
        torch.arange(frequencies, dtype=torch.float32),
        observed,
        observed_indices,
        time_count=time_count,
        config=config,
    )
    result = solve_transfer_dg_system(system, parent, basis, config=config)
    assert result.accepted
    assert result.future_truth_used is False
    assert result.candidate_objective < result.parent_objective
    torch.testing.assert_close(
        result.coefficients, true_coefficients, atol=3.0e-4, rtol=3.0e-4
    )


def test_no_improvement_rolls_back_bit_exactly_to_parent():
    torch.manual_seed(419)
    frequencies, height, width, rank, time_count = 3, 21, 21, 2, 15
    parent = hard_free_surface_pairs(torch.randn(frequencies, 2, height, width))
    basis = hard_free_surface_pairs(torch.randn(rank, frequencies, 2, height, width) * 0.01)
    observed_indices = (1, 2)
    observed = irfft_observed_frames(
        parent, time_count=time_count, observed_indices=observed_indices
    )
    config = TransferDGConfig(
        element_intervals=10,
        volume_weight=0.0,
        flux_weight=0.0,
        ridge_weight=1.0e-4,
    )
    system = build_transfer_dg_linear_system(
        parent,
        basis,
        torch.full((height, width), 2000.0),
        torch.arange(frequencies, dtype=torch.float32),
        observed,
        observed_indices,
        time_count=time_count,
        config=config,
    )
    result = solve_transfer_dg_system(system, parent, basis, config=config)
    assert not result.accepted
    assert torch.count_nonzero(result.coefficients) == 0
    assert torch.equal(result.candidate, parent)


def test_reduced_solve_enforces_registered_correction_trust_ratio():
    parent = torch.ones(1, 2, 3, 3)
    basis = torch.ones(1, 1, 2, 3, 3)
    system = TransferDGLinearSystem(
        design=torch.ones(1, 1),
        residual=torch.tensor([-1.0]),
        parent_objective=torch.tensor(1.0),
        observation_scale=torch.tensor(1.0),
        volume_scale=None,
        flux_scale=None,
    )
    config = TransferDGConfig(
        element_intervals=1,
        ridge_weight=1.0e-8,
        trust_ratio=0.1,
    )
    result = solve_transfer_dg_system(system, parent, basis, config=config)
    assert result.accepted
    assert result.correction_ratio <= 0.1 * (1.0 + 1.0e-6)
