from __future__ import annotations

import inspect

import torch

from saved_time_phase_operator_v4.instance_adaptation.head_coefficient_prior import (
    CausalHeadCoefficientPrior,
    HeadCoefficientPriorConfig,
    solve_prior_centered_system,
)
from saved_time_phase_operator_v4.instance_adaptation.transfer_dg_adapt import (
    TransferDGConfig,
    TransferDGLinearSystem,
    solve_transfer_dg_system,
)


def _system(design: torch.Tensor, residual: torch.Tensor) -> TransferDGLinearSystem:
    return TransferDGLinearSystem(
        design=design.float(),
        residual=residual.float(),
        parent_objective=residual.float().square().sum(),
        observation_scale=torch.tensor(1.0),
        volume_scale=None,
        flux_scale=None,
    )


def _parent_basis(rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    parent = torch.ones(1, 2, 2, 2)
    basis = torch.zeros(rank, 1, 2, 2, 2)
    for index in range(rank):
        basis[index].flatten()[index % basis[index].numel()] = 1.0
    return parent, basis


def test_zero_initialized_prior_mean_and_bounded_outputs():
    torch.manual_seed(503)
    config = HeadCoefficientPriorConfig(
        input_dim=7,
        coefficient_dim=5,
        hidden_dim=11,
        precision_min=0.01,
        precision_max=2.0,
        maximum_trust_ratio=0.05,
        initial_trust_fraction=0.2,
    )
    model = CausalHeadCoefficientPrior(config)
    result = model(torch.randn(4, 7))
    assert torch.count_nonzero(result.mean) == 0
    assert bool((result.precision >= 0.01).all())
    assert bool((result.precision <= 2.0).all())
    torch.testing.assert_close(result.trust_ratio, torch.full((4,), 0.01))


def test_zero_precision_matches_uncentered_ridge_coefficients():
    system = _system(torch.tensor([[1.0]]), torch.tensor([-1.0]))
    parent, basis = _parent_basis(1)
    config = TransferDGConfig(ridge_weight=1.0e-4, trust_ratio=10.0)
    baseline = solve_transfer_dg_system(system, parent, basis, config=config)
    centered = solve_prior_centered_system(
        system,
        parent,
        basis,
        torch.zeros(1),
        torch.zeros(1),
        trust_ratio=10.0,
        ridge_weight=1.0e-4,
    )
    assert baseline.accepted and centered.accepted
    torch.testing.assert_close(centered.coefficients, baseline.coefficients)
    torch.testing.assert_close(centered.candidate, baseline.candidate)


def test_prior_resolves_an_online_design_null_direction():
    system = _system(torch.tensor([[1.0, 0.0]]), torch.tensor([-1.0]))
    parent, basis = _parent_basis(2)
    result = solve_prior_centered_system(
        system,
        parent,
        basis,
        torch.tensor([1.0, 2.0]),
        torch.tensor([0.0, 10.0]),
        trust_ratio=10.0,
        ridge_weight=0.0,
    )
    assert result.accepted
    torch.testing.assert_close(result.coefficients, torch.tensor([1.0, 2.0]))
    assert result.online_objective_after <= result.online_objective_before
    assert result.posterior_objective_after < result.posterior_objective_before


def test_trust_projection_caps_the_physical_correction():
    system = _system(torch.zeros(1, 1), torch.zeros(1))
    parent = torch.ones(1, 2, 2, 2)
    basis = parent[None].clone()
    result = solve_prior_centered_system(
        system,
        parent,
        basis,
        torch.ones(1),
        torch.full((1,), 100.0),
        trust_ratio=0.05,
        ridge_weight=0.0,
    )
    assert result.accepted
    assert result.correction_ratio <= 0.05 * (1.0 + 1.0e-6)


def test_online_regression_rolls_back_bit_exactly():
    system = _system(torch.ones(1, 1), torch.zeros(1))
    parent, basis = _parent_basis(1)
    result = solve_prior_centered_system(
        system,
        parent,
        basis,
        torch.ones(1),
        torch.full((1,), 100.0),
        trust_ratio=10.0,
        ridge_weight=0.0,
        online_absolute_tolerance=0.0,
    )
    assert not result.accepted
    assert torch.count_nonzero(result.coefficients) == 0
    assert torch.equal(result.candidate, parent)
    assert result.online_objective_after == result.online_objective_before


def test_deployment_solver_has_no_future_target_argument():
    parameters = inspect.signature(solve_prior_centered_system).parameters
    forbidden = {"future_truth", "future_target", "target", "oracle_coefficients"}
    assert forbidden.isdisjoint(parameters)
