from __future__ import annotations

import torch

from saved_time_phase_operator_v4.instance_adaptation.chonknoris import (
    ReducedCholeskyPredictor,
    ReducedChonknorisConfig,
    reduced_chonknoris_solve,
    supervise_cholesky_factor,
    supervise_cholesky_linearizations,
)


def _linear_problem():
    matrix = torch.tensor(
        [[3.0, 0.5], [0.0, 2.0], [1.0, -1.0]], dtype=torch.float32
    )
    target = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)

    def residual(state):
        return matrix @ state - target

    return residual


def test_predicted_factor_is_lower_triangular_with_positive_diagonal():
    predictor = ReducedCholeskyPredictor(context_dim=4, state_dim=3, hidden_dim=8)
    factor = predictor(torch.zeros(4), 1.0e-2, torch.zeros(2))
    assert factor.shape == (3, 3)
    assert torch.equal(factor, torch.tril(factor))
    assert bool((factor.diagonal() > 0.0).all())
    assert torch.linalg.cholesky_ex(factor @ factor.mT).info.item() == 0


def test_exact_reduced_chonknoris_is_residual_monotone():
    result = reduced_chonknoris_solve(
        _linear_problem(),
        torch.zeros(2),
        config=ReducedChonknorisConfig(
            iterations=4,
            initial_relaxation=1.0e-3,
            minimum_relative_improvement=1.0e-8,
        ),
    )
    assert result.accepted_iterations >= 1
    assert all(
        after < before
        for before, after in zip(
            result.residual_history, result.residual_history[1:]
        )
    )
    assert all(value < 1.0 for value in result.contraction_history)
    # The three-equation/two-state system is inconsistent, so its least-squares
    # optimum has a non-zero residual; the solver should still remove most error.
    assert result.residual_history[-1] < result.residual_history[0] * 0.2


def test_minimum_relative_improvement_rejects_negligible_exact_step():
    result = reduced_chonknoris_solve(
        _linear_problem(),
        torch.zeros(2),
        config=ReducedChonknorisConfig(
            iterations=1,
            initial_relaxation=1.0e6,
            relaxation_factors=(1.0,),
            minimum_relaxation=1.0e6,
            maximum_relaxation=1.0e6,
            initial_step_size=1.0,
            step_factors=(1.0,),
            minimum_step_size=1.0,
            maximum_step_size=1.0,
            minimum_relative_improvement=0.02,
        ),
    )
    assert result.accepted_iterations == 0
    assert result.stopped_reason == "no_contracting_step"
    assert result.residual_history == (result.residual_history[0],)


def test_factor_supervision_is_finite_and_updates_predictor():
    predictor = ReducedCholeskyPredictor(context_dim=4, state_dim=2, hidden_dim=8)
    supervision = supervise_cholesky_factor(
        predictor,
        _linear_problem(),
        torch.zeros(2),
        torch.zeros(4),
        (1.0e-3, 1.0e-2),
    )
    assert torch.isfinite(supervision.loss)
    assert supervision.condition_number >= 1.0
    supervision.loss.backward()
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in predictor.parameters()
    )


def test_batched_factor_supervision_matches_individual_mean():
    predictor = ReducedCholeskyPredictor(context_dim=4, state_dim=2, hidden_dim=8)
    residual_fn = _linear_problem()
    states = (torch.zeros(2), torch.tensor([0.2, -0.1]))
    residuals = torch.stack([residual_fn(state) for state in states])
    jacobians = torch.stack(
        [torch.func.jacfwd(residual_fn)(state) for state in states]
    )
    contexts = torch.zeros(2, 4)
    relaxations = (1.0e-3, 1.0e-2)
    batched = supervise_cholesky_linearizations(
        predictor, residuals, jacobians, contexts, relaxations
    )
    individual = [
        supervise_cholesky_factor(
            predictor, residual_fn, state, contexts[index], relaxations
        )
        for index, state in enumerate(states)
    ]
    assert torch.allclose(
        batched.loss, torch.stack([item.loss for item in individual]).mean()
    )
    assert torch.allclose(
        batched.factor_relative_error,
        torch.stack([item.factor_relative_error for item in individual]).mean(),
    )
    assert torch.allclose(
        batched.operator_relative_error,
        torch.stack([item.operator_relative_error for item in individual]).mean(),
    )
    assert batched.condition_number == max(
        item.condition_number for item in individual
    )


def test_bad_learned_factor_falls_back_to_exact_cholesky():
    predictor = ReducedCholeskyPredictor(context_dim=4, state_dim=2, hidden_dim=8)
    with torch.no_grad():
        # A huge predicted factor produces a numerically negligible learned step.
        predictor.network[-1].bias[-1] = 12.0
    result = reduced_chonknoris_solve(
        _linear_problem(),
        torch.zeros(2),
        config=ReducedChonknorisConfig(
            iterations=2,
            minimum_relative_improvement=1.0e-4,
            exact_factor_fallback=True,
        ),
        factor_predictor=predictor,
        context=torch.zeros(4),
    )
    assert result.exact_factor_fallbacks >= 1
    assert result.accepted_iterations >= 1
    assert result.residual_history[-1] < result.residual_history[0]
