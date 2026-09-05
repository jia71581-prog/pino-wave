"""Reduced CHONKNORIS primitives for causal wavefield adaptation.

The full wavefield contains millions of degrees of freedom, so forming the
dense Gauss--Newton matrix used by the reference CHONKNORIS implementation is
not viable here.  The deployment adapter, however, has only ``latent_dim + 1``
state variables (the latent shift and residual gate).  This module applies the
same Tikhonov-regularized residual iteration in that reduced space:

    (J.T @ J / m + lambda I) delta = -J.T @ r / m.

Offline meta-training can regress the lower Cholesky factor of this small SPD
matrix.  Online adaptation then needs one residual-gradient evaluation per
iteration; an exact reduced Cholesky is available as a guarded fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


TensorResidual = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ReducedChonknorisConfig:
    """Controls the contractive reduced Newton--Kantorovich iteration."""

    iterations: int = 4
    initial_relaxation: float = 1.0e-2
    initial_step_size: float = 1.0
    relaxation_factors: tuple[float, ...] = (0.5, 1.0, 2.0)
    step_factors: tuple[float, ...] = (0.5, 1.0, 2.0)
    minimum_relaxation: float = 1.0e-6
    maximum_relaxation: float = 1.0e2
    minimum_step_size: float = 1.0e-3
    maximum_step_size: float = 2.0
    residual_tolerance: float = 1.0e-8
    minimum_relative_improvement: float = 1.0e-6
    maximum_condition_number: float = 1.0e8
    exact_factor_fallback: bool = True
    residual_pool_size: int = 4
    physics_point_count: int = 128

    def __post_init__(self) -> None:
        positive = (
            self.initial_relaxation,
            self.initial_step_size,
            self.minimum_relaxation,
            self.maximum_relaxation,
            self.minimum_step_size,
            self.maximum_step_size,
            self.residual_tolerance,
            self.maximum_condition_number,
        )
        if self.iterations < 0 or any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("CHONKNORIS iteration configuration must be finite and positive")
        if self.residual_pool_size <= 0 or self.physics_point_count <= 0:
            raise ValueError("CHONKNORIS residual sizes must be positive")
        if self.minimum_relaxation > self.maximum_relaxation:
            raise ValueError("minimum relaxation exceeds maximum relaxation")
        if self.minimum_step_size > self.maximum_step_size:
            raise ValueError("minimum step size exceeds maximum step size")
        if not 0.0 <= self.minimum_relative_improvement < 1.0:
            raise ValueError("minimum relative improvement must lie in [0,1)")
        if not self.relaxation_factors or not self.step_factors:
            raise ValueError("CHONKNORIS search factors cannot be empty")
        if any(not math.isfinite(value) or value <= 0 for value in self.relaxation_factors):
            raise ValueError("relaxation factors must be finite and positive")
        if any(not math.isfinite(value) or value <= 0 for value in self.step_factors):
            raise ValueError("step factors must be finite and positive")


@dataclass(frozen=True)
class CholeskySupervision:
    """Detached exact target and differentiable learned-factor loss."""

    loss: torch.Tensor
    factor_relative_error: torch.Tensor
    operator_relative_error: torch.Tensor
    condition_number: float
    relaxation_count: int


@dataclass(frozen=True)
class ReducedChonknorisResult:
    """Auditable state and contraction history for one online solve."""

    state: torch.Tensor
    accepted_iterations: int
    residual_history: tuple[float, ...]
    relaxation_history: tuple[float, ...]
    step_size_history: tuple[float, ...]
    contraction_history: tuple[float, ...]
    condition_history: tuple[float, ...]
    learned_factor_steps: int
    exact_factor_fallbacks: int
    stopped_reason: str


class ReducedCholeskyPredictor(nn.Module):
    """Predict a positive-diagonal lower Cholesky factor in reduced state space."""

    def __init__(
        self,
        *,
        context_dim: int,
        state_dim: int,
        hidden_dim: int = 64,
        minimum_diagonal: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if min(int(context_dim), int(state_dim), int(hidden_dim)) <= 0:
            raise ValueError("reduced Cholesky dimensions must be positive")
        if not math.isfinite(float(minimum_diagonal)) or float(minimum_diagonal) <= 0.0:
            raise ValueError("minimum Cholesky diagonal must be finite and positive")
        self.context_dim = int(context_dim)
        self.state_dim = int(state_dim)
        self.minimum_diagonal = float(minimum_diagonal)
        self.tril_size = self.state_dim * (self.state_dim + 1) // 2
        self.network = nn.Sequential(
            nn.Linear(self.context_dim + 3, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.tril_size + 1),
        )
        # Start from a finite isotropic SPD factor.  Offline Cholesky
        # supervision moves this head toward problem-conditioned factors.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        indices = torch.tril_indices(self.state_dim, self.state_dim)
        self.register_buffer("tril_rows", indices[0], persistent=False)
        self.register_buffer("tril_cols", indices[1], persistent=False)

    def forward(
        self,
        context: torch.Tensor,
        relaxation: torch.Tensor | float,
        residual_stats: torch.Tensor,
    ) -> torch.Tensor:
        value = torch.as_tensor(context)
        squeeze = value.ndim == 1
        if squeeze:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[1] != self.context_dim:
            raise ValueError("Cholesky context must be [batch,context_dim]")
        stats = torch.as_tensor(residual_stats, dtype=value.dtype, device=value.device)
        if stats.ndim == 1:
            stats = stats.unsqueeze(0)
        if stats.shape != (value.shape[0], 2):
            raise ValueError("residual statistics must be [batch,2]")
        lam = torch.as_tensor(relaxation, dtype=value.dtype, device=value.device)
        if lam.ndim == 0:
            lam = lam.expand(value.shape[0])
        if lam.shape != (value.shape[0],) or torch.any(lam <= 0.0):
            raise ValueError("relaxation must contain one positive value per batch")
        features = torch.cat((value, lam.log10()[:, None], stats), dim=-1)
        packed = self.network(features)
        raw_factor, log_scale = packed[:, :-1], packed[:, -1]
        factor = value.new_zeros((value.shape[0], self.state_dim, self.state_dim))
        factor[:, self.tril_rows, self.tril_cols] = raw_factor
        diagonal = torch.arange(self.state_dim, device=value.device)
        # Zero network output is the Tikhonov/gradient-descent limit
        # sqrt(lambda) I, rather than an arbitrary unit-scale factor.  This is
        # already the exact target when the zero-initialized residual head has
        # negligible Jacobian at the start of offline meta-training.
        relaxation_scale = lam.sqrt()[:, None]
        factor = factor * relaxation_scale[:, :, None]
        softplus_zero = math.log(2.0)
        positive_diagonal = (
            relaxation_scale
            * F.softplus(raw_factor[:, self.tril_rows == self.tril_cols])
            / softplus_zero
            + self.minimum_diagonal
        )
        factor[:, diagonal, diagonal] = positive_diagonal
        factor = factor * torch.exp(log_scale.clamp(-12.0, 12.0))[:, None, None]
        return factor[0] if squeeze else factor


def residual_statistics(residual: torch.Tensor) -> torch.Tensor:
    """Return stable log-RMS and log-maximum summaries for factor conditioning."""

    value = torch.as_tensor(residual)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[1] == 0:
        raise ValueError("residual must be [batch,residual_dim]")
    tiny = torch.finfo(value.dtype).tiny
    rms = value.square().mean(dim=-1).sqrt().clamp_min(tiny).log10()
    maximum = value.abs().amax(dim=-1).clamp_min(tiny).log10()
    return torch.stack((rms, maximum), dim=-1)


def _linearize(
    residual_fn: TensorResidual,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a residual and its forward-mode Jacobian with respect to state."""

    point = torch.as_tensor(state)
    if point.ndim != 1 or point.numel() == 0:
        raise ValueError("reduced CHONKNORIS state must be a non-empty vector")
    residual = residual_fn(point)
    if residual.ndim != 1 or residual.numel() == 0:
        raise ValueError("reduced residual function must return a non-empty vector")
    jacobian = torch.func.jacfwd(residual_fn)(point)
    if jacobian.shape != (residual.numel(), point.numel()):
        raise ValueError("reduced residual Jacobian has an invalid shape")
    if not torch.isfinite(residual).all() or not torch.isfinite(jacobian).all():
        raise FloatingPointError("nonfinite reduced residual linearization")
    return residual, jacobian


def _normal_matrix(
    jacobian: torch.Tensor,
    relaxation: float,
) -> torch.Tensor:
    rows = max(int(jacobian.shape[0]), 1)
    work = jacobian.to(torch.float64)
    identity = torch.eye(work.shape[1], dtype=work.dtype, device=work.device)
    return work.mT @ work / float(rows) + float(relaxation) * identity


def _exact_factor(
    jacobian: torch.Tensor,
    relaxation: float,
) -> tuple[torch.Tensor, float]:
    normal = _normal_matrix(jacobian, relaxation)
    factor, info = torch.linalg.cholesky_ex(normal, upper=False)
    if int(info.max()) != 0 or not torch.isfinite(factor).all():
        raise FloatingPointError("exact reduced Cholesky factorization failed")
    condition = float(torch.linalg.cond(normal))
    return factor, condition


def supervise_cholesky_factor(
    predictor: ReducedCholeskyPredictor,
    residual_fn: TensorResidual,
    state: torch.Tensor,
    context: torch.Tensor,
    relaxations: Sequence[float],
) -> CholeskySupervision:
    """Regress exact reduced Hessian factors along an offline flow trajectory.

    The Jacobian and exact targets are detached.  Gradients therefore update
    only the learned factor model (and, if desired, its context encoder), without
    introducing second-order derivatives through the wavefield network.
    """

    residual, jacobian = _linearize(residual_fn, state.detach())
    return supervise_cholesky_linearizations(
        predictor,
        residual.unsqueeze(0),
        jacobian.unsqueeze(0),
        context.unsqueeze(0) if context.ndim == 1 else context,
        relaxations,
    )


def supervise_cholesky_linearizations(
    predictor: ReducedCholeskyPredictor,
    residuals: torch.Tensor,
    jacobians: torch.Tensor,
    contexts: torch.Tensor,
    relaxations: Sequence[float],
) -> CholeskySupervision:
    """Supervise a batch of independent reduced residual linearizations."""

    residual = torch.as_tensor(residuals)
    jacobian = torch.as_tensor(
        jacobians, dtype=residual.dtype, device=residual.device
    )
    context = torch.as_tensor(
        contexts, dtype=residual.dtype, device=residual.device
    )
    if residual.ndim != 2 or residual.shape[1] == 0:
        raise ValueError("batched residuals must be [batch,residual_dim]")
    if jacobian.ndim != 3 or jacobian.shape[:2] != residual.shape:
        raise ValueError("batched Jacobians must be [batch,residual_dim,state_dim]")
    if context.ndim != 2 or context.shape[0] != residual.shape[0]:
        raise ValueError("batched contexts must be [batch,context_dim]")
    if jacobian.shape[2] != predictor.state_dim:
        raise ValueError("batched Jacobian state dimension is incompatible")
    if context.shape[1] != predictor.context_dim:
        raise ValueError("batched context dimension is incompatible")
    if not (
        torch.isfinite(residual).all()
        and torch.isfinite(jacobian).all()
        and torch.isfinite(context).all()
    ):
        raise FloatingPointError("batched Cholesky supervision inputs are nonfinite")

    stats = residual_statistics(residual.detach())
    factor_terms: list[torch.Tensor] = []
    operator_terms: list[torch.Tensor] = []
    condition_tensors: list[torch.Tensor] = []
    work = jacobian.detach().to(torch.float64)
    identity = torch.eye(
        work.shape[-1], dtype=work.dtype, device=work.device
    ).unsqueeze(0)
    normal_base = work.transpose(-1, -2) @ work / float(max(work.shape[1], 1))
    for relaxation in tuple(float(value) for value in relaxations):
        if not math.isfinite(relaxation) or relaxation <= 0.0:
            raise ValueError("factor-supervision relaxations must be finite and positive")
        normal = normal_base + float(relaxation) * identity
        target, info = torch.linalg.cholesky_ex(normal, upper=False)
        if int(info.max()) != 0 or not torch.isfinite(target).all():
            raise FloatingPointError("batched exact Cholesky factorization failed")
        condition = torch.linalg.cond(normal)
        target = target.to(dtype=context.dtype, device=context.device).detach()
        predicted = predictor(context, relaxation, stats.to(context))
        factor_denominator = target.square().mean(dim=(-2, -1)).clamp_min(1.0e-12)
        factor_terms.append(
            ((predicted - target).square().mean(dim=(-2, -1)) / factor_denominator).mean()
        )
        target_operator = target @ target.mT
        predicted_operator = predicted @ predicted.mT
        operator_denominator = target_operator.square().mean(dim=(-2, -1)).clamp_min(1.0e-12)
        operator_terms.append(
            (
                (predicted_operator - target_operator).square().mean(dim=(-2, -1))
                / operator_denominator
            ).mean()
        )
        condition_tensors.append(condition)
    factor_error = torch.stack(factor_terms).mean()
    operator_error = torch.stack(operator_terms).mean()
    return CholeskySupervision(
        loss=factor_error + operator_error,
        factor_relative_error=factor_error,
        operator_relative_error=operator_error,
        condition_number=float(torch.stack(condition_tensors).amax()),
        relaxation_count=len(condition_tensors),
    )


def _search_values(
    center: float,
    factors: Sequence[float],
    lower: float,
    upper: float,
) -> tuple[float, ...]:
    return tuple(
        sorted(
            {
                min(max(float(center) * float(factor), float(lower)), float(upper))
                for factor in factors
            }
        )
    )


def _solve_factor(factor: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    work_factor = factor.to(torch.float64)
    work_gradient = gradient.to(torch.float64)
    delta = torch.cholesky_solve((-work_gradient)[:, None], work_factor, upper=False)[:, 0]
    return delta.to(dtype=gradient.dtype, device=gradient.device)


def reduced_chonknoris_solve(
    residual_fn: TensorResidual,
    initial_state: torch.Tensor,
    *,
    config: ReducedChonknorisConfig = ReducedChonknorisConfig(),
    factor_predictor: ReducedCholeskyPredictor | None = None,
    context: torch.Tensor | None = None,
) -> ReducedChonknorisResult:
    """Run a residual-monotone reduced CHONKNORIS iteration.

    Learned factors are attempted first.  If all learned-factor candidates fail
    the residual-decrease gate, the exact small Cholesky factor is tried when
    ``exact_factor_fallback`` is enabled.  No state update survives unless the
    same residual objective contracts.
    """

    state = torch.as_tensor(initial_state).detach().clone()
    if state.ndim != 1 or state.numel() == 0 or not torch.isfinite(state).all():
        raise ValueError("initial reduced state must be a finite non-empty vector")
    if (factor_predictor is None) != (context is None):
        raise ValueError("learned-factor prediction requires both predictor and context")
    if factor_predictor is not None and factor_predictor.state_dim != state.numel():
        raise ValueError("learned factor state dimension does not match deployment state")

    with torch.no_grad():
        initial_residual = residual_fn(state)
        if initial_residual.ndim != 1 or not torch.isfinite(initial_residual).all():
            raise FloatingPointError("initial CHONKNORIS residual is invalid")
        initial_objective = float(0.5 * initial_residual.square().mean())
    residual_history = [initial_objective]
    relaxation_history: list[float] = []
    step_size_history: list[float] = []
    contraction_history: list[float] = []
    condition_history: list[float] = []
    relaxation = float(config.initial_relaxation)
    step_size = float(config.initial_step_size)
    learned_steps = 0
    exact_fallbacks = 0
    stopped_reason = "iteration_budget"

    for _ in range(int(config.iterations)):
        current_objective = residual_history[-1]
        if current_objective <= float(config.residual_tolerance) ** 2:
            stopped_reason = "residual_tolerance"
            break
        point = state.detach().requires_grad_(True)
        residual = residual_fn(point)
        if residual.ndim != 1 or residual.numel() == 0 or not torch.isfinite(residual).all():
            stopped_reason = "nonfinite_residual"
            break
        objective = 0.5 * residual.square().mean()
        gradient = torch.autograd.grad(objective, point, create_graph=False)[0].detach()
        if not torch.isfinite(gradient).all():
            stopped_reason = "nonfinite_gradient"
            break
        relaxation_options = _search_values(
            relaxation,
            config.relaxation_factors,
            config.minimum_relaxation,
            config.maximum_relaxation,
        )
        step_options = _search_values(
            step_size,
            config.step_factors,
            config.minimum_step_size,
            config.maximum_step_size,
        )
        best: tuple[float, torch.Tensor, float, float, float, bool] | None = None

        def consider_factor(
            factor: torch.Tensor,
            *,
            candidate_relaxation: float,
            condition: float,
            learned: bool,
        ) -> None:
            nonlocal best
            if not math.isfinite(condition) or condition > float(config.maximum_condition_number):
                return
            try:
                delta = _solve_factor(factor, gradient)
            except (RuntimeError, FloatingPointError):
                return
            if not torch.isfinite(delta).all():
                return
            for candidate_step in step_options:
                candidate_state = state + float(candidate_step) * delta
                with torch.no_grad():
                    candidate_residual = residual_fn(candidate_state)
                    if not torch.isfinite(candidate_residual).all():
                        continue
                    candidate_objective = float(0.5 * candidate_residual.square().mean())
                if best is None or candidate_objective < best[0]:
                    best = (
                        candidate_objective,
                        candidate_state.detach(),
                        float(candidate_relaxation),
                        float(candidate_step),
                        float(condition),
                        bool(learned),
                    )

        if factor_predictor is not None and context is not None:
            stats = residual_statistics(residual.detach())[0].to(context)
            for candidate_relaxation in relaxation_options:
                predicted_factor = factor_predictor(
                    context.detach(), candidate_relaxation, stats
                )
                predicted_operator = predicted_factor @ predicted_factor.mT
                condition = float(torch.linalg.cond(predicted_operator.detach().to(torch.float64)))
                consider_factor(
                    predicted_factor,
                    candidate_relaxation=candidate_relaxation,
                    condition=condition,
                    learned=True,
                )

        improvement_gate = current_objective * (
            1.0 - float(config.minimum_relative_improvement)
        )
        learned_succeeded = best is not None and best[0] < improvement_gate
        if not learned_succeeded and config.exact_factor_fallback:
            exact_fallbacks += int(factor_predictor is not None)
            try:
                _, jacobian = _linearize(residual_fn, state.detach())
                for candidate_relaxation in relaxation_options:
                    exact_factor, condition = _exact_factor(jacobian, candidate_relaxation)
                    consider_factor(
                        exact_factor,
                        candidate_relaxation=candidate_relaxation,
                        condition=condition,
                        learned=False,
                    )
            except (RuntimeError, FloatingPointError):
                pass

        if best is None or best[0] >= improvement_gate:
            stopped_reason = "no_contracting_step"
            break
        candidate_objective, state, relaxation, step_size, condition, learned = best
        residual_history.append(candidate_objective)
        relaxation_history.append(relaxation)
        step_size_history.append(step_size)
        contraction_history.append(
            candidate_objective / max(current_objective, torch.finfo(torch.float64).tiny)
        )
        condition_history.append(condition)
        learned_steps += int(learned)

    return ReducedChonknorisResult(
        state=state.detach(),
        accepted_iterations=len(relaxation_history),
        residual_history=tuple(residual_history),
        relaxation_history=tuple(relaxation_history),
        step_size_history=tuple(step_size_history),
        contraction_history=tuple(contraction_history),
        condition_history=tuple(condition_history),
        learned_factor_steps=learned_steps,
        exact_factor_fallbacks=exact_fallbacks,
        stopped_reason=stopped_reason,
    )


__all__ = [
    "CholeskySupervision",
    "ReducedCholeskyPredictor",
    "ReducedChonknorisConfig",
    "ReducedChonknorisResult",
    "reduced_chonknoris_solve",
    "residual_statistics",
    "supervise_cholesky_factor",
    "supervise_cholesky_linearizations",
]
