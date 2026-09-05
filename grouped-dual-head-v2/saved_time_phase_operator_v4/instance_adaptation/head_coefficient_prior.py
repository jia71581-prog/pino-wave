"""Causal coefficient priors and prior-centered convex instance adaptation."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .transfer_dg_adapt import TransferDGLinearSystem


@dataclass(frozen=True)
class HeadCoefficientPriorConfig:
    input_dim: int
    coefficient_dim: int = 64
    hidden_dim: int = 128
    precision_min: float = 1.0e-4
    precision_max: float = 10.0
    maximum_trust_ratio: float = 0.05
    initial_trust_fraction: float = 0.2

    def __post_init__(self) -> None:
        if min(self.input_dim, self.coefficient_dim, self.hidden_dim) <= 0:
            raise ValueError("coefficient-prior dimensions must be positive")
        if self.precision_min < 0.0 or self.precision_max <= self.precision_min:
            raise ValueError("coefficient-prior precision bounds are invalid")
        if self.maximum_trust_ratio <= 0.0:
            raise ValueError("maximum trust ratio must be positive")
        if not 0.0 < self.initial_trust_fraction < 1.0:
            raise ValueError("initial trust fraction must lie in (0,1)")


@dataclass(frozen=True)
class HeadCoefficientPrior:
    mean: torch.Tensor
    precision: torch.Tensor
    trust_ratio: torch.Tensor


class CausalHeadCoefficientPrior(nn.Module):
    """Predict a diagonal Gaussian-style prior from deployment-safe features."""

    def __init__(self, config: HeadCoefficientPriorConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(config.hidden_dim, config.coefficient_dim)
        self.precision_head = nn.Linear(config.hidden_dim, config.coefficient_dim)
        self.trust_head = nn.Linear(config.hidden_dim, 1)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.precision_head.weight)
        nn.init.zeros_(self.precision_head.bias)
        nn.init.zeros_(self.trust_head.weight)
        initial = float(config.initial_trust_fraction)
        nn.init.constant_(self.trust_head.bias, math.log(initial / (1.0 - initial)))

    def forward(self, deployment_features: torch.Tensor) -> HeadCoefficientPrior:
        value = torch.as_tensor(deployment_features)
        if value.ndim != 2 or value.shape[1] != self.config.input_dim:
            raise ValueError("deployment features must be [batch,input_dim]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("deployment features must be finite")
        hidden = self.encoder(value)
        mean = self.mean_head(hidden)
        fraction = torch.sigmoid(self.precision_head(hidden))
        precision = self.config.precision_min + (
            self.config.precision_max - self.config.precision_min
        ) * fraction
        trust_ratio = self.config.maximum_trust_ratio * torch.sigmoid(
            self.trust_head(hidden)
        ).squeeze(-1)
        return HeadCoefficientPrior(mean, precision, trust_ratio)


@dataclass(frozen=True)
class PriorCenteredAdaptationResult:
    coefficients: torch.Tensor
    candidate: torch.Tensor
    accepted: bool
    online_objective_before: float
    online_objective_after: float
    proposed_online_objective: float
    posterior_objective_before: float
    posterior_objective_after: float
    proposed_posterior_objective: float
    correction_ratio: float
    proposed_correction_ratio: float
    condition_number: float
    future_truth_used: bool = False


def _objective_values(
    system: TransferDGLinearSystem,
    coefficients: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_precision: torch.Tensor,
    ridge_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = system.design @ coefficients + system.residual
    online = residual.square().sum()
    posterior = (
        online
        + (prior_precision * (coefficients - prior_mean).square()).sum()
        + float(ridge_weight) * coefficients.square().sum()
    )
    return online, posterior


def solve_prior_centered_system(
    system: TransferDGLinearSystem,
    parent_coefficients: torch.Tensor,
    correction_basis: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_precision: torch.Tensor,
    *,
    trust_ratio: float,
    ridge_weight: float = 1.0e-4,
    online_relative_tolerance: float = 0.0,
    online_absolute_tolerance: float = 1.0e-8,
) -> PriorCenteredAdaptationResult:
    """Solve a prior-centered ridge system with online-only safety rollback.

    The API intentionally accepts no future target or oracle coefficient tensor.
    """

    design = torch.as_tensor(system.design).float()
    residual = torch.as_tensor(system.residual, device=design.device).float()
    parent = torch.as_tensor(
        parent_coefficients, device=design.device, dtype=design.dtype
    )
    basis = torch.as_tensor(
        correction_basis, device=design.device, dtype=design.dtype
    )
    mean = torch.as_tensor(prior_mean, device=design.device, dtype=design.dtype).flatten()
    precision = torch.as_tensor(
        prior_precision, device=design.device, dtype=design.dtype
    ).flatten()
    rank = int(design.shape[1])
    if design.ndim != 2 or residual.shape != (design.shape[0],):
        raise ValueError("online design and residual shapes disagree")
    if basis.ndim != 5 or basis.shape[0] != rank or tuple(basis.shape[1:]) != tuple(parent.shape):
        raise ValueError("correction basis must be [rank,F,2,Z,X] and match parent")
    if mean.shape != (rank,) or precision.shape != (rank,):
        raise ValueError("prior mean and precision must contain one value per coefficient")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (design, residual, parent, basis, mean, precision)
    ):
        raise ValueError("prior-centered solve inputs must be finite")
    if bool((precision < 0.0).any()):
        raise ValueError("prior precision must be nonnegative")
    if trust_ratio <= 0.0 or ridge_weight < 0.0:
        raise ValueError("trust ratio must be positive and ridge nonnegative")
    if online_relative_tolerance < 0.0 or online_absolute_tolerance < 0.0:
        raise ValueError("online objective tolerances must be nonnegative")

    normal = design.T @ design
    normal = normal + torch.diag(precision)
    normal = normal + float(ridge_weight) * torch.eye(
        rank, device=design.device, dtype=design.dtype
    )
    right = -(design.T @ residual) + precision * mean
    factor, info = torch.linalg.cholesky_ex(normal)
    if int(info.max()) == 0:
        proposed = torch.cholesky_solve(right[:, None], factor)[:, 0]
    else:
        proposed = torch.linalg.lstsq(normal, right[:, None]).solution[:, 0]

    correction = torch.einsum("k,kfczx->fczx", proposed, basis)
    proposed_ratio = float(correction.norm() / parent.norm().clamp_min(1.0e-12))
    if proposed_ratio > float(trust_ratio):
        proposed = proposed * (float(trust_ratio) / proposed_ratio)
        correction = torch.einsum("k,kfczx->fczx", proposed, basis)
        proposed_ratio = float(correction.norm() / parent.norm().clamp_min(1.0e-12))

    zeros = torch.zeros_like(proposed)
    online_before, posterior_before = _objective_values(
        system, zeros, mean, precision, ridge_weight
    )
    online_proposed, posterior_proposed = _objective_values(
        system, proposed, mean, precision, ridge_weight
    )
    online_limit = (
        online_before * (1.0 + float(online_relative_tolerance))
        + float(online_absolute_tolerance)
    )
    accepted = bool(
        torch.isfinite(online_proposed)
        and torch.isfinite(posterior_proposed)
        and posterior_proposed < posterior_before
        and online_proposed <= online_limit
        and proposed_ratio <= float(trust_ratio) * (1.0 + 1.0e-6)
    )
    if accepted:
        coefficients = proposed
        candidate = parent + correction
        online_after = online_proposed
        posterior_after = posterior_proposed
        ratio = proposed_ratio
    else:
        coefficients = zeros
        candidate = parent.clone()
        online_after = online_before
        posterior_after = posterior_before
        ratio = 0.0

    singular_values = torch.linalg.svdvals(normal.double())
    condition_number = float(
        singular_values.max() / singular_values.min().clamp_min(1.0e-16)
    )
    return PriorCenteredAdaptationResult(
        coefficients=coefficients.detach(),
        candidate=candidate.detach(),
        accepted=accepted,
        online_objective_before=float(online_before),
        online_objective_after=float(online_after),
        proposed_online_objective=float(online_proposed),
        posterior_objective_before=float(posterior_before),
        posterior_objective_after=float(posterior_after),
        proposed_posterior_objective=float(posterior_proposed),
        correction_ratio=ratio,
        proposed_correction_ratio=proposed_ratio,
        condition_number=condition_number,
    )


__all__ = [
    "CausalHeadCoefficientPrior",
    "HeadCoefficientPrior",
    "HeadCoefficientPriorConfig",
    "PriorCenteredAdaptationResult",
    "solve_prior_centered_system",
]
