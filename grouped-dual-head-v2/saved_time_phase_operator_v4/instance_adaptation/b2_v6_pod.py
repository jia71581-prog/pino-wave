"""Offline residual POD and online self-supervised coefficient adaptation."""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from .b2_fe_weak_adapter import apply_decoder_channel_scales
from .b2_v6_encoding import enriched_physical_conditioning
from .fe_weak_residual import normalized_fe_weak_loss


def fit_residual_pod(residuals: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit orthonormal modes from [N,T,1,Z,X] residuals via the N x N Gram matrix."""
    if residuals.ndim != 5 or not 0 < rank <= residuals.shape[0]:
        raise ValueError("invalid residual tensor or POD rank")
    matrix = residuals.reshape(residuals.shape[0], -1).double()
    gram = matrix @ matrix.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)[:rank]
    values = eigenvalues[order].clamp_min(1.0e-16)
    vectors = eigenvectors[:, order]
    modes = (vectors.T @ matrix) / values.sqrt()[:, None]
    return modes.reshape(rank, *residuals.shape[1:]).float(), values.float()


def pod_coefficients(residuals: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
    if residuals.ndim != 5 or modes.ndim != 5 or residuals.shape[1:] != modes.shape[1:]:
        raise ValueError("residual/mode shape mismatch")
    return residuals.reshape(residuals.shape[0], -1).float() @ modes.reshape(
        modes.shape[0], -1
    ).float().T


def fit_ridge_prior(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    ridge: float = 1.0e-3,
) -> dict[str, torch.Tensor]:
    x = torch.as_tensor(features).double()
    y = torch.as_tensor(targets).double()
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("ridge features/targets must share a sample axis")
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1.0e-6)
    normalized = (x - mean) / scale
    design = torch.cat((normalized, torch.ones(len(x), 1, dtype=x.dtype)), dim=1)
    eye = torch.eye(design.shape[1], dtype=x.dtype)
    eye[-1, -1] = 0.0
    weight = torch.linalg.solve(design.T @ design + ridge * eye, design.T @ y)
    return {"mean": mean.float(), "scale": scale.float(), "weight": weight.float()}


def predict_ridge_prior(bundle: dict[str, torch.Tensor], features: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(features).float()
    normalized = (x - bundle["mean"].to(x)) / bundle["scale"].to(x)
    design = torch.cat((normalized, torch.ones(*x.shape[:-1], 1, device=x.device)), dim=-1)
    return design @ bundle["weight"].to(x)


def summary_encoding_features(
    arm: str,
    base_cond: torch.Tensor,
    velocity_mps: torch.Tensor,
    source_f0_hz: torch.Tensor,
    source_t0_s: torch.Tensor,
    *,
    model=None,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return record-level E0/E1/E2 features available at deployment."""
    base = torch.as_tensor(base_cond).float()
    if arm == "E0":
        field = base
    elif arm == "E1":
        field = enriched_physical_conditioning(
            base, velocity_mps, source_f0_hz, source_t0_s
        )
    elif arm == "E2":
        if model is None or initial_state is None:
            raise ValueError("E2 requires the frozen B2 model and initial state")
        hidden = model.encoder(torch.cat((initial_state, base), dim=1))
        return torch.cat(
            (hidden.mean((-2, -1)), hidden.std((-2, -1), unbiased=False)), dim=1
        )
    else:
        raise ValueError(f"unknown encoding arm: {arm}")
    return torch.cat(
        (field.mean((-2, -1)), field.std((-2, -1), unbiased=False)), dim=1
    )


def apply_pod_correction(
    parent: torch.Tensor, modes: torch.Tensor, coefficients: torch.Tensor
) -> torch.Tensor:
    if parent.ndim != 5 or modes.ndim != 5 or coefficients.ndim != 1:
        raise ValueError("invalid parent/modes/coefficients")
    if modes.shape[0] != coefficients.numel() or modes.shape[1:] != parent.shape[1:]:
        raise ValueError("POD correction shape mismatch")
    return parent + torch.einsum("r,rtczx->tczx", coefficients, modes)[None]


@dataclass(frozen=True)
class PODAdaptConfig:
    observed_weight: float = 1.0
    weak_weight: float = 0.1
    prior_weight: float = 1.0e-3
    trust_ratio: float = 0.05
    steps: int = 12
    learning_rate: float = 0.05
    coarsen: int = 4
    cpml_margin_fine: int = 20


def adapt_pod_coefficients(
    parent: torch.Tensor,
    modes: torch.Tensor,
    prior_coefficients: torch.Tensor,
    observed_true: torch.Tensor,
    velocity_mps: torch.Tensor,
    *,
    source_off_frame: int,
    dt_s: float,
    dx_m: float,
    dz_m: float,
    config: PODAdaptConfig = PODAdaptConfig(),
) -> tuple[torch.Tensor, dict]:
    started = time.perf_counter()
    parent = parent.detach()
    modes = modes.to(parent).detach()
    prior = prior_coefficients.to(parent).detach().reshape(-1)
    observed_count = observed_true.shape[1]
    if observed_true.shape != parent[:, :observed_count].shape:
        raise ValueError("observed prefix shape mismatch")
    velocity = velocity_mps.to(parent)
    _, weak_scale = normalized_fe_weak_loss(
        parent[:, :, 0], velocity, dt_s=dt_s, dx_m=dx_m, dz_m=dz_m,
        coarsen=config.coarsen, source_off_frame=source_off_frame,
        cpml_margin_fine=config.cpml_margin_fine,
    )
    observed_scale = observed_true.square().mean().detach().clamp_min(1.0e-8)
    coefficients = prior.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([coefficients], lr=config.learning_rate)

    def objective():
        candidate = apply_pod_correction(parent, modes, coefficients)
        observed = (candidate[:, :observed_count] - observed_true).square().mean() / observed_scale
        weak, _ = normalized_fe_weak_loss(
            candidate[:, :, 0], velocity, reference_scale=weak_scale,
            dt_s=dt_s, dx_m=dx_m, dz_m=dz_m, coarsen=config.coarsen,
            source_off_frame=source_off_frame, cpml_margin_fine=config.cpml_margin_fine,
        )
        prior_loss = (coefficients - prior).square().mean()
        total = config.observed_weight * observed + config.weak_weight * weak + config.prior_weight * prior_loss
        return total, observed, weak, prior_loss, candidate

    with torch.no_grad():
        parent_objective = objective()[0]
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        total = objective()[0]
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite POD adaptation objective")
        total.backward()
        optimizer.step()
    with torch.no_grad():
        total, observed, weak, prior_loss, candidate = objective()
        correction = candidate - parent
        ratio = float(correction.norm() / parent.norm().clamp_min(1.0e-16))
        if ratio > config.trust_ratio:
            coefficients.mul_(config.trust_ratio / max(ratio, 1.0e-16))
            total, observed, weak, prior_loss, candidate = objective()
            ratio = float((candidate - parent).norm() / parent.norm().clamp_min(1.0e-16))
        accepted = bool(torch.isfinite(total) and total < parent_objective and ratio <= config.trust_ratio * 1.000001)
        if not accepted:
            coefficients.zero_()
            candidate = parent
            ratio = 0.0
    return candidate.detach(), {
        "accepted": accepted,
        "rank": int(modes.shape[0]),
        "parent_objective": float(parent_objective),
        "adapted_objective": float(total),
        "observed": float(observed),
        "weak": float(weak),
        "prior_loss": float(prior_loss),
        "correction_ratio": ratio,
        "elapsed_s": time.perf_counter() - started,
        "future_truth_used": False,
    }
