"""Convex low-dimensional B2 instance adaptation with a Q1-FE weak feature."""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from torch import nn
from torch.nn import functional as F

from .fe_weak_residual import normalized_fe_weak_loss


@dataclass(frozen=True)
class FEWeakAdapterConfig:
    observed_weight: float = 1.0
    weak_weight: float = 0.1
    ridge_weight: float = 1.0e-4
    trust_ratio: float = 0.05
    steps: int = 8
    learning_rate: float = 0.05
    coarsen: int = 4
    cpml_margin_fine: int = 20


def extract_parent_and_decoder_channel_responses(
    model: nn.Module,
    base_seq: torch.Tensor,
    cond: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return parent [B,T,1,Z,X] and 64 linear decoder-channel responses."""
    if base_seq.ndim != 5 or base_seq.shape[2] != 1:
        raise ValueError("base_seq must be [B,T,1,Z,X]")
    if base_seq.shape[0] != 1:
        raise ValueError("instance adapter requires batch size one")
    final_conv = model.decoder[-1]
    if not isinstance(final_conv, nn.Conv2d) or final_conv.out_channels != 1:
        raise ValueError("B2 decoder must end in a one-channel Conv2d")
    hidden = model.encoder(torch.cat((initial_state, cond), dim=1))
    cond_latent = model.cond_projection(cond)
    frames = []
    responses = []
    channel_kernel = final_conv.weight[0, :, :, :][:, None]
    for frame_index in range(base_seq.shape[1]):
        base_frame = base_seq[:, frame_index]
        base_latent = model.base_projection(base_frame)
        driven = model.step_norm(hidden + cond_latent + base_latent)
        activation = model.decoder[1](model.decoder[0](driven))
        correction = final_conv(activation)
        frames.append(base_frame + model.gate * correction)
        per_channel = F.conv2d(
            activation,
            channel_kernel,
            bias=None,
            padding=final_conv.padding,
            groups=activation.shape[1],
        )
        responses.append(model.gate * per_channel)
        hidden = model._step_anchored(hidden, cond_latent, base_latent)
    return torch.stack(frames, dim=1), torch.stack(responses, dim=1)


def apply_decoder_channel_scales(
    parent: torch.Tensor,
    responses: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    if responses.ndim != 5 or parent.ndim != 5:
        raise ValueError("parent/responses must be [B,T,C,Z,X] tensors")
    if coefficients.ndim != 1 or coefficients.numel() != responses.shape[2]:
        raise ValueError("coefficient count must match decoder channels")
    delta = torch.einsum("btcij,c->btij", responses, coefficients)[:, :, None]
    return parent + delta


def adapt_decoder_channel_scales(
    model: nn.Module,
    base_seq: torch.Tensor,
    cond: torch.Tensor,
    initial_state: torch.Tensor,
    observed_true: torch.Tensor,
    velocity_mps: torch.Tensor,
    *,
    source_off_frame: int,
    dt_s: float,
    dx_m: float,
    dz_m: float,
    config: FEWeakAdapterConfig = FEWeakAdapterConfig(),
) -> tuple[torch.Tensor, dict]:
    """Adapt 64 linear decoder scales using observations and FE weak defect only."""
    started = time.perf_counter()
    with torch.no_grad():
        parent, responses = extract_parent_and_decoder_channel_responses(
            model, base_seq, cond, initial_state
        )
    parent = parent.detach()
    responses = responses.detach()
    observed_count = observed_true.shape[1]
    if observed_true.shape != parent[:, :observed_count].shape:
        raise ValueError("observed_true must match the visible parent prefix")
    velocity = velocity_mps.to(parent.device, dtype=parent.dtype)
    parent_pressure = parent[:, :, 0]
    _, weak_scale = normalized_fe_weak_loss(
        parent_pressure,
        velocity,
        dt_s=dt_s,
        dx_m=dx_m,
        dz_m=dz_m,
        coarsen=config.coarsen,
        source_off_frame=source_off_frame,
        cpml_margin_fine=config.cpml_margin_fine,
    )
    observed_scale = observed_true.square().mean().detach().clamp_min(1.0e-8)
    coefficients = torch.zeros(
        responses.shape[2], device=parent.device, dtype=parent.dtype, requires_grad=True
    )
    optimizer = torch.optim.Adam([coefficients], lr=config.learning_rate)

    def objective() -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        candidate = apply_decoder_channel_scales(parent, responses, coefficients)
        observed = (
            candidate[:, :observed_count] - observed_true
        ).square().mean() / observed_scale
        weak, _ = normalized_fe_weak_loss(
            candidate[:, :, 0],
            velocity,
            reference_scale=weak_scale,
            dt_s=dt_s,
            dx_m=dx_m,
            dz_m=dz_m,
            coarsen=config.coarsen,
            source_off_frame=source_off_frame,
            cpml_margin_fine=config.cpml_margin_fine,
        )
        ridge = coefficients.square().mean()
        total = (
            config.observed_weight * observed
            + config.weak_weight * weak
            + config.ridge_weight * ridge
        )
        return total, {"observed": observed, "weak": weak, "ridge": ridge}, candidate

    with torch.no_grad():
        parent_objective, parent_parts, _ = objective()
    trace = []
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        total, parts, _ = objective()
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite FE weak adaptation objective")
        total.backward()
        optimizer.step()
        trace.append({
            "step": step + 1,
            "objective": float(total.detach()),
            **{name: float(value.detach()) for name, value in parts.items()},
        })
    with torch.no_grad():
        candidate_objective, candidate_parts, candidate = objective()
        correction = candidate - parent
        correction_ratio = float(
            correction.square().sum().sqrt()
            / parent.square().sum().clamp_min(1.0e-16).sqrt()
        )
        if correction_ratio > config.trust_ratio:
            coefficients.mul_(config.trust_ratio / max(correction_ratio, 1.0e-16))
            candidate_objective, candidate_parts, candidate = objective()
            correction = candidate - parent
            correction_ratio = float(
                correction.square().sum().sqrt()
                / parent.square().sum().clamp_min(1.0e-16).sqrt()
            )
        accepted = bool(
            torch.isfinite(candidate_objective)
            and candidate_objective < parent_objective
            and correction_ratio <= config.trust_ratio * (1.0 + 1.0e-6)
        )
        if not accepted:
            coefficients.zero_()
            candidate = parent
            candidate_objective = parent_objective
            candidate_parts = parent_parts
            correction_ratio = 0.0
    report = {
        "schema": "b2_fe_weak_instance_adaptation_v1",
        "accepted": accepted,
        "trainable_parameters": int(coefficients.numel()),
        "source_off_frame": int(source_off_frame),
        "parent_objective": float(parent_objective),
        "adapted_objective": float(candidate_objective),
        "objective_parts": {
            name: float(value) for name, value in candidate_parts.items()
        },
        "correction_ratio": correction_ratio,
        "coeff_norm": float(coefficients.norm()),
        "steps": config.steps,
        "elapsed_s": time.perf_counter() - started,
        "future_truth_used": False,
        "physics_contract": "Q1-FE weak residual is a mismatched-discretization feature, not truth",
        "trace": trace,
    }
    return candidate.detach(), report
