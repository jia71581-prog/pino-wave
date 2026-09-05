"""Local observed-prefix-to-future residual meta-adaptation for B2-v11."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch
from torch import nn
from torch.nn import functional as F


def prefix_residual_context(
    parent: torch.Tensor,
    observed_true: torch.Tensor,
    *,
    anchor_frames: int = 8,
) -> torch.Tensor:
    """Summarize only newly observed parent residuals as four local maps."""
    if parent.ndim != 5 or observed_true.ndim != 5:
        raise ValueError("parent and observations must be [B,T,1,Z,X]")
    if parent.shape[0] != observed_true.shape[0] or parent.shape[2:] != observed_true.shape[2:]:
        raise ValueError("parent/observation shapes do not match")
    if parent.shape[2] != 1 or observed_true.shape[1] > parent.shape[1]:
        raise ValueError("local meta adapter requires one channel and a proper prefix")
    anchor = int(anchor_frames)
    count = int(observed_true.shape[1])
    if count < anchor + 2:
        raise ValueError("at least two informative frames after the anchors are required")
    residual = observed_true - parent[:, :count]
    informative = residual[:, anchor:, 0]
    last = informative[:, -1]
    slope = informative[:, -1] - informative[:, -2]
    mean = informative.mean(dim=1)
    rms = informative.square().mean(dim=1).clamp_min(0.0).sqrt()
    return torch.stack((last, slope, mean, rms), dim=1)


def _gather_time(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, _, channels, height, width = value.shape
    gather = indices[:, :, None, None, None].expand(
        batch, indices.shape[1], channels, height, width
    )
    return torch.gather(value, dim=1, index=gather)


def query_features(
    parent: torch.Tensor,
    physics_conditioning: torch.Tensor,
    context: torch.Tensor,
    query_indices: torch.Tensor,
    *,
    context_count: torch.Tensor,
    time_s: torch.Tensor,
    source_f0_hz: torch.Tensor,
    source_t0_s: torch.Tensor,
) -> torch.Tensor:
    """Build local deployment features for arbitrary future query frames."""
    if parent.ndim != 5 or parent.shape[2] != 1:
        raise ValueError("parent must be [B,T,1,Z,X]")
    batch, total, _, height, width = parent.shape
    conditioning = torch.as_tensor(physics_conditioning, device=parent.device).float()
    if conditioning.ndim != 4 or conditioning.shape[0] != batch or conditioning.shape[-2:] != (height, width):
        raise ValueError("physics conditioning must be [B,C,Z,X]")
    local = torch.as_tensor(context, device=parent.device).float()
    if local.shape != (batch, 4, height, width):
        raise ValueError("context must be [B,4,Z,X]")
    indices = torch.as_tensor(query_indices, dtype=torch.long, device=parent.device)
    if indices.ndim == 1:
        indices = indices[None].expand(batch, -1)
    if indices.ndim != 2 or indices.shape[0] != batch:
        raise ValueError("query_indices must be [Q] or [B,Q]")
    if bool((indices < 0).any()) or bool((indices >= total).any()):
        raise ValueError("query index is outside the parent sequence")
    counts = torch.as_tensor(context_count, dtype=torch.long, device=parent.device).reshape(-1)
    if counts.shape != (batch,) or bool((counts < 1).any()) or bool((counts >= total).any()):
        raise ValueError("context_count must contain one proper prefix per record")
    previous = (indices - 1).clamp_min(0)
    following = (indices + 1).clamp_max(total - 1)
    current_field = _gather_time(parent, indices)[:, :, 0]
    previous_delta = _gather_time(parent, previous)[:, :, 0] - current_field
    following_delta = _gather_time(parent, following)[:, :, 0] - current_field

    axis = torch.as_tensor(time_s, dtype=parent.dtype, device=parent.device).flatten()
    if axis.shape != (total,) or not bool(torch.isfinite(axis).all()):
        raise ValueError("time_s must match the parent time axis")
    query_time = axis[indices]
    t_min, t_max = axis[0], axis[-1]
    time_norm = 2.0 * (query_time - t_min) / (t_max - t_min).clamp_min(1.0e-8) - 1.0
    horizon = (indices.float() - (counts[:, None].float() - 1.0)) / max(total - 1, 1)
    f0 = torch.as_tensor(source_f0_hz, dtype=parent.dtype, device=parent.device).reshape(batch, 1)
    t0 = torch.as_tensor(source_t0_s, dtype=parent.dtype, device=parent.device).reshape(batch, 1)
    phase = 2.0 * math.pi * f0 * (query_time - t0)
    dynamic = torch.stack((time_norm, horizon, torch.sin(phase), torch.cos(phase)), dim=2)

    query_count = indices.shape[1]
    spatial = (height, width)
    conditioning_expanded = conditioning[:, None].expand(-1, query_count, -1, -1, -1)
    context_expanded = local[:, None].expand(-1, query_count, -1, -1, -1)
    parent_features = torch.stack((current_field, previous_delta, following_delta), dim=2)
    dynamic_maps = dynamic[:, :, :, None, None].expand(-1, -1, -1, *spatial)
    features = torch.cat(
        (conditioning_expanded, context_expanded, parent_features, dynamic_maps), dim=2
    )
    return features.reshape(batch * query_count, features.shape[2], height, width)


class _ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, dilation: int = 1):
        super().__init__()
        groups = 8 if out_channels % 8 == 0 else 4
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int = 1):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 4
        self.first = _ConvNormAct(channels, channels, dilation=dilation)
        self.second = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value + self.second(self.first(value)), inplace=True)


class LocalResidualMetaOperator(nn.Module):
    """Small U-Net predicting a spatially local future residual direction."""

    def __init__(
        self,
        *,
        physics_channels: int = 20,
        base_width: int = 16,
        correction_cap: float = 0.25,
    ) -> None:
        super().__init__()
        width = int(base_width)
        if width < 16 or width % 8:
            raise ValueError("base_width must be a multiple of 8 and at least 16")
        if correction_cap <= 0.0:
            raise ValueError("correction_cap must be positive")
        self.physics_channels = int(physics_channels)
        self.input_channels = self.physics_channels + 11
        self.correction_cap = float(correction_cap)
        self.stem = _ConvNormAct(self.input_channels, width)
        self.enc0 = _ResidualBlock(width)
        self.down1 = _ConvNormAct(width, 2 * width, stride=2)
        self.enc1 = _ResidualBlock(2 * width)
        self.down2 = _ConvNormAct(2 * width, 3 * width, stride=2)
        self.bottleneck = nn.Sequential(
            _ResidualBlock(3 * width, dilation=2),
            _ResidualBlock(3 * width, dilation=3),
        )
        self.up1 = _ConvNormAct(5 * width, 2 * width)
        self.dec1 = _ResidualBlock(2 * width)
        self.up0 = _ConvNormAct(3 * width, width)
        self.dec0 = _ResidualBlock(width)
        self.output = nn.Conv2d(width, 1, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != self.input_channels:
            raise ValueError("local residual features have the wrong shape")
        x0 = self.enc0(self.stem(features))
        x1 = self.enc1(self.down1(x0))
        x2 = self.bottleneck(self.down2(x1))
        y1 = F.interpolate(x2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y1 = self.dec1(self.up1(torch.cat((y1, x1), dim=1)))
        y0 = F.interpolate(y1, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        y0 = self.dec0(self.up0(torch.cat((y0, x0), dim=1)))
        correction = self.correction_cap * torch.tanh(self.output(y0))
        correction = correction.clone()
        correction[:, :, 0, :] = 0.0
        return correction


def temporal_polynomial_basis(query_time_s: torch.Tensor, full_time_s: torch.Tensor) -> torch.Tensor:
    axis = torch.as_tensor(full_time_s, dtype=query_time_s.dtype, device=query_time_s.device).flatten()
    value = 2.0 * (query_time_s - axis[0]) / (axis[-1] - axis[0]).clamp_min(1.0e-8) - 1.0
    return torch.stack((torch.ones_like(value), value, value.square()), dim=-1)


def fit_temporal_polynomial(
    directions: torch.Tensor,
    target_residual: torch.Tensor,
    query_time_s: torch.Tensor,
    full_time_s: torch.Tensor,
    *,
    ridge_fraction: float = 1.0e-3,
) -> torch.Tensor:
    """Differentiably fit three temporal scale coefficients per record."""
    if directions.shape != target_residual.shape or directions.ndim != 5:
        raise ValueError("directions and residual target must match [B,Q,1,Z,X]")
    batch, query_count = directions.shape[:2]
    times = torch.as_tensor(query_time_s, dtype=directions.dtype, device=directions.device)
    if times.ndim == 1:
        times = times[None].expand(batch, -1)
    if times.shape != (batch, query_count):
        raise ValueError("query times must match directions")
    basis = temporal_polynomial_basis(times, full_time_s)
    prior = directions.new_tensor((1.0, 0.0, 0.0))
    coefficients = []
    for index in range(batch):
        design = (
            directions[index, :, 0, :, :, None]
            * basis[index, :, None, None, :]
        ).reshape(-1, 3)
        target = target_residual[index, :, 0].reshape(-1)
        gram = design.T @ design
        scale = torch.trace(gram) / 3.0
        regularization = float(ridge_fraction) * scale.clamp_min(1.0e-8)
        system = gram + regularization * torch.eye(3, device=gram.device, dtype=gram.dtype)
        rhs = design.T @ target + regularization * prior
        coefficients.append(torch.linalg.solve(system, rhs))
    return torch.stack(coefficients)


def apply_temporal_polynomial(
    directions: torch.Tensor,
    coefficients: torch.Tensor,
    query_time_s: torch.Tensor,
    full_time_s: torch.Tensor,
) -> torch.Tensor:
    if directions.ndim != 5 or coefficients.shape != (directions.shape[0], 3):
        raise ValueError("direction/coefficient shapes do not match")
    times = torch.as_tensor(query_time_s, dtype=directions.dtype, device=directions.device)
    if times.ndim == 1:
        times = times[None].expand(directions.shape[0], -1)
    basis = temporal_polynomial_basis(times, full_time_s)
    scale = torch.einsum("bqj,bj->bq", basis, coefficients)
    return directions * scale[:, :, None, None, None]


@dataclass(frozen=True)
class LocalMetaAdaptConfig:
    anchor_frames: int = 8
    validation_tail_frames: int = 4
    ridge_fraction: float = 1.0e-3
    trust_ratio: float = 0.05
    minimum_observed_gain: float = 0.0


def adapt_local_meta(
    model: nn.Module,
    parent: torch.Tensor,
    physics_conditioning: torch.Tensor,
    observed_true: torch.Tensor,
    *,
    time_s: torch.Tensor,
    source_f0_hz: torch.Tensor,
    source_t0_s: torch.Tensor,
    config: LocalMetaAdaptConfig = LocalMetaAdaptConfig(),
) -> tuple[torch.Tensor, dict[str, object]]:
    """Fit three online coefficients from a held-out prefix tail."""
    started = time.perf_counter()
    parent = parent.detach()
    observed = observed_true.to(parent).detach()
    batch, total = parent.shape[:2]
    count = int(observed.shape[1])
    support_count = count - int(config.validation_tail_frames)
    if batch != 1:
        raise ValueError("deployment adaptation currently requires one record")
    if support_count < int(config.anchor_frames) + 2 or count >= total:
        raise ValueError("observed prefix cannot provide support, tail, and future")
    support_context = prefix_residual_context(
        parent, observed[:, :support_count], anchor_frames=config.anchor_frames
    )
    validation_indices = torch.arange(support_count, count, device=parent.device)
    validation_features = query_features(
        parent,
        physics_conditioning,
        support_context,
        validation_indices,
        context_count=torch.tensor([support_count], device=parent.device),
        time_s=time_s,
        source_f0_hz=source_f0_hz,
        source_t0_s=source_t0_s,
    )
    validation_directions = model(validation_features).reshape(
        1, len(validation_indices), 1, *parent.shape[-2:]
    )
    validation_target = observed[:, support_count:count] - parent[:, support_count:count]
    axis = torch.as_tensor(time_s, dtype=parent.dtype, device=parent.device)
    coefficients = fit_temporal_polynomial(
        validation_directions,
        validation_target,
        axis[validation_indices],
        axis,
        ridge_fraction=config.ridge_fraction,
    )
    fitted_validation = apply_temporal_polynomial(
        validation_directions, coefficients, axis[validation_indices], axis
    )
    parent_observed_loss = float(validation_target.square().mean())
    adapted_observed_loss = float((fitted_validation - validation_target).square().mean())
    observed_gain = (
        0.0
        if parent_observed_loss <= 1.0e-30
        else 1.0 - adapted_observed_loss / parent_observed_loss
    )
    candidate = parent.clone()
    accepted = bool(observed_gain > config.minimum_observed_gain)
    correction_ratio = 0.0
    if accepted:
        full_context = prefix_residual_context(
            parent, observed, anchor_frames=config.anchor_frames
        )
        future_indices = torch.arange(count, total, device=parent.device)
        future_features = query_features(
            parent,
            physics_conditioning,
            full_context,
            future_indices,
            context_count=torch.tensor([count], device=parent.device),
            time_s=axis,
            source_f0_hz=source_f0_hz,
            source_t0_s=source_t0_s,
        )
        future_directions = model(future_features).reshape(
            1, len(future_indices), 1, *parent.shape[-2:]
        )
        correction = apply_temporal_polynomial(
            future_directions, coefficients, axis[future_indices], axis
        )
        parent_future = parent[:, count:]
        correction_ratio = float(correction.norm() / parent_future.norm().clamp_min(1.0e-16))
        if correction_ratio > config.trust_ratio:
            correction = correction * (float(config.trust_ratio) / max(correction_ratio, 1.0e-16))
            correction_ratio = float(correction.norm() / parent_future.norm().clamp_min(1.0e-16))
        if not bool(torch.isfinite(correction).all()):
            accepted = False
            correction_ratio = 0.0
        else:
            candidate[:, count:] = parent_future + correction
    if not accepted:
        candidate = parent.clone()
        coefficients = torch.zeros_like(coefficients)
    if not torch.equal(candidate[:, :count], parent[:, :count]):
        raise AssertionError("local meta adaptation modified an observed frame")
    return candidate.detach(), {
        "accepted": accepted,
        "observed_count": count,
        "support_count": support_count,
        "validation_tail_frames": int(config.validation_tail_frames),
        "coefficients": coefficients.detach().cpu().tolist(),
        "parent_observed_loss": parent_observed_loss,
        "adapted_observed_loss": adapted_observed_loss,
        "observed_gain": observed_gain,
        "correction_ratio": correction_ratio,
        "elapsed_s": time.perf_counter() - started,
        "future_truth_used": False,
    }


__all__ = [
    "LocalMetaAdaptConfig",
    "LocalResidualMetaOperator",
    "adapt_local_meta",
    "apply_temporal_polynomial",
    "fit_temporal_polynomial",
    "prefix_residual_context",
    "query_features",
    "temporal_polynomial_basis",
]
