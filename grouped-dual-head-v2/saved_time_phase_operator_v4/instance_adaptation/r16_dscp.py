"""Deployment-safe rank-16 temporal residual coefficient predictor.

The deployment API is intentionally label-free.  Future truth is not accepted by
``deployment_features``, ``predict_coefficients``, or ``forward``.  Training code
may construct coefficient targets separately, but it must never route those
targets through this module's feature path.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn


FAMILIES = ("uniform", "layered", "marmousi")
FAMILY_TO_INDEX = {name: index for index, name in enumerate(FAMILIES)}
INPUT_CHANNELS = 29
RANK = 16
HEIGHT = 201
WIDTH = 201
TIME_COUNT = 401
CONDITION_MAXIMUM = 1.0e3
CAUSAL_RAMP_STEPS = 4
MODEL_MACS_201 = 45_451_125
TEMPORAL_PROJECTION_MACS_201 = 259_212_816
TEMPORAL_MATERIALIZATION_MACS_201 = 259_212_816


class DSCPContractError(RuntimeError):
    """A deployment or frozen-artifact contract was violated."""


@dataclass(frozen=True)
class RouteDecision:
    name: str
    index: int
    one_hot: torch.Tensor
    frac_dx: float
    frac_dz: float
    abstain: bool


def predictor_parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def analytic_model_macs(height: int = HEIGHT, width: int = WIDTH) -> int:
    points = int(height) * int(width)
    if points <= 0:
        raise ValueError("spatial dimensions must be positive")
    return int(points * (29 * 9 + 29 * 16 + 16 * 9 + 16 * 16))


def analytic_temporal_macs(
    time_count: int = TIME_COUNT,
    rank: int = RANK,
    height: int = HEIGHT,
    width: int = WIDTH,
) -> int:
    values = (int(time_count), int(rank), int(height), int(width))
    if any(value <= 0 for value in values):
        raise ValueError("temporal MAC dimensions must be positive")
    return math.prod(values)


def _as_bchw(value: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 3:
        tensor = tensor[:, None]
    if tensor.ndim != 4 or tensor.shape[1] != 1:
        raise ValueError(f"{name} must have shape [batch,1,height,width]")
    return tensor


def velocity_route(velocity_mps: torch.Tensor) -> tuple[RouteDecision, ...]:
    """Route solely from velocity gradients; gray-zone media abstain."""
    velocity = _as_bchw(velocity_mps, "velocity_mps")
    if not bool(torch.isfinite(velocity).all()) or bool((velocity <= 0).any()):
        raise ValueError("velocity must be finite and positive")
    dx = (velocity[..., 1:] - velocity[..., :-1]).abs()
    dz = (velocity[..., 1:, :] - velocity[..., :-1, :]).abs()
    frac_x = (dx > 1.0e-6).to(torch.float64).mean(dim=(1, 2, 3))
    frac_z = (dz > 1.0e-6).to(torch.float64).mean(dim=(1, 2, 3))
    output: list[RouteDecision] = []
    for batch_index, (fx_tensor, fz_tensor) in enumerate(zip(frac_x, frac_z)):
        fx, fz = float(fx_tensor), float(fz_tensor)
        constant = fx == 0.0 and fz == 0.0
        if constant:
            name, index, abstain = "uniform", 0, False
        elif fx <= 0.05 and fz <= 0.10:
            name, index, abstain = "layered", 1, False
        elif fx >= 0.15 and fz >= 0.50:
            name, index, abstain = "marmousi", 2, False
        else:
            name, index, abstain = "abstain", -1, True
        one_hot = torch.zeros(3, dtype=velocity.dtype, device=velocity.device)
        if not abstain:
            one_hot[index] = 1.0
        output.append(RouteDecision(name, index, one_hot, fx, fz, abstain))
    return tuple(output)


def c1_causal_mask(
    time_count: int,
    k1: int,
    *,
    ramp_steps: int = CAUSAL_RAMP_STEPS,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """C1 smoothstep gate: exactly zero through k1, one after four steps."""
    count, onset, width = int(time_count), int(k1), int(ramp_steps)
    if count <= 0 or not 0 <= onset < count or width <= 0:
        raise ValueError("invalid time_count, k1, or ramp_steps")
    indices = torch.arange(count, device=device, dtype=torch.float64)
    u = ((indices - float(onset)) / float(width)).clamp(0.0, 1.0)
    mask = u.square() * (3.0 - 2.0 * u)
    mask[: onset + 1] = 0.0
    return mask.to(dtype=dtype)


def basis_condition(basis: torch.Tensor, k1: int) -> float:
    value = torch.as_tensor(basis, device="cpu", dtype=torch.float64)
    if value.ndim != 2 or value.shape[1] != RANK:
        raise ValueError("basis must have shape [time,16]")
    restricted = value[int(k1) + 1 :]
    if restricted.shape[0] < RANK:
        return math.inf
    singular = torch.linalg.svdvals(restricted)
    smallest = float(singular[-1])
    largest = float(singular[0])
    if not math.isfinite(smallest) or smallest <= 0.0:
        return math.inf
    return largest / smallest


def temporal_projection(parent_wavefield: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    parent = torch.as_tensor(parent_wavefield)
    temporal = torch.as_tensor(basis, device=parent.device, dtype=parent.dtype)
    if parent.ndim != 4 or temporal.ndim != 2 or parent.shape[1] != temporal.shape[0]:
        raise ValueError("parent [B,T,H,W] and basis [T,R] are required")
    if temporal.shape[1] != RANK:
        raise ValueError("temporal basis rank must be 16")
    return torch.einsum("tr,bthw->brhw", temporal, parent)


def _coordinate_maps(
    x_m: torch.Tensor,
    z_m: torch.Tensor,
    *,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(x_m, device=device, dtype=dtype).flatten()
    z = torch.as_tensor(z_m, device=device, dtype=dtype).flatten()
    if x.numel() != WIDTH or z.numel() != HEIGHT:
        raise ValueError("x_m and z_m must contain 201 coordinates")
    x_span = (x[-1] - x[0]).abs().clamp_min(torch.finfo(dtype).eps)
    z_span = (z[-1] - z[0]).abs().clamp_min(torch.finfo(dtype).eps)
    x_norm = (2.0 * (x - x[0]) / x_span - 1.0)[None, None, None, :]
    z_norm = (2.0 * (z - z[0]) / z_span - 1.0)[None, None, :, None]
    return x_norm.expand(batch, 1, HEIGHT, WIDTH), z_norm.expand(batch, 1, HEIGHT, WIDTH)


def deployment_features(
    velocity_mps: torch.Tensor,
    source_map: torch.Tensor,
    travel_time_map: torch.Tensor,
    x_m: torch.Tensor,
    z_m: torch.Tensor,
    observed_k0: torch.Tensor,
    observed_k1: torch.Tensor,
    parent_wavefield: torch.Tensor,
    bases: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
) -> tuple[torch.Tensor, tuple[RouteDecision, ...], torch.Tensor]:
    """Create exactly 29 permitted deployment channels.

    ``bases`` is the immutable three-family train-only artifact.  Family labels,
    split/sample/group identifiers, future truth, and oracle weights are absent.
    """
    velocity = _as_bchw(velocity_mps, "velocity_mps")
    source = _as_bchw(source_map, "source_map").to(velocity)
    travel = _as_bchw(travel_time_map, "travel_time_map").to(velocity)
    observed0 = _as_bchw(observed_k0, "observed_k0").to(velocity)
    observed1 = _as_bchw(observed_k1, "observed_k1").to(velocity)
    parent = torch.as_tensor(parent_wavefield, device=velocity.device, dtype=velocity.dtype)
    temporal_bases = torch.as_tensor(bases, device=velocity.device, dtype=velocity.dtype)
    batch, _, height, width = velocity.shape
    if (height, width) != (HEIGHT, WIDTH):
        raise ValueError("R16-DSCP is frozen to 201x201")
    if parent.shape != (batch, TIME_COUNT, HEIGHT, WIDTH):
        raise ValueError("parent_wavefield must have shape [B,401,201,201]")
    if temporal_bases.shape != (3, TIME_COUNT, RANK):
        raise ValueError("bases must have shape [3,401,16]")
    indices0 = torch.as_tensor(k0, dtype=torch.long, device=velocity.device).flatten()
    indices1 = torch.as_tensor(k1, dtype=torch.long, device=velocity.device).flatten()
    if indices0.numel() != batch or indices1.numel() != batch:
        raise ValueError("k0 and k1 must have one value per record")
    if bool((indices1 != indices0 + 1).any()) or bool((indices0 < 0).any()) or bool(
        (indices1 >= TIME_COUNT).any()
    ):
        raise ValueError("registered observed indices must be adjacent and in range")

    decisions = velocity_route(velocity)
    log_velocity = velocity.log()
    dx = torch.zeros_like(log_velocity)
    dz = torch.zeros_like(log_velocity)
    dx[..., 1:] = log_velocity[..., 1:] - log_velocity[..., :-1]
    dz[..., 1:, :] = log_velocity[..., 1:, :] - log_velocity[..., :-1, :]
    x_map, z_map = _coordinate_maps(
        x_m, z_m, batch=batch, device=velocity.device, dtype=velocity.dtype
    )
    selected_projections = torch.zeros(
        (batch, RANK, HEIGHT, WIDTH), device=velocity.device, dtype=velocity.dtype
    )
    route_maps = torch.zeros(
        (batch, 3, HEIGHT, WIDTH), device=velocity.device, dtype=velocity.dtype
    )
    conditions = torch.full((batch,), math.inf, dtype=torch.float64)
    for index, decision in enumerate(decisions):
        if decision.abstain:
            continue
        family_basis = temporal_bases[decision.index]
        selected_projections[index : index + 1] = temporal_projection(
            parent[index : index + 1], family_basis
        )
        route_maps[index] = decision.one_hot[:, None, None]
        conditions[index] = basis_condition(family_basis, int(indices1[index]))

    gather0 = parent[torch.arange(batch, device=velocity.device), indices0][:, None]
    gather1 = parent[torch.arange(batch, device=velocity.device), indices1][:, None]
    error0, error1 = observed0 - gather0, observed1 - gather1
    field_scale = parent.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1.0e-12)
    travel_scale = travel.amax(dim=(2, 3), keepdim=True).clamp_min(1.0e-6)
    channels = torch.cat(
        (
            log_velocity,
            dx,
            dz,
            source,
            travel / travel_scale,
            x_map,
            z_map,
            error0 / field_scale,
            error1 / field_scale,
            (error1 - error0) / field_scale,
            selected_projections / field_scale,
            route_maps,
        ),
        dim=1,
    )
    if channels.shape != (batch, INPUT_CHANNELS, HEIGHT, WIDTH):
        raise DSCPContractError("deployment feature channel contract changed")
    if not bool(torch.isfinite(channels).all()):
        raise FloatingPointError("deployment features contain NaN or Inf")
    return channels, decisions, conditions


class R16DSCP(nn.Module):
    """The exact 1,202-parameter depthwise/pointwise predictor."""

    def __init__(self, bases: torch.Tensor, coefficient_scales: torch.Tensor) -> None:
        super().__init__()
        basis_value = torch.as_tensor(bases, dtype=torch.float32).contiguous()
        scale_value = torch.as_tensor(coefficient_scales, dtype=torch.float32).contiguous()
        if basis_value.shape != (3, TIME_COUNT, RANK):
            raise ValueError("bases must have shape [3,401,16]")
        if scale_value.shape != (3, RANK) or bool((scale_value <= 0).any()):
            raise ValueError("coefficient_scales must be positive [3,16]")
        self.register_buffer("bases", basis_value, persistent=True)
        self.register_buffer("coefficient_scales", scale_value, persistent=True)
        self.depthwise_in = nn.Conv2d(
            INPUT_CHANNELS, INPUT_CHANNELS, 3, padding=1, groups=INPUT_CHANNELS, bias=True
        )
        self.pointwise_hidden = nn.Conv2d(INPUT_CHANNELS, RANK, 1, bias=True)
        self.activation = nn.SiLU()
        self.depthwise_hidden = nn.Conv2d(RANK, RANK, 3, padding=1, groups=RANK, bias=True)
        self.pointwise_out = nn.Conv2d(RANK, RANK, 1, bias=True)
        nn.init.zeros_(self.pointwise_out.weight)
        nn.init.zeros_(self.pointwise_out.bias)
        if predictor_parameter_count(self) != 1202:
            raise DSCPContractError("predictor parameter count is not 1,202")

    def coefficient_head(self, features: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(features)
        if value.ndim != 4 or value.shape[1] != INPUT_CHANNELS:
            raise ValueError("features must have shape [batch,29,height,width]")
        value = self.depthwise_in(value)
        value = self.activation(self.pointwise_hidden(value))
        value = self.depthwise_hidden(value)
        return self.pointwise_out(value).tanh()

    def predict_coefficients(
        self,
        velocity_mps: torch.Tensor,
        source_map: torch.Tensor,
        travel_time_map: torch.Tensor,
        x_m: torch.Tensor,
        z_m: torch.Tensor,
        observed_k0: torch.Tensor,
        observed_k1: torch.Tensor,
        parent_wavefield: torch.Tensor,
        k0: torch.Tensor,
        k1: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[RouteDecision, ...], torch.Tensor]:
        features, decisions, conditions = deployment_features(
            velocity_mps,
            source_map,
            travel_time_map,
            x_m,
            z_m,
            observed_k0,
            observed_k1,
            parent_wavefield,
            self.bases,
            k0,
            k1,
        )
        unit = self.coefficient_head(features)
        coefficients = torch.zeros_like(unit)
        for index, decision in enumerate(decisions):
            if not decision.abstain and float(conditions[index]) <= CONDITION_MAXIMUM:
                coefficients[index] = unit[index] * self.coefficient_scales[decision.index, :, None, None]
        if not bool(torch.isfinite(coefficients).all()):
            raise FloatingPointError("predicted coefficients contain NaN or Inf")
        return coefficients, decisions, conditions

    def forward(
        self,
        velocity_mps: torch.Tensor,
        source_map: torch.Tensor,
        travel_time_map: torch.Tensor,
        x_m: torch.Tensor,
        z_m: torch.Tensor,
        observed_k0: torch.Tensor,
        observed_k1: torch.Tensor,
        parent_wavefield: torch.Tensor,
        k0: torch.Tensor,
        k1: torch.Tensor,
    ) -> torch.Tensor:
        coefficients, decisions, conditions = self.predict_coefficients(
            velocity_mps,
            source_map,
            travel_time_map,
            x_m,
            z_m,
            observed_k0,
            observed_k1,
            parent_wavefield,
            k0,
            k1,
        )
        parent = torch.as_tensor(parent_wavefield)
        corrected = parent.clone()
        onset = torch.as_tensor(k1, dtype=torch.long, device=parent.device).flatten()
        for index, decision in enumerate(decisions):
            if decision.abstain or float(conditions[index]) > CONDITION_MAXIMUM:
                continue
            basis = self.bases[decision.index].to(parent)
            correction = torch.einsum("tr,rhw->thw", basis, coefficients[index])
            mask = c1_causal_mask(
                TIME_COUNT, int(onset[index]), device=parent.device, dtype=parent.dtype
            )
            correction = correction * mask[:, None, None]
            correction[:, 0, :] = 0.0
            corrected[index] += correction
        if not bool(torch.isfinite(corrected).all()):
            raise FloatingPointError("corrected output contains NaN or Inf")
        return corrected


class RidgePointwiseBaseline(nn.Conv2d):
    """Fixed 480-parameter 1x1 baseline; fit by lambda=1e-4 ridge."""

    def __init__(self) -> None:
        super().__init__(INPUT_CHANNELS, RANK, kernel_size=1, bias=True)
        if predictor_parameter_count(self) != 480:
            raise DSCPContractError("ridge baseline parameter count is not 480")

    @torch.no_grad()
    def fit_closed_form(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        *,
        ridge_lambda: float = 1.0e-4,
    ) -> None:
        x = torch.as_tensor(features, device="cpu", dtype=torch.float64)
        y = torch.as_tensor(targets, device="cpu", dtype=torch.float64)
        if x.ndim != 4 or y.ndim != 4 or x.shape[0] != y.shape[0]:
            raise ValueError("ridge features and targets must be matched BCHW tensors")
        design = x.permute(0, 2, 3, 1).reshape(-1, INPUT_CHANNELS)
        target = y.permute(0, 2, 3, 1).reshape(-1, RANK)
        ones = torch.ones((design.shape[0], 1), dtype=design.dtype)
        design = torch.cat((design, ones), dim=1)
        regularizer = torch.eye(INPUT_CHANNELS + 1, dtype=design.dtype) * float(ridge_lambda)
        regularizer[-1, -1] = 0.0
        solution = torch.linalg.solve(design.T @ design + regularizer, design.T @ target)
        self.weight.copy_(solution[:-1].T[:, :, None, None].to(self.weight))
        self.bias.copy_(solution[-1].to(self.bias))


def frozen_loss(
    corrected_future: torch.Tensor,
    truth_future: torch.Tensor,
    coefficients: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Train-only supervised loss.  This function is not used in deployment."""
    prediction = torch.as_tensor(corrected_future).float()
    truth = torch.as_tensor(truth_future, device=prediction.device).float()
    if prediction.shape != truth.shape or prediction.ndim != 4:
        raise ValueError("future prediction and truth must match [B,T,H,W]")
    error = prediction - truth
    denominator = truth.square().sum(dim=(2, 3)).clamp_min(1.0e-30)
    frame_rel2_sq = error.square().sum(dim=(2, 3)) / denominator
    late_start = (2 * prediction.shape[1]) // 3
    frame_term = frame_rel2_sq.mean()
    late_term = frame_rel2_sq[:, late_start:].mean()
    if prediction.shape[1] < 2:
        temporal_term = prediction.new_zeros(())
    else:
        delta_error = error[:, 1:] - error[:, :-1]
        delta_truth = truth[:, 1:] - truth[:, :-1]
        temporal_term = delta_error.square().sum() / delta_truth.square().sum().clamp_min(1.0e-30)
    coefficient_term = torch.as_tensor(coefficients).float().square().mean()
    total = frame_term + 0.25 * late_term + 0.10 * temporal_term + 1.0e-4 * coefficient_term
    return {
        "total": total,
        "frame_relative_l2_squared": frame_term,
        "late_third_relative_l2_squared": late_term,
        "temporal_difference": temporal_term,
        "normalized_coefficient_energy": coefficient_term,
    }


__all__ = [
    "CAUSAL_RAMP_STEPS",
    "CONDITION_MAXIMUM",
    "DSCPContractError",
    "FAMILIES",
    "INPUT_CHANNELS",
    "MODEL_MACS_201",
    "R16DSCP",
    "RANK",
    "RidgePointwiseBaseline",
    "RouteDecision",
    "TEMPORAL_MATERIALIZATION_MACS_201",
    "TEMPORAL_PROJECTION_MACS_201",
    "analytic_model_macs",
    "analytic_temporal_macs",
    "basis_condition",
    "c1_causal_mask",
    "deployment_features",
    "frozen_loss",
    "predictor_parameter_count",
    "temporal_projection",
    "velocity_route",
]
