"""Differentiable straight-ray travel-time feature for phase alignment."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class RayTravelTime:
    seconds: torch.Tensor
    distance_m: torch.Tensor
    path_velocity_mps: torch.Tensor
    endpoint_velocity_mps: torch.Tensor
    mean_slowness_s_per_m: torch.Tensor


def _validate_extent(extent: tuple[float, float], name: str) -> tuple[float, float]:
    lower, upper = (float(extent[0]), float(extent[1]))
    if not lower < upper:
        raise ValueError(f"{name} extent must be increasing")
    return lower, upper


def _inside(value: torch.Tensor, lower: float, upper: float) -> bool:
    tolerance = max(abs(upper - lower), 1.0) * 1.0e-6
    return bool(torch.all((value >= lower - tolerance) & (value <= upper + tolerance)))


def straight_ray_travel_time(
    velocity_mps: torch.Tensor,
    source_xy_m: torch.Tensor,
    query_xy_m: torch.Tensor,
    *,
    x_extent_m: tuple[float, float],
    z_extent_m: tuple[float, float],
    record_to_medium: torch.Tensor | None = None,
    samples: int = 12,
) -> RayTravelTime:
    """Integrate sampled slowness on each straight source-to-query segment."""
    velocity = torch.as_tensor(velocity_mps)
    if velocity.ndim != 4 or velocity.shape[1] != 1:
        raise ValueError("velocity_mps must be [medium,1,z,x]")
    if not velocity.dtype.is_floating_point:
        velocity = velocity.float()
    source = torch.as_tensor(source_xy_m, dtype=velocity.dtype, device=velocity.device)
    query = torch.as_tensor(query_xy_m, dtype=velocity.dtype, device=velocity.device)
    if source.ndim != 2 or source.shape[-1] != 2:
        raise ValueError("source_xy_m must be [record,2]")
    if query.ndim != 3 or query.shape[0] != source.shape[0] or query.shape[-1] != 2:
        raise ValueError("query_xy_m must be [record,query,2]")
    if samples < 2:
        raise ValueError("at least two ray samples are required")
    x0, x1 = _validate_extent(x_extent_m, "x")
    z0, z1 = _validate_extent(z_extent_m, "z")
    if not _inside(source[:, 0], x0, x1) or not _inside(source[:, 1], z0, z1):
        raise ValueError("source lies outside the physical domain")
    if not _inside(query[..., 0], x0, x1) or not _inside(query[..., 1], z0, z1):
        raise ValueError("query lies outside the physical domain")
    if not torch.isfinite(velocity).all() or torch.any(velocity <= 0):
        raise ValueError("velocity must be finite and positive")

    records = source.shape[0]
    if record_to_medium is None:
        if velocity.shape[0] == 1:
            mapping = torch.zeros(records, dtype=torch.long, device=velocity.device)
        elif velocity.shape[0] == records:
            mapping = torch.arange(records, dtype=torch.long, device=velocity.device)
        else:
            raise ValueError("record_to_medium is required when medium and record counts differ")
    else:
        mapping = torch.as_tensor(record_to_medium, dtype=torch.long, device=velocity.device)
        if mapping.shape != (records,) or torch.any(mapping < 0) or torch.any(mapping >= velocity.shape[0]):
            raise ValueError("record_to_medium contains invalid medium indices")

    alpha = torch.linspace(0.0, 1.0, samples, dtype=velocity.dtype, device=velocity.device)
    ray = source[:, None, None, :] + alpha[None, None, :, None] * (
        query[:, :, None, :] - source[:, None, None, :]
    )
    grid_x = 2.0 * (ray[..., 0] - x0) / (x1 - x0) - 1.0
    grid_z = 2.0 * (ray[..., 1] - z0) / (z1 - z0) - 1.0
    grid = torch.stack((grid_x, grid_z), dim=-1)
    record_velocity = velocity[mapping]
    sampled_velocity = F.grid_sample(
        record_velocity,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(1)
    mean_slowness = sampled_velocity.reciprocal().mean(dim=-1)
    distance = torch.linalg.vector_norm(query - source[:, None, :], dim=-1)
    return RayTravelTime(
        seconds=distance * mean_slowness,
        distance_m=distance,
        path_velocity_mps=mean_slowness.reciprocal(),
        endpoint_velocity_mps=sampled_velocity[..., -1],
        mean_slowness_s_per_m=mean_slowness,
    )


def dense_query_coordinates(
    x_m: torch.Tensor,
    z_m: torch.Tensor,
    *,
    records: int,
) -> torch.Tensor:
    x = torch.as_tensor(x_m)
    z = torch.as_tensor(z_m, dtype=x.dtype, device=x.device)
    if x.ndim != 1 or z.ndim != 1 or records <= 0:
        raise ValueError("x/z must be vectors and records must be positive")
    zz, xx = torch.meshgrid(z, x, indexing="ij")
    flat = torch.stack((xx.reshape(-1), zz.reshape(-1)), dim=-1)
    return flat[None].expand(records, -1, -1)


__all__ = ["RayTravelTime", "dense_query_coordinates", "straight_ray_travel_time"]

