"""Coarsened Q1 finite-element weak residual for acoustic-wave diagnostics.

The dataset generator is LWC84/CPML rather than Q1 FEM.  Consequently this
residual is an online physics feature and adaptation objective, not numerical
ground truth.  Boundary and source-active regions are excluded explicitly.
"""
from __future__ import annotations

import math

import torch


def q1_element_matrices(
    dx: float,
    dz: float,
    *,
    device=None,
    dtype=torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return consistent mass and Laplacian stiffness matrices for one Q1 cell."""
    if dx <= 0.0 or dz <= 0.0:
        raise ValueError("element spacings must be positive")
    points = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))
    mass = torch.zeros((4, 4), device=device, dtype=dtype)
    stiffness = torch.zeros_like(mass)
    det_jacobian = float(dx * dz / 4.0)
    for eta in points:
        for xi in points:
            shape = torch.tensor(
                [
                    0.25 * (1.0 - xi) * (1.0 - eta),
                    0.25 * (1.0 + xi) * (1.0 - eta),
                    0.25 * (1.0 + xi) * (1.0 + eta),
                    0.25 * (1.0 - xi) * (1.0 + eta),
                ],
                device=device,
                dtype=dtype,
            )
            d_dxi = torch.tensor(
                [
                    -0.25 * (1.0 - eta),
                    0.25 * (1.0 - eta),
                    0.25 * (1.0 + eta),
                    -0.25 * (1.0 + eta),
                ],
                device=device,
                dtype=dtype,
            )
            d_deta = torch.tensor(
                [
                    -0.25 * (1.0 - xi),
                    -0.25 * (1.0 + xi),
                    0.25 * (1.0 + xi),
                    0.25 * (1.0 - xi),
                ],
                device=device,
                dtype=dtype,
            )
            gradient = torch.stack((2.0 * d_dxi / dx, 2.0 * d_deta / dz), dim=1)
            mass += det_jacobian * torch.outer(shape, shape)
            stiffness += det_jacobian * (gradient @ gradient.T)
    return mass, stiffness


def _element_nodes(field: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            field[..., :-1, :-1],
            field[..., :-1, 1:],
            field[..., 1:, 1:],
            field[..., 1:, :-1],
        ),
        dim=-1,
    )


def _assemble_nodes(local: torch.Tensor, height: int, width: int) -> torch.Tensor:
    result = torch.zeros(
        (*local.shape[:-3], height, width),
        device=local.device,
        dtype=local.dtype,
    )
    result[..., :-1, :-1] += local[..., 0]
    result[..., :-1, 1:] += local[..., 1]
    result[..., 1:, 1:] += local[..., 2]
    result[..., 1:, :-1] += local[..., 3]
    return result


def q1_acoustic_weak_residual(
    pressure: torch.Tensor,
    velocity_mps: torch.Tensor,
    *,
    dt_s: float,
    dx_m: float,
    dz_m: float,
    coarsen: int = 4,
    source_off_frame: int = 8,
    cpml_margin_fine: int = 20,
) -> torch.Tensor:
    """Assemble interior nodal residual of inv(v)^2 p_tt - Laplacian(p)=0.

    ``pressure`` is [B,T,Z,X] on the stored 10 m grid.  The residual is
    evaluated after source shutoff on a strided Q1 test mesh.  Top boundary and
    left/right/bottom CPML regions are removed from the returned tensor.
    """
    if pressure.ndim != 4 or velocity_mps.ndim != 3:
        raise ValueError("pressure must be [B,T,Z,X], velocity [B,Z,X]")
    if pressure.shape[0] != velocity_mps.shape[0] or pressure.shape[-2:] != velocity_mps.shape[-2:]:
        raise ValueError("pressure and velocity batch/grid mismatch")
    if pressure.shape[1] < 3 or coarsen <= 0 or dt_s <= 0.0:
        raise ValueError("invalid time or coarsening contract")
    sampled = pressure[..., ::coarsen, ::coarsen]
    velocity = velocity_mps[..., ::coarsen, ::coarsen].clamp_min(1.0)
    batch, time_count, height, width = sampled.shape
    center_indices = torch.arange(1, time_count - 1, device=pressure.device)
    active = center_indices >= int(source_off_frame)
    if not bool(active.any()):
        raise ValueError("source_off_frame leaves no weak-residual time steps")
    p_tt = (
        sampled[:, 2:] - 2.0 * sampled[:, 1:-1] + sampled[:, :-2]
    ) / float(dt_s * dt_s)
    center = sampled[:, 1:-1]
    p_tt = p_tt[:, active]
    center = center[:, active]
    mass, stiffness = q1_element_matrices(
        dx_m * coarsen,
        dz_m * coarsen,
        device=pressure.device,
        dtype=pressure.dtype,
    )
    center_nodes = _element_nodes(center)
    acceleration_nodes = _element_nodes(p_tt)
    inverse_v2_element = _element_nodes(velocity.reciprocal().square()).mean(dim=-1)
    dynamic = inverse_v2_element[..., None] * torch.einsum(
        "ij,...j->...i", mass, acceleration_nodes
    )
    spatial = torch.einsum("ij,...j->...i", stiffness, center_nodes)
    residual = _assemble_nodes(dynamic + spatial, height, width)
    margin = max(1, int(math.ceil(cpml_margin_fine / coarsen)))
    if height <= margin + 2 or width <= 2 * margin + 2:
        raise ValueError("coarsened grid is too small for the CPML exclusion")
    return residual[..., 1:-margin, margin:-margin]


def normalized_fe_weak_loss(
    pressure: torch.Tensor,
    velocity_mps: torch.Tensor,
    *,
    reference_scale: torch.Tensor | None = None,
    **residual_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = q1_acoustic_weak_residual(
        pressure, velocity_mps, **residual_kwargs
    )
    energy = residual.square().mean()
    scale = energy.detach().clamp_min(1.0e-16) if reference_scale is None else reference_scale
    return energy / scale.clamp_min(1.0e-16), scale
