"""Differentiable causal physics losses for one-source acoustic instances."""
from __future__ import annotations

import torch


def lwc84_residual(field, velocity_mps, *, dt: float, dx: float, dz: float, observed_indices: tuple[int, int]):
    """Mean squared homogeneous acoustic residual strictly after the two visible frames."""
    if field.ndim != 4 or velocity_mps.ndim != 4:
        raise ValueError("field [B,T,Z,X] and velocity [B,1,Z,X] are required")
    start = int(observed_indices[1]) + 1
    if start < 1 or field.shape[1] - 1 <= start:
        return field.sum() * 0.0
    center = field[:, 1:-1, 1:-1, 1:-1]
    dtt = (field[:, 2:, 1:-1, 1:-1] - 2 * center + field[:, :-2, 1:-1, 1:-1]) / (dt * dt)
    dxx = (field[:, 1:-1, 1:-1, 2:] - 2 * center + field[:, 1:-1, 1:-1, :-2]) / (dx * dx)
    dzz = (field[:, 1:-1, 2:, 1:-1] - 2 * center + field[:, 1:-1, :-2, 1:-1]) / (dz * dz)
    residual = dtt - velocity_mps[:, :, 1:-1, 1:-1].square() * (dxx + dzz)
    valid = max(0, start - 1)
    return residual[:, valid:].square().mean() if valid < residual.shape[1] else field.sum() * 0.0
