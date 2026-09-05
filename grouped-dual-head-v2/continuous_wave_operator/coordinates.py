from __future__ import annotations

import torch

from .config import DomainConfig


def _validate_query_shape(query_coords: torch.Tensor) -> None:
    if query_coords.ndim < 1 or query_coords.shape[-1] != 3:
        raise ValueError("query_coords must end with physical (x, z, t) coordinates")
    if not query_coords.is_floating_point():
        raise TypeError("query_coords must be floating point")


def normalize_query_coordinates(query_coords: torch.Tensor, domain: DomainConfig) -> torch.Tensor:
    """Map physical ``(x,z,t)`` coordinates to the closed interval ``[-1,1]``."""

    _validate_query_shape(query_coords)
    scale = query_coords.new_tensor([domain.lx_m, domain.lz_m, domain.t_end_s])
    return 2.0 * query_coords / scale - 1.0


def physical_output_gate(query_coords: torch.Tensor, domain: DomainConfig) -> torch.Tensor:
    """Smoothly impose zero pressure and zero time slope at ``t=0``, and pressure at ``z=0``."""

    _validate_query_shape(query_coords)
    z = query_coords[..., 1]
    time = query_coords[..., 2]
    time_gate = 1.0 - torch.exp(-torch.square(time / domain.initial_gate_scale_s))
    surface_gate = 1.0 - torch.exp(-z / domain.surface_gate_scale_m)
    return time_gate * surface_gate
