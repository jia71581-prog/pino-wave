"""Boundary-consistent pressure projections for acoustic operator outputs."""
from __future__ import annotations

import torch
from torch import nn

from fno_acoustic.data_generation.free_surface import pad_pressure_halo


def project_pressure_free_surface(field: torch.Tensor) -> torch.Tensor:
    """Project any ``[..., z, x]`` pressure field onto ``p(z=0)=0``."""
    value = torch.as_tensor(field)
    if value.ndim < 2 or value.shape[-2] < 1:
        raise ValueError("pressure field must have trailing [z,x] dimensions")
    projected = value.clone()
    projected[..., 0, :] = 0.0
    return projected


def free_surface_violation(field: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return maximum and RMS pressure on the top physical row."""
    value = torch.as_tensor(field)
    if value.ndim < 2:
        raise ValueError("pressure field must have trailing [z,x] dimensions")
    top = value[..., 0, :].float()
    return {
        "maximum_absolute": top.abs().amax(),
        "rms": top.square().mean().sqrt(),
    }


def odd_top_three_side_halo(field: torch.Tensor, *, radius: int = 20) -> torch.Tensor:
    """Apply the generator's odd top extension and zero side/bottom halo."""
    return pad_pressure_halo(torch.as_tensor(field), radius=radius, side_mode="zero")


def crop_pressure_halo(field: torch.Tensor, *, radius: int = 20) -> torch.Tensor:
    value = torch.as_tensor(field)
    width = int(radius)
    if width <= 0 or value.shape[-2] <= 2 * width or value.shape[-1] <= 2 * width:
        raise ValueError("halo radius does not fit padded pressure field")
    return value[..., width:-width, width:-width]


class HardFreeSurfaceWrapper(nn.Module):
    """Parameter-free wrapper enforcing the dataset's top pressure boundary."""

    def __init__(self, parent: nn.Module) -> None:
        super().__init__()
        self.parent = parent

    def forward(self, *args, **kwargs):
        return project_pressure_free_surface(self.parent(*args, **kwargs))

    def forward_anchored(self, *args, **kwargs):
        return project_pressure_free_surface(
            self.parent.forward_anchored(*args, **kwargs)
        )


__all__ = [
    "HardFreeSurfaceWrapper",
    "crop_pressure_halo",
    "free_surface_violation",
    "odd_top_three_side_halo",
    "project_pressure_free_surface",
]
