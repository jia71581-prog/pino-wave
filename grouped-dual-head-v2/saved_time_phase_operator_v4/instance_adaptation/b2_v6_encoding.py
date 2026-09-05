"""Velocity/source encoders for offline B2-v6 pretraining ablations."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def enriched_physical_conditioning(
    base_cond: torch.Tensor,
    velocity_mps: torch.Tensor,
    source_f0_hz: torch.Tensor,
    source_t0_s: torch.Tensor,
    *,
    stored_dx_m: float = 10.0,
) -> torch.Tensor:
    """Extend B2's 7 channels with source phase and multiscale medium features.

    Output channels (20 total): base7, f0, t0, sin/cos arrival phase,
    log points-per-wavelength, 5x5/17x17 velocity contrast, and coarse
    log-velocity gradient magnitude, four boundary-proximity maps, and a
    per-instance CPML reference-speed proxy.
    """
    cond = torch.as_tensor(base_cond).float()
    velocity = torch.as_tensor(velocity_mps, device=cond.device).float()
    if cond.ndim != 4 or cond.shape[1] != 7:
        raise ValueError("base_cond must be [B,7,Z,X]")
    if velocity.ndim == 4 and velocity.shape[1] == 1:
        velocity = velocity[:, 0]
    if velocity.shape != (cond.shape[0], cond.shape[2], cond.shape[3]):
        raise ValueError("velocity must match the conditioning batch/grid")
    if stored_dx_m <= 0.0:
        raise ValueError("stored_dx_m must be positive")
    batch, _, height, width = cond.shape
    f0 = torch.as_tensor(source_f0_hz, device=cond.device, dtype=cond.dtype).reshape(-1)
    t0 = torch.as_tensor(source_t0_s, device=cond.device, dtype=cond.dtype).reshape(-1)
    if f0.numel() != batch or t0.numel() != batch or bool((f0 <= 0).any()):
        raise ValueError("f0/t0 must contain one valid value per record")
    f0_norm = ((f0 - 20.0) / 10.0).clamp(-2.0, 2.0)
    t0_norm = ((t0 - 0.10) / 0.05).clamp(-2.0, 2.0)
    f0_map = f0_norm[:, None, None, None].expand(batch, 1, height, width)
    t0_map = t0_norm[:, None, None, None].expand(batch, 1, height, width)
    travel_s = cond[:, 4:5].clamp_min(0.0)
    phase = 2.0 * math.pi * f0[:, None, None, None] * travel_s
    phase_features = torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)
    points_per_wavelength = velocity / (
        f0[:, None, None] * float(stored_dx_m)
    ).clamp_min(1.0e-6)
    wavelength_feature = torch.log2(points_per_wavelength.clamp(1.0, 64.0) / 8.0)
    wavelength_feature = wavelength_feature[:, None].clamp(-3.0, 3.0)
    velocity_field = velocity[:, None]
    blur5 = F.avg_pool2d(velocity_field, kernel_size=5, stride=1, padding=2)
    blur17 = F.avg_pool2d(velocity_field, kernel_size=17, stride=1, padding=8)
    contrast5 = ((velocity_field - blur5) / 1000.0).clamp(-4.0, 4.0)
    contrast17 = ((velocity_field - blur17) / 1000.0).clamp(-4.0, 4.0)
    log_blur = torch.log(blur17.clamp_min(1.0))
    grad_z = F.pad(log_blur[:, :, 2:] - log_blur[:, :, :-2], (0, 0, 1, 1)) * 0.5
    grad_x = F.pad(log_blur[:, :, :, 2:] - log_blur[:, :, :, :-2], (1, 1, 0, 0)) * 0.5
    coarse_interface = (20.0 * torch.sqrt(grad_x.square() + grad_z.square())).clamp(0.0, 4.0)
    x_m = torch.arange(width, device=cond.device, dtype=cond.dtype) * stored_dx_m
    z_m = torch.arange(height, device=cond.device, dtype=cond.dtype) * stored_dx_m
    thickness_m = 200.0
    left = (1.0 - x_m / thickness_m).clamp(0.0, 1.0)[None, None, None, :].expand(batch, 1, height, width)
    right = (1.0 - (x_m[-1] - x_m) / thickness_m).clamp(0.0, 1.0)[None, None, None, :].expand(batch, 1, height, width)
    top = (1.0 - z_m / thickness_m).clamp(0.0, 1.0)[None, None, :, None].expand(batch, 1, height, width)
    bottom = (1.0 - (z_m[-1] - z_m) / thickness_m).clamp(0.0, 1.0)[None, None, :, None].expand(batch, 1, height, width)
    cpml_reference_speed = (velocity.amax((-2, -1)) / 6000.0).clamp(0.0, 1.5)
    cpml_reference_map = cpml_reference_speed[:, None, None, None].expand(batch, 1, height, width)
    return torch.cat(
        (
            cond,
            f0_map,
            t0_map,
            phase_features,
            wavelength_feature,
            contrast5,
            contrast17,
            coarse_interface,
            left,
            right,
            bottom,
            top,
            cpml_reference_map,
        ),
        dim=1,
    )


class ParentLatentConditioner(nn.Module):
    """Project reusable frozen-parent medium/source representations into B2."""

    def __init__(
        self,
        *,
        base_channels: int = 7,
        parent_width: int = 64,
        projected_width: int = 8,
    ) -> None:
        super().__init__()
        if min(base_channels, parent_width, projected_width) <= 0:
            raise ValueError("conditioner widths must be positive")
        self.base_channels = int(base_channels)
        self.parent_width = int(parent_width)
        self.projected_width = int(projected_width)
        self.medium_projection = nn.Conv2d(parent_width, projected_width, 1)
        self.source_map_projection = nn.Conv2d(parent_width, projected_width, 1)
        self.source_hidden_projection = nn.Sequential(
            nn.Linear(parent_width, projected_width),
            nn.GELU(),
            nn.Linear(projected_width, projected_width),
        )

    @property
    def output_channels(self) -> int:
        return self.base_channels + 3 * self.projected_width

    def forward(
        self,
        base_cond: torch.Tensor,
        medium_fullres: torch.Tensor,
        source_map_field: torch.Tensor,
        source_hidden: torch.Tensor,
    ) -> torch.Tensor:
        base = torch.as_tensor(base_cond).float()
        medium = torch.as_tensor(medium_fullres, device=base.device, dtype=base.dtype)
        source_map = torch.as_tensor(source_map_field, device=base.device, dtype=base.dtype)
        hidden = torch.as_tensor(source_hidden, device=base.device, dtype=base.dtype)
        if base.ndim != 4 or base.shape[1] != self.base_channels:
            raise ValueError("base conditioning shape mismatch")
        expected = (base.shape[0], self.parent_width, base.shape[2], base.shape[3])
        if medium.shape != expected or source_map.shape != expected:
            raise ValueError("parent spatial feature shape mismatch")
        if hidden.shape != (base.shape[0], self.parent_width):
            raise ValueError("parent source hidden shape mismatch")
        hidden_map = self.source_hidden_projection(hidden)[:, :, None, None].expand(
            base.shape[0], self.projected_width, base.shape[2], base.shape[3]
        )
        return torch.cat(
            (
                base,
                self.medium_projection(medium),
                self.source_map_projection(source_map),
                hidden_map,
            ),
            dim=1,
        )
