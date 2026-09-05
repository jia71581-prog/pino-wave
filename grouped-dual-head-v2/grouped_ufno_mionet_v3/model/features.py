"""Periodic, Gabor, and propagation-phase coordinate features."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn

from .travel_time import RayTravelTime


@dataclass(frozen=True)
class PhaseFeatureBundle:
    raw: torch.Tensor
    tau_s: torch.Tensor
    phase_rad: torch.Tensor
    causal_feature: torch.Tensor
    gabor: torch.Tensor


def build_phase_features(
    coords_xyz_m: torch.Tensor,
    source_parameters: torch.Tensor,
    travel: RayTravelTime,
    *,
    domain_x_m: float,
    domain_z_m: float,
    domain_t_s: float,
    fourier_bands: int,
    gabor_scales_s: Sequence[float],
) -> PhaseFeatureBundle:
    coords = torch.as_tensor(coords_xyz_m)
    source = torch.as_tensor(source_parameters, dtype=coords.dtype, device=coords.device)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError("coords_xyz_m must be [record,query,3]")
    if source.shape != (coords.shape[0], 5):
        raise ValueError("source_parameters must be [record,5]")
    if travel.seconds.shape != coords.shape[:2]:
        raise ValueError("travel-time shape does not match coordinates")
    if domain_x_m <= 0 or domain_z_m <= 0 or domain_t_s <= 0 or fourier_bands <= 0:
        raise ValueError("domain scales and Fourier-band count must be positive")
    scales = tuple(float(value) for value in gabor_scales_s)
    if not scales or any(value <= 0 for value in scales):
        raise ValueError("Gabor scales must be a nonempty positive sequence")

    x = coords[..., 0]
    z = coords[..., 1]
    time = coords[..., 2]
    source_x = source[:, None, 0]
    source_z = source[:, None, 1]
    f0 = source[:, None, 2].expand_as(time)
    t0 = source[:, None, 3]
    dx = x - source_x
    dz = z - source_z
    radius = torch.sqrt(dx.square() + dz.square() + 1.0e-12)
    relative_time = time - t0
    tau = relative_time - travel.seconds
    phase = 2.0 * math.pi * f0 * tau
    causal = torch.sigmoid(tau / max(domain_t_s * 0.01, 1.0e-6))

    normalized_periodic = torch.stack(
        (x / domain_x_m, z / domain_z_m, tau / domain_t_s),
        dim=-1,
    )
    frequencies = (2.0 ** torch.arange(fourier_bands, device=coords.device, dtype=coords.dtype)) * math.pi
    angles = normalized_periodic.unsqueeze(-1) * frequencies
    fourier = torch.cat(
        (torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)),
        dim=-1,
    )
    gabor_values = []
    for scale in scales:
        window = torch.exp(-0.5 * (tau / scale).square())
        gabor_values.extend((window * torch.sin(phase), window * torch.cos(phase)))
    gabor = torch.stack(gabor_values, dim=-1)
    base = torch.stack(
        (
            x / domain_x_m,
            z / domain_z_m,
            time / domain_t_s,
            dx / domain_x_m,
            dz / domain_z_m,
            radius / math.sqrt(domain_x_m**2 + domain_z_m**2),
            relative_time / domain_t_s,
            travel.seconds / domain_t_s,
            tau / domain_t_s,
            f0 / 50.0,
            travel.path_velocity_mps / 6000.0,
            travel.endpoint_velocity_mps / 6000.0,
            travel.mean_slowness_s_per_m * 6000.0,
            causal,
        ),
        dim=-1,
    )
    raw = torch.cat((base, fourier, torch.sin(phase)[..., None], torch.cos(phase)[..., None], gabor), dim=-1)
    return PhaseFeatureBundle(raw=raw, tau_s=tau, phase_rad=phase, causal_feature=causal, gabor=gabor)


class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, first: bool, omega0: float = 30.0):
        super().__init__()
        if in_features <= 0 or out_features <= 0 or omega0 <= 0:
            raise ValueError("SineLayer dimensions and omega0 must be positive")
        self.omega0 = float(omega0)
        self.first = bool(first)
        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.first:
            bound = 1.0 / self.linear.in_features
        else:
            bound = math.sqrt(6.0 / self.linear.in_features) / self.omega0
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega0 * self.linear(value))


class PhaseAlignedCoordinateEncoder(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        rank: int,
        fourier_bands: int = 6,
        gabor_scales_s: Sequence[float] = (0.01, 0.025, 0.05, 0.1),
        omega0: float = 30.0,
    ) -> None:
        super().__init__()
        self.fourier_bands = int(fourier_bands)
        self.gabor_scales_s = tuple(float(value) for value in gabor_scales_s)
        raw_features = 14 + 6 * self.fourier_bands + 2 + 2 * len(self.gabor_scales_s)
        self.raw_projection = nn.Linear(raw_features, width)
        self.periodic_in = SineLayer(raw_features, width, first=True, omega0=omega0)
        self.periodic_hidden = SineLayer(width, width, first=False, omega0=omega0)
        self.periodic_out = nn.Linear(width, rank)
        self.norm = nn.LayerNorm(width)

    def forward(
        self,
        coords_xyz_m: torch.Tensor,
        source_parameters: torch.Tensor,
        travel: RayTravelTime,
        *,
        domain_x_m: float,
        domain_z_m: float,
        domain_t_s: float,
    ) -> tuple[torch.Tensor, PhaseFeatureBundle]:
        bundle = build_phase_features(
            coords_xyz_m,
            source_parameters,
            travel,
            domain_x_m=domain_x_m,
            domain_z_m=domain_z_m,
            domain_t_s=domain_t_s,
            fourier_bands=self.fourier_bands,
            gabor_scales_s=self.gabor_scales_s,
        )
        periodic = self.periodic_hidden(self.periodic_in(bundle.raw))
        hidden = self.norm(periodic + self.raw_projection(bundle.raw))
        return self.periodic_out(hidden), bundle


__all__ = [
    "PhaseAlignedCoordinateEncoder",
    "PhaseFeatureBundle",
    "SineLayer",
    "build_phase_features",
]
