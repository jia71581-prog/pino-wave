"""Frequency-local Background-WFP with a separate exterior-CPML auxiliary head."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from grouped_ufno_mionet_v3.model.spectral import LearnedComplexSpectralConv2d


def _groups(width: int) -> int:
    for value in (8, 4, 2, 1):
        if width % value == 0:
            return value
    return 1


class ComplexFrequencyWindowConv2d(nn.Module):
    """Mix neighboring full-support spatial Fourier modes in a compact window."""

    def __init__(self, channels: int, *, rank: int, radius: int) -> None:
        super().__init__()
        width, latent, window = int(channels), int(rank), int(radius)
        if min(width, latent) <= 0 or window <= 0:
            raise ValueError("WFP channels/rank/radius must be positive")
        self.channels = width
        self.rank = latent
        self.radius = window
        self.channel_in = nn.Conv2d(width, latent, 1, bias=False)
        self.channel_out = nn.Conv2d(latent, width, 1, bias=False)
        kernel = 2 * window + 1
        scale = 1.0 / math.sqrt(latent * kernel * kernel)
        self.weight_real = nn.Parameter(
            torch.empty(latent, latent, kernel, kernel).uniform_(-scale, scale)
        )
        self.weight_imag = nn.Parameter(
            torch.empty(latent, latent, kernel, kernel).uniform_(-scale, scale)
        )

    def _pad(self, value: torch.Tensor) -> torch.Tensor:
        radius = self.radius
        # Spatial-frequency z is signed and periodic. The one-sided rFFT x axis
        # is not periodic, so its two ends are zero padded rather than wrapped.
        value = F.pad(value, (radius, radius, 0, 0))
        return F.pad(value, (0, 0, radius, radius), mode="circular")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != self.channels:
            raise ValueError(f"WFP input must be [B,{self.channels},Z,X]")
        shape = value.shape[-2:]
        with torch.autocast(device_type=value.device.type, enabled=False):
            latent = self.channel_in(value.float())
            spectrum = torch.fft.rfft2(latent, norm="ortho")
            real, imag = self._pad(spectrum.real), self._pad(spectrum.imag)
            real_out = F.conv2d(real, self.weight_real) - F.conv2d(
                imag, self.weight_imag
            )
            imag_out = F.conv2d(real, self.weight_imag) + F.conv2d(
                imag, self.weight_real
            )
            mixed = torch.complex(real_out, imag_out)
            field = torch.fft.irfft2(mixed, s=shape, norm="ortho")
            return self.channel_out(field)


class _WFPBlock(nn.Module):
    def __init__(self, width: int, *, rank: int, radius: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_groups(width), width)
        self.frequency = ComplexFrequencyWindowConv2d(
            width, rank=rank, radius=radius
        )
        self.local = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.Conv2d(width, width, 1),
        )
        self.film = nn.Linear(5, 2 * width)

    def forward(self, value: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(scalars.float()).chunk(2, dim=1)
        driven = self.norm(value)
        driven = driven * (1.0 + 0.1 * torch.tanh(scale)[:, :, None, None])
        driven = driven + 0.1 * shift[:, :, None, None]
        update = self.frequency(driven) + self.local(driven)
        return value + F.gelu(update)


class _FNOBlock(nn.Module):
    def __init__(self, width: int, *, rank: int, modes: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_groups(width), width)
        self.in_projection = nn.Conv2d(width, rank, 1)
        self.spectral = LearnedComplexSpectralConv2d(
            rank, rank, modes_y=modes, modes_x=modes
        )
        self.out_projection = nn.Conv2d(rank, width, 1)
        self.local = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.Conv2d(width, width, 1),
        )
        self.film = nn.Linear(5, 2 * width)

    def forward(self, value: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(scalars.float()).chunk(2, dim=1)
        driven = self.norm(value)
        driven = driven * (1.0 + 0.1 * torch.tanh(scale)[:, :, None, None])
        driven = driven + 0.1 * shift[:, :, None, None]
        update = self.out_projection(self.spectral(self.in_projection(driven)))
        update = update + self.local(driven)
        return value + F.gelu(update)


class BackgroundFrequencyOperator(nn.Module):
    """Predict one complex temporal coefficient and its CPML auxiliary state.

    Both arms use the same MIONet-style separate medium/source encoders and the
    same two output heads. Only the global propagation block differs, making
    ``wfp`` versus ``fno`` a paired architectural comparison.
    """

    def __init__(
        self,
        *,
        medium_channels: int,
        source_channels: int,
        width: int = 32,
        rank: int = 16,
        depth: int = 4,
        arm: str = "wfp",
        radii: tuple[int, ...] = (1, 2, 3, 4),
        fno_modes: int = 32,
        physical_shape: tuple[int, int] = (201, 201),
        cpml_layers: int = 20,
    ) -> None:
        super().__init__()
        if arm not in {"wfp", "fno"}:
            raise ValueError("arm must be wfp or fno")
        if depth <= 0 or width <= 0 or rank <= 0:
            raise ValueError("operator dimensions must be positive")
        if len(radii) < depth:
            raise ValueError("one WFP radius is required per block")
        self.arm = arm
        self.width = int(width)
        self.physical_shape = tuple(int(value) for value in physical_shape)
        self.cpml_layers = int(cpml_layers)
        self.medium_stem = nn.Sequential(
            nn.Conv2d(int(medium_channels), width, 3, padding=1),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
        )
        self.source_stem = nn.Sequential(
            nn.Conv2d(int(source_channels), width, 3, padding=1),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
        )
        self.fuse = nn.Conv2d(3 * width, width, 1)
        if arm == "wfp":
            self.blocks = nn.ModuleList(
                [_WFPBlock(width, rank=rank, radius=int(radii[i])) for i in range(depth)]
            )
        else:
            self.blocks = nn.ModuleList(
                [_FNOBlock(width, rank=rank, modes=fno_modes) for _ in range(depth)]
            )
        self.physical_head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, 2, 1)
        )
        self.cpml_head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, 10, 1)
        )

    def forward(
        self,
        medium: torch.Tensor,
        source: torch.Tensor,
        scalars: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if medium.ndim != 4 or source.ndim != 4 or scalars.ndim != 2:
            raise ValueError("operator inputs must be [B,C,Z,X], [B,C,Z,X], [B,5]")
        if medium.shape[0] != source.shape[0] or medium.shape[-2:] != source.shape[-2:]:
            raise ValueError("medium and source batches/grids must match")
        medium_latent = self.medium_stem(medium)
        source_latent = self.source_stem(source)
        value = self.fuse(
            torch.cat((medium_latent, source_latent, medium_latent * source_latent), dim=1)
        )
        for block in self.blocks:
            value = block(value, scalars)
        z, x = self.physical_shape
        start = self.cpml_layers
        physical = self.physical_head(value[..., :z, start : start + x])
        physical = physical.clone()
        physical[..., 0, :] = 0.0
        cpml = self.cpml_head(value)
        cpml = cpml.clone()
        # Pressure real/imaginary channels satisfy the exact top and outer edge.
        cpml[:, :2, 0, :] = 0.0
        cpml[:, :2, -1, :] = 0.0
        cpml[:, :2, :, 0] = 0.0
        cpml[:, :2, :, -1] = 0.0
        return physical, cpml


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


__all__ = [
    "BackgroundFrequencyOperator",
    "ComplexFrequencyWindowConv2d",
    "parameter_count",
]
