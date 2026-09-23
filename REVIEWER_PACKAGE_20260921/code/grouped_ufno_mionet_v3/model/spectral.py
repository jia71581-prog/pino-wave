"""True learned-complex Fourier contractions for the V3 medium and dense paths."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LearnedComplexSpectralConv2d(nn.Module):
    """Contract retained Fourier modes with independent learned complex matrices."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        modes_y: int,
        modes_x: int,
        fft_norm: str = "ortho",
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or modes_y <= 0 or modes_x <= 0:
            raise ValueError("channels and retained modes must be positive")
        if fft_norm not in {"backward", "forward", "ortho"}:
            raise ValueError(f"unsupported FFT normalization: {fft_norm}")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_y = int(modes_y)
        self.modes_x = int(modes_x)
        self.fft_norm = fft_norm
        shape = (self.in_channels, self.out_channels, self.modes_y, self.modes_x, 2)
        self.weight_top = nn.Parameter(torch.empty(shape, dtype=torch.float32))
        self.weight_bottom = nn.Parameter(torch.empty(shape, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        scale = 1.0 / math.sqrt(self.in_channels * self.out_channels)
        nn.init.uniform_(self.weight_top, -scale, scale)
        nn.init.uniform_(self.weight_bottom, -scale, scale)

    def retained_modes(self, height: int, width: int) -> tuple[int, int]:
        if height < 2 or width < 1:
            raise ValueError("spectral input grid is too small")
        modes_y = min(self.modes_y, height // 2)
        modes_x = min(self.modes_x, width // 2 + 1)
        if modes_y < 1 or modes_x < 1:
            raise ValueError("input has no usable retained Fourier modes")
        return modes_y, modes_x

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != self.in_channels:
            raise ValueError(
                f"spectral input must be [batch,{self.in_channels},y,x], got {tuple(value.shape)}"
            )
        height, width = value.shape[-2:]
        modes_y, modes_x = self.retained_modes(height, width)
        with torch.autocast(device_type=value.device.type, enabled=False):
            value_fft = torch.fft.rfft2(value.float(), norm=self.fft_norm)
            output_fft = torch.zeros(
                value.shape[0],
                self.out_channels,
                height,
                width // 2 + 1,
                dtype=value_fft.dtype,
                device=value.device,
            )
            top = torch.view_as_complex(self.weight_top.contiguous())[
                :, :, :modes_y, :modes_x
            ]
            bottom = torch.view_as_complex(self.weight_bottom.contiguous())[
                :, :, :modes_y, :modes_x
            ]
            output_fft[:, :, :modes_y, :modes_x] = torch.einsum(
                "bixy,ioxy->boxy",
                value_fft[:, :, :modes_y, :modes_x],
                top,
            )
            output_fft[:, :, -modes_y:, :modes_x] = torch.einsum(
                "bixy,ioxy->boxy",
                value_fft[:, :, -modes_y:, :modes_x],
                bottom,
            )
            return torch.fft.irfft2(
                output_fft,
                s=(height, width),
                norm=self.fft_norm,
            )


class EquivariantSpectralConv2d(nn.Module):
    """Share complex weights across equal-|k| modes for an isotropic spectral kernel.

    Uses the full ``fft2`` with a circular wavenumber cutoff rather than the
    half-plane ``rfft2`` with a square mode window: a square window and a
    half-plane layout are not themselves invariant under 90 degree rotation,
    so radial weight sharing alone does not buy equivariance.  Buckets key on
    the exact integer ``ky^2 + kx^2``, which is a C4 and reflection invariant of
    the signed frequency lattice, making the resulting operator equivariant to
    the wave equation's spatial translations, rotations, and reflections up to
    floating point.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        cutoff: int,
        fft_norm: str = "ortho",
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or cutoff <= 0:
            raise ValueError("channels and wavenumber cutoff must be positive")
        if fft_norm not in {"backward", "forward", "ortho"}:
            raise ValueError(f"unsupported FFT normalization: {fft_norm}")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.cutoff = int(cutoff)
        self.fft_norm = fft_norm
        self._lattice: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.bucket_count = self._radii(2 * self.cutoff + 1, 2 * self.cutoff + 1)[2]
        shape = (self.in_channels, self.out_channels, self.bucket_count, 2)
        self.weight = nn.Parameter(torch.empty(shape, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        scale = 1.0 / math.sqrt(self.in_channels * self.out_channels)
        nn.init.uniform_(self.weight, -scale, scale)

    def _radii(self, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return the retained-mode mask, per-mode bucket index, and bucket count."""
        frequency_y = torch.fft.fftfreq(height) * height
        frequency_x = torch.fft.fftfreq(width) * width
        squared = (
            frequency_y[:, None].round().long() ** 2
            + frequency_x[None, :].round().long() ** 2
        )
        mask = squared <= self.cutoff * self.cutoff
        retained = torch.unique(squared[mask])
        table = torch.full((int(squared.max()) + 1,), 0, dtype=torch.long)
        table[retained] = torch.arange(retained.numel())
        return mask, torch.where(mask, table[squared], torch.zeros_like(squared)), int(retained.numel())

    def _cached_lattice(self, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Cache the flat positions of retained modes and their bucket ids.

        Only the modes inside the circular cutoff are contracted.  Masking the
        full ``height x width`` lattice instead would make the contraction scale
        with the grid rather than with the retained mode count, which costs an
        order of magnitude at the 201x201 production grid.
        """
        key = (int(height), int(width))
        cached = self._lattice.get(key)
        if cached is None or cached[0].device != device:
            mask, index, count = self._radii(height, width)
            if count != self.bucket_count:
                raise ValueError(
                    f"grid {key} yields {count} radial buckets, expected {self.bucket_count}; "
                    "the cutoff must be resolvable on this grid"
                )
            flat = mask.reshape(-1).nonzero(as_tuple=True)[0]
            cached = (flat.to(device), index.reshape(-1)[flat].to(device))
            self._lattice[key] = cached
        return cached

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != self.in_channels:
            raise ValueError(
                f"spectral input must be [batch,{self.in_channels},y,x], got {tuple(value.shape)}"
            )
        height, width = value.shape[-2:]
        if min(height, width) < 2 * self.cutoff + 1:
            raise ValueError("input grid is too small to resolve the wavenumber cutoff")
        mask, index = self._cached_lattice(height, width, value.device)
        with torch.autocast(device_type=value.device.type, enabled=False):
            value_fft = torch.fft.fft2(value.float(), norm=self.fft_norm)
            flat = value_fft.reshape(*value_fft.shape[:2], -1)[:, :, mask]
            kernel = torch.view_as_complex(self.weight.contiguous())[:, :, index]
            contracted = torch.einsum("bik,iok->bok", flat, kernel)
            output_fft = torch.zeros(
                value.shape[0],
                self.out_channels,
                height * width,
                dtype=value_fft.dtype,
                device=value.device,
            )
            output_fft[:, :, mask] = contracted
            return torch.fft.ifft2(
                output_fft.reshape(value.shape[0], self.out_channels, height, width),
                norm=self.fft_norm,
            ).real


def _group_count(width: int) -> int:
    for groups in (8, 4, 2, 1):
        if width % groups == 0:
            return groups
    return 1


class ComplexSpectralResidualBlock(nn.Module):
    """Low-rank learned spectral path plus the useful V2 local residual path."""

    def __init__(
        self,
        *,
        width: int,
        spectral_rank: int,
        modes_y: int,
        modes_x: int,
        equivariant: bool = False,
    ) -> None:
        super().__init__()
        if width <= 0 or spectral_rank <= 0:
            raise ValueError("width and spectral rank must be positive")
        self.in_projection = nn.Conv2d(width, spectral_rank, kernel_size=1)
        self.spectral: nn.Module
        if equivariant:
            self.spectral = EquivariantSpectralConv2d(
                spectral_rank,
                spectral_rank,
                cutoff=min(int(modes_y), int(modes_x)),
            )
        else:
            self.spectral = LearnedComplexSpectralConv2d(
                spectral_rank,
                spectral_rank,
                modes_y=modes_y,
                modes_x=modes_x,
            )
        self.out_projection = nn.Conv2d(spectral_rank, width, kernel_size=1)
        self.local_depthwise = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            padding=1,
            groups=width,
        )
        self.local_pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(_group_count(width), width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError("residual block input must be [batch,channel,y,x]")
        spectral = self.out_projection(self.spectral(self.in_projection(value).float()))
        local = self.local_pointwise(self.local_depthwise(value))
        return value + F.gelu(self.norm(spectral + local))


__all__ = [
    "ComplexSpectralResidualBlock",
    "EquivariantSpectralConv2d",
    "LearnedComplexSpectralConv2d",
]
