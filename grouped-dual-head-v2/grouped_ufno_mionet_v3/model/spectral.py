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
    ) -> None:
        super().__init__()
        if width <= 0 or spectral_rank <= 0:
            raise ValueError("width and spectral rank must be positive")
        self.in_projection = nn.Conv2d(width, spectral_rank, kernel_size=1)
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


__all__ = ["ComplexSpectralResidualBlock", "LearnedComplexSpectralConv2d"]
