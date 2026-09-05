from __future__ import annotations

import torch
from torch import nn


class SpectralConv2d(nn.Module):
    """Resolution-agnostic real-to-real Fourier convolution."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        if min(in_channels, out_channels, modes) <= 0:
            raise ValueError("channels and modes must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = (in_channels * out_channels) ** -0.5
        shape = (in_channels, out_channels, modes, modes)
        self.weight_top = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_bottom = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def _multiply(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bihw,iohw->bohw", inputs, weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("spectral input must have shape [B,C,H,W]")
        batch, _, height, width = inputs.shape
        spectrum = torch.fft.rfft2(inputs, norm="ortho")
        output = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=spectrum.dtype,
            device=inputs.device,
        )
        modes_h = min(self.modes, max(1, height // 2))
        modes_w = min(self.modes, width // 2 + 1)
        output[:, :, :modes_h, :modes_w] = self._multiply(
            spectrum[:, :, :modes_h, :modes_w],
            self.weight_top[:, :, :modes_h, :modes_w],
        )
        output[:, :, -modes_h:, :modes_w] = self._multiply(
            spectrum[:, :, -modes_h:, :modes_w],
            self.weight_bottom[:, :, :modes_h, :modes_w],
        )
        return torch.fft.irfft2(output, s=(height, width), norm="ortho")


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class SpectralResidualBlock(nn.Module):
    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes)
        self.local = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(_group_count(width), width)
        self.activation = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = self.spectral(inputs) + self.local(inputs)
        return inputs + self.activation(self.norm(update))
