from __future__ import annotations
import torch
from torch import nn


class SpectralResidualBlock(nn.Module):
    """Small resolution-agnostic spectral/local residual block.

    The Fourier mask deliberately has no hard-coded spatial size, so odd 201
    grids and off-grid smoke tests follow the same path.
    """
    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.local = nn.Conv2d(channels, channels, 3, padding=1)
        self.point = nn.Conv2d(channels, channels, 1)
        self.modes = int(modes)
        self.norm = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        # cuFFT FP16 cannot transform odd 201x201 grids.  Keep only the FFT
        # in FP32 while AMP still accelerates convolutions and MLPs.
        with torch.autocast(device_type=x.device.type, enabled=False):
            fy = torch.fft.rfft2(x.float(), norm="ortho")
            mask = torch.zeros_like(fy)
            my = min(self.modes, h)
            mx = min(self.modes, fy.shape[-1])
            mask[..., :my, :mx] = fy[..., :my, :mx]
            spectral = torch.fft.irfft2(mask, s=(h, w), norm="ortho")
        spectral = spectral.to(dtype=x.dtype)
        return x + torch.nn.functional.gelu(self.norm(self.local(x) + self.point(spectral)))
