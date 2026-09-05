from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .spectral_layers import SpectralResidualBlock


@dataclass(frozen=True)
class MediumCache:
    local_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    global_tokens: torch.Tensor
    normalized_velocity: torch.Tensor


class MediumEncoder(nn.Module):
    """Encode a velocity model once into reusable local and global features."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        width = config.width
        self.input_projection = nn.Conv2d(3, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            SpectralResidualBlock(width, modes) for modes in config.spectral_modes
        )
        self.downsamples = nn.ModuleList(
            nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1) for _ in range(3)
        )

    def forward(self, velocity_mps: torch.Tensor) -> MediumCache:
        if velocity_mps.ndim != 4 or velocity_mps.shape[1] != 1:
            raise ValueError("velocity_mps must have shape [B,1,H,W]")
        if not torch.isfinite(velocity_mps).all() or torch.any(velocity_mps <= 0.0):
            raise ValueError("velocity_mps must be finite and positive")
        ratio = velocity_mps / self.config.reference_velocity_mps
        inputs = torch.cat((ratio, ratio.square(), ratio.reciprocal()), dim=1)
        features = self.input_projection(inputs)
        scales: list[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            features = block(features)
            scales.append(features)
            if index < len(self.downsamples):
                features = self.downsamples[index](features)
        pooled = F.adaptive_avg_pool2d(
            scales[-1], (self.config.token_grid_size, self.config.token_grid_size)
        )
        tokens = pooled.flatten(2).transpose(1, 2).contiguous()
        return MediumCache(
            local_features=(scales[0], scales[1], scales[2], scales[3]),
            global_tokens=tokens,
            normalized_velocity=ratio,
        )
