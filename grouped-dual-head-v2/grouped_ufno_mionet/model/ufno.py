from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from .spectral import SpectralResidualBlock


@dataclass
class MediumEncoding:
    pyramid: tuple[torch.Tensor, ...]
    tokens: torch.Tensor
    medium_rank: torch.Tensor


class UFNOMediumEncoder(nn.Module):
    """Differentiable, multi-scale velocity encoder for 201x201 media."""
    def __init__(self, width=96, modes=(24, 16, 12, 8), rank=64, token_grid=8):
        super().__init__()
        self.width, self.rank, self.token_grid = width, rank, token_grid
        self.lift = nn.Conv2d(1, width, 3, padding=1)
        self.blocks = nn.ModuleList([SpectralResidualBlock(width, m) for m in modes])
        self.down = nn.ModuleList([nn.Conv2d(width, width, 3, stride=2, padding=1) for _ in range(3)])
        self.rank_proj = nn.Linear(width, rank)

    def forward(self, velocity: torch.Tensor) -> MediumEncoding:
        if velocity.ndim != 4 or velocity.shape[1] != 1:
            raise ValueError("velocity must have shape [medium,1,z,x]")
        x = velocity.float() / 2000.0 - 1.0
        x = self.lift(x)
        pyramid = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            pyramid.append(x)
            if i < len(self.down):
                x = self.down[i](x)
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.token_grid, self.token_grid))
        tokens = pooled.flatten(2).transpose(1, 2)
        medium_rank = self.rank_proj(tokens.mean(1))
        return MediumEncoding(tuple(pyramid), tokens, medium_rank)
