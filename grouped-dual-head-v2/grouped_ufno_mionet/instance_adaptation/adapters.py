"""Small identity-initialized adaptation layers."""
from __future__ import annotations

import torch
from torch import nn


class LowRankLinearDelta(nn.Module):
    """Frozen linear map plus a zero-initialized rank-limited update."""

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.down = nn.Parameter(torch.empty(rank, base.in_features))
        self.up = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.down, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.down.T @ self.up.T)


class SpatialTemporalFiLM(nn.Module):
    """Rank-basis FiLM transform whose initial map is the exact identity."""

    def __init__(self, rank: int):
        super().__init__()
        self.scale_delta = nn.Parameter(torch.zeros(1, 1, rank, 1, 1))
        self.bias_delta = nn.Parameter(torch.zeros(1, 1, rank, 1, 1))

    def forward(self, basis: torch.Tensor) -> torch.Tensor:
        if basis.ndim != 5 or basis.shape[2] != self.scale_delta.shape[2]:
            raise ValueError("basis must have [batch,time,rank,z,x] axes")
        return basis * (1 + self.scale_delta) + self.bias_delta
