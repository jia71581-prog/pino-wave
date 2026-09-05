from __future__ import annotations
import math
import torch
from torch import nn


class CoordinateTrunk(nn.Module):
    def __init__(self, width=96, rank=64, domain_x_m=2000., domain_z_m=2000.):
        super().__init__()
        self.domain_x_m, self.domain_z_m = domain_x_m, domain_z_m
        frequencies = 2.0 ** torch.arange(6, dtype=torch.float32) * math.pi
        self.register_buffer("frequencies", frequencies)
        self.proj = nn.Sequential(nn.Linear(3 + 3 * 12, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.rank = nn.Linear(width, rank)

    def forward(self, coords: torch.Tensor):
        if coords.ndim == 2:
            coords = coords.unsqueeze(0)
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError("coords must have shape [records,queries,3] (x,z,t)")
        c = coords.float().clone()
        c[..., 0] = c[..., 0] / self.domain_x_m
        c[..., 1] = c[..., 1] / self.domain_z_m
        c[..., 2] = c[..., 2] / 1.2
        angles = c.unsqueeze(-1) * self.frequencies
        features = torch.cat((c, torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)), dim=-1)
        hidden = self.proj(features)
        return hidden, self.rank(hidden)
