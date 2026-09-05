from __future__ import annotations
import torch
from torch import nn


class DenseWavefieldHead(nn.Module):
    def __init__(self, width: int, rank: int):
        super().__init__()
        self.spatial = nn.Conv2d(width, rank, 1)
        self.source_bias = nn.Linear(width, rank)
        self.time = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, rank))

    def forward(self, local_feature: torch.Tensor, source_hidden: torch.Tensor, time_s: torch.Tensor, domain_z=2000.0):
        # [N,R,Z,X], source conditioning is a multiplicative residual bias.
        spatial = self.spatial(local_feature) + self.source_bias(source_hidden)[:, :, None, None]
        if time_s.ndim == 1:
            time_s = time_s[None].expand(local_feature.shape[0], -1)
        n, t = time_s.shape
        coords = torch.stack((torch.zeros_like(time_s), torch.zeros_like(time_s), time_s), dim=-1)
        time = self.time(coords).view(n, t, -1)
        return torch.einsum("ntr,nrzx->ntzx", time, spatial)
