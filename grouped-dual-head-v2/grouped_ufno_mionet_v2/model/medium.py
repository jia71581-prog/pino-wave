from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from grouped_ufno_mionet.model.spectral import SpectralResidualBlock


@dataclass
class MediumEncoding:
    pyramid: tuple[torch.Tensor, ...]
    tokens: torch.Tensor
    rank: torch.Tensor


class NormalizedMediumEncoder(nn.Module):
    def __init__(self, width=64, rank=64, modes=(16, 12, 8, 8), token_grid=8):
        super().__init__(); self.lift=nn.Conv2d(1,width,3,padding=1)
        self.blocks=nn.ModuleList([SpectralResidualBlock(width,m) for m in modes])
        self.down=nn.ModuleList([nn.Conv2d(width,width,3,stride=2,padding=1) for _ in range(3)])
        self.rank_proj=nn.Linear(width,rank); self.token_grid=token_grid

    def forward(self, velocity_normalized):
        x=self.lift(velocity_normalized.float()); pyramid=[]
        for index,block in enumerate(self.blocks):
            x=block(x); pyramid.append(x)
            if index < len(self.down): x=self.down[index](x)
        pooled=torch.nn.functional.adaptive_avg_pool2d(x,(self.token_grid,self.token_grid))
        tokens=pooled.flatten(2).transpose(1,2)
        return MediumEncoding(tuple(pyramid),tokens,self.rank_proj(tokens.mean(1)))
