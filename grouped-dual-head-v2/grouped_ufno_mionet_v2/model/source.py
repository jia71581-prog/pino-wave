from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class SourceEncoderV2(nn.Module):
    def __init__(self,width=64,rank=64):
        super().__init__(); self.parameter_mlp=nn.Sequential(nn.Linear(5,width),nn.GELU(),nn.Linear(width,width))
        self.local_proj=nn.Linear(width,width)
        self.map_encoder=nn.Sequential(nn.Conv2d(1,width,5,padding=2),nn.GELU(),nn.Conv2d(width,width,3,padding=1),nn.GELU())
        self.map_proj=nn.Linear(width,width); self.rank_proj=nn.Linear(width,rank)

    def forward(self,source_normalized,source_map,medium,record_to_medium):
        if source_map.ndim != 4 or source_map.shape[1] != 1: raise ValueError("source_map must be [record,1,z,x]")
        if torch.any(source_map < 0) or not torch.allclose(source_map.sum((-2,-1)),torch.ones_like(source_map.sum((-2,-1))),atol=2e-4): raise ValueError("source maps require unit mass")
        local_field=medium.pyramid[0][record_to_medium]
        grid=torch.stack((source_normalized[:,0]*2-1,source_normalized[:,1]*2-1),-1).view(-1,1,1,2)
        local=F.grid_sample(local_field,grid,align_corners=True).flatten(1)
        map_field=self.map_encoder(source_map.float())
        map_local=F.grid_sample(map_field,grid,align_corners=True).flatten(1)
        parameters=source_normalized.clone(); parameters[:,:2]=0; parameters[:,4]=0
        hidden=self.parameter_mlp(parameters)+self.local_proj(local)+self.map_proj(map_local)
        return torch.nn.functional.gelu(hidden),self.rank_proj(hidden),map_field
