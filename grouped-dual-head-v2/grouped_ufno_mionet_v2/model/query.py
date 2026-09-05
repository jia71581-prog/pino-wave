from __future__ import annotations
import math,torch
from torch import nn
from torch.nn import functional as F


class ContinuousQueryHeadV2(nn.Module):
    def __init__(self,width=64,rank=64,heads=4):
        super().__init__(); freq=2**torch.arange(8,dtype=torch.float32)*math.pi; self.register_buffer("frequencies",freq)
        self.trunk=nn.Sequential(nn.Linear(71,width),nn.GELU(),nn.Linear(width,width),nn.GELU())
        self.trunk_rank=nn.Linear(width,rank); self.attention=nn.MultiheadAttention(width,heads,batch_first=True)
        self.local=nn.Linear(width,width); self.output=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,1)); self.base_scale=nn.Parameter(torch.tensor(.1))

    def forward(self,coords_normalized,source_normalized,medium,medium_index,source_hidden,source_rank):
        dx=coords_normalized[...,0]-source_normalized[:,None,0]
        dz=coords_normalized[...,1]-source_normalized[:,None,1]
        radius=torch.sqrt(dx.square()+dz.square()+1e-8)
        relative_time=coords_normalized[...,2]-source_normalized[:,None,3]
        relative=torch.stack((dx,dz,radius,relative_time),-1)
        angles=relative.unsqueeze(-1)*self.frequencies
        frequency=source_normalized[:,None,2].expand_as(relative_time)
        phase=2*math.pi*60*frequency*relative_time
        features=torch.cat((relative,torch.sin(angles).flatten(-2),torch.cos(angles).flatten(-2),
                            frequency[...,None],torch.sin(phase)[...,None],torch.cos(phase)[...,None]),-1)
        hidden=self.trunk(features); trunk_rank=self.trunk_rank(hidden)
        base=torch.einsum("br,br,bqr->bq",medium.rank[medium_index],source_rank,trunk_rank)
        grid=coords_normalized[...,:2]*2-1
        local=F.grid_sample(medium.pyramid[0][medium_index],grid[:,None],align_corners=True).squeeze(2).transpose(1,2)
        query=hidden+source_hidden[:,None]+self.local(local)
        attended,_=self.attention(query,medium.tokens[medium_index],medium.tokens[medium_index],need_weights=False)
        return base+self.base_scale*self.output(attended+query).squeeze(-1)
