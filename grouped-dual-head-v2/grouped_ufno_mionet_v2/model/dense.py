from __future__ import annotations
import math,torch
from torch import nn
from torch.nn import functional as F


class SeparableResidual(nn.Module):
    def __init__(self,width):
        super().__init__(); self.depthwise=nn.Conv2d(width,width,3,padding=1,groups=width); self.pointwise=nn.Conv2d(width,width,1); self.norm=nn.GroupNorm(8 if width%8==0 else 1,width)
    def forward(self,x): return x+F.gelu(self.norm(self.pointwise(self.depthwise(x))))


class TimeConditionedDenseDecoder(nn.Module):
    def __init__(self,width=64,time_block=8):
        super().__init__(); self.time_block=int(time_block); self.fuse=nn.Conv2d(width*4,width,1); self.source=nn.Linear(width,width); self.map_proj=nn.Conv2d(width,width,1)
        frequencies=2**torch.arange(6,dtype=torch.float32)*math.pi; self.register_buffer("frequencies",frequencies)
        geometry_frequencies=2**torch.arange(8,dtype=torch.float32)*math.pi; self.register_buffer("geometry_frequencies",geometry_frequencies)
        self.geometry_proj=nn.Conv2d(51,width,1)
        self.time_mlp=nn.Sequential(nn.Linear(17,width),nn.GELU(),nn.Linear(width,2*width))
        self.blocks=nn.Sequential(SeparableResidual(width),SeparableResidual(width)); self.output=nn.Conv2d(width,1,1)

    def _geometry(self,source_normalized,height,width,dtype,device):
        x=torch.linspace(0,1,width,dtype=dtype,device=device)[None,None,None,:]
        z=torch.linspace(0,1,height,dtype=dtype,device=device)[None,None,:,None]
        dx=x-source_normalized[:,0,None,None,None]
        dz=z-source_normalized[:,1,None,None,None]
        dx=dx.expand(-1,-1,height,-1); dz=dz.expand(-1,-1,-1,width)
        radius=torch.sqrt(dx.square()+dz.square()+1e-8)
        raw=torch.cat((dx,dz,radius),1)
        angles=raw.unsqueeze(2)*self.geometry_frequencies[None,None,:,None,None]
        return torch.cat((raw,torch.sin(angles).flatten(1,2),torch.cos(angles).flatten(1,2)),1)

    def forward(self,medium,mapping,source_hidden,source_normalized,map_field,time_normalized):
        if time_normalized.shape[1] > self.time_block: raise ValueError("time block exceeds configured dense time block")
        full_size=medium.pyramid[0].shape[-2:]; levels=[medium.pyramid[0][mapping]]
        levels.extend(F.interpolate(level[mapping],size=full_size,mode="bilinear",align_corners=True) for level in medium.pyramid[1:])
        geometry=self._geometry(source_normalized,*full_size,levels[0].dtype,levels[0].device)
        spatial=(self.fuse(torch.cat(levels,1))+self.source(source_hidden)[:,:,None,None]
                 +self.map_proj(map_field)+self.geometry_proj(geometry))
        angles=time_normalized[...,None]*self.frequencies; time_features=torch.cat((time_normalized[...,None],torch.sin(angles),torch.cos(angles)),-1)
        relative_time=time_normalized-source_normalized[:,None,3]
        frequency=source_normalized[:,None,2].expand_as(relative_time)
        phase=2*math.pi*60*frequency*relative_time
        time_features=torch.cat((time_features,relative_time[...,None],frequency[...,None],torch.sin(phase)[...,None],torch.cos(phase)[...,None]),-1)
        scale,bias=self.time_mlp(time_features).chunk(2,-1)
        conditioned=spatial[:,None]*(1+scale[:,:,:,None,None])+bias[:,:,:,None,None]
        batch,count,channels,height,width=conditioned.shape; decoded=self.blocks(conditioned.reshape(batch*count,channels,height,width))
        with torch.autocast(device_type=decoded.device.type,enabled=False): output=self.output(decoded.float())
        return output.reshape(batch,count,height,width)
