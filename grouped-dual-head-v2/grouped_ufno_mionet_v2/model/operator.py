from __future__ import annotations
import torch
from torch import nn
from .medium import NormalizedMediumEncoder
from .source import SourceEncoderV2
from .query import ContinuousQueryHeadV2
from .dense import TimeConditionedDenseDecoder


class DualHeadWaveOperator(nn.Module):
    def __init__(self,width=64,rank=64,modes=(16,12,8,8),heads=4,dense_time_block=8,domain_x_m=2000.,domain_z_m=2000.):
        super().__init__(); self.domain_x_m=domain_x_m; self.domain_z_m=domain_z_m
        self.medium_encoder=NormalizedMediumEncoder(width,rank,modes); self.source_encoder=SourceEncoderV2(width,rank); self.query_head=ContinuousQueryHeadV2(width,rank,heads); self.dense_decoder=TimeConditionedDenseDecoder(width,dense_time_block)

    def prepare(self,velocity_mps,source,source_map,normalizer,record_to_medium=None):
        source=torch.as_tensor(source,dtype=torch.float32,device=next(self.parameters()).device); velocity=torch.as_tensor(velocity_mps,dtype=torch.float32,device=source.device); source_map=torch.as_tensor(source_map,dtype=torch.float32,device=source.device)
        if record_to_medium is None: record_to_medium=torch.arange(len(source),device=source.device) if len(velocity)>1 else torch.zeros(len(source),dtype=torch.long,device=source.device)
        else: record_to_medium=torch.as_tensor(record_to_medium,dtype=torch.long,device=source.device)
        medium=self.medium_encoder(normalizer.encode_velocity(velocity)); source_norm=normalizer.encode_source(source)
        source_hidden,source_rank,map_field=self.source_encoder(source_norm,source_map,medium,record_to_medium)
        return source,source_norm,record_to_medium,medium,source_hidden,source_rank,map_field

    def query_normalized(self,velocity_mps,source,source_map,coords,normalizer,record_to_medium=None,chunk_size=None):
        cache=self.prepare(velocity_mps,source,source_map,normalizer,record_to_medium); source,source_normalized,mapping,medium,hidden,rank,_=cache
        coords=torch.as_tensor(coords,dtype=torch.float32,device=source.device); norm=coords.clone(); norm[...,0]/=self.domain_x_m; norm[...,1]/=self.domain_z_m; norm[...,2]/=1.2
        step=coords.shape[1] if chunk_size is None else int(chunk_size); output=[]
        for start in range(0,coords.shape[1],step):
            current=norm[:,start:start+step]; raw=self.query_head(current,source_normalized,medium,mapping,hidden,rank)
            surface=torch.tanh(coords[:,start:start+step,1].clamp_min(0)/20).square(); output.append(raw*surface)
        return torch.cat(output,1)

    def query_pressure(self,**kwargs):
        source=torch.as_tensor(kwargs["source"],dtype=torch.float32,device=next(self.parameters()).device); normalized=self.query_normalized(**kwargs)
        return kwargs["normalizer"].decode_pressure(normalized.float(),source[:,4:5])

    def _dense_from_cache(self,cache,time_s):
        source,source_normalized,mapping,medium,hidden,_,map_field=cache; time=torch.as_tensor(time_s,dtype=torch.float32,device=source.device)
        if time.ndim==1: time=time[None].expand(len(source),-1)
        normalized=self.dense_decoder(medium,mapping,hidden,source_normalized,map_field,time/1.2)
        z=torch.linspace(0,self.domain_z_m,normalized.shape[-2],device=normalized.device); surface=torch.tanh(z/20).square()[None,None,:,None]
        return normalized*surface

    def dense_normalized(self,velocity_mps,source,source_map,time_s,normalizer,record_to_medium=None):
        return self._dense_from_cache(self.prepare(velocity_mps,source,source_map,normalizer,record_to_medium),time_s)

    def predict_wavefield(self,velocity_mps,source,source_map,time_s,normalizer,record_to_medium=None):
        source_tensor=torch.as_tensor(source,dtype=torch.float32,device=next(self.parameters()).device); times=torch.as_tensor(time_s,dtype=torch.float32,device=source_tensor.device)
        if times.ndim==1: times=times[None].expand(len(source_tensor),-1)
        cache=self.prepare(velocity_mps,source_tensor,source_map,normalizer,record_to_medium); block=self.dense_decoder.time_block
        normalized=torch.cat([self._dense_from_cache(cache,times[:,start:start+block]) for start in range(0,times.shape[1],block)],1)
        return normalizer.decode_pressure(normalized.float(),source_tensor[:,4,None,None,None])
