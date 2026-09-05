"""V6 RAM-only record cache, smoke reuse, and complete run identity."""
from __future__ import annotations
from dataclasses import dataclass
import math,time
from typing import Any,Callable,Mapping,Sequence
import torch
from .r16_dscp import c1_causal_mask

from .r16_dscp_engine_v3 import AccessLedger,canonical_sha,smoke_gates
from .r16_dscp_engine_v4 import CandidateSpool,V4Prepared
from .r16_dscp_engine_v5 import V5ProductionBackend

HOST_RESERVE_BYTES=16*1024**3
ROLE_CAPACITY={"smoke":3,"pilot":48,"scale-probe-1":3,"scale-probe-4":3,"long":216,"final-train-confirm":1,"validation-once":1,"test-once":1}

class CacheBindingRefusal(RuntimeError):pass
class CacheMemoryRefusal(RuntimeError):pass

@dataclass(frozen=True)
class V6CacheKey:
    split:str;sample_id:str;group_id:str;nontruth_sha256:str
    parent_sha256:str;basis_sha256:str;feature_code_sha256:str
    def digest(self)->str:return canonical_sha(self.__dict__)

@dataclass
class V6CacheEntry:
    key:V6CacheKey; public:Any; parent:torch.Tensor; travel:torch.Tensor
    args:tuple[Any,...]; features:torch.Tensor; route_index:int; condition:float
    truth:torch.Tensor; bytes:int

def tensor_bytes(value:Any)->int:
    if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
    if isinstance(value,Mapping):return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,(tuple,list)):return sum(tensor_bytes(v) for v in value)
    return 0

def mem_available_bytes()->int:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):return int(line.split()[1])*1024
    raise CacheMemoryRefusal("MemAvailable unavailable")

def stage_memory_gate(role:str,record_bytes:int,available:int|None=None)->Mapping[str,Any]:
    if role not in ROLE_CAPACITY:raise ValueError(role)
    observed=mem_available_bytes() if available is None else int(available);cache_bytes=ROLE_CAPACITY[role]*int(record_bytes);required=cache_bytes+HOST_RESERVE_BYTES
    return {"role":role,"capacity":ROLE_CAPACITY[role],"record_bytes":int(record_bytes),"cache_bytes":cache_bytes,"host_reserve_bytes":HOST_RESERVE_BYTES,"mem_available_bytes":observed,"required_bytes":required,"passed":observed>=required}

class V6RecordCache:
    """Process-local RAM cache. It deliberately has no serialization API."""
    def __init__(self,role:str,*,available:Callable[[],int]=mem_available_bytes):
        if role not in ROLE_CAPACITY:raise ValueError(role)
        self.role=role;self.capacity=ROLE_CAPACITY[role];self.available=available;self.entries={};self.current_bytes=0;self.peak_bytes=0;self.peak_items=0;self.hits=0;self.misses=0
    def admit(self,expected_bytes:int)->None:
        required=int(expected_bytes)+HOST_RESERVE_BYTES
        if self.available()<required:raise CacheMemoryRefusal(f"host cache requires {required} bytes")
    def put(self,entry:V6CacheEntry)->None:
        digest=entry.key.digest()
        if digest in self.entries:raise CacheBindingRefusal("duplicate cache key")
        if len(self.entries)>=self.capacity:raise CacheMemoryRefusal("role cache capacity exceeded")
        self.admit(self.current_bytes+entry.bytes);self.entries[digest]=entry;self.current_bytes+=entry.bytes;self.peak_bytes=max(self.peak_bytes,self.current_bytes);self.peak_items=max(self.peak_items,len(self.entries));self.misses+=1
    def get(self,key:V6CacheKey,ledger:AccessLedger)->V6CacheEntry:
        digest=key.digest()
        if digest not in self.entries:raise CacheBindingRefusal("cache key miss or binding drift")
        self.hits+=1;ledger.add("cache_hit",key_sha256=digest,cache_bytes=self.current_bytes);return self.entries[digest]
    def release(self,key:V6CacheKey)->None:
        entry=self.entries.pop(key.digest());self.current_bytes-=entry.bytes
    def clear(self)->None:self.entries.clear();self.current_bytes=0
    def payload(self)->Mapping[str,Any]:return {"role":self.role,"capacity":self.capacity,"current_bytes":self.current_bytes,"peak_bytes":self.peak_bytes,"peak_items":self.peak_items,"hits":self.hits,"misses":self.misses,"serialization":False}

class V6ProductionBackend(V5ProductionBackend):
    def __init__(self,*args,cache_role:str,parent_sha256:str,basis_sha256:str,feature_code_sha256:str,nontruth_by_sample:Mapping[str,str],cache_available:Callable[[],int]=mem_available_bytes,**kwargs):
        super().__init__(*args,**kwargs);self.spool=CandidateSpool(self.run_dir,run_digest=self.lineage.run_digest,rank=int(__import__("os").environ.get("RANK","0")),owned_root="/dev/shm/r16_dscp_v6");self.cache=V6RecordCache(cache_role,available=cache_available);self.parent_binding=parent_sha256;self.basis_binding=basis_sha256;self.feature_binding=feature_code_sha256;self.nontruth_by_sample=dict(nontruth_by_sample);self.index_keys={}
    def _key(self,public)->V6CacheKey:
        if public.sample_id not in self.nontruth_by_sample:raise CacheBindingRefusal("nontruth hash unavailable")
        return V6CacheKey(self.split,public.sample_id,public.group_id,self.nontruth_by_sample[public.sample_id],self.parent_binding,self.basis_binding,self.feature_binding)
    def preload(self,index:int)->V6CacheEntry:
        prepared=super().prepare(index,"preload");truth=super().open_truth(prepared).detach().cpu().float();key=self._key(prepared.public);size=tensor_bytes((prepared.public.velocity_mps,prepared.public.source_parameters,prepared.public.source_map,prepared.public.observed_wavefield,prepared.parent,prepared.travel,prepared.args,prepared.features,truth));entry=V6CacheEntry(key,prepared.public,prepared.parent,prepared.travel,tuple(x.detach().cpu() if isinstance(x,torch.Tensor) else x for x in prepared.args),prepared.features,prepared.route_index,prepared.condition,truth,size);self.cleanup(prepared);self.cache.put(entry);self.index_keys[int(index)]=key;return entry
    def cached_prepare(self,index:int,key_suffix:str="cached")->V4Prepared:
        ledger=AccessLedger(self.split);entry=self.cache.get(self.index_keys[int(index)],ledger);args=tuple(x.to(self.device) if isinstance(x,torch.Tensor) else x for x in entry.args);coefficient=self._coefficients(entry.features.to(self.device).float(),entry.route_index,entry.route_index<0,entry.condition);adapted=self._materialize(args[7],coefficient,entry.route_index,entry.public.observed_indices[1]);ledger.add("autocast",enabled=self.autocast_enabled,dtype=self.autocast_dtype);(seal,spool_s)=self.timer.call(lambda:self.spool.seal(f"{entry.public.sample_id}:{key_suffix}",adapted.detach().cpu(),ledger));self.protocol.spool_io_s.append(spool_s);self.active.append(seal["path"]);return V4Prepared(entry.public,entry.parent,entry.travel,args,entry.features,entry.route_index,entry.condition,coefficient.detach().cpu(),seal,ledger)
    def _cached_truth(self,prepared:V4Prepared)->torch.Tensor:return self.cache.get(self._key(prepared.public),prepared.ledger).truth
    def update_cached(self,index:int)->Mapping[str,Any]:
        prepared=self.cached_prepare(index,"update");truth=self._cached_truth(prepared);losses,_=self._forward_loss(prepared,truth[None]);norms=self._finite_backward(losses["total"],reset=True,step=True);self.cleanup(prepared);return {"loss":float(losses["total"]),**norms,"ledger_digest":prepared.ledger.digest()}
    def measure_cached(self,index:int)->Mapping[str,Any]:
        prepared=self.cached_prepare(index,"scale");truth=self._cached_truth(prepared);losses,_=self._forward_loss(prepared,truth[None]);norms=self._finite_backward(losses["total"],reset=True,step=False);self.optimizer.zero_grad(set_to_none=True);self.cleanup(prepared);return {"loss":float(losses["total"]),**norms,"sample_id":prepared.public.sample_id,"ledger_digest":prepared.ledger.digest()}
    def backward_cached(self,index:int,scale:float)->Mapping[str,Any]:
        prepared=self.cached_prepare(index,"long");truth=self._cached_truth(prepared);losses,_=self._forward_loss(prepared,truth[None]);norms=self._finite_backward(losses["total"]*float(scale),reset=False,step=False,clip=False);self.cleanup(prepared);return {"loss":float(losses["total"]),**norms,"ledger_digest":prepared.ledger.digest()}
    def score_cached(self,index:int,release:bool=False)->Mapping[str,Any]:
        prepared=self.cached_prepare(index,"score");truth=self._cached_truth(prepared);row=self.score(prepared,truth)
        if release:self.cache.release(self.index_keys.pop(int(index)))
        return row
    def ridge_score_cached(self,index:int,ridge)->Mapping[str,Any]:
        prepared=self.cached_prepare(index,"ridge-score-base");coefficient=ridge(prepared.features).squeeze(0);adapted=prepared.parent.clone()
        if prepared.route_index>=0:
            correction=torch.einsum("tr,rhw->thw",self.candidate.bases[prepared.route_index].cpu(),coefficient);k1=prepared.public.observed_indices[1];correction*=c1_causal_mask(401,k1)[:,None,None];correction[:,0]=0;adapted+=correction
        self.cleanup(prepared);ledger=AccessLedger(self.split);ledger.add("cache_hit",key_sha256=self.index_keys[index].digest());seal=self.spool.seal(f"{prepared.public.sample_id}:ridge-score",adapted,ledger);self.active.append(seal["path"]);ridge_prepared=V4Prepared(prepared.public,prepared.parent,prepared.travel,prepared.args,prepared.features,prepared.route_index,prepared.condition,coefficient[None],seal,ledger);return self.score(ridge_prepared,self._cached_truth(ridge_prepared))

def run_v6_smoke(*,backend:V6ProductionBackend,records:Sequence[int],quick_gate:Callable[[Sequence[Mapping[str,Any]]],bool],checkpoint:Callable[[],Mapping[str,Any]],resource_snapshot:Callable[[Mapping[str,Any]],Mapping[str,Any]],terminal:Callable[[Mapping[str,Any]],None],clock:Callable[[],float]=time.monotonic)->Mapping[str,Any]:
    started=clock();[backend.preload(index) for index in records];training_deadline=min(started+420.,started+600.-180.);losses=[];updates=0;early=False
    while updates<192 and clock()<training_deadline:
        losses.append(float(backend.update_cached(records[updates%len(records)])["loss"]));updates+=1
        if updates%32==0:
            quick=[backend.score_cached(index) for index in records]
            if losses[-1]<=.2*losses[0] and quick_gate(quick):early=True;break
    if started+600.-clock()<180.:payload={"status":"fail_budget","updates":updates,"cache":backend.cache.payload()};terminal(payload);backend.cache.clear();return payload
    scores=[backend.score_cached(index) for index in records];saved=dict(checkpoint());resources=dict(resource_snapshot(saved));resources["wall_s"]=clock()-started;gates,metrics=smoke_gates(losses[0],losses[-1],scores,resources);total=clock()-started;status="passed" if all(g["passed"] for g in gates.values()) and total<=600 else "fail_budget" if total>600 else "fail_gate";payload={"schema":"r16_dscp_v6_smoke_terminal_v1","candidate":"r16_dscp_v6","mode":"smoke","status":status,"decision":status,"run_digest":backend.lineage.run_digest,"effective":{"parent":backend.parent_binding,"basis":backend.basis_binding,"feature":backend.feature_binding},"input_checkpoint_sha256":backend.lineage.input_checkpoint_sha256,"updates":updates,"early_stop":early,"cache":backend.cache.payload(),"access_ledger_digest":canonical_sha([row["ledger_digest"] for row in scores]),"latency":backend.protocol.payload(),"peak_vram_bytes":backend.peak_bytes,"gates":gates,"metrics":metrics,"checkpoint":saved};terminal(payload);backend.cache.clear();return payload

def complete_run_identity(lineage:Mapping[str,Any],*,authorization_sha256:str,engine_sha256:str,script_sha256:str,config_sha256:str,panels_sha256:str,basis_sha256:str,parent_sha256:str,input_checkpoint_sha256:str)->Mapping[str,Any]:return {**dict(lineage),"authorization_sha256":authorization_sha256,"engine_sha256":engine_sha256,"script_sha256":script_sha256,"config_sha256_effective":config_sha256,"panels_sha256_effective":panels_sha256,"basis_sha256_effective":basis_sha256,"parent_sha256_effective":parent_sha256,"input_checkpoint_sha256":input_checkpoint_sha256}

__all__=["CacheBindingRefusal","CacheMemoryRefusal","HOST_RESERVE_BYTES","ROLE_CAPACITY","V6CacheEntry","V6CacheKey","V6ProductionBackend","V6RecordCache","complete_run_identity","mem_available_bytes","run_v6_smoke","stage_memory_gate","tensor_bytes"]
