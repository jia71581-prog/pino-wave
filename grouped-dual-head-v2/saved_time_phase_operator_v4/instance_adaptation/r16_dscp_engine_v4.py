"""V4 canonical shapes, actual-coefficient loss, and sealed candidate spool."""
from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib, os
from pathlib import Path
from typing import Any, Mapping
import torch

from .data_guard import GuardedOnsetRecord
from .r16_dscp import CONDITION_MAXIMUM, R16DSCP, RidgePointwiseBaseline, basis_condition, c1_causal_mask, deployment_features
from .r16_dscp_training_v2 import atomic_torch_save, checkpoint_payload, cosine_learning_rate, exact_microblocked_loss, save_best_last, sha256_file, weighted_coefficient_target
from .r16_dscp_engine_v3 import AccessLedger, GateFailure, LeakageRefusal, Lineage, PatienceState, StreamingRidge, aggregate_metrics, authorized_future_truth, metric_record, tensor_sha

class ShapeRefusal(ValueError):pass
class SpoolRefusal(RuntimeError):pass

class V4GuardedOnsetLoader:
    """Normalize guarded HDF5 observations `[2,H,W]` to v4 `[2,1,H,W]`."""
    def __init__(self,dataset:Any):self.dataset=dataset
    def __len__(self)->int:return len(self.dataset)
    def __getitem__(self,index:int)->GuardedOnsetRecord:
        record=self.dataset[int(index)];observed=torch.as_tensor(record.observed_wavefield)
        if observed.ndim!=3 or observed.shape[0]!=2:raise ShapeRefusal("guarded source observations must be [2,H,W]")
        return replace(record,observed_wavefield=observed[:,None].contiguous())
    def close(self)->None:
        close=getattr(self.dataset,"close",None)
        if close is not None:close()

@dataclass(frozen=True)
class CanonicalPublic:
    velocity_mps:torch.Tensor; source_parameters:torch.Tensor; source_map:torch.Tensor
    time_s:torch.Tensor; x_m:torch.Tensor; z_m:torch.Tensor
    observed_indices:tuple[int,int]; observed_wavefield:torch.Tensor
    source_index:int; sample_id:str; group_id:str; input_digest:str
    dense_travel_time_s:torch.Tensor|None

def _unique_channel(value:torch.Tensor,name:str)->torch.Tensor:
    x=torch.as_tensor(value)
    if x.ndim!=3 or x.shape[0]!=1:raise ShapeRefusal(f"{name} must be [1,H,W]")
    return x.squeeze(0).contiguous()

def canonical_public(record:GuardedOnsetRecord)->CanonicalPublic:
    velocity=_unique_channel(record.velocity_mps,"velocity_mps");source_map=_unique_channel(record.source_map,"source_map");observed=torch.as_tensor(record.observed_wavefield)
    if observed.ndim!=4 or observed.shape[:2]!=(2,1) or tuple(observed.shape[2:])!=tuple(velocity.shape):raise ShapeRefusal("observed_wavefield must be [2,1,H,W]")
    source=torch.as_tensor(record.source_parameters)
    if source.shape!=(5,):raise ShapeRefusal("source_parameters must be [5]")
    travel=None if record.dense_travel_time_s is None else torch.as_tensor(record.dense_travel_time_s)
    if travel is not None:
        if travel.ndim==3 and travel.shape[0]==1:travel=travel.squeeze(0)
        if travel.shape!=velocity.shape:raise ShapeRefusal("travel time must resolve to [H,W]")
    return CanonicalPublic(velocity,source.contiguous(),source_map,record.time_s,record.x_m,record.z_m,tuple(record.observed_indices),observed.squeeze(1).contiguous(),record.source_index,record.sample_id,record.group_id,record.input_digest,travel)

def deployment_args(public:CanonicalPublic,parent:torch.Tensor,travel:torch.Tensor,device:torch.device)->tuple[Any,...]:
    p=torch.as_tensor(parent)
    if p.ndim!=3:raise ShapeRefusal("parent must be [T,H,W]")
    if public.velocity_mps.ndim!=2 or public.source_map.ndim!=2 or public.observed_wavefield.ndim!=3:raise ShapeRefusal("canonical public fields changed")
    t=torch.as_tensor(travel)
    if t.shape!=public.velocity_mps.shape:raise ShapeRefusal("travel must be [H,W]")
    k0,k1=public.observed_indices
    args=(public.velocity_mps[None,None],public.source_map[None,None],t[None,None],public.x_m,public.z_m,public.observed_wavefield[0][None,None],public.observed_wavefield[1][None,None],p[None],torch.tensor([k0]),torch.tensor([k1]))
    result=tuple(x.to(device) if isinstance(x,torch.Tensor) else x for x in args)
    if any(x.ndim!=4 for x in (result[0],result[1],result[2],result[5],result[6],result[7])):raise ShapeRefusal("model spatial inputs must be 4-D")
    return result

def canonical_parent_arrays(public:CanonicalPublic)->tuple[Any,Any,Any]:
    velocity=public.velocity_mps.detach().cpu().numpy();source=public.source_parameters.detach().cpu().numpy();source_map=public.source_map.detach().cpu().numpy()
    if velocity.ndim!=2 or source_map.ndim!=2 or source.shape!=(5,):raise ShapeRefusal("parent arrays are not canonical")
    return velocity,source,source_map

def canonical_travel(public:CanonicalPublic,solver:Any)->torch.Tensor:
    velocity,source,_=canonical_parent_arrays(public);result=solver(velocity,source)
    travel=torch.as_tensor(result)
    if travel.shape!=public.velocity_mps.shape:raise ShapeRefusal("eikonal solver must return [H,W]")
    return travel

def forward_actual_coefficients(model:R16DSCP,args:tuple[Any,...])->tuple[torch.Tensor,torch.Tensor]:
    coefficients,decisions,conditions=model.predict_coefficients(*args);parent=torch.as_tensor(args[7]);corrected=parent.clone();k1=torch.as_tensor(args[-1]).flatten()
    for index,decision in enumerate(decisions):
        if decision.abstain or float(conditions[index])>CONDITION_MAXIMUM:continue
        correction=torch.einsum("tr,rhw->thw",model.bases[decision.index].to(parent),coefficients[index]);correction*=c1_causal_mask(parent.shape[1],int(k1[index]),device=parent.device,dtype=parent.dtype)[:,None,None];correction[:,0,:]=0;corrected[index]+=correction
    return corrected,coefficients

def actual_coefficient_loss(model:R16DSCP,args:tuple[Any,...],truth_future:torch.Tensor)->tuple[Mapping[str,torch.Tensor],torch.Tensor,torch.Tensor]:
    corrected,coefficients=forward_actual_coefficients(model,args);k1=int(torch.as_tensor(args[-1])[0]);losses=exact_microblocked_loss(corrected[:,k1+1:].float(),torch.as_tensor(truth_future,device=corrected.device).float(),coefficients)
    return losses,corrected,coefficients

class CandidateSpool:
    def __init__(self,run_dir:str|Path,*,run_digest:str|None=None,rank:int=0,owned_root:str|Path|None=None):
        self.run_dir=Path(run_dir).resolve();digest=str(run_digest or hashlib.sha256(str(self.run_dir).encode()).hexdigest());base=Path(owned_root).resolve() if owned_root is not None else Path("/dev/shm/r16_dscp_v4").resolve();self.root=(base/digest/f"rank{int(rank)}").resolve();self.root.mkdir(parents=True,exist_ok=True)
        if self.root.parent.parent!=base or self.root.name!=f"rank{int(rank)}":raise SpoolRefusal("invalid owned spool root")
    def _path(self,key:str)->Path:
        safe=hashlib.sha256(str(key).encode()).hexdigest()+".pt";path=(self.root/safe).resolve()
        if path.parent!=self.root:raise SpoolRefusal("spool path escaped root")
        return path
    def seal(self,key:str,candidate:torch.Tensor,ledger:AccessLedger)->Mapping[str,Any]:
        path=self._path(key)
        if path.exists():raise FileExistsError(f"candidate spool exists: {path}")
        atomic_torch_save({"schema":"r16_dscp_v4_candidate_v1","candidate":torch.as_tensor(candidate).detach().cpu().contiguous(),"tensor_sha256":tensor_sha(candidate)},path)
        record={"path":str(path),"file_sha256":sha256_file(path),"tensor_sha256":tensor_sha(candidate),"serialized_bytes":path.stat().st_size};ledger.add("candidate_sealed",serialized=True,**record);return record
    def validate(self,record:Mapping[str,Any],ledger:AccessLedger)->Path:
        path=Path(record["path"]).resolve()
        if path.parent!=self.root or not path.is_file():raise SpoolRefusal("owned spool binding invalid")
        if sha256_file(path)!=record["file_sha256"]:raise SpoolRefusal("sealed candidate file hash mismatch")
        events=[e for e in ledger.events if e["event"]=="candidate_sealed" and e.get("serialized")]
        if not events or events[-1].get("file_sha256")!=record["file_sha256"]:raise LeakageRefusal("serialized seal ledger missing")
        return path
    def load(self,record:Mapping[str,Any],ledger:AccessLedger)->torch.Tensor:
        payload=torch.load(self.validate(record,ledger),map_location="cpu",weights_only=False);candidate=payload["candidate"]
        if tensor_sha(candidate)!=record["tensor_sha256"]:raise SpoolRefusal("sealed tensor hash mismatch")
        ledger.add("candidate_loaded_for_score",file_sha256=record["file_sha256"]);return candidate
    def cleanup(self,record:Mapping[str,Any],ledger:AccessLedger)->None:
        path=self.validate(record,ledger);path.unlink();descriptor=os.open(str(self.root),os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
        ledger.add("candidate_spool_cleanup",path=str(path),owned=True)

def open_truth_after_spool(*,spool:CandidateSpool,seal:Mapping[str,Any],path:str|Path,index:int,k1:int,mode:str,split:str,ledger:AccessLedger,authorization:Mapping[str,Any],lineage:Lineage,once_token:Path|None=None)->torch.Tensor:
    spool.validate(seal,ledger);return authorized_future_truth(path,index,k1,mode=mode,split=split,ledger=ledger,authorization=authorization,expected_lineage=lineage,once_token=once_token)

def score_same_sealed_candidate(spool:CandidateSpool,seal:Mapping[str,Any],ledger:AccessLedger)->torch.Tensor:return spool.load(seal,ledger)

@dataclass(frozen=True)
class V4Prepared:
    public:CanonicalPublic; parent:torch.Tensor; travel:torch.Tensor; args:tuple[Any,...]
    features:torch.Tensor; route_index:int; condition:float; coefficients:torch.Tensor
    seal:Mapping[str,Any]; ledger:AccessLedger

class V4ProductionBackend:
    """Canonical production callbacks with one owned sealed candidate per record."""
    def __init__(self,*,public_loader:Any,parent_predictor:Any,travel_builder:Any,candidate:R16DSCP,optimizer:torch.optim.Optimizer,authorization:Mapping[str,Any],lineage:Lineage,source_h5:str|Path,run_dir:str|Path,device:torch.device,split:str,family_by_sample:Mapping[str,str],once_token:Path|None=None):
        self.public_loader=public_loader;self.parent_predictor=parent_predictor;self.travel_builder=travel_builder;self.candidate=candidate;self.optimizer=optimizer;self.authorization=authorization;self.lineage=lineage;self.checkpoint_identity=lineage.__dict__;self.source_h5=Path(source_h5);self.run_dir=Path(run_dir);self.device=device;self.split=str(split);self.family_by_sample=dict(family_by_sample);self.once_token=once_token;self.spool=CandidateSpool(self.run_dir,run_digest=lineage.run_digest,rank=int(os.environ.get("RANK","0")));self.ridge=StreamingRidge();self.active=[];self.spool_max_bytes=0;self.started=__import__("time").monotonic();self.latencies=[];self.e2e_latencies=[];self.peak_bytes=0
    def prepare(self,index:int,key_suffix:str="candidate")->V4Prepared:
        if self.active:raise SpoolRefusal("previous owned spool must be scored/cleaned before next record")
        timer=__import__("time").perf_counter();public=canonical_public(self.public_loader[int(index)]);parent=torch.as_tensor(self.parent_predictor(public)).detach().cpu().float();travel=canonical_travel(public,self.travel_builder).float();args=deployment_args(public,parent,travel,self.device);adapted,coefficients=forward_actual_coefficients(self.candidate,args);self.e2e_latencies.append(__import__("time").perf_counter()-timer)
        features,decisions,conditions=deployment_features(*args[:-2],self.candidate.bases,args[-2],args[-1]);decision=decisions[0];ledger=AccessLedger(self.split);ledger.add("public_bundle_ready",input_digest=public.input_digest,split=self.split);ledger.add("features_ready",sha256=tensor_sha(features),serialized=False)
        seal=self.spool.seal(f"{public.sample_id}:{key_suffix}",adapted.detach().cpu(),ledger);self.spool_max_bytes=max(self.spool_max_bytes,int(seal["serialized_bytes"]));self.active.append(seal["path"])
        if self.device.type=="cuda":self.peak_bytes=max(self.peak_bytes,int(torch.cuda.max_memory_reserved(self.device)))
        return V4Prepared(public,parent,travel,args,features.detach().cpu(),decision.index,float(conditions[0]),coefficients.detach().cpu(),seal,ledger)
    def open_truth(self,prepared:V4Prepared)->torch.Tensor:
        return open_truth_after_spool(spool=self.spool,seal=prepared.seal,path=self.source_h5,index=prepared.public.source_index,k1=prepared.public.observed_indices[1],mode=self.lineage.mode,split=self.split,ledger=prepared.ledger,authorization=self.authorization,lineage=self.lineage,once_token=self.once_token)
    def update(self,prepared:V4Prepared)->Mapping[str,Any]:
        if self.split!="train":raise LeakageRefusal("optimizer update requires train split")
        truth=self.open_truth(prepared).to(self.device)[None];self.optimizer.zero_grad(set_to_none=True);losses,_,coefficients=actual_coefficient_loss(self.candidate,prepared.args,truth);losses["total"].backward();grad=float(torch.nn.utils.clip_grad_norm_(self.candidate.parameters(),1.0));self.optimizer.step();self.cleanup(prepared)
        return {"loss":float(losses["total"]),"coefficient_energy":float(losses["normalized_coefficient_energy"]),"grad_norm":grad,"ledger_digest":prepared.ledger.digest()}
    def measure_gradient(self,prepared:V4Prepared)->Mapping[str,Any]:
        truth=self.open_truth(prepared).to(self.device)[None];self.optimizer.zero_grad(set_to_none=True);losses,_,_=actual_coefficient_loss(self.candidate,prepared.args,truth);losses["total"].backward();grad=float(torch.nn.utils.clip_grad_norm_(self.candidate.parameters(),1.0));self.optimizer.zero_grad(set_to_none=True);self.cleanup(prepared);return {"loss":float(losses["total"]),"coefficient_energy":float(losses["normalized_coefficient_energy"]),"grad_norm":grad,"sample_id":prepared.public.sample_id,"ledger_digest":prepared.ledger.digest()}
    def backward_only(self,prepared:V4Prepared,loss_scale:float)->Mapping[str,Any]:
        if self.split!="train":raise LeakageRefusal("long backward requires train split")
        truth=self.open_truth(prepared).to(self.device)[None];losses,_,_=actual_coefficient_loss(self.candidate,prepared.args,truth);scaled=losses["total"]*float(loss_scale);scaled.backward();self.cleanup(prepared);return {"loss":float(losses["total"]),"scaled_loss":float(scaled),"ledger_digest":prepared.ledger.digest()}
    def score(self,prepared:V4Prepared,truth:torch.Tensor,cleanup:bool=True)->Mapping[str,Any]:
        timer=__import__("time").perf_counter();adapted=score_same_sealed_candidate(self.spool,prepared.seal,prepared.ledger)[0,prepared.public.observed_indices[1]+1:];parent=prepared.parent[prepared.public.observed_indices[1]+1:];row=metric_record(adapted,parent,torch.as_tensor(truth).float(),family=self.family_by_sample[prepared.public.sample_id],condition=prepared.condition,abstain=prepared.route_index<0);self.latencies.append(__import__("time").perf_counter()-timer)
        row.update(sample_id=prepared.public.sample_id,candidate_file_sha256=prepared.seal["file_sha256"],ledger_digest=prepared.ledger.digest(),training_only=self.split=="train")
        if cleanup:self.cleanup(prepared)
        return row
    def ridge_update(self,prepared:V4Prepared,truth:torch.Tensor)->None:
        if self.split!="train":raise LeakageRefusal("ridge fit requires train split")
        if prepared.route_index>=0:
            full_truth=torch.cat((prepared.parent[:prepared.public.observed_indices[1]+1],torch.as_tensor(truth).float().cpu()),0);target=weighted_coefficient_target(self.candidate.bases[prepared.route_index],prepared.parent,full_truth,k1=prepared.public.observed_indices[1]);self.ridge.update(prepared.features,target[None]);del target,full_truth
        self.cleanup(prepared)
    def ridge_finalize(self)->RidgePointwiseBaseline:return self.ridge.solve()
    def prepare_ridge(self,index:int,ridge:RidgePointwiseBaseline)->V4Prepared:
        prepared=self.prepare(index,"ridge");coefficient=ridge(prepared.features).squeeze(0);adapted=prepared.parent.clone()
        if prepared.route_index>=0:
            correction=torch.einsum("tr,rhw->thw",self.candidate.bases[prepared.route_index].cpu(),coefficient);k1=prepared.public.observed_indices[1];correction*=c1_causal_mask(401,k1)[:,None,None];correction[:,0]=0;adapted+=correction
        self.spool.cleanup(prepared.seal,prepared.ledger);self.active.clear();seal=self.spool.seal(f"{prepared.public.sample_id}:ridge-score",adapted,prepared.ledger);self.active.append(seal["path"]);return V4Prepared(prepared.public,prepared.parent,prepared.travel,prepared.args,prepared.features,prepared.route_index,prepared.condition,coefficient[None],seal,prepared.ledger)
    def cleanup(self,prepared:V4Prepared)->None:
        self.spool.cleanup(prepared.seal,prepared.ledger);self.active.clear()
    def preserve_failure_spool(self)->None:
        if len(self.active)>1:raise SpoolRefusal("more than one failure spool retained")
    def checkpoint(self,progress:Mapping[str,Any],is_best:bool=True)->Mapping[str,Any]:
        payload=checkpoint_payload(self.candidate,self.optimizer,run_identity=self.checkpoint_identity,sampler_order=list(range(len(self.public_loader))),progress=progress);return save_best_last(payload,self.run_dir,is_best=is_best)
    def resources(self,saved:Mapping[str,Any],world_size:int=1)->Mapping[str,Any]:
        values=sorted(self.latencies);e2e=sorted(self.e2e_latencies);p95=lambda x:x[max(0,__import__("math").ceil(.95*len(x))-1)] if x else float("inf");e2e_mean=sum(e2e)/len(e2e) if e2e else float("inf");gate=dual_space_gate(workspace=self.run_dir,tmpfs=self.spool.root,checkpoint_bytes=int(saved.get("size_bytes",0)),spool_max_bytes=self.spool_max_bytes,world_size=world_size)
        return {"wall_s":__import__("time").monotonic()-self.started,"peak_bytes":self.peak_bytes,"checkpoint_bytes":int(saved.get("size_bytes",0)),"space_passed":gate["passed"],"workspace_gate":gate,"spool_max_bytes":self.spool_max_bytes,"adapter_mean_s":sum(values)/len(values) if values else float("inf"),"adapter_p95_s":p95(values),"e2e_mean_s":e2e_mean,"e2e_p95_s":p95(e2e),"e2e_mean_ratio":e2e_mean/12.652311868034303,"e2e_p95_ratio":p95(e2e)/12.656147628091276,"fields_serialized":False,"coefficient_maps_serialized":False}

@dataclass(frozen=True)
class LongResumeState:
    epoch:int=0; global_step:int=0; group_index:int=0
    best:float=float("inf"); bad_epochs:int=0; best_epoch:int=-1

def frozen_global_groups(records:list[Any])->list[list[Any]]:
    if len(records)%4:raise GateFailure("long records must form groups of four")
    return [records[i:i+4] for i in range(0,len(records),4)]

def validate_world_size(decision:Mapping[str,Any],world_size:int,cuda_visible:str)->None:
    chosen=int(decision.get("selected_gpus",0))
    if decision.get("status")!="passed" or chosen not in {1,4} or chosen!=int(world_size):raise GateFailure("scale decision/world mismatch")
    if len([x for x in str(cuda_visible).split(",") if x])!=chosen:raise GateFailure("visible GPU count mismatch")

def long_command(chosen:int)->str:
    prefix="env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="
    if chosen==1:return prefix+"0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python scripts/train_r16_dscp_v4.py --mode long --world-size 1"
    if chosen==4:return prefix+"0,1,2,3 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. torchrun --standalone --nproc-per-node=4 scripts/train_r16_dscp_v4.py --mode long --world-size 4"
    raise GateFailure("chosen world size must be 1 or 4")

def dual_space_gate(*,workspace:str|Path,tmpfs:str|Path,checkpoint_bytes:int,spool_max_bytes:int,world_size:int)->Mapping[str,Any]:
    import shutil
    workspace_free=int(shutil.disk_usage(workspace).free);tmpfs_free=int(shutil.disk_usage(tmpfs).free);workspace_required=2*1024**3+3*int(checkpoint_bytes)+64*1024**2;tmpfs_required=int(world_size)*(int(spool_max_bytes)+64*1024**2)
    return {"workspace_free_bytes":workspace_free,"workspace_required_bytes":workspace_required,"workspace_passed":workspace_free>=workspace_required,"tmpfs_free_bytes":tmpfs_free,"tmpfs_required_bytes":tmpfs_required,"tmpfs_passed":tmpfs_free>=tmpfs_required,"world_size":int(world_size),"passed":workspace_free>=workspace_required and tmpfs_free>=tmpfs_required}

def tmpfs_constrained_world(decision:Mapping[str,Any],gate1:Mapping[str,Any],gate4:Mapping[str,Any])->int:
    selected=int(decision.get("selected_gpus",1))
    if selected==4 and not gate4["tmpfs_passed"]:selected=1
    if selected==1 and not gate1["tmpfs_passed"]:raise GateFailure("tmpfs cannot hold one candidate spool")
    return selected

class V4LongRunner:
    def __init__(self,*,world_size:int,rank:int,train_records:list[Any],calibration_records:list[Any],zero_grad:Any,backward_record:Any,optimizer_step:Any,set_lr:Any,evaluate_record:Any,gather:Any,checkpoint:Any,terminal:Any,barrier:Any=lambda:None):
        self.world_size=int(world_size);self.rank=int(rank);self.groups=frozen_global_groups(train_records);self.calibration=list(calibration_records);self.zero_grad=zero_grad;self.backward_record=backward_record;self.optimizer_step=optimizer_step;self.set_lr=set_lr;self.evaluate_record=evaluate_record;self.gather=gather;self.checkpoint=checkpoint;self.terminal=terminal;self.barrier=barrier
    def run(self,state:LongResumeState=LongResumeState(),max_epochs:int=20,patience:int=4)->Mapping[str,Any]:
        if self.world_size not in {1,4} or not 0<=self.rank<self.world_size:raise GateFailure("invalid long world/rank")
        patience_state=PatienceState(state.best,state.bad_epochs,state.best_epoch);step=state.global_step;history=[]
        try:
            for epoch in range(state.epoch,max_epochs):
                self.set_lr(cosine_learning_rate(epoch,max_epochs));start=state.group_index if epoch==state.epoch else 0
                for group_index in range(start,len(self.groups)):
                    self.zero_grad();group=self.groups[group_index];assigned=group if self.world_size==1 else [group[self.rank]];scale=.25 if self.world_size==1 else 1.;local_failure=None
                    try:[self.backward_record(record,scale) for record in assigned]
                    except Exception as exc:local_failure={"rank":self.rank,"reason":str(exc),"epoch":epoch,"group_index":group_index}
                    statuses=self.gather([local_failure] if local_failure else [{"ok":True,"rank":self.rank}]);failures=[status for status in statuses if "reason" in status]
                    if failures:raise GateFailure(f"global backward failure: {failures}")
                    self.optimizer_step();step+=1
                    if self.rank==0:self.checkpoint(LongResumeState(epoch,step,group_index+1,patience_state.best,patience_state.bad_epochs,patience_state.best_epoch),False)
                    self.barrier()
                local=[self.evaluate_record(record) for record in self.calibration[self.rank::self.world_size]];ordered=self.gather(local);stop=False
                if self.rank==0:
                    order={str(record):index for index,record in enumerate(self.calibration)};ordered.sort(key=lambda row:order[str(row["sample_id"])] if str(row["sample_id"]) in order else len(order));
                    if len(ordered)!=len(self.calibration):raise GateFailure("calibration gather incomplete")
                    primary=float(aggregate_metrics(ordered)["joint_aggregate_rel_l2"]);improved,stop=patience_state.update(primary,epoch,patience);self.checkpoint(LongResumeState(epoch+1,step,0,patience_state.best,patience_state.bad_epochs,patience_state.best_epoch),improved);history.append({"epoch":epoch,"primary":primary,"improved":improved})
                flags=self.gather([{"stop":stop}] if self.rank==0 else []);self.barrier()
                if flags and bool(flags[0]["stop"]):break
            result={"status":"completed_train","global_step":step,"best":patience_state.best,"best_epoch":patience_state.best_epoch,"history":history}
            if self.rank==0:self.terminal(result)
            self.barrier();return result
        except Exception as exc:
            failures=self.gather([{"rank":self.rank,"reason":str(exc),"global_step":step}])
            if self.rank==0:self.terminal({"status":"failed","rank_failures":failures})
            self.barrier();raise

def validate_resume_directory(stage_dir:str|Path,*,authorization_sha256:str,effective_bindings:Mapping[str,str],load_last:Any)->Mapping[str,Any]:
    root=Path(stage_dir).resolve();terminal=root/"terminal.json";last=root/"last.pt";identity=root/"run_identity.json"
    if not root.is_dir() or not last.is_file() or not identity.is_file():raise GateFailure("resume directory incomplete")
    import json
    if terminal.is_file() and json.loads(terminal.read_text()).get("status") in {"passed","completed_train"}:raise GateFailure("completed long cannot resume")
    run_identity=json.loads(identity.read_text())
    if run_identity.get("authorization_sha256")!=authorization_sha256:raise GateFailure("resume authorization mismatch")
    for key,value in effective_bindings.items():
        if run_identity.get(key)!=value:raise GateFailure(f"resume binding mismatch: {key}")
    payload=load_last(last)
    if payload.get("run_identity",{}).get("authorization_sha256")!=authorization_sha256:raise GateFailure("checkpoint lineage mismatch")
    return payload

__all__=["CandidateSpool","CanonicalPublic","LongResumeState","ShapeRefusal","SpoolRefusal","V4GuardedOnsetLoader","V4LongRunner","V4Prepared","V4ProductionBackend","actual_coefficient_loss","canonical_parent_arrays","canonical_public","canonical_travel","deployment_args","dual_space_gate","forward_actual_coefficients","frozen_global_groups","long_command","open_truth_after_spool","score_same_sealed_candidate","tmpfs_constrained_world","validate_resume_directory","validate_world_size"]
