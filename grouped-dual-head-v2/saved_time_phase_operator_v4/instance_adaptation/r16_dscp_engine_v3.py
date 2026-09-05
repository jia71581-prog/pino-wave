"""Executable, fail-closed stage engine for R16-DSCP v3."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, math, os, time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import h5py, numpy as np, torch
from torch import nn

from .r16_dscp import R16DSCP, RidgePointwiseBaseline, basis_condition, c1_causal_mask, deployment_features, velocity_route
from .r16_dscp_training_v2 import (
    BASIS_FILE_SHA256, BASIS_TENSOR_SHA256, PANELS_SHA256, PARENT_SHA256,
    BindingRefusal, StageGateRefusal, exact_microblocked_loss, predictor_parameter_state,
    atomic_json_exclusive, checkpoint_payload, cosine_learning_rate, save_best_last, space_gate, weighted_coefficient_target,
)
from .data_guard import GuardedOnsetDataset, GuardedOnsetRecord

CANDIDATE="r16_dscp_v3"
AUTH_SCHEMA="r16_dscp_v3_authorization_v1"
TERMINAL_SCHEMA="r16_dscp_v3_terminal_v1"

class LeakageRefusal(BindingRefusal): pass
class GateFailure(RuntimeError): pass

def assert_stage_split(mode:str,split:str)->None:
    train_modes={"smoke","pilot","scale-probe-1","scale-probe-4","long","final-train-confirm"}
    expected="validation" if mode=="validation-once" else "test_id" if mode=="test-once" else "train"
    if mode not in train_modes|{"validation-once","test-once"} or split!=expected: raise LeakageRefusal(f"mode {mode} forbids split {split}")

@dataclass(frozen=True)
class DispatchPhase:
    action:str; split:str; panel_role:str

def stage_dispatch(mode:str)->tuple[DispatchPhase,...]:
    table={
        "smoke":(("update","train","smoke"),("eval","train","smoke")),
        "pilot":(("update","train","pilot_fit"),("ridge_fit","train","pilot_fit"),("eval","train","pilot_confirm")),
        "scale-probe-1":(("ephemeral_update","train","pilot_fit_first12"),),
        "scale-probe-4":(("ephemeral_update_ddp","train","pilot_fit_first12"),),
        "long":(("update","train","long_fit"),("eval","train","long_calibration")),
        "final-train-confirm":(("eval","train","final_train_confirm"),),
        "validation-once":(("eval","validation","complete_registered_validation"),),
        "test-once":(("eval","test_id","complete_registered_test_id"),),
    }
    if mode not in table: raise StageGateRefusal(f"unknown stage mode: {mode}")
    return tuple(DispatchPhase(*row) for row in table[mode])

@dataclass(frozen=True)
class PublicBundle:
    velocity_mps:torch.Tensor; source_parameters:torch.Tensor; source_map:torch.Tensor
    time_s:torch.Tensor; x_m:torch.Tensor; z_m:torch.Tensor
    observed_indices:tuple[int,int]; observed_wavefield:torch.Tensor
    dense_travel_time_s:torch.Tensor|None; input_digest:str
    source_index:int; sample_id:str; group_id:str

class GuardedPublicLoader:
    """Production wrapper exposing only parent inputs and two registered frames."""
    def __init__(self,source_h5:str|Path,manifest:Any,*,split:str,sample_ids:Sequence[str]|None=None,travel_time_h5:str|Path|None=None,dataset_factory:Callable[...,Any]=GuardedOnsetDataset):
        self.split=str(split); self.dataset=dataset_factory(source_h5,manifest,split=self.split,sample_ids=sample_ids,travel_time_h5=travel_time_h5)
    def __len__(self)->int:return len(self.dataset)
    def __getitem__(self,index:int)->PublicBundle:
        r=self.dataset[index]
        if tuple(r.audit.requested_indices)!=tuple(r.observed_indices) or r.audit.payload()["future_truth_used"]: raise LeakageRefusal("public loader accessed outside exact onset pair")
        return PublicBundle(r.velocity_mps,r.source_parameters,r.source_map,r.time_s,r.x_m,r.z_m,tuple(r.observed_indices),r.observed_wavefield,r.dense_travel_time_s,r.input_digest,int(r.source_index),str(r.sample_id),str(r.group_id))
    def close(self)->None:self.dataset.close()

def canonical_sha(value:Any)->str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()

def tensor_sha(value:torch.Tensor)->str:
    x=torch.as_tensor(value).detach().cpu().contiguous()
    h=hashlib.sha256(); h.update(str(x.dtype).encode()); h.update(json.dumps(list(x.shape)).encode()); h.update(x.numpy().tobytes()); return h.hexdigest()

@dataclass(frozen=True)
class Lineage:
    candidate:str; mode:str; run_digest:str; code_sha256:str; config_sha256:str
    panels_sha256:str; basis_sha256:str; basis_tensor_sha256:str; parent_sha256:str
    input_checkpoint_sha256:str

def validate_authorization(payload:Mapping[str,Any], expected:Lineage)->None:
    required={"schema":AUTH_SCHEMA,"status":"authorized",**expected.__dict__}
    if any(payload.get(k)!=v for k,v in required.items()): raise BindingRefusal("authorization lineage/hash mismatch")
    if payload.get("gates_sha256") is None or payload.get("authorization_digest")!=canonical_sha({k:v for k,v in payload.items() if k!="authorization_digest"}):
        raise BindingRefusal("authorization gates/digest mismatch")

class AccessLedger:
    def __init__(self, split:str): self.split=str(split); self.events=[]
    def add(self,event:str,**payload:Any)->None: self.events.append({"ordinal":len(self.events),"event":event,**payload})
    def require_before(self,first:str,second:str)->None:
        positions={row["event"]:row["ordinal"] for row in self.events}
        if first not in positions or second not in positions or positions[first]>=positions[second]: raise LeakageRefusal(f"ledger order requires {first} before {second}")
    def digest(self)->str:return canonical_sha(self.events)

def read_exact_observations(path:str|Path,index:int,k0:int,k1:int,*,split:str,ledger:AccessLedger)->torch.Tensor:
    if split not in {"train","validation","test_id"}: raise LeakageRefusal("unregistered split")
    if k1!=k0+1: raise LeakageRefusal("observations must be adjacent")
    rows=[]
    with h5py.File(path,"r",swmr=True) as h:
        for k in (k0,k1): rows.append(np.asarray(h["wavefield"][index,k],dtype=np.float32)); ledger.add("observed_truth_read",index=k)
    ledger.add("observations_complete",indices=[k0,k1]); return torch.from_numpy(np.stack(rows))

def read_future_truth(path:str|Path,index:int,k1:int,*,split:str,ledger:AccessLedger,once_token:Path|None=None,token_payload:Mapping[str,Any]|None=None)->torch.Tensor:
    if split not in {"train","validation","test_id"}: raise LeakageRefusal("unregistered split")
    ledger.require_before("candidate_sealed","future_truth_read_authorized")
    if split in {"validation","test_id"}:
        if once_token is None:
            if not any(row["event"]=="once_token_already_claimed" for row in ledger.events):raise LeakageRefusal("sealed split requires once token")
        elif token_payload is None:raise LeakageRefusal("sealed split token payload absent")
        else:atomic_json_exclusive(dict(token_payload),once_token); ledger.add("once_token_claimed",path=str(once_token))
    with h5py.File(path,"r",swmr=True) as h: value=np.asarray(h["wavefield"][index,k1+1:],dtype=np.float32)
    ledger.add("future_truth_read",start=k1+1,frames=value.shape[0]); return torch.from_numpy(value)

def seal_candidate(value:torch.Tensor,ledger:AccessLedger)->str:
    digest=tensor_sha(value); ledger.add("candidate_sealed",sha256=digest,serialized=False); return digest

def authorized_future_truth(path:str|Path,index:int,k1:int,*,mode:str,split:str,ledger:AccessLedger,authorization:Mapping[str,Any],expected_lineage:Lineage,once_token:Path|None=None)->torch.Tensor:
    assert_stage_split(mode,split);validate_authorization(authorization,expected_lineage)
    if not any(row["event"]=="candidate_sealed" for row in ledger.events):raise LeakageRefusal("candidate must be sealed before future authorization")
    ledger.add("future_truth_read_authorized",split=split,authorization_digest=authorization["authorization_digest"])
    token_payload=None
    if split in {"validation","test_id"}:
        if once_token is None:raise LeakageRefusal("sealed split once token path absent")
        token_payload={"schema":"r16_dscp_v3_once_token_v1","candidate":CANDIDATE,"mode":mode,"split":split,"run_digest":expected_lineage.run_digest,"authorization_digest":authorization["authorization_digest"]}
        if once_token.exists():
            observed=json.loads(once_token.read_text())
            if observed!=token_payload:raise LeakageRefusal("once token binding mismatch")
            ledger.add("once_token_already_claimed",path=str(once_token));once_token=None;token_payload=None
    return read_future_truth(path,index,k1,split=split,ledger=ledger,once_token=once_token,token_payload=token_payload)

def relative_l2(pred:torch.Tensor,truth:torch.Tensor)->float:
    p=pred.double(); t=truth.double(); return float(torch.linalg.vector_norm(p-t)/torch.linalg.vector_norm(t).clamp_min(1e-30))

def metric_record(candidate:torch.Tensor,parent:torch.Tensor,truth:torch.Tensor,*,family:str,condition:float,abstain:bool)->dict[str,Any]:
    c=torch.as_tensor(candidate).float(); p=torch.as_tensor(parent).float(); t=torch.as_tensor(truth).float(); n=t.shape[0]
    thirds=[(0,n//3),(n//3,2*n//3),(2*n//3,n)]
    frame=torch.sqrt((c-t).square().flatten(1).sum(1)/t.square().flatten(1).sum(1).clamp_min(1e-30))
    pf=torch.sqrt((p-t).square().flatten(1).sum(1)/t.square().flatten(1).sum(1).clamp_min(1e-30))
    correction=c-p
    fft_e=torch.fft.rfft2(c-t).abs().square(); fft_t=torch.fft.rfft2(t).abs().square().clamp_min(1e-30)
    width=fft_e.shape[-1]; cuts=(0,width//3,2*width//3,width)
    bands=[float(torch.sqrt(fft_e[...,cuts[i]:cuts[i+1]].sum()/fft_t[...,cuts[i]:cuts[i+1]].sum())) for i in range(3)];parent_fft=torch.fft.rfft2(p-t).abs().square();parent_bands=[float(torch.sqrt(parent_fft[...,cuts[i]:cuts[i+1]].sum()/fft_t[...,cuts[i]:cuts[i+1]].sum())) for i in range(3)]
    return {"family":family,"aggregate_rel_l2":relative_l2(c,t),"parent_rel_l2":relative_l2(p,t),
            "mean_frame_rel_l2":float(frame.mean()),"nonworse":bool(float(frame.mean())<=float(pf.mean())),
            "time_bands":{name:float(frame[a:b].mean()) for name,(a,b) in zip(("early","mid","late"),thirds)},
            "parent_time_bands":{name:float(pf[a:b].mean()) for name,(a,b) in zip(("early","mid","late"),thirds)},
            "spectrum_bands":dict(zip(("low","mid","high"),bands)),
            "parent_spectrum_bands":dict(zip(("low","mid","high"),parent_bands)),
            "parent_mean_frame_rel_l2":float(pf.mean()),
            "correction_energy_ratio":float(correction.double().square().sum()/p.double().square().sum().clamp_min(1e-30)),
            "condition":float(condition),"abstain":bool(abstain),"finite":bool(torch.isfinite(c).all())}

def aggregate_metrics(records:Sequence[Mapping[str,Any]])->dict[str,Any]:
    if not records: raise ValueError("records required")
    families=sorted(set(r["family"] for r in records))
    mean=lambda rows,key:sum(float(r[key]) for r in rows)/len(rows)
    return {"count":len(records),"joint_aggregate_rel_l2":mean(records,"aggregate_rel_l2"),
            "parent_joint_aggregate_rel_l2":mean(records,"parent_rel_l2"),"nonworse":sum(bool(r["nonworse"]) for r in records),
            "by_family":{f:{"count":len(rows:=[r for r in records if r["family"]==f]),"aggregate_rel_l2":mean(rows,"aggregate_rel_l2"),"parent_rel_l2":mean(rows,"parent_rel_l2")} for f in families},
            "digest":canonical_sha(list(records))}

def pilot_gate(candidate:Mapping[str,Any],ridge:Mapping[str,Any])->dict[str,Any]:
    joint=(candidate["parent_joint_aggregate_rel_l2"]-candidate["joint_aggregate_rel_l2"])/candidate["parent_joint_aggregate_rel_l2"]
    family={f:(x["parent_rel_l2"]-x["aggregate_rel_l2"])/x["parent_rel_l2"] for f,x in candidate["by_family"].items()}
    ridge_gain=(ridge["parent_joint_aggregate_rel_l2"]-ridge["joint_aggregate_rel_l2"])/ridge["parent_joint_aggregate_rel_l2"]
    checks={"joint":joint>=.01,"families":all(v>=.005 for v in family.values()),"nonworse":candidate["nonworse"]>=23,"ridge_margin":joint-ridge_gain>=.0025}
    return {"passed":all(checks.values()),"checks":checks,"joint_gain":joint,"family_gain":family,"ridge_gain":ridge_gain}

class StreamingRidge:
    def __init__(self,lam:float=1e-4): self.xtx=torch.zeros(30,30,dtype=torch.float64); self.xty=torch.zeros(30,16,dtype=torch.float64); self.lam=lam
    def update(self,features:torch.Tensor,target:torch.Tensor)->None:
        x=features.permute(0,2,3,1).reshape(-1,29).double(); y=target.permute(0,2,3,1).reshape(-1,16).double(); x=torch.cat((x,torch.ones(x.shape[0],1)),1); self.xtx+=x.T@x; self.xty+=x.T@y
    def solve(self)->RidgePointwiseBaseline:
        reg=torch.eye(30,dtype=torch.float64)*self.lam;reg[-1,-1]=0; w=torch.linalg.solve(self.xtx+reg,self.xty); model=RidgePointwiseBaseline(); model.weight.data.copy_(w[:-1].T[:,:,None,None].float());model.bias.data.copy_(w[-1].float());return model

def ddp_host_aggregate(shards:Sequence[Sequence[Mapping[str,Any]]])->dict[str,Any]:
    records=sorted([dict(r) for shard in shards for r in shard],key=lambda r:(r.get("sample_id",""),canonical_sha(r)))
    return aggregate_metrics(records)

def train_update(model:R16DSCP,optimizer:torch.optim.Optimizer,deployment_args:Sequence[Any],truth_loader:Callable[[],torch.Tensor],ledger:AccessLedger,*,device:torch.device)->dict[str,Any]:
    optimizer.zero_grad(set_to_none=True); started=time.perf_counter()
    amp=torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True)
    with amp: corrected=model(*deployment_args)
    candidate_future=corrected[:,int(torch.as_tensor(deployment_args[-1])[0])+1:]
    candidate_hash=seal_candidate(candidate_future,ledger); truth=truth_loader().to(device)[None]
    losses=exact_microblocked_loss(candidate_future.float(),truth.float(),torch.zeros(1,16,201,201,device=device))
    if not torch.isfinite(losses["total"]): raise GateFailure("nonfinite loss")
    losses["total"].backward(); grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
    if not math.isfinite(grad): raise GateFailure("nonfinite gradient")
    optimizer.step(); return {"loss":float(losses["total"]),"grad_norm":grad,"candidate_sha256":candidate_hash,"wall_s":time.perf_counter()-started}

def scale_decision(one:Mapping[str,Any],four:Mapping[str,Any])->dict[str,Any]:
    speed=float(one["wall_s"])/float(four["wall_s"]); efficiency=speed/4; gpu_ratio=float(four.get("gpu_seconds",4*float(four["wall_s"])))/float(one.get("gpu_seconds",one["wall_s"]))
    match=all(one[key]==four[key] for key in ("metric_digest","model_sha256","data_sha256","schedule_sha256"))
    passed=speed>=3 and efficiency>=.75 and gpu_ratio<=1.35 and match and max(four["per_gpu_peak_bytes"])<=int(6.5*1024**3)
    return {"selected_gpus":4 if passed else 1,"passed":passed,"speedup":speed,"efficiency":efficiency,"gpu_seconds_ratio":gpu_ratio,"hash_match":match}

def model_digest(model:nn.Module)->str:return canonical_sha({k:tensor_sha(v) for k,v in sorted(model.state_dict().items())})

def run_scale_probe(*,mode:str,records:Sequence[Any],measure:Callable[[Any],Mapping[str,Any]],model_hash:Callable[[],str],resource_snapshot:Callable[[],Mapping[str,Any]],lineage:Lineage,run_dir:Path,warmup:int=1,timed:int=3,world_size:int=1,rank:int=0)->Mapping[str,Any]:
    if mode not in {"scale-probe-1","scale-probe-4"} or len(records)!=12:raise GateFailure("scale requires 12 records")
    before=model_hash();repeats=[];rows=[]
    for repeat in range(warmup+timed):
        started=time.perf_counter();current=[dict(measure(record)) for record in records[rank::world_size]];elapsed=time.perf_counter()-started
        if repeat>=warmup:repeats.append(elapsed);rows.extend({**row,"repeat":repeat-warmup,"rank":rank} for row in current)
    after=model_hash()
    if before!=after:raise GateFailure("scale changed model")
    ordered=sorted(rows,key=lambda r:(r["repeat"],str(r["sample_id"])));report={"schema":"r16_dscp_v3_scale_rank_v1","candidate":CANDIDATE,"mode":mode,"status":"passed","rank":rank,"world_size":world_size,"wall_s":sorted(repeats)[len(repeats)//2],"repeat_wall_s":repeats,"gpu_seconds":sum(repeats)*world_size,"model_sha256_before":before,"model_sha256_after":after,"model_sha256":before,"rows":rows,"metric_digest":canonical_sha([(r["sample_id"],r["repeat"],float(r["loss"])) for r in ordered]),"data_sha256":canonical_sha([str(r) for r in records]),"schedule_sha256":canonical_sha({"warmup":warmup,"timed":timed}),"ledger_digest":canonical_sha(sorted(r["ledger_digest"] for r in rows)),"resources":dict(resource_snapshot()),"per_gpu_peak_bytes":[int(resource_snapshot()["peak_bytes"])],"finite":all(math.isfinite(float(r["loss"])) for r in rows),"fields_serialized":False}
    report["data_sha256"]=canonical_sha([r["sample_id"] for r in ordered if r["repeat"]==0])
    report["lineage"]=lineage.__dict__
    if world_size==1:atomic_json_exclusive(report,run_dir/"terminal.json")
    return report

def aggregate_scale_ranks(reports:Sequence[Mapping[str,Any]],frozen_order:Sequence[str])->Mapping[str,Any]:
    world=int(reports[0]["world_size"]) if reports else 0
    if len(reports)!=world or {int(r["rank"]) for r in reports}!=set(range(world)) or any(r.get("status")!="passed" for r in reports):raise GateFailure("rank failure")
    rows=[dict(x) for report in reports for x in report["rows"]];order={s:i for i,s in enumerate(frozen_order)};rows.sort(key=lambda r:(r["repeat"],order[r["sample_id"]]))
    if [(r["repeat"],r["sample_id"]) for r in rows]!=[(n,s) for n in range(3) for s in frozen_order]:raise GateFailure("gather order mismatch")
    models={r["model_sha256_before"] for r in reports}|{r["model_sha256_after"] for r in reports}
    if len(models)!=1:raise GateFailure("model hash mismatch")
    return {"schema":"r16_dscp_v3_scale_aggregate_v1","status":"passed","world_size":world,"wall_s":max(float(r["wall_s"]) for r in reports),"gpu_seconds":sum(float(r["gpu_seconds"])/world for r in reports),"metric_digest":canonical_sha([(r["sample_id"],r["repeat"],float(r["loss"])) for r in rows]),"model_sha256":models.pop(),"data_sha256":canonical_sha(list(frozen_order)),"schedule_sha256":canonical_sha({"warmup":1,"timed":3}),"ledger_digest":canonical_sha([r["ledger_digest"] for r in reports]),"per_gpu_peak_bytes":[int(r["resources"]["peak_bytes"]) for r in sorted(reports,key=lambda x:x["rank"])],"finite":all(r["finite"] for r in reports)}

def ephemeral_scale_probe(model:nn.Module, optimizer_factory:Callable, step:Callable[[nn.Module,Any],None],*,warmup:int=1,timed:int=3)->dict[str,Any]:
    before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; durations=[]
    for run in range(warmup+timed):
        ephemeral=type(model)(model.bases.detach().cpu(),model.coefficient_scales.detach().cpu()).to(next(model.parameters()).device);ephemeral.load_state_dict(before);opt=optimizer_factory(list(ephemeral.parameters()));start=time.perf_counter();step(ephemeral,opt)
        if run>=warmup:durations.append(time.perf_counter()-start)
    after=model.state_dict();
    if any(not torch.equal(before[k],after[k].detach().cpu()) for k in before):raise GateFailure("scale probe polluted input model")
    return {"warmup":warmup,"timed":timed,"wall_s":sum(durations)/len(durations),"input_model_sha256":canonical_sha({k:tensor_sha(v) for k,v in before.items()})}

@torch.inference_mode()
def evaluate_only(model:nn.Module, callback:Callable[[nn.Module],Any])->Any:return callback(model)

@dataclass
class PatienceState:
    best:float=math.inf; bad_epochs:int=0; best_epoch:int=-1
    def update(self,value:float,epoch:int,patience:int=4)->tuple[bool,bool]:
        improved=float(value)<self.best
        if improved:self.best=float(value);self.bad_epochs=0;self.best_epoch=int(epoch)
        else:self.bad_epochs+=1
        return improved,self.bad_epochs>=patience

def gate_terminal(mode:str,lineage:Lineage,gates:Mapping[str,Mapping[str,Any]],metrics:Mapping[str,Any],**extra:Any)->dict[str,Any]:
    passed=all(bool(v.get("passed")) for v in gates.values());return {"schema":TERMINAL_SCHEMA,"candidate":CANDIDATE,"mode":mode,"status":"passed" if passed else "fail_gate","lineage":lineage.__dict__,"gates":dict(gates),"metrics":dict(metrics),**extra}

@dataclass(frozen=True)
class PreparedRecord:
    public:PublicBundle; parent:torch.Tensor; deployment_args:tuple[Any,...]
    candidate_sha256:str; ledger:AccessLedger; features:torch.Tensor|None=None
    route_index:int=-1; k1:int=-1; family:str="unknown"

def prepare_public_record(public:PublicBundle,*,parent_predictor:Callable[[PublicBundle],torch.Tensor],deployment_builder:Callable[[PublicBundle,torch.Tensor],tuple[Any,...]],candidate_predictor:Callable[[tuple[Any,...]],torch.Tensor],split:str)->PreparedRecord:
    ledger=AccessLedger(split);ledger.add("public_bundle_ready",input_digest=public.input_digest,observed_indices=list(public.observed_indices))
    parent=parent_predictor(public);ledger.add("parent_ready",sha256=tensor_sha(parent),serialized=False)
    args=deployment_builder(public,parent);ledger.add("features_ready",sha256=canonical_sha([public.input_digest,tensor_sha(parent)]),serialized=False)
    output=candidate_predictor(args);digest=seal_candidate(output,ledger)
    return PreparedRecord(public,parent,args,digest,ledger)

class ProductionSmokePilotBackend:
    """Real smoke/pilot callbacks; all field caches remain host RAM only."""
    def __init__(self,*,public_loader:GuardedPublicLoader,parent_predictor:Callable[[PublicBundle],torch.Tensor],travel_builder:Callable[[PublicBundle],torch.Tensor],candidate:R16DSCP,optimizer:torch.optim.Optimizer,authorization:Mapping[str,Any],lineage:Lineage,source_h5:str|Path,run_dir:str|Path,device:torch.device,family_by_sample:Mapping[str,str],split:str="train",once_token:Path|None=None):
        self.public_loader=public_loader;self.parent_predictor=parent_predictor;self.travel_builder=travel_builder;self.candidate=candidate;self.optimizer=optimizer;self.authorization=authorization;self.lineage=lineage;self.source_h5=Path(source_h5);self.run_dir=Path(run_dir);self.device=device;self.family_by_sample=dict(family_by_sample);self.split=str(split);self.once_token=once_token;self.ridge=StreamingRidge();self.started=time.monotonic();self.latencies=[];self.e2e_latencies=[];self.peak_bytes=0
    def _args(self,public:PublicBundle,parent:torch.Tensor)->tuple[Any,...]:
        k0,k1=public.observed_indices;travel=self.travel_builder(public)
        return (public.velocity_mps[None,None].cpu(),public.source_map[None,None].cpu(),travel[None,None].cpu(),public.x_m.cpu(),public.z_m.cpu(),public.observed_wavefield[0][None,None].cpu(),public.observed_wavefield[1][None,None].cpu(),parent[None].cpu(),torch.tensor([k0]),torch.tensor([k1]))
    def _device_args(self,args:tuple[Any,...])->tuple[Any,...]:return tuple(value.to(self.device) if isinstance(value,torch.Tensor) else value for value in args)
    def prepare(self,index:int)->PreparedRecord:
        e2e_started=time.perf_counter();public=self.public_loader[int(index)];parent=self.parent_predictor(public).detach().cpu().float();args=self._args(public,parent);device_args=self._device_args(args)
        with torch.inference_mode():
            features,decisions,_=deployment_features(*device_args[:-2],self.candidate.bases,device_args[-2],device_args[-1]);output=self.candidate(*device_args).detach().cpu()
        ledger=AccessLedger("train");ledger.add("public_bundle_ready",input_digest=public.input_digest,observed_indices=list(public.observed_indices));ledger.add("parent_ready",sha256=tensor_sha(parent),serialized=False);ledger.add("features_ready",sha256=tensor_sha(features),serialized=False);digest=seal_candidate(output,ledger);decision=decisions[0]
        self.e2e_latencies.append(time.perf_counter()-e2e_started);return PreparedRecord(public,parent,args,digest,ledger,features.detach().cpu(),decision.index,public.observed_indices[1],self.family_by_sample[public.sample_id])
    def open_train_truth(self,prepared:PreparedRecord)->torch.Tensor:
        if self.split!="train":raise LeakageRefusal("open_train_truth requires train split")
        return authorized_future_truth(self.source_h5,prepared.public.source_index,prepared.k1,mode=self.lineage.mode,split="train",ledger=prepared.ledger,authorization=self.authorization,expected_lineage=self.lineage)
    def open_truth(self,prepared:PreparedRecord)->torch.Tensor:
        return authorized_future_truth(self.source_h5,prepared.public.source_index,prepared.k1,mode=self.lineage.mode,split=self.split,ledger=prepared.ledger,authorization=self.authorization,expected_lineage=self.lineage,once_token=self.once_token)
    def update(self,prepared:PreparedRecord)->Mapping[str,Any]:
        def loader():return self.open_train_truth(prepared)
        result=train_update(self.candidate,self.optimizer,self._device_args(prepared.deployment_args),loader,prepared.ledger,device=self.device);self.latencies.append(float(result["wall_s"]));self._peak();return result
    def measure_gradient(self,prepared:PreparedRecord)->Mapping[str,Any]:
        self.optimizer.zero_grad(set_to_none=True);started=time.perf_counter();args=self._device_args(prepared.deployment_args)
        with torch.autocast(device_type=self.device.type,dtype=torch.bfloat16,enabled=True):corrected=self.candidate(*args)
        truth=self.open_train_truth(prepared).to(self.device)[None];loss=exact_microblocked_loss(corrected[:,prepared.k1+1:].float(),truth.float(),torch.zeros(1,16,201,201,device=self.device))["total"];loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(self.candidate.parameters(),1.0));self.optimizer.zero_grad(set_to_none=True)
        if not math.isfinite(grad) or not torch.isfinite(loss):raise GateFailure("scale nonfinite")
        self.latencies.append(time.perf_counter()-started);self._peak();return {"loss":float(loss),"grad_norm":grad,"ledger_digest":prepared.ledger.digest(),"sample_id":prepared.public.sample_id}
    def _peak(self)->None:
        if self.device.type=="cuda":self.peak_bytes=max(self.peak_bytes,int(torch.cuda.max_memory_reserved(self.device)))
    @torch.inference_mode()
    def score(self,model:R16DSCP,prepared:PreparedRecord,truth:torch.Tensor)->Mapping[str,Any]:
        start=time.perf_counter();corrected=model(*self._device_args(prepared.deployment_args))[0,prepared.k1+1:].float().cpu();elapsed=time.perf_counter()-start;parent=prepared.parent[prepared.k1+1:];self.latencies.append(elapsed);self._peak()
        row=metric_record(corrected,parent,truth.float().cpu(),family=prepared.family,condition=basis_condition(model.bases[prepared.route_index],prepared.k1) if prepared.route_index>=0 else math.inf,abstain=prepared.route_index<0);row["latency_s"]=elapsed;return row
    def ridge_update(self,prepared:PreparedRecord,truth:torch.Tensor)->None:
        if prepared.route_index<0:return
        full_truth=torch.cat((prepared.parent[:prepared.k1+1],truth.float().cpu()),0);target=weighted_coefficient_target(self.candidate.bases[prepared.route_index],prepared.parent,full_truth,k1=prepared.k1);self.ridge.update(prepared.features,target[None]);del target,full_truth
    def ridge_finalize(self)->RidgePointwiseBaseline:return self.ridge.solve()
    @torch.inference_mode()
    def ridge_score(self,ridge:RidgePointwiseBaseline,prepared:PreparedRecord,truth:torch.Tensor)->Mapping[str,Any]:
        if prepared.route_index<0:corrected=prepared.parent.clone()
        else:
            coefficient=ridge(prepared.features).squeeze(0);basis=self.candidate.bases[prepared.route_index].cpu();correction=torch.einsum("tr,rhw->thw",basis,coefficient);correction*=c1_causal_mask(401,prepared.k1)[:,None,None];correction[:,0]=0;corrected=prepared.parent+correction
        return metric_record(corrected[prepared.k1+1:],prepared.parent[prepared.k1+1:],truth.float().cpu(),family=prepared.family,condition=basis_condition(self.candidate.bases[prepared.route_index],prepared.k1) if prepared.route_index>=0 else math.inf,abstain=prepared.route_index<0)
    def checkpoint(self,progress:Mapping[str,Any],is_best:bool=True)->Mapping[str,Any]:
        payload=checkpoint_payload(self.candidate,self.optimizer,run_identity=self.lineage.__dict__,sampler_order=list(range(len(self.public_loader))),progress=progress);return save_best_last(payload,self.run_dir,is_best=is_best)
    def resources(self,saved:Mapping[str,Any])->Mapping[str,Any]:
        gate=space_gate(int(saved["size_bytes"]),self.run_dir);values=sorted(self.latencies);p95=values[max(0,math.ceil(.95*len(values))-1)] if values else math.inf;e2e=sorted(self.e2e_latencies);e2e_p95=e2e[max(0,math.ceil(.95*len(e2e))-1)] if e2e else math.inf;e2e_mean=sum(e2e)/len(e2e) if e2e else math.inf
        return {"wall_s":time.monotonic()-self.started,"peak_bytes":self.peak_bytes,"checkpoint_bytes":int(saved["size_bytes"]),"space_passed":gate["passed"],"adapter_mean_s":sum(values)/len(values) if values else math.inf,"adapter_p95_s":p95,"e2e_mean_s":e2e_mean,"e2e_p95_s":e2e_p95,"e2e_mean_ratio":e2e_mean/12.652311868034303,"e2e_p95_ratio":e2e_p95/12.656147628091276,"parent_sha256":PARENT_SHA256,"fields_serialized":False,"coefficient_maps_serialized":False}
    def release(self,prepared:PreparedRecord)->None:del prepared

def _family_gain(records:Sequence[Mapping[str,Any]])->dict[str,float]:
    out={}
    for family in ("uniform","layered","marmousi"):
        rows=[r for r in records if r["family"]==family]
        if not rows: continue
        parent=sum(float(r["parent_mean_frame_rel_l2"]) for r in rows)/len(rows);candidate=sum(float(r["mean_frame_rel_l2"]) for r in rows)/len(rows)
        out[family]=(parent-candidate)/max(parent,1e-30)
    return out

def smoke_gates(initial_loss:float,final_loss:float,score_records:Sequence[Mapping[str,Any]],resources:Mapping[str,Any])->tuple[dict[str,Any],dict[str,Any]]:
    oracle={"uniform":.2224565778108675,"layered":.37334545199603053,"marmousi":.29789271567937464};gain=_family_gain(score_records);reduction=(initial_loss-final_loss)/max(abs(initial_loss),1e-30)
    gates={"loss":{"value":reduction,"threshold":.80,"passed":reduction>=.80},"oracle_gain":{"value":gain,"threshold_fraction":.5,"passed":all(gain.get(f,-math.inf)>=.5*v for f,v in oracle.items())},"aggregate_nonworse":{"passed":sum(float(r["aggregate_rel_l2"]) for r in score_records)<=sum(float(r["parent_rel_l2"]) for r in score_records)},"finite":{"passed":all(bool(r.get("finite")) for r in score_records)},"wall":{"value":resources["wall_s"],"threshold":600.,"passed":resources["wall_s"]<=600},"vram":{"value":resources["peak_bytes"],"threshold":8*1024**3,"passed":resources["peak_bytes"]<=8*1024**3},"checkpoint":{"value":resources["checkpoint_bytes"],"threshold":2*1024**2,"passed":resources["checkpoint_bytes"]<=2*1024**2},"space":{"passed":bool(resources["space_passed"])}}
    return gates,{"initial_loss":initial_loss,"final_loss":final_loss,"loss_reduction":reduction,"family_gain":gain,"records":list(score_records)}

def run_smoke_stage(*,records:Sequence[Any],update:Callable[[Any],Mapping[str,Any]],score:Callable[[],Sequence[Mapping[str,Any]]],checkpoint:Callable[[],Mapping[str,Any]],lineage:Lineage,run_dir:Path,max_updates:int=500,max_seconds:float=600.,resource_snapshot:Callable[[Mapping[str,Any]],Mapping[str,Any]],terminal_extra:Callable[[],Mapping[str,Any]]=lambda:{})->Mapping[str,Any]:
    if not records:raise GateFailure("smoke records empty")
    started=time.monotonic();losses=[];updates=0
    while updates<max_updates and time.monotonic()-started<max_seconds:
        result=update(records[updates%len(records)]);losses.append(float(result["loss"]));updates+=1
    scores=list(score());saved=dict(checkpoint());resources=dict(resource_snapshot(saved));resources["wall_s"]=time.monotonic()-started
    gates,metrics=smoke_gates(losses[0],losses[-1],scores,resources);terminal=gate_terminal("smoke",lineage,gates,metrics,updates=updates,checkpoint=saved,resources=resources,promotion_authorization_created=False,**dict(terminal_extra()))
    atomic_json_exclusive(terminal,run_dir/"terminal.json");return terminal

def pilot_gates(candidate_records:Sequence[Mapping[str,Any]],ridge_records:Sequence[Mapping[str,Any]],resources:Mapping[str,Any])->tuple[dict[str,Any],dict[str,Any]]:
    cand=aggregate_metrics(candidate_records);ridge=aggregate_metrics(ridge_records);core=pilot_gate(cand,ridge)
    late=all(float(c["time_bands"]["late"])<=float(c["parent_time_bands"]["late"]) for c in candidate_records);high=all(float(c["spectrum_bands"]["high"])<=float(c["parent_spectrum_bands"]["high"]) for c in candidate_records);energy=max(float(c["correction_energy_ratio"]) for c in candidate_records)
    gates={"core":{"values":core,"passed":core["passed"]},"late":{"passed":late},"high":{"passed":high},"energy":{"value":energy,"threshold":.35,"passed":energy<=.35},"finite":{"passed":all(bool(c["finite"]) for c in candidate_records)},"adapter_latency":{"mean":resources["adapter_mean_s"],"p95":resources["adapter_p95_s"],"passed":resources["adapter_mean_s"]<=.35 and resources["adapter_p95_s"]<=.50},"e2e":{"passed":resources["e2e_mean_ratio"]<=1.05 and resources["e2e_p95_ratio"]<=1.05},"vram":{"passed":resources["peak_bytes"]<=8*1024**3},"checkpoint":{"passed":resources["checkpoint_bytes"]<=2*1024**2},"space":{"passed":bool(resources["space_passed"])}}
    return gates,{"candidate":cand,"ridge":ridge,"record_metrics":list(candidate_records),"ridge_record_metrics":list(ridge_records)}

def run_pilot_stage(*,fit_records:Sequence[Any],confirm_records:Sequence[Any],candidate_update:Callable[[Any],Mapping[str,Any]],ridge_update:Callable[[Any],None],ridge_solve:Callable[[],Any],candidate_score:Callable[[Sequence[Any]],Sequence[Mapping[str,Any]]],ridge_score:Callable[[Any,Sequence[Any]],Sequence[Mapping[str,Any]]],checkpoint:Callable[[],Mapping[str,Any]],lineage:Lineage,run_dir:Path,max_updates:int=1000,max_seconds:float=1800.,resource_snapshot:Callable[[Mapping[str,Any]],Mapping[str,Any]],terminal_extra:Callable[[],Mapping[str,Any]]=lambda:{})->Mapping[str,Any]:
    if not fit_records or not confirm_records:raise GateFailure("pilot panels empty")
    for record in fit_records:ridge_update(record)
    ridge=ridge_solve();started=time.monotonic();updates=0
    while updates<max_updates and time.monotonic()-started<max_seconds:candidate_update(fit_records[updates%len(fit_records)]);updates+=1
    cand=list(candidate_score(confirm_records));ridge_metrics=list(ridge_score(ridge,confirm_records));saved=dict(checkpoint());resources=dict(resource_snapshot(saved));resources["wall_s"]=time.monotonic()-started
    gates,metrics=pilot_gates(cand,ridge_metrics,resources);terminal=gate_terminal("pilot",lineage,gates,metrics,updates=updates,checkpoint=saved,resources=resources,promotion_authorization_created=False,**dict(terminal_extra()))
    atomic_json_exclusive(terminal,run_dir/"terminal.json");return terminal

def run_long_stage(*,fit_records:Sequence[Any],calibration_records:Sequence[Any],update:Callable[[Any],Mapping[str,Any]],evaluate:Callable[[Sequence[Any]],Sequence[Mapping[str,Any]]],set_lr:Callable[[float],None],checkpoint:Callable[[Mapping[str,Any],bool],Mapping[str,Any]],resource_snapshot:Callable[[Mapping[str,Any]],Mapping[str,Any]],lineage:Lineage,run_dir:Path,start_epoch:int=0,max_epochs:int=20,patience:int=4,max_seconds:float=7200.,resume_state:Mapping[str,Any]|None=None,terminal_extra:Callable[[],Mapping[str,Any]]=lambda:{})->Mapping[str,Any]:
    if not fit_records or not calibration_records:raise GateFailure("long panels empty")
    state=PatienceState(**dict(resume_state or {}));history=[];started=time.monotonic();last={}
    try:
        for epoch in range(int(start_epoch),int(max_epochs)):
            lr=cosine_learning_rate(epoch,max_epochs);set_lr(lr);losses=[]
            for record in fit_records:
                if time.monotonic()-started>max_seconds:raise GateFailure("long wall budget exceeded")
                result=update(record);loss=float(result["loss"])
                if not math.isfinite(loss):raise GateFailure("long nonfinite loss")
                losses.append(loss)
            scored=list(evaluate(calibration_records));aggregate=aggregate_metrics(scored);primary=float(aggregate["joint_aggregate_rel_l2"])
            improved,stop=state.update(primary,epoch,patience);last=dict(checkpoint({"epoch":epoch,"lr":lr,"primary":primary,"bad_epochs":state.bad_epochs,"best":state.best,"best_epoch":state.best_epoch},improved));resources=dict(resource_snapshot(last))
            gates={"finite":{"passed":all(bool(r["finite"]) for r in scored)},"space":{"passed":bool(resources["space_passed"])},"vram":{"passed":resources["peak_bytes"]<=8*1024**3},"wall":{"passed":time.monotonic()-started<=max_seconds}}
            history.append({"epoch":epoch,"lr":lr,"mean_loss":sum(losses)/len(losses),"primary":primary,"improved":improved,"bad_epochs":state.bad_epochs,"gates":gates})
            if not all(g["passed"] for g in gates.values()):raise GateFailure("long epoch resource/finite gate failed")
            if stop:break
        terminal={"schema":TERMINAL_SCHEMA,"candidate":CANDIDATE,"mode":"long","status":"completed_train","lineage":lineage.__dict__,"epochs_completed":len(history),"best_epoch":state.best_epoch,"best_primary":state.best,"history":history,"checkpoint":last,"resources":dict(resource_snapshot(last)),**dict(terminal_extra())};atomic_json_exclusive(terminal,run_dir/"terminal.json");return terminal
    except Exception as exc:
        if not (run_dir/"terminal.json").exists():atomic_json_exclusive({"schema":TERMINAL_SCHEMA,"candidate":CANDIDATE,"mode":"long","status":"failed","reason":str(exc),"lineage":lineage.__dict__,"history":history,**dict(terminal_extra())},run_dir/"terminal.json")
        raise

def evaluation_gates(mode:str,records:Sequence[Mapping[str,Any]],resources:Mapping[str,Any],*,traditional_runtime_s:float=20.34137312322855)->tuple[dict[str,Any],dict[str,Any]]:
    aggregate=aggregate_metrics(records);late=all(float(r["time_bands"]["late"])<=float(r["parent_time_bands"]["late"]) for r in records);high=all(float(r["spectrum_bands"]["high"])<=float(r["parent_spectrum_bands"]["high"]) for r in records);finite=all(bool(r["finite"]) for r in records)
    gates={"late":{"passed":late},"high":{"passed":high},"finite":{"passed":finite},"e2e":{"passed":resources["e2e_mean_ratio"]<=1.05 and resources["e2e_p95_ratio"]<=1.05},"vram":{"passed":resources["peak_bytes"]<=int(6.5*1024**3)}}
    if mode=="final-train-confirm":
        joint=(aggregate["parent_joint_aggregate_rel_l2"]-aggregate["joint_aggregate_rel_l2"])/aggregate["parent_joint_aggregate_rel_l2"];family={f:(x["parent_rel_l2"]-x["aggregate_rel_l2"])/x["parent_rel_l2"] for f,x in aggregate["by_family"].items()};gates.update(joint={"value":joint,"threshold":.01,"passed":joint>=.01},families={"value":family,"threshold":.005,"passed":all(v>=.005 for v in family.values())},nonworse={"value":aggregate["nonworse"],"threshold":23,"passed":aggregate["nonworse"]>=23})
    else:
        gates["absolute_accuracy"]={"aggregate":aggregate["joint_aggregate_rel_l2"],"families":{f:x["aggregate_rel_l2"] for f,x in aggregate["by_family"].items()},"threshold":.05,"passed":aggregate["joint_aggregate_rel_l2"]<=.05 and all(x["aggregate_rel_l2"]<=.05 for x in aggregate["by_family"].values())};gates["protocol_speed"]={"mean":traditional_runtime_s/resources["e2e_mean_s"],"p95":traditional_runtime_s/resources["e2e_p95_s"],"threshold":10.,"passed":traditional_runtime_s/resources["e2e_mean_s"]>=10 and traditional_runtime_s/resources["e2e_p95_s"]>=10}
    return gates,{"aggregate":aggregate,"records":list(records)}

def run_evaluation_stage(*,mode:str,records:Sequence[Any],evaluate_one:Callable[[Any],Mapping[str,Any]],resource_snapshot:Callable[[],Mapping[str,Any]],lineage:Lineage,run_dir:Path,metadata_digest:str,candidate_lock_payload:Mapping[str,Any]|None=None,required_validation_terminal:Mapping[str,Any]|None=None,terminal_extra:Callable[[],Mapping[str,Any]]=lambda:{})->Mapping[str,Any]:
    if mode not in {"final-train-confirm","validation-once","test-once"}:raise GateFailure("invalid evaluation mode")
    if mode=="test-once" and (required_validation_terminal is None or required_validation_terminal.get("status")!="passed"):raise GateFailure("test requires passed validation terminal")
    if not records:raise GateFailure("evaluation records empty")
    scored=[dict(evaluate_one(record)) for record in records];resources=dict(resource_snapshot());gates,metrics=evaluation_gates(mode,scored,resources);terminal=gate_terminal(mode,lineage,gates,metrics,resources=resources,metadata_digest=metadata_digest,optimizer_steps=0,backward_calls=0,**dict(terminal_extra()))
    if terminal["status"]=="passed" and mode=="final-train-confirm":
        if candidate_lock_payload is None:raise GateFailure("passed final confirmation requires candidate lock payload")
        atomic_json_exclusive(dict(candidate_lock_payload),run_dir/"candidate_lock.json");terminal["candidate_lock_sha256"]=hashlib.sha256((run_dir/"candidate_lock.json").read_bytes()).hexdigest()
    atomic_json_exclusive(terminal,run_dir/"terminal.json");return terminal

__all__=["AccessLedger","AUTH_SCHEMA","CANDIDATE","DispatchPhase","GateFailure","GuardedOnsetDataset","GuardedOnsetRecord","GuardedPublicLoader","LeakageRefusal","Lineage","PatienceState","PreparedRecord","ProductionSmokePilotBackend","PublicBundle","StreamingRidge","aggregate_metrics","aggregate_scale_ranks","assert_stage_split","authorized_future_truth","canonical_sha","ddp_host_aggregate","ephemeral_scale_probe","evaluate_only","evaluation_gates","gate_terminal","metric_record","model_digest","pilot_gate","pilot_gates","prepare_public_record","read_exact_observations","read_future_truth","run_evaluation_stage","run_long_stage","run_pilot_stage","run_scale_probe","run_smoke_stage","scale_decision","seal_candidate","smoke_gates","stage_dispatch","tensor_sha","train_update","validate_authorization"]
