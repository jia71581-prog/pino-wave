"""V5 BF16 execution, finite-gradient gates, protocol timing, and bounded smoke."""
from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass
import math, time
from typing import Any, Callable, Mapping, Sequence
import torch

from .r16_dscp import CONDITION_MAXIMUM, c1_causal_mask, deployment_features
from .r16_dscp_engine_v3 import AccessLedger, GateFailure, canonical_sha, smoke_gates
from .r16_dscp_engine_v4 import CandidateSpool, LongResumeState, V4Prepared, V4ProductionBackend, frozen_global_groups, canonical_public, canonical_travel, deployment_args
from .r16_dscp_training_v2 import exact_microblocked_loss, sha256_file

class NonFiniteStepRejection(RuntimeError):pass
class OOMStepRejection(RuntimeError):pass
class SmokeBudgetRejection(RuntimeError):pass
class BindingDriftRejection(RuntimeError):pass

def model_state_digest(model:torch.nn.Module)->str:return canonical_sha({name:sha256_file_tensor(value) for name,value in sorted(model.state_dict().items())})
def sha256_file_tensor(value:torch.Tensor)->str:
    import hashlib
    tensor=torch.as_tensor(value).detach().cpu().contiguous();return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

def verify_before_data(authorization:Mapping[str,Any],*,input_checkpoint:str|None,parent:str,basis:str,config:str,code:str)->Mapping[str,str]:
    observed={"parent_sha256":sha256_file(parent),"basis_sha256":sha256_file(basis),"config_sha256":sha256_file(config),"code_sha256":sha256_file(code),"input_checkpoint_sha256":"none" if input_checkpoint is None else sha256_file(input_checkpoint)}
    for key,value in observed.items():
        if str(authorization.get(key))!=value:raise BindingDriftRejection(f"effective binding drift before data/CUDA: {key}")
    return observed

def verify_loaded_state(model:torch.nn.Module,authorization:Mapping[str,Any])->str:
    digest=model_state_digest(model)
    if authorization.get("input_state_sha256") not in {None,digest}:raise BindingDriftRejection("loaded model state hash drift")
    return digest

@dataclass
class ProtocolLatency:
    parent_s:list[float]; feature_s:list[float]; adapter_s:list[float]; spool_io_s:list[float]
    def __init__(self):self.parent_s=[];self.feature_s=[];self.adapter_s=[];self.spool_io_s=[]
    @staticmethod
    def _summary(values:Sequence[float])->Mapping[str,float]:
        ordered=sorted(float(v) for v in values)
        if not ordered:return {"mean_s":math.inf,"p95_s":math.inf}
        return {"mean_s":sum(ordered)/len(ordered),"p95_s":ordered[max(0,math.ceil(.95*len(ordered))-1)]}
    def payload(self)->Mapping[str,Any]:
        adapter=self._summary(self.adapter_s);parent=self._summary(self.parent_s);e2e=self._summary([p+a+f for p,a,f in zip(self.parent_s,self.adapter_s,self.feature_s)])
        return {"parent":parent,"feature":self._summary(self.feature_s),"adapter":adapter,"e2e":e2e,"spool_io_excluded":self._summary(self.spool_io_s),"finite":all(math.isfinite(v) and v>=0 for group in (self.parent_s,self.feature_s,self.adapter_s,self.spool_io_s) for v in group)}

class SegmentTimer:
    def __init__(self,device:torch.device,clock:Callable[[],float]=time.perf_counter):self.device=device;self.clock=clock
    def call(self,function:Callable[[],Any])->tuple[Any,float]:
        if self.device.type=="cuda":
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);torch.cuda.synchronize(self.device);start.record();result=function();end.record();torch.cuda.synchronize(self.device);return result,float(start.elapsed_time(end))/1000.
        started=self.clock();result=function();return result,float(self.clock()-started)

class V5ProductionBackend(V4ProductionBackend):
    def __init__(self,*args,autocast_factory:Callable[[],Any]|None=None,timer:SegmentTimer|None=None,**kwargs):
        super().__init__(*args,**kwargs);self.spool=CandidateSpool(self.run_dir,run_digest=self.lineage.run_digest,rank=int(__import__("os").environ.get("RANK","0")),owned_root="/dev/shm/r16_dscp_v5");self.autocast_factory=autocast_factory or (lambda:torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True) if self.device.type=="cuda" else nullcontext());self.timer=timer or SegmentTimer(self.device);self.protocol=ProtocolLatency();self.autocast_dtype="torch.bfloat16";self.autocast_enabled=self.device.type=="cuda" or autocast_factory is not None
    def _coefficients(self,features:torch.Tensor,route_index:int,abstain:bool,condition:float)->torch.Tensor:
        with self.autocast_factory():unit=self.candidate.coefficient_head(features)
        coefficient=torch.zeros_like(unit,dtype=torch.float32)
        if not abstain and condition<=CONDITION_MAXIMUM:coefficient[0]=unit[0].float()*self.candidate.coefficient_scales[route_index,:,None,None].float()
        return coefficient
    def _materialize(self,parent:torch.Tensor,coefficient:torch.Tensor,route_index:int,k1:int)->torch.Tensor:
        corrected=parent.float().clone()
        if route_index>=0:
            correction=torch.einsum("tr,rhw->thw",self.candidate.bases[route_index].float(),coefficient[0]);correction*=c1_causal_mask(parent.shape[1],k1,device=parent.device,dtype=torch.float32)[:,None,None];correction[:,0]=0;corrected[0]+=correction
        return corrected
    def prepare(self,index:int,key_suffix:str="candidate")->V4Prepared:
        if self.active:raise RuntimeError("previous spool active")
        public=canonical_public(self.public_loader[int(index)]);parent,parent_s=self.timer.call(lambda:torch.as_tensor(self.parent_predictor(public)).detach().cpu().float())
        def feature_call():
            travel=canonical_travel(public,self.travel_builder).float();args=deployment_args(public,parent,travel,self.device);features,decisions,conditions=deployment_features(*args[:-2],self.candidate.bases.float(),args[-2],args[-1]);return travel,args,features.float(),decisions[0],float(conditions[0])
        (travel,args,features,decision,condition),feature_s=self.timer.call(feature_call)
        (coefficient,adapted),adapter_s=self.timer.call(lambda:(lambda c:(c,self._materialize(args[7],c,decision.index,args[-1].item())))(self._coefficients(features,decision.index,decision.abstain,condition)))
        ledger=AccessLedger(self.split);ledger.add("public_bundle_ready",split=self.split,input_digest=public.input_digest);ledger.add("autocast",enabled=self.autocast_enabled,dtype=self.autocast_dtype,model_compute="bf16",basis_materialization="float32",loss_reduction="float32_or_float64")
        (seal,spool_s)=self.timer.call(lambda:self.spool.seal(f"{public.sample_id}:{key_suffix}",adapted.detach().cpu(),ledger));self.active.append(seal["path"]);self.spool_max_bytes=max(self.spool_max_bytes,int(seal["serialized_bytes"]));self.protocol.parent_s.append(parent_s);self.protocol.feature_s.append(feature_s);self.protocol.adapter_s.append(adapter_s);self.protocol.spool_io_s.append(spool_s)
        return V4Prepared(public,parent,travel,args,features.detach().cpu(),decision.index,condition,coefficient.detach().cpu(),seal,ledger)
    def _forward_loss(self,prepared:V4Prepared,truth:torch.Tensor):
        features=prepared.features.to(self.device).float();abstain=prepared.route_index<0;coefficient=self._coefficients(features,prepared.route_index,abstain,prepared.condition);adapted=self._materialize(prepared.args[7],coefficient,prepared.route_index,prepared.public.observed_indices[1]);k1=prepared.public.observed_indices[1];return exact_microblocked_loss(adapted[:,k1+1:].float(),truth.to(self.device).float(),coefficient),coefficient
    def _finite_backward(self,loss:torch.Tensor,*,reset:bool,step:bool,clip:bool=True)->Mapping[str,float]:
        if reset:self.optimizer.zero_grad(set_to_none=True)
        parameters=[p for p in self.candidate.parameters() if p.requires_grad]
        try:
            if not torch.isfinite(loss):raise NonFiniteStepRejection("nonfinite loss")
            loss.backward()
            gradients=[p.grad for p in parameters if p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all() for g in gradients):raise NonFiniteStepRejection("nonfinite or absent gradient")
            unclipped=math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients))
            if not math.isfinite(unclipped):raise NonFiniteStepRejection("nonfinite unclipped grad norm")
            if clip:torch.nn.utils.clip_grad_norm_(parameters,1.0)
            clipped=math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in parameters if p.grad is not None))
            if not math.isfinite(clipped):raise NonFiniteStepRejection("nonfinite clipped grad norm")
            if step:self.optimizer.step()
            return {"unclipped_grad_norm":unclipped,"clipped_grad_norm":clipped}
        except RuntimeError as exc:
            self.optimizer.zero_grad(set_to_none=True)
            if "out of memory" in str(exc).lower():
                if self.device.type=="cuda":torch.cuda.empty_cache()
                raise OOMStepRejection(str(exc)) from exc
            raise
    def update(self,prepared:V4Prepared)->Mapping[str,Any]:
        truth=self.open_truth(prepared);losses,coefficient=self._forward_loss(prepared,truth[None]);norms=self._finite_backward(losses["total"],reset=True,step=True);prepared.ledger.add("step",autocast_enabled=self.autocast_enabled,autocast_dtype=self.autocast_dtype,**norms);self.cleanup(prepared);return {"loss":float(losses["total"]),"coefficient_energy":float(losses["normalized_coefficient_energy"]),**norms,"ledger_digest":prepared.ledger.digest()}
    def measure_gradient(self,prepared:V4Prepared)->Mapping[str,Any]:
        truth=self.open_truth(prepared);losses,_=self._forward_loss(prepared,truth[None]);norms=self._finite_backward(losses["total"],reset=True,step=False);self.optimizer.zero_grad(set_to_none=True);self.cleanup(prepared);return {"loss":float(losses["total"]),"coefficient_energy":float(losses["normalized_coefficient_energy"]),**norms,"sample_id":prepared.public.sample_id,"ledger_digest":prepared.ledger.digest()}
    def backward_only(self,prepared:V4Prepared,loss_scale:float)->Mapping[str,Any]:
        truth=self.open_truth(prepared);losses,_=self._forward_loss(prepared,truth[None]);norms=self._finite_backward(losses["total"]*float(loss_scale),reset=False,step=False,clip=False);self.cleanup(prepared);return {"loss":float(losses["total"]),**norms,"ledger_digest":prepared.ledger.digest()}
    def resources(self,saved:Mapping[str,Any],world_size:int=1)->Mapping[str,Any]:
        result=dict(super().resources(saved,world_size));timing=self.protocol.payload();e2e_mean=timing["e2e"]["mean_s"];e2e_p95=timing["e2e"]["p95_s"];result.update(protocol_timing=timing,adapter_mean_s=timing["adapter"]["mean_s"],adapter_p95_s=timing["adapter"]["p95_s"],e2e_mean_s=e2e_mean,e2e_p95_s=e2e_p95,e2e_mean_ratio=e2e_mean/12.652311868034303,e2e_p95_ratio=e2e_p95/12.656147628091276,protocol_latency_finite=timing["finite"]);return result

def run_v5_smoke(*,records:Sequence[Any],update:Callable[[Any],Mapping[str,Any]],quick_score:Callable[[],tuple[Sequence[Mapping[str,Any]],bool]],final_score:Callable[[],Sequence[Mapping[str,Any]]],checkpoint:Callable[[],Mapping[str,Any]],resource_snapshot:Callable[[Mapping[str,Any]],Mapping[str,Any]],terminal:Callable[[Mapping[str,Any]],None],clock:Callable[[],float]=time.monotonic,max_updates:int=192,overall_seconds:float=600.,final_reserve_s:float=180.)->Mapping[str,Any]:
    if max_updates!=192:raise ValueError("v5 smoke max_updates frozen at 192")
    started=clock();overall_deadline=started+overall_seconds;training_deadline=min(started+420.,overall_deadline-final_reserve_s);losses=[];updates=0;early=False
    while updates<max_updates and clock()<training_deadline:
        result=update(records[updates%len(records)]);losses.append(float(result["loss"]));updates+=1
        if updates%32==0:
            _,quick_pass=quick_score()
            if losses[-1]<=.2*losses[0] and quick_pass:early=True;break
    if overall_deadline-clock()<final_reserve_s:
        payload={"status":"fail_budget","updates":updates,"reason":"final reserve below 180s","total_s":clock()-started};terminal(payload);return payload
    scores=list(final_score());saved=dict(checkpoint());resources=dict(resource_snapshot(saved));resources["wall_s"]=clock()-started;gates,metrics=smoke_gates(losses[0],losses[-1],scores,resources);total=clock()-started
    status="passed" if all(g["passed"] for g in gates.values()) and total<=overall_seconds else "fail_gate" if total<=overall_seconds else "fail_budget";payload={"status":status,"updates":updates,"early_stop":early,"training_deadline_s":420.,"final_reserve_s":final_reserve_s,"total_s":total,"gates":gates,"metrics":metrics,"checkpoint":saved,"resources":resources};terminal(payload);return payload

def v5_long_command(*,world_size:int,authorization:str,preregistration:str,config:str,scale_decision:str,input_checkpoint:str,resume:bool=False)->Mapping[str,str]:
    prefix="env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=";common=f" scripts/train_r16_dscp_v5.py --mode long --world-size {world_size} --authorization {authorization} --preregistration {preregistration} --config {config} --scale-decision {scale_decision} --input-checkpoint {input_checkpoint}"+(" --resume" if resume else "")
    if world_size==1:command=prefix+"0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python"+common
    elif world_size==4:command=prefix+"0,1,2,3 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. torchrun --standalone --nproc-per-node=4"+common
    else:raise ValueError("world size must be 1 or 4")
    return {"command":command,"command_sha256":canonical_sha(command),"world_size":str(world_size),"resume":str(bool(resume)).lower()}

class V5LongRunner:
    def __init__(self,*,world_size:int,rank:int,train_records:list[Any],calibration_records:list[Any],zero_grad:Callable,backward_record:Callable,optimizer_step:Callable,set_lr:Callable,evaluate:Callable,gather:Callable,checkpoint:Callable,terminal:Callable,clock:Callable[[],float]=time.monotonic):
        self.world_size=world_size;self.rank=rank;self.groups=frozen_global_groups(train_records);self.calibration=calibration_records;self.zero_grad=zero_grad;self.backward_record=backward_record;self.optimizer_step=optimizer_step;self.set_lr=set_lr;self.evaluate=evaluate;self.gather=gather;self.checkpoint=checkpoint;self.terminal=terminal;self.clock=clock
    def run(self,state:LongResumeState=LongResumeState(),max_epochs:int=20)->Mapping[str,Any]:
        wall_limit=7200. if self.world_size==1 else 2700.;gpu_limit=10800.;reserve=180.;started=self.clock();step=state.global_step;best=state.best;best_epoch=state.best_epoch;bad=state.bad_epochs
        try:
            for epoch in range(state.epoch,max_epochs):
                self.set_lr(3e-5+.5*(3e-3-3e-5)*(1+math.cos(math.pi*epoch/max(max_epochs-1,1))));start=state.group_index if epoch==state.epoch else 0
                for group_index in range(start,len(self.groups)):
                    elapsed=self.clock()-started
                    if elapsed>=wall_limit-reserve or self.world_size*elapsed>=gpu_limit:raise SmokeBudgetRejection("long reserve/GPU-seconds exhausted before group")
                    self.zero_grad();assigned=self.groups[group_index] if self.world_size==1 else [self.groups[group_index][self.rank]];scale=.25 if self.world_size==1 else 1.;local_failure=None
                    try:[self.backward_record(record,scale) for record in assigned]
                    except Exception as exc:local_failure={"rank":self.rank,"reason":str(exc),"epoch":epoch,"group_index":group_index}
                    statuses=self.gather([local_failure] if local_failure else [{"ok":True,"rank":self.rank}]);failures=[status for status in statuses if "reason" in status]
                    if failures:self.zero_grad();raise GateFailure(f"global backward failure: {failures}")
                    elapsed=self.clock()-started
                    if elapsed>=wall_limit-reserve or self.world_size*elapsed>=gpu_limit:self.zero_grad();raise SmokeBudgetRejection("long reserve/GPU-seconds exhausted before step")
                    self.optimizer_step();step+=1
                    if self.rank==0:self.checkpoint(LongResumeState(epoch,step,group_index+1,best,bad,best_epoch),False)
                rows=self.gather([self.evaluate(record) for record in self.calibration[self.rank::self.world_size]])
                if self.rank==0:
                    metric=sum(float(r["aggregate_rel_l2"]) for r in rows)/len(rows);improved=metric<best
                    if improved:best=metric;best_epoch=epoch;bad=0
                    else:bad+=1
                    self.checkpoint(LongResumeState(epoch+1,step,0,best,bad,best_epoch),improved)
                flags=self.gather([{"stop":bad>=4}] if self.rank==0 else [])
                if flags and flags[0].get("stop"):break
            elapsed=self.clock()-started
            if elapsed>wall_limit or self.world_size*elapsed>gpu_limit:raise SmokeBudgetRejection("long terminal budget exceeded")
            result={"status":"completed_train","world_size":self.world_size,"wall_s":elapsed,"gpu_seconds":self.world_size*elapsed,"global_step":step,"best_metric":best,"best_epoch":best_epoch,"final_reserve_s":reserve}
            if self.rank==0:self.terminal(result)
            return result
        except Exception as exc:
            self.zero_grad();status="failed_budget" if isinstance(exc,SmokeBudgetRejection) else "failed";payload={"status":status,"reason":str(exc),"world_size":self.world_size,"wall_s":self.clock()-started,"gpu_seconds":self.world_size*(self.clock()-started),"global_step":step}
            if self.rank==0:self.terminal(payload)
            raise

def long_terminal(*,status:str,decision:Mapping[str,Any],run_digest:str,effective:Mapping[str,str],input_checkpoint:Mapping[str,Any],best:Mapping[str,Any],last:Mapping[str,Any],world_size:int,wall_s:float,gpu_seconds:float,epoch:int,best_metric:float,access_digest:str,latency:Mapping[str,Any],peak_vram:int,gates:Mapping[str,Any])->Mapping[str,Any]:
    required=("path","sha256","size_bytes")
    if any(key not in best or key not in last for key in required):raise BindingDriftRejection("long terminal checkpoint binding incomplete")
    return {"schema":"r16_dscp_v5_long_terminal_v1","candidate":"r16_dscp_v5","mode":"long","status":status,"decision":dict(decision),"run_digest":run_digest,"effective":dict(effective),"input_checkpoint":dict(input_checkpoint),"best_checkpoint":dict(best),"last_checkpoint":dict(last),"world_size":world_size,"wall_s":wall_s,"gpu_seconds":gpu_seconds,"gpu_hours":gpu_seconds/3600.,"epoch":epoch,"best_metric":best_metric,"access_ledger_digest":access_digest,"latency":dict(latency),"peak_vram_bytes":peak_vram,"gates":dict(gates)}

__all__=["BindingDriftRejection","NonFiniteStepRejection","OOMStepRejection","ProtocolLatency","SegmentTimer","SmokeBudgetRejection","V5LongRunner","V5ProductionBackend","long_terminal","model_state_digest","run_v5_smoke","v5_long_command","verify_before_data","verify_loaded_state"]
