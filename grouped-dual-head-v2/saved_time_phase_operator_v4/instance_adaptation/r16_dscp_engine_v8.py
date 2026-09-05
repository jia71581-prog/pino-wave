"""V8 terminal schemas and exception-safe terminal payloads."""
from __future__ import annotations
from typing import Any,Callable,Mapping,Sequence
import time

from .r16_dscp_engine_v3 import canonical_sha,smoke_gates
from .r16_dscp_engine_v4 import CandidateSpool
from .r16_dscp_engine_v7 import V7ProductionBackend,validate_checkpoint_identity

ROLLBACK="zero_grad_cleanup_owned_spool_preserve_run_best_last_log_never_touch_protected_never_fake_success"

class TerminalBindingRefusal(RuntimeError):pass

class V8ProductionBackend(V7ProductionBackend):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.spool=CandidateSpool(self.run_dir,run_digest=self.lineage.run_digest,rank=int(__import__("os").environ.get("RANK","0")),owned_root="/dev/shm/r16_dscp_v8")

def complete_identity_v8(**values:Any)->dict[str,Any]:
    payload={"schema":"r16_dscp_v8_checkpoint_identity_v1","candidate":"r16_dscp_v8",**values};payload["identity_digest"]=canonical_sha(payload);validate_checkpoint_identity(payload);return payload

def _checkpoint(checkpoint:Mapping[str,Any]|None,success:bool)->Mapping[str,Any]|None:
    if checkpoint is None:
        if success:raise TerminalBindingRefusal("successful terminal requires checkpoint")
        return None
    if success and any(not checkpoint.get(key) for key in ("best","last")):raise TerminalBindingRefusal("success requires best and last")
    return dict(checkpoint)

def terminal_base(*,mode:str,status:str,lineage:Mapping[str,Any],effective:Mapping[str,Any],parent:Mapping[str,Any],checkpoint:Mapping[str,Any]|None=None,**extra:Any)->dict[str,Any]:
    success=status in {"passed","completed_train"};return {"schema":f"r16_dscp_v8_{mode.replace('-','_')}_terminal_v1","candidate":"r16_dscp_v8","mode":mode,"status":status,"decision":status,"lineage":dict(lineage),"effective":dict(effective),"parent":dict(parent),"checkpoint":_checkpoint(checkpoint,success),"rollback":ROLLBACK,**extra}

def failure_terminal(*,mode:str,error:BaseException,lineage:Mapping[str,Any],authorization:Mapping[str,Any],effective:Mapping[str,Any],parent:Mapping[str,Any],checkpoint:Mapping[str,Any]|None=None)->dict[str,Any]:
    return terminal_base(mode=mode,status="failed",lineage=lineage,effective=effective,parent=parent,checkpoint=checkpoint,error_type=type(error).__name__,error_message=str(error),authorization_digest=authorization.get("authorization_digest"),gates={"passed":False})

def run_v8_smoke(*,backend:Any,records:Sequence[int],quick_gate:Callable[[Sequence[Mapping[str,Any]]],bool],checkpoint:Callable[[],Mapping[str,Any]],resource_snapshot:Callable[[Mapping[str,Any]],Mapping[str,Any]],terminal:Callable[[Mapping[str,Any]],None],lineage:Mapping[str,Any],effective:Mapping[str,Any],parent:Mapping[str,Any],clock:Callable[[],float]=time.monotonic)->Mapping[str,Any]:
    started=clock();[backend.preload(i) for i in records];deadline=min(started+420,started+600-180);losses=[];updates=0;early=False
    while updates<192 and clock()<deadline:
        losses.append(float(backend.update_cached(records[updates%len(records)])["loss"]));updates+=1
        if updates%32==0:
            quick=[backend.score_cached(i) for i in records]
            if losses[-1]<=.2*losses[0] and quick_gate(quick):early=True;break
    cache=backend.cache.payload();latency=backend.protocol.payload();peak=backend.peak_bytes
    if started+600-clock()<180:
        payload=terminal_base(mode="smoke",status="fail_budget",lineage=lineage,effective=effective,parent=parent,checkpoint=None,updates=updates,early_stop=early,access_ledger_digest=None,cache=cache,latency=latency,peak_vram_bytes=peak,gates={"budget":{"passed":False}});terminal(payload);backend.cache.clear();return payload
    scores=[backend.score_cached(i) for i in records];saved=dict(checkpoint());resources=dict(resource_snapshot(saved));resources["wall_s"]=clock()-started;gates,metrics=smoke_gates(losses[0],losses[-1],scores,resources);total=clock()-started;status="passed" if all(g["passed"] for g in gates.values()) and total<=600 else "fail_budget" if total>600 else "fail_gate";payload=terminal_base(mode="smoke",status=status,lineage=lineage,effective=effective,parent=parent,checkpoint=saved if status=="passed" else saved,updates=updates,early_stop=early,total_s=total,access_ledger_digest=canonical_sha([r["ledger_digest"] for r in scores]),cache=cache,latency=latency,peak_vram_bytes=peak,gates=gates,metrics=metrics);terminal(payload);backend.cache.clear();return payload

def long_terminal_v8(*,status:str,lineage:Mapping[str,Any],effective:Mapping[str,Any],parent:Mapping[str,Any],decision:Mapping[str,Any],input_checkpoint:Mapping[str,Any],best:Mapping[str,Any]|None,last:Mapping[str,Any]|None,world_size:int,wall_s:float,gpu_seconds:float,epoch:int,best_metric:float,access:Mapping[str,Any],cache:Mapping[str,Any],latency:Mapping[str,Any],peak_vram:int,gates:Mapping[str,Any])->Mapping[str,Any]:
    checkpoint=None if best is None or last is None else {"best":best,"last":last}
    return terminal_base(mode="long",status=status,lineage=lineage,effective=effective,parent=parent,checkpoint=checkpoint,scale_decision=dict(decision),selected_world_size=decision.get("selected_gpus"),actual_world_size=world_size,input_checkpoint=dict(input_checkpoint),wall_s=wall_s,gpu_seconds=gpu_seconds,gpu_hours=gpu_seconds/3600.,epoch=epoch,best_metric=best_metric,access=dict(access),cache=dict(cache),latency=dict(latency),peak_vram_bytes=peak_vram,gates=dict(gates))

__all__=["ROLLBACK","TerminalBindingRefusal","V8ProductionBackend","complete_identity_v8","failure_terminal","long_terminal_v8","run_v8_smoke","terminal_base"]
