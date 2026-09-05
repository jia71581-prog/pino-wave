from __future__ import annotations
from contextlib import AbstractContextManager
import math
import pytest, torch

from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import R16DSCP
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v5 import *
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v4 import V4ProductionBackend
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import exact_microblocked_loss
from scripts.train_r16_dscp_v5 import V5StageDispatcher
from scripts.train_r16_dscp_v5 import V5ProductionDispatcher,backend_after_strict_verification,validate_final_from_long
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import GateFailure,pilot_gates
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v4 import LongResumeState

def model():
 q=torch.linalg.qr(torch.randn(401,16,generator=torch.Generator().manual_seed(372),dtype=torch.float64))[0].float();return R16DSCP(q.repeat(3,1,1),torch.ones(3,16))

class Marker(AbstractContextManager):
 def __init__(self,state):self.state=state
 def __enter__(self):self.state.append("enter");return self
 def __exit__(self,*a):self.state.append("exit")

def bare_backend(candidate=None):
 backend=object.__new__(V5ProductionBackend);backend.candidate=candidate or model();backend.device=torch.device("cpu");backend.optimizer=torch.optim.AdamW(backend.candidate.parameters(),lr=1e-3);backend.autocast_enabled=True;backend.autocast_dtype="torch.bfloat16";return backend

def test_actual_r16_controlled_autocast_actual_coefficient_loss_and_zero_init():
 state=[];backend=bare_backend();backend.autocast_factory=lambda:Marker(state);features=torch.randn(1,29,5,7);zero=backend._coefficients(features,0,False,1.);assert torch.count_nonzero(zero)==0 and state==["enter","exit"]
 backend.candidate.pointwise_out.bias.data.fill_(.2);coefficient=backend._coefficients(features,0,False,1.);parent=torch.zeros(1,401,5,7);adapted=backend._materialize(parent,coefficient,0,2);losses=exact_microblocked_loss(adapted[:,3:],adapted[:,3:].detach(),coefficient);assert losses["normalized_coefficient_energy"]>0;losses["total"].backward();assert torch.linalg.vector_norm(backend.candidate.pointwise_out.bias.grad)>0

def test_nonfinite_loss_rejected_without_step_or_param_change():
 backend=bare_backend();before=[p.detach().clone() for p in backend.candidate.parameters()];parameter=next(backend.candidate.parameters());loss=parameter.sum()*torch.tensor(float("nan"))
 with pytest.raises(NonFiniteStepRejection):backend._finite_backward(loss,reset=True,step=True)
 assert all(torch.equal(a,b) for a,b in zip(before,backend.candidate.parameters())) and all(p.grad is None for p in backend.candidate.parameters())

def test_nonfinite_gradient_rejected_without_step():
 backend=bare_backend();parameter=next(backend.candidate.parameters());handle=parameter.register_hook(lambda grad:grad*float("nan"));before=[p.detach().clone() for p in backend.candidate.parameters()]
 with pytest.raises(NonFiniteStepRejection):backend._finite_backward(parameter.square().sum(),reset=True,step=True)
 handle.remove();assert all(torch.equal(a,b) for a,b in zip(before,backend.candidate.parameters()))

def test_oom_typed_rejection_clears_gradients():
 class OOM(torch.autograd.Function):
  @staticmethod
  def forward(ctx,x):return x.sum()
  @staticmethod
  def backward(ctx,g):raise RuntimeError("CUDA out of memory")
 backend=bare_backend();parameter=next(backend.candidate.parameters())
 with pytest.raises(OOMStepRejection):backend._finite_backward(OOM.apply(parameter),reset=True,step=True)
 assert all(p.grad is None for p in backend.candidate.parameters())

def test_protocol_latency_finite_and_spool_excluded():
 latency=ProtocolLatency();latency.parent_s.extend([1.,2.]);latency.feature_s.extend([.1,.2]);latency.adapter_s.extend([.2,.3]);latency.spool_io_s.extend([9.,10.]);payload=latency.payload();assert payload["finite"] and payload["adapter"]["mean_s"]==pytest.approx(.25) and payload["e2e"]["mean_s"]<3 and payload["spool_io_excluded"]["mean_s"]==pytest.approx(9.5)

def test_v5_resources_replace_v4_infinite_latency(monkeypatch):
 backend=object.__new__(V5ProductionBackend);backend.protocol=ProtocolLatency();backend.protocol.parent_s=[1.];backend.protocol.feature_s=[.1];backend.protocol.adapter_s=[.2];backend.protocol.spool_io_s=[99.]
 monkeypatch.setattr(V4ProductionBackend,"resources",lambda self,saved,world_size=1:{"adapter_mean_s":math.inf,"e2e_mean_ratio":math.inf})
 resources=backend.resources({});assert math.isfinite(resources["adapter_mean_s"]) and math.isfinite(resources["e2e_mean_ratio"]) and resources["protocol_timing"]["spool_io_excluded"]["mean_s"]==99

def score_rows():
 return [{"family":family,"mean_frame_rel_l2":.1,"parent_mean_frame_rel_l2":.4,"aggregate_rel_l2":.1,"parent_rel_l2":.4,"finite":True} for family in ("uniform","layered","marmousi")]

def resources(saved):return {"peak_bytes":1,"checkpoint_bytes":1,"space_passed":True,"wall_s":1}

def test_smoke_early_stop_at_32_and_final_reserve():
 now=[0.];calls=[]
 def update(record):calls.append(record);return {"loss":1. if len(calls)==1 else .1}
 terminal=[];result=run_v5_smoke(records=[0,1,2],update=update,quick_score=lambda:(score_rows(),True),final_score=score_rows,checkpoint=lambda:{"size_bytes":1},resource_snapshot=resources,terminal=terminal.append,clock=lambda:now[0]);assert result["status"]=="passed" and result["updates"]==32 and result["early_stop"] and terminal

def test_smoke_max192_and_deadline_fail_budget():
 now=[0.];count=[];result=run_v5_smoke(records=[0,1,2],update=lambda r:count.append(r) or {"loss":1.},quick_score=lambda:(score_rows(),False),final_score=score_rows,checkpoint=lambda:{},resource_snapshot=resources,terminal=lambda p:None,clock=lambda:now[0]);assert result["updates"]==192
 now=[0.]
 def slow(r):now[0]+=11;return {"loss":1.}
 failed=run_v5_smoke(records=[0,1,2],update=slow,quick_score=lambda:(score_rows(),False),final_score=score_rows,checkpoint=lambda:{},resource_snapshot=resources,terminal=lambda p:None,clock=lambda:now[0]);assert failed["status"]=="fail_budget" and failed["total_s"]<=600
 now=[0.];calls=[]
 def final_slow():now[0]=601.;return score_rows()
 overtime=run_v5_smoke(records=[0],update=lambda r:calls.append(1) or {"loss":1. if len(calls)==1 else .1},quick_score=lambda:(score_rows(),True),final_score=final_slow,checkpoint=lambda:{},resource_snapshot=resources,terminal=lambda p:None,clock=lambda:now[0]);assert overtime["status"]=="fail_budget" and overtime["total_s"]>600

def test_dispatcher_refuses_v4_backend():
 with pytest.raises(TypeError):V5StageDispatcher(object())

def test_checkpoint_drift_refused_before_backend_or_data(tmp_path):
 paths={name:tmp_path/name for name in ("input","parent","basis","config","code")}
 for name,path in paths.items():path.write_text(name)
 import hashlib
 digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();authorization={"input_checkpoint_sha256":digest(paths["input"]),"parent_sha256":digest(paths["parent"]),"basis_sha256":digest(paths["basis"]),"config_sha256":digest(paths["config"]),"code_sha256":digest(paths["code"])};called=[];paths["input"].write_text("drift")
 with pytest.raises(BindingDriftRejection):backend_after_strict_verification(authorization,input_checkpoint=paths["input"],parent=paths["parent"],basis=paths["basis"],config=paths["config"],code=paths["code"],backend_factory=lambda:called.append(1))
 assert called==[]

def test_loaded_state_hash_rechecked():
 candidate=model();digest=model_state_digest(candidate);assert verify_loaded_state(candidate,{"input_state_sha256":digest})==digest
 with pytest.raises(BindingDriftRejection):verify_loaded_state(candidate,{"input_state_sha256":"bad"})

def test_pilot_latency_resources_are_finite_and_gate_reachable():
 records=[{"sample_id":str(i),"family":("uniform","layered","marmousi")[i%3],"aggregate_rel_l2":.1,"parent_rel_l2":.3,"nonworse":True,"time_bands":{"late":.1},"parent_time_bands":{"late":.3},"spectrum_bands":{"high":.1},"parent_spectrum_bands":{"high":.3},"correction_energy_ratio":.1,"finite":True} for i in range(24)];ridge=[{**r,"aggregate_rel_l2":.12} for r in records];resources={"adapter_mean_s":.1,"adapter_p95_s":.2,"e2e_mean_ratio":1.,"e2e_p95_ratio":1.,"peak_bytes":1,"checkpoint_bytes":1,"space_passed":True};gates,_=pilot_gates(records,ridge,resources);assert all(g["passed"] for g in gates.values())

def test_v5_long_command_contains_all_bindings():
 kwargs=dict(authorization="results/r16_dscp_v5/authorizations/long.json",preregistration="pre.json",config="cfg.yaml",scale_decision="scale.json",input_checkpoint="best.pt")
 one=v5_long_command(world_size=1,resume=True,**kwargs);four=v5_long_command(world_size=4,resume=False,**kwargs)
 for token in ("--authorization results/r16_dscp_v5/authorizations/long.json","--preregistration pre.json","--config cfg.yaml","--scale-decision scale.json","--input-checkpoint best.pt"):
  assert token in one["command"] and token in four["command"]
 assert "--resume" in one["command"] and "torchrun" in four["command"] and one["command_sha256"]

def metric(sample):return {"sample_id":str(sample),"aggregate_rel_l2":.1}

def test_v5_long_budget_reserve_world1_and_world4():
 for world,limit in ((1,7021.),(4,2521.)):
  now=[0.];steps=[];term=[]
  def backward(r,s):now[0]=limit
  runner=V5LongRunner(world_size=world,rank=0,train_records=list(range(4)),calibration_records=[0],zero_grad=lambda:None,backward_record=backward,optimizer_step=lambda:steps.append(1),set_lr=lambda x:None,evaluate=lambda r:metric(r),gather=lambda rows:rows,checkpoint=lambda *a:None,terminal=term.append,clock=lambda:now[0])
  with pytest.raises(SmokeBudgetRejection):runner.run(max_epochs=1)
  assert steps==[] and term[0]["status"]=="failed_budget"

def test_long_terminal_full_binding_and_final_wrong_checkpoint(tmp_path):
 best=tmp_path/"best.pt";last=tmp_path/"last.pt";best.write_bytes(b"best");last.write_bytes(b"last")
 import hashlib
 item=lambda p:{"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"size_bytes":p.stat().st_size}
 terminal=long_terminal(status="completed_train",decision={"selected_gpus":1},run_digest="r",effective={"code":"c"},input_checkpoint=item(best),best=item(best),last=item(last),world_size=1,wall_s=1,gpu_seconds=1,epoch=2,best_metric=.1,access_digest="a",latency={"finite":True},peak_vram=1,gates={"passed":True});assert terminal["schema"]=="r16_dscp_v5_long_terminal_v1" and terminal["gpu_hours"]==pytest.approx(1/3600)
 with pytest.raises(BindingDriftRejection):validate_final_from_long({"input_checkpoint_sha256":"bad"},terminal)
 validate_final_from_long({"input_checkpoint_sha256":terminal["best_checkpoint"]["sha256"]},terminal)

def test_v5_production_dispatcher_refuses_foreign_backend():
 with pytest.raises(TypeError):V5ProductionDispatcher(object())

def test_v5_dispatcher_reaches_finite_backend_paths():
 backend=object.__new__(V5ProductionBackend);calls=[];backend.prepare=lambda i,suffix="candidate":(i,suffix);backend.update=lambda p:calls.append("update") or {};backend.measure_gradient=lambda p:calls.append("scale") or {};backend.backward_only=lambda p,s:calls.append(("long",s)) or {}
 dispatcher=V5StageDispatcher(backend);dispatcher.update(1);dispatcher.measure_gradient(1);dispatcher.backward_only(1,.25);assert calls==["update","scale",("long",.25)]

def test_v5_long_synchronized_failure_no_step():
 steps=[];term=[]
 runner=V5LongRunner(world_size=1,rank=0,train_records=list(range(4)),calibration_records=[0],zero_grad=lambda:None,backward_record=lambda r,s:(_ for _ in ()).throw(NonFiniteStepRejection("nan")),optimizer_step=lambda:steps.append(1),set_lr=lambda x:None,evaluate=lambda r:metric(r),gather=lambda rows:rows,checkpoint=lambda *a:None,terminal=term.append,clock=lambda:0.)
 with pytest.raises(GateFailure):runner.run(max_epochs=1)
 assert steps==[] and term[0]["status"]=="failed"

def test_v5_config_and_production_source_no_foreign_backend():
 import inspect,yaml,scripts.train_r16_dscp_v5 as cli
 source=inspect.getsource(cli);assert "build_v5_backend" in source and "V4ProductionBackend" not in source and "complete_unpromoted" not in source
 config=yaml.safe_load((cli.ROOT/"configs/r16_dscp_v5.yaml").read_text());assert config["schedule"]["smoke_max_updates"]==192 and config["schedule"]["smoke_training_deadline_s"]==420 and config["schedule"]["smoke_final_reserve_s"]==180
 for name,command in config["commands"].items():
  if name not in {"prep","authorize","scale_decide"}:assert command.startswith("env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=")
