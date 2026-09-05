from __future__ import annotations
from types import SimpleNamespace
import pytest
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v7 import V7ProductionBackend
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v8 import *
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v5 import NonFiniteStepRejection,OOMStepRejection
from scripts.train_r16_dscp_v8 import V8StageDispatcher,execute_with_boundary,validate_long_cli

LINEAGE={"run_digest":"r"};EFFECTIVE={"code":"c"};PARENT={"sha256":"p"}
def rows():return [{"family":f,"mean_frame_rel_l2":.1,"parent_mean_frame_rel_l2":.4,"aggregate_rel_l2":.1,"parent_rel_l2":.4,"finite":True,"ledger_digest":f} for f in ("uniform","layered","marmousi")]
class Cache:
 def payload(self):return {"hits":1}
 def clear(self):self.cleared=True
class Backend:
 def __init__(self):self.cache=Cache();self.lineage=SimpleNamespace();self.protocol=SimpleNamespace(payload=lambda:{"finite":True});self.peak_bytes=1;self.n=0
 def preload(self,i):pass
 def update_cached(self,i):self.n+=1;return {"loss":1. if self.n==1 else .1}
 def score_cached(self,i):return rows()[i]

def test_normal_smoke_terminal_is_v8():
 backend=Backend();terminal=[];result=run_v8_smoke(backend=backend,records=[0,1,2],quick_gate=lambda r:True,checkpoint=lambda:{"best":"b","last":"l"},resource_snapshot=lambda s:{"peak_bytes":1,"checkpoint_bytes":1,"space_passed":True},terminal=terminal.append,lineage=LINEAGE,effective=EFFECTIVE,parent=PARENT,clock=lambda:0.)
 assert result["schema"]=="r16_dscp_v8_smoke_terminal_v1" and result["candidate"]=="r16_dscp_v8" and result["status"]=="passed"

def test_fail_budget_terminal_allows_no_checkpoint():
 backend=Backend();now=[0.];original=backend.update_cached
 def update(i):now[0]+=11;return original(i)
 backend.update_cached=update;terminal=[];result=run_v8_smoke(backend=backend,records=[0,1,2],quick_gate=lambda r:False,checkpoint=lambda:{"best":"b","last":"l"},resource_snapshot=lambda s:{},terminal=terminal.append,lineage=LINEAGE,effective=EFFECTIVE,parent=PARENT,clock=lambda:now[0]);assert result["status"]=="fail_budget" and result["checkpoint"] is None

def test_fail_budget_builder_without_checkpoint():
 payload=terminal_base(mode="smoke",status="fail_budget",lineage=LINEAGE,effective=EFFECTIVE,parent=PARENT,checkpoint=None,gates={"budget":{"passed":False}});assert payload["checkpoint"] is None and payload["schema"]=="r16_dscp_v8_smoke_terminal_v1"

@pytest.mark.parametrize("error",[RuntimeError("factory"),NonFiniteStepRejection("nan"),OOMStepRejection("oom")])
def test_exception_boundary_writes_atomic_failure(tmp_path,error):
 auth={"authorization_digest":"a"};lineage={"run_digest":"r"}
 if str(error)=="factory":factory=lambda:(_ for _ in ()).throw(error);body=lambda b:None
 else:
  backend=object.__new__(V8ProductionBackend);backend.optimizer=SimpleNamespace(zero_grad=lambda **k:None);backend.cache=Cache();backend.public_loader=SimpleNamespace(close=lambda:None);backend.preserve_failure_spool=lambda:None;factory=lambda:backend;body=lambda b:(_ for _ in ()).throw(error)
 with pytest.raises(type(error)):execute_with_boundary(mode="pilot",run_dir=tmp_path,authorization=auth,lineage=lineage,effective=EFFECTIVE,parent=PARENT,factory=factory,body=body)
 terminal=__import__("json").loads((tmp_path/"terminal.json").read_text());assert terminal["schema"]=="r16_dscp_v8_pilot_terminal_v1" and terminal["status"]=="failed" and terminal["error_type"]==type(error).__name__

def test_terminal_write_failure_reports_stderr_and_preserves_run(tmp_path,capsys):
 error=RuntimeError("update")
 with pytest.raises(RuntimeError):execute_with_boundary(mode="smoke",run_dir=tmp_path,authorization={},lineage=LINEAGE,effective=EFFECTIVE,parent=PARENT,factory=lambda:(_ for _ in ()).throw(error),body=lambda b:None,terminal_writer=lambda p:(_ for _ in ()).throw(OSError("disk")))
 assert "terminal write failed" in capsys.readouterr().err and tmp_path.exists()

def test_long_cli_mismatch_and_completed_resume_refusal():
 auth={"selected_world_size":4};decision={"selected_gpus":4};validate_long_cli(arg_world_size=4,actual_world_size=4,cuda_visible="0,1,2,3",authorization=auth,scale_decision=decision,resume=False,terminal_exists=False)
 for kwargs in ({"arg_world_size":1,"actual_world_size":4,"cuda_visible":"0,1,2,3","authorization":auth,"scale_decision":decision,"resume":False,"terminal_exists":False},{"arg_world_size":4,"actual_world_size":4,"cuda_visible":"0,1,2,3","authorization":auth,"scale_decision":decision,"resume":True,"terminal_exists":True}):
  with pytest.raises(ValueError):validate_long_cli(**kwargs)

def test_long_terminal_full_and_failure_optional_checkpoint():
 base=dict(lineage=LINEAGE,effective=EFFECTIVE,parent=PARENT,decision={"selected_gpus":1},input_checkpoint={"sha256":"i"},world_size=1,wall_s=1.,gpu_seconds=1.,epoch=2,best_metric=.1,access={},cache={},latency={},peak_vram=1,gates={"passed":True})
 success=long_terminal_v8(status="completed_train",best={"path":"b"},last={"path":"l"},**base);assert success["schema"]=="r16_dscp_v8_long_terminal_v1" and success["checkpoint"]
 failed=long_terminal_v8(status="failed_budget",best=None,last=None,**base);assert failed["checkpoint"] is None and failed["status"]=="failed_budget"

def test_dispatcher_refuses_v7_backend():
 with pytest.raises(TypeError):V8StageDispatcher(object.__new__(V7ProductionBackend))

def test_v8_config_production_resume_keeps_authorization_immutable():
 import inspect,yaml,scripts.train_r16_dscp_v8 as cli
 source=inspect.getsource(cli.run_authorized);resume_block=source.split("if resume:",1)[1].split("if run.exists",1)[0];assert 'auth["input_checkpoint_path"]' not in resume_block and "resume_checkpoint=resume_path" in source
 factory=inspect.getsource(cli.build_v8_backend);assert "V8ProductionBackend" in factory and "V7ProductionBackend(" not in factory
 config=yaml.safe_load((cli.ROOT/"configs/r16_dscp_v8.yaml").read_text());assert config["only_change"]=="terminal_exception_long_cli_contract"
