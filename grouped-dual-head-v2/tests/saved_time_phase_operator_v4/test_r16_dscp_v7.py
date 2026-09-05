from __future__ import annotations
import copy,random
import numpy as np,pytest,torch
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import R16DSCP
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v4 import LongResumeState
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v6 import V6ProductionBackend
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v7 import *
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import BindingRefusal,checkpoint_payload,make_optimizer,save_best_last
from scripts.train_r16_dscp_v7 import V7StageDispatcher

def identity(**updates):
 values=dict(mode="long",run_digest="r",authorization_sha256="a",engine_sha256="e",script_sha256="s",config_sha256="c",panels_sha256="p",basis_sha256="b",parent_sha256="par",input_checkpoint_sha256="in");values.update(updates);return complete_checkpoint_identity(**values)
def model():
 q=torch.linalg.qr(torch.randn(401,16,generator=torch.Generator().manual_seed(372),dtype=torch.float64))[0].float();return R16DSCP(q.repeat(3,1,1),torch.ones(3,16))
def step(m,o):
 x=torch.randn(1,29,5,7);y=torch.randn(1,16,5,7);o.zero_grad();loss=(m.coefficient_head(x)-y).square().mean();loss.backward();o.step()

def test_complete_identity_required_and_digest_tamper_refused():
 complete=identity();validate_checkpoint_identity(complete)
 for field in IDENTITY_FIELDS:
  broken=dict(complete);broken.pop(field)
  with pytest.raises(BindingRefusal):validate_checkpoint_identity(broken)
 broken=dict(complete);broken["authorization_sha256"]="bad"
 with pytest.raises(BindingRefusal):validate_checkpoint_identity(broken)

def test_actual_r16_adamw_interruption_resume_next_step_equivalence(tmp_path):
 random.seed(372);np.random.seed(372);torch.manual_seed(372);reference=model();initial=copy.deepcopy(reference.state_dict());ro=make_optimizer(list(reference.parameters()))
 for _ in range(4):step(reference,ro)
 random.seed(372);np.random.seed(372);torch.manual_seed(372);interrupted=model();interrupted.load_state_dict(initial);io=make_optimizer(list(interrupted.parameters()))
 for _ in range(2):step(interrupted,io)
 ident=identity();state=LongResumeState(epoch=1,global_step=2,group_index=3,best=.2,bad_epochs=1,best_epoch=0);payload=checkpoint_payload(interrupted,io,run_identity=ident,sampler_order=[0,1,2,3],progress=state.__dict__);saved=save_best_last(payload,tmp_path,is_best=True)
 resumed=model();resume_optimizer=make_optimizer(list(resumed.parameters()));loaded,resume_state=resume_v7_checkpoint(saved["last"],resumed,resume_optimizer,expected_identity=ident);assert resume_state==state
 for _ in range(2):step(resumed,resume_optimizer)
 assert all(torch.equal(a,b) for a,b in zip(reference.parameters(),resumed.parameters())) and loaded["run_identity"]==ident

@pytest.mark.parametrize("field",["authorization_sha256","engine_sha256","script_sha256","config_sha256_effective","panels_sha256_effective","basis_sha256_effective","parent_sha256_effective","input_checkpoint_sha256"])
def test_resume_rejects_each_effective_binding_tamper(tmp_path,field):
 m=model();o=make_optimizer(list(m.parameters()));ident=identity();payload=checkpoint_payload(m,o,run_identity=ident,sampler_order=[0],progress=LongResumeState().__dict__);saved=save_best_last(payload,tmp_path,is_best=True);wrong=dict(ident);wrong[field]="wrong";wrong["identity_digest"]=__import__("saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3",fromlist=["canonical_sha"]).canonical_sha({k:v for k,v in wrong.items() if k!="identity_digest"})
 resumed=model();optimizer=make_optimizer(list(resumed.parameters()))
 with pytest.raises(BindingRefusal):resume_v7_checkpoint(saved["last"],resumed,optimizer,expected_identity=wrong)

def test_checkpoint_terminal_identity_and_dispatcher_refuses_v6():
 bound=checkpoint_terminal_binding({"best":"b","last":"l","size_bytes":1},identity());assert bound["checkpoint_identity_digest"]==identity()["identity_digest"]
 with pytest.raises(TypeError):V7StageDispatcher(object())
 with pytest.raises(TypeError):V7StageDispatcher(object.__new__(V6ProductionBackend))

def test_v7_config_and_production_factory_assign_complete_identity():
 import inspect,yaml,scripts.train_r16_dscp_v7 as cli
 source=inspect.getsource(cli);assert "V7ProductionBackend" in source and "complete_identity=identity" in source and "V6ProductionBackend(" not in source and "complete_unpromoted" not in source
 config=yaml.safe_load((cli.ROOT/"configs/r16_dscp_v7.yaml").read_text());assert config["only_change"]=="complete_checkpoint_identity_wiring" and len(config["identity_required"])>=12
 for name,command in config["commands"].items():
  if name not in {"prep","authorize","scale_decide"}:assert command.startswith("env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=")
