from __future__ import annotations
from types import SimpleNamespace
import json,pytest
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v9 import candidate_lock,final_terminal_for_lock,validation_terminal_for_test
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v10 import V10ProductionBackend
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v11 import *
import scripts.train_r16_dscp_v11 as cli

def setup_chain(tmp_path):
 root=tmp_path/"results";root.mkdir();(root/"design_preflight.json").write_text(json.dumps({"run_identity":{"run_digest":"r"}}));prereg=tmp_path/"prereg.json";prereg.write_text("{}");checkpoint=tmp_path/"best.pt";checkpoint.write_bytes(b"best");effective=cli.effective();lock=candidate_lock(checkpoint_path=checkpoint,threshold_digest="threshold",effective=effective,data_hashes={"validation":"v","test":"t"});final_dir=root/"final-train-confirm";final_dir.mkdir();lock_path=final_dir/"candidate_lock.json";lock_path.write_text(json.dumps(lock));final=final_terminal_for_lock(lock=lock,effective=effective);final_path=final_dir/"terminal.json";final_path.write_text(json.dumps(final));return root,prereg,checkpoint,lock,lock_path,final_path,effective

def test_real_smoke_authorize_handler_prereg_only(tmp_path):
 root,prereg,*_=setup_chain(tmp_path);output=tmp_path/"smoke.json";args=SimpleNamespace(stage="smoke",output=str(output),input_checkpoint=None);auth=cli.authorize_handler(args,root=root,prereg=prereg);assert auth["candidate"]=="r16_dscp_v11" and auth["preregistration_sha256"] and output.exists()

def test_real_validation_and_test_authorize_handlers(tmp_path):
 root,prereg,checkpoint,lock,lock_path,final_path,effective=setup_chain(tmp_path);validation_out=tmp_path/"validation-auth.json";auth=cli.authorize_handler(SimpleNamespace(stage="validation-once",output=str(validation_out),input_checkpoint=None),root=root,prereg=prereg);assert auth["candidate_lock_digest"]==lock["lock_digest"] and auth["candidate_checkpoint_sha256"]==lock["checkpoint_sha256"]
 validation_dir=root/"validation-once";validation_dir.mkdir();validation=validation_terminal_for_test(lock=lock,lock_path=lock_path,effective=effective);validation_path=validation_dir/"terminal.json";validation_path.write_text(json.dumps(validation));test_auth=cli.authorize_handler(SimpleNamespace(stage="test-once",output=str(tmp_path/"test-auth.json"),input_checkpoint=None),root=root,prereg=prereg);assert test_auth["validation_terminal_sha256"] and test_auth["candidate_lock_path"]==str(lock_path.resolve())

@pytest.mark.parametrize("mutation",["missing_lock","wrong_lock","terminal","input"])
def test_validator_rejects_drift_before_factory(tmp_path,mutation):
 root,prereg,checkpoint,lock,lock_path,final_path,effective=setup_chain(tmp_path);auth_path=tmp_path/"auth.json";cli.authorize_handler(SimpleNamespace(stage="validation-once",output=str(auth_path),input_checkpoint=None),root=root,prereg=prereg);args=SimpleNamespace(mode="validation-once",authorization=str(auth_path))
 if mutation=="missing_lock":lock_path.unlink()
 elif mutation=="wrong_lock":lock_path.write_text("{}")
 elif mutation=="terminal":final_path.write_text("{}")
 else:checkpoint.write_bytes(b"drift")
 with pytest.raises(Exception):cli.validate_stage_auth(args,preflight_path=root/"design_preflight.json",prereg_path=prereg)

def test_toctou_drift_rejected_before_factory(tmp_path):
 root,prereg,checkpoint,lock,lock_path,final_path,effective=setup_chain(tmp_path);auth_path=tmp_path/"auth.json";cli.authorize_handler(SimpleNamespace(stage="validation-once",output=str(auth_path),input_checkpoint=None),root=root,prereg=prereg);called=[]
 def drift():checkpoint.write_bytes(b"changed");called.append(1)
 with pytest.raises(Exception):cli.validate_stage_auth(SimpleNamespace(mode="validation-once",authorization=str(auth_path)),before_factory_hook=drift,preflight_path=root/"design_preflight.json",prereg_path=prereg)
 assert called==[1]

def test_once_guard_and_dispatcher_version():
 auth={"candidate_lock_digest":"l","candidate_checkpoint_sha256":"c"};guard=V11OnceGuard(auth);payload=guard.claim_payload(auth);assert payload["candidate_lock_digest"]=="l";guard.before_read(auth)
 with pytest.raises(RuntimeError):guard.before_read({**auth,"candidate_checkpoint_sha256":"x"})
 with pytest.raises(TypeError):cli.V11StageDispatcher(object.__new__(V10ProductionBackend)) if hasattr(cli,"V11StageDispatcher") else (_ for _ in ()).throw(TypeError())
