from __future__ import annotations
import json
from pathlib import Path
import pytest
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v8 import V8ProductionBackend
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v9 import *
from scripts.train_r16_dscp_v9 import V9ProductionWiring,V9StageDispatcher,smoke_authorization,test_authorization as build_test_authorization,validation_authorization

EFFECTIVE={"code":"c","config":"cfg","basis":"b","panels":"p","parent":"par","data":"d"};DATA={"validation":"v","test":"t"}
def fixtures(tmp_path):
 checkpoint=tmp_path/"best.pt";checkpoint.write_bytes(b"checkpoint");lock=candidate_lock(checkpoint_path=checkpoint,threshold_digest="threshold",effective=EFFECTIVE,data_hashes=DATA);lock_path=tmp_path/"candidate_lock.json";lock_path.write_text(json.dumps(lock));final=final_terminal_for_lock(lock=lock,effective=EFFECTIVE);final_path=tmp_path/"final.json";final_path.write_text(json.dumps(final));return checkpoint,lock,lock_path,final,final_path

def test_valid_validation_and_test_chain(tmp_path):
 checkpoint,lock,lock_path,final,final_path=fixtures(tmp_path);authorization=validation_authorization(final_terminal_path=final_path,lock_path=lock_path,effective=EFFECTIVE);validate_validation_chain(authorization=authorization,final_terminal=final,final_terminal_path=final_path,lock=lock,lock_path=lock_path,expected_effective=EFFECTIVE)
 validation=validation_terminal_for_test(lock=lock,lock_path=lock_path,effective=EFFECTIVE);validation_path=tmp_path/"validation.json";validation_path.write_text(json.dumps(validation));test_auth=build_test_authorization(validation_terminal_path=validation_path,lock_path=lock_path,effective=EFFECTIVE);validate_test_chain(authorization=test_auth,validation_terminal=validation,validation_terminal_path=validation_path,lock=lock,lock_path=lock_path,expected_effective=EFFECTIVE)

def test_missing_and_wrong_lock_rejected(tmp_path):
 _,lock,lock_path,final,final_path=fixtures(tmp_path);auth=validation_authorization(final_terminal_path=final_path,lock_path=lock_path,effective=EFFECTIVE);lock_path.unlink()
 with pytest.raises(SealedAuthorizationRefusal):validate_validation_chain(authorization=auth,final_terminal=final,final_terminal_path=final_path,lock=lock,lock_path=lock_path,expected_effective=EFFECTIVE)
 _,lock,lock_path,final,final_path=fixtures(tmp_path);wrong=dict(lock);wrong["threshold_digest"]="wrong";lock_path.write_text(json.dumps(wrong))
 with pytest.raises(SealedAuthorizationRefusal):validate_validation_chain(authorization={},final_terminal=final,final_terminal_path=final_path,lock=wrong,lock_path=lock_path,expected_effective=EFFECTIVE)

def test_terminal_best_path_same_sha_wrong_rejected(tmp_path):
 _,lock,lock_path,final,final_path=fixtures(tmp_path);final["best_checkpoint"]["sha256"]="wrong";final_path.write_text(json.dumps(final))
 with pytest.raises(SealedAuthorizationRefusal):validation_authorization(final_terminal_path=final_path,lock_path=lock_path,effective=EFFECTIVE)

def test_current_checkpoint_file_drift_rejected(tmp_path):
 checkpoint,lock,lock_path,final,final_path=fixtures(tmp_path);auth=validation_authorization(final_terminal_path=final_path,lock_path=lock_path,effective=EFFECTIVE);checkpoint.write_bytes(b"drift")
 with pytest.raises(SealedAuthorizationRefusal):validate_validation_chain(authorization=auth,final_terminal=final,final_terminal_path=final_path,lock=lock,lock_path=lock_path,expected_effective=EFFECTIVE)

def test_test_different_lock_and_failed_validation_rejected(tmp_path):
 _,lock,lock_path,_,_=fixtures(tmp_path);validation=validation_terminal_for_test(lock=lock,lock_path=lock_path,effective=EFFECTIVE);validation_path=tmp_path/"validation.json";validation_path.write_text(json.dumps(validation));auth=build_test_authorization(validation_terminal_path=validation_path,lock_path=lock_path,effective=EFFECTIVE)
 different=tmp_path/"different.json";different.write_text(lock_path.read_text())
 with pytest.raises(SealedAuthorizationRefusal):validate_test_chain(authorization=auth,validation_terminal=validation,validation_terminal_path=validation_path,lock=lock,lock_path=different,expected_effective=EFFECTIVE)
 validation["status"]="fail_gate";validation_path.write_text(json.dumps(validation))
 with pytest.raises(SealedAuthorizationRefusal):validate_test_chain(authorization=auth,validation_terminal=validation,validation_terminal_path=validation_path,lock=lock,lock_path=lock_path,expected_effective=EFFECTIVE)

def test_effective_drift_and_guard_after_claim_rejected(tmp_path):
 _,lock,lock_path,final,final_path=fixtures(tmp_path);auth=validation_authorization(final_terminal_path=final_path,lock_path=lock_path,effective=EFFECTIVE)
 with pytest.raises(SealedAuthorizationRefusal):validate_validation_chain(authorization=auth,final_terminal=final,final_terminal_path=final_path,lock=lock,lock_path=lock_path,expected_effective={**EFFECTIVE,"code":"drift"})
 guard=SealedAuthorizationGuard(auth);guard.claim(auth);changed={**auth,"threshold_digest":"changed"}
 with pytest.raises(SealedAuthorizationRefusal):guard.before_read(changed)

def test_smoke_authorization_still_prereg_only(tmp_path):
 prereg=tmp_path/"prereg.json";prereg.write_text("{}");auth=smoke_authorization(prereg);assert set(auth)=={"mode","preregistration_path","preregistration_sha256"}

def test_dispatcher_refuses_v8_backend():
 with pytest.raises(TypeError):V9StageDispatcher(object.__new__(V8ProductionBackend))

def test_production_wiring_rejects_invalid_chain_before_factory(tmp_path):
 _,lock,lock_path,final,final_path=fixtures(tmp_path);called=[];wiring=V9ProductionWiring(lambda:called.append(1))
 with pytest.raises(SealedAuthorizationRefusal):wiring.run("validation-once",{},effective=EFFECTIVE,final_terminal_path=final_path,lock_path=lock_path)
 assert called==[]
