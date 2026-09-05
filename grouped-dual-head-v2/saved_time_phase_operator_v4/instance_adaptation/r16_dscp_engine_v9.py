"""V9 sealed-evaluation authorization chain and immutable lock guard."""
from __future__ import annotations
from pathlib import Path
import json
from typing import Any,Mapping

from .r16_dscp_engine_v3 import canonical_sha
from .r16_dscp_engine_v4 import CandidateSpool
from .r16_dscp_engine_v8 import V8ProductionBackend
from .r16_dscp_engine_v7 import validate_checkpoint_identity
from .r16_dscp_training_v2 import BindingRefusal,sha256_file

class SealedAuthorizationRefusal(BindingRefusal):pass

def candidate_lock(*,checkpoint_path:str|Path,threshold_digest:str,effective:Mapping[str,str],data_hashes:Mapping[str,str])->dict[str,Any]:
    path=Path(checkpoint_path).resolve()
    if not path.is_file():raise SealedAuthorizationRefusal("candidate checkpoint absent")
    payload={"schema":"r16_dscp_v9_candidate_lock_v1","candidate":"r16_dscp_v9","checkpoint_path":str(path),"checkpoint_sha256":sha256_file(path),"threshold_digest":str(threshold_digest),"effective":dict(effective),"data_hashes":dict(data_hashes)};payload["lock_digest"]=canonical_sha(payload);return payload

def final_terminal_for_lock(*,lock:Mapping[str,Any],effective:Mapping[str,str])->dict[str,Any]:
    return {"schema":"r16_dscp_v9_final_terminal_v1","candidate":"r16_dscp_v9","mode":"final-train-confirm","status":"passed","best_checkpoint":{"path":lock["checkpoint_path"],"sha256":lock["checkpoint_sha256"]},"candidate_lock_digest":lock["lock_digest"],"threshold_digest":lock["threshold_digest"],"effective":dict(effective)}

def _validate_lock(lock:Mapping[str,Any],lock_path:str|Path,expected_effective:Mapping[str,str])->None:
    required=("checkpoint_path","checkpoint_sha256","threshold_digest","effective","data_hashes","lock_digest")
    if any(key not in lock for key in required):raise SealedAuthorizationRefusal("candidate lock incomplete")
    if canonical_sha({k:v for k,v in lock.items() if k!="lock_digest"})!=lock["lock_digest"]:raise SealedAuthorizationRefusal("candidate lock digest mismatch")
    if dict(lock["effective"])!=dict(expected_effective):raise SealedAuthorizationRefusal("candidate lock effective drift")
    checkpoint=Path(lock["checkpoint_path"]).resolve()
    if not checkpoint.is_file() or sha256_file(checkpoint)!=lock["checkpoint_sha256"]:raise SealedAuthorizationRefusal("candidate checkpoint current hash drift")
    observed_path=Path(lock_path).resolve()
    if not observed_path.is_file() or json.loads(observed_path.read_text())!=dict(lock):raise SealedAuthorizationRefusal("candidate lock file content mismatch")

def validate_validation_chain(*,authorization:Mapping[str,Any],final_terminal:Mapping[str,Any],final_terminal_path:str|Path,lock:Mapping[str,Any],lock_path:str|Path,expected_effective:Mapping[str,str])->None:
    _validate_lock(lock,lock_path,expected_effective)
    if final_terminal.get("status")!="passed" or final_terminal.get("mode")!="final-train-confirm":raise SealedAuthorizationRefusal("final confirmation not passed")
    best=final_terminal.get("best_checkpoint",{})
    if best.get("path")!=lock["checkpoint_path"] or best.get("sha256")!=lock["checkpoint_sha256"]:raise SealedAuthorizationRefusal("final terminal best differs from lock")
    if final_terminal.get("candidate_lock_digest")!=lock["lock_digest"] or final_terminal.get("threshold_digest")!=lock["threshold_digest"] or dict(final_terminal.get("effective",{}))!=dict(expected_effective):raise SealedAuthorizationRefusal("final terminal lock/effective mismatch")
    if sha256_file(best["path"])!=best["sha256"]:raise SealedAuthorizationRefusal("final best current file drift")
    bindings={"final_terminal_path":str(Path(final_terminal_path).resolve()),"final_terminal_sha256":sha256_file(final_terminal_path),"candidate_lock_path":str(Path(lock_path).resolve()),"candidate_lock_sha256":sha256_file(lock_path),"candidate_lock_digest":lock["lock_digest"],"candidate_checkpoint_path":lock["checkpoint_path"],"candidate_checkpoint_sha256":lock["checkpoint_sha256"],"threshold_digest":lock["threshold_digest"],"effective":dict(expected_effective),"data_hashes":dict(lock["data_hashes"])}
    for key,value in bindings.items():
        if authorization.get(key)!=value:raise SealedAuthorizationRefusal(f"validation authorization binding mismatch: {key}")

def validation_terminal_for_test(*,lock:Mapping[str,Any],lock_path:str|Path,effective:Mapping[str,str],status:str="passed")->dict[str,Any]:
    return {"schema":"r16_dscp_v9_validation_terminal_v1","candidate":"r16_dscp_v9","mode":"validation-once","status":status,"candidate_checkpoint":{"path":lock["checkpoint_path"],"sha256":lock["checkpoint_sha256"]},"candidate_lock_path":str(Path(lock_path).resolve()),"candidate_lock_sha256":sha256_file(lock_path),"candidate_lock_digest":lock["lock_digest"],"threshold_digest":lock["threshold_digest"],"effective":dict(effective)}

def validate_test_chain(*,authorization:Mapping[str,Any],validation_terminal:Mapping[str,Any],validation_terminal_path:str|Path,lock:Mapping[str,Any],lock_path:str|Path,expected_effective:Mapping[str,str])->None:
    _validate_lock(lock,lock_path,expected_effective)
    if validation_terminal.get("status")!="passed":raise SealedAuthorizationRefusal("validation did not pass")
    checkpoint=validation_terminal.get("candidate_checkpoint",{})
    if checkpoint!={"path":lock["checkpoint_path"],"sha256":lock["checkpoint_sha256"]}:raise SealedAuthorizationRefusal("validation checkpoint differs from lock")
    if validation_terminal.get("candidate_lock_path")!=str(Path(lock_path).resolve()) or validation_terminal.get("candidate_lock_sha256")!=sha256_file(lock_path) or validation_terminal.get("candidate_lock_digest")!=lock["lock_digest"] or validation_terminal.get("threshold_digest")!=lock["threshold_digest"] or dict(validation_terminal.get("effective",{}))!=dict(expected_effective):raise SealedAuthorizationRefusal("validation terminal lock/effective mismatch")
    bindings={"validation_terminal_path":str(Path(validation_terminal_path).resolve()),"validation_terminal_sha256":sha256_file(validation_terminal_path),"candidate_lock_path":str(Path(lock_path).resolve()),"candidate_lock_sha256":sha256_file(lock_path),"candidate_lock_digest":lock["lock_digest"],"candidate_checkpoint_path":lock["checkpoint_path"],"candidate_checkpoint_sha256":lock["checkpoint_sha256"],"threshold_digest":lock["threshold_digest"],"effective":dict(expected_effective)}
    for key,value in bindings.items():
        if authorization.get(key)!=value:raise SealedAuthorizationRefusal(f"test authorization mismatch: {key}")

class SealedAuthorizationGuard:
    def __init__(self,authorization:Mapping[str,Any]):self.authorization_digest=canonical_sha(dict(authorization));self.claimed=False
    def validate(self,authorization:Mapping[str,Any])->None:
        if canonical_sha(dict(authorization))!=self.authorization_digest:raise SealedAuthorizationRefusal("authorization changed after guard creation")
    def claim(self,authorization:Mapping[str,Any])->None:self.validate(authorization);self.claimed=True
    def before_read(self,authorization:Mapping[str,Any])->None:
        self.validate(authorization)
        if not self.claimed:raise SealedAuthorizationRefusal("once token not claimed")

class V9ProductionBackend(V8ProductionBackend):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.spool=CandidateSpool(self.run_dir,run_digest=self.lineage.run_digest,rank=int(__import__("os").environ.get("RANK","0")),owned_root="/dev/shm/r16_dscp_v9")

def complete_identity_v9(**values:Any)->dict[str,Any]:
    payload={"schema":"r16_dscp_v9_checkpoint_identity_v1","candidate":"r16_dscp_v9",**values};payload["identity_digest"]=canonical_sha(payload);validate_checkpoint_identity(payload);return payload

__all__=["SealedAuthorizationGuard","SealedAuthorizationRefusal","V9ProductionBackend","candidate_lock","complete_identity_v9","final_terminal_for_lock","validate_test_chain","validate_validation_chain","validation_terminal_for_test"]
