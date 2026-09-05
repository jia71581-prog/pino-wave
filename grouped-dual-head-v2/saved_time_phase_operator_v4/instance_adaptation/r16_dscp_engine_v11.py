"""V11 versioned backend and immutable sealed authorization guard."""
from __future__ import annotations
from typing import Any,Mapping
from .r16_dscp_engine_v3 import canonical_sha
from .r16_dscp_engine_v4 import CandidateSpool
from .r16_dscp_engine_v10 import V10ProductionBackend
from .r16_dscp_engine_v7 import validate_checkpoint_identity

class V11ProductionBackend(V10ProductionBackend):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.spool=CandidateSpool(self.run_dir,run_digest=self.lineage.run_digest,rank=int(__import__("os").environ.get("RANK","0")),owned_root="/dev/shm/r16_dscp_v11")

def complete_identity_v11(**values:Any)->dict[str,Any]:
    payload={"schema":"r16_dscp_v11_checkpoint_identity_v1","candidate":"r16_dscp_v11",**values};payload["identity_digest"]=canonical_sha(payload);validate_checkpoint_identity(payload);return payload

def terminal_v11(mode:str,status:str,payload:Mapping[str,Any])->dict[str,Any]:return {**dict(payload),"schema":f"r16_dscp_v11_{mode.replace('-','_')}_terminal_v1","candidate":"r16_dscp_v11","mode":mode,"status":status,"decision":status}

class V11OnceGuard:
    def __init__(self,authorization:Mapping[str,Any]):
        self.digest=canonical_sha(dict(authorization));self.lock_digest=authorization.get("candidate_lock_digest");self.checkpoint_digest=authorization.get("candidate_checkpoint_sha256");self.claimed=False
    def validate(self,authorization:Mapping[str,Any]):
        if canonical_sha(dict(authorization))!=self.digest or authorization.get("candidate_lock_digest")!=self.lock_digest or authorization.get("candidate_checkpoint_sha256")!=self.checkpoint_digest:raise RuntimeError("sealed authorization/lock/checkpoint changed")
    def claim_payload(self,authorization:Mapping[str,Any])->dict[str,Any]:
        self.validate(authorization);self.claimed=True;return {"schema":"r16_dscp_v11_once_token_v1","authorization_digest":self.digest,"candidate_lock_digest":self.lock_digest,"candidate_checkpoint_sha256":self.checkpoint_digest}
    def before_read(self,authorization:Mapping[str,Any]):
        self.validate(authorization)
        if not self.claimed:raise RuntimeError("once token not claimed")

__all__=["V11OnceGuard","V11ProductionBackend","complete_identity_v11","terminal_v11"]
