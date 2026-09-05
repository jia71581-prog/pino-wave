"""V10 versioned backend, identity, and terminal schemas."""
from __future__ import annotations
from typing import Any,Mapping
from .r16_dscp_engine_v3 import canonical_sha
from .r16_dscp_engine_v4 import CandidateSpool
from .r16_dscp_engine_v7 import validate_checkpoint_identity
from .r16_dscp_engine_v9 import V9ProductionBackend

class V10ProductionBackend(V9ProductionBackend):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.spool=CandidateSpool(self.run_dir,run_digest=self.lineage.run_digest,rank=int(__import__("os").environ.get("RANK","0")),owned_root="/dev/shm/r16_dscp_v10")

def complete_identity_v10(**values:Any)->dict[str,Any]:
    payload={"schema":"r16_dscp_v10_checkpoint_identity_v1","candidate":"r16_dscp_v10",**values};payload["identity_digest"]=canonical_sha(payload);validate_checkpoint_identity(payload);return payload

def terminal_v10(mode:str,status:str,payload:Mapping[str,Any])->dict[str,Any]:
    return {**dict(payload),"schema":f"r16_dscp_v10_{mode.replace('-','_')}_terminal_v1","candidate":"r16_dscp_v10","mode":mode,"status":status,"decision":status}

__all__=["V10ProductionBackend","complete_identity_v10","terminal_v10"]
