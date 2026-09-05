"""V7 complete checkpoint identity wiring and strict resume."""
from __future__ import annotations
from typing import Any,Mapping
import torch

from .r16_dscp_engine_v3 import canonical_sha
from .r16_dscp_engine_v4 import CandidateSpool,LongResumeState
from .r16_dscp_engine_v6 import V6ProductionBackend
from .r16_dscp_training_v2 import BindingRefusal,load_checkpoint

IDENTITY_FIELDS=("schema","candidate","mode","run_digest","authorization_sha256","engine_sha256","script_sha256","config_sha256_effective","panels_sha256_effective","basis_sha256_effective","parent_sha256_effective","input_checkpoint_sha256")

def complete_checkpoint_identity(*,mode:str,run_digest:str,authorization_sha256:str,engine_sha256:str,script_sha256:str,config_sha256:str,panels_sha256:str,basis_sha256:str,parent_sha256:str,input_checkpoint_sha256:str)->dict[str,str]:
    payload={"schema":"r16_dscp_v7_checkpoint_identity_v1","candidate":"r16_dscp_v7","mode":mode,"run_digest":run_digest,"authorization_sha256":authorization_sha256,"engine_sha256":engine_sha256,"script_sha256":script_sha256,"config_sha256_effective":config_sha256,"panels_sha256_effective":panels_sha256,"basis_sha256_effective":basis_sha256,"parent_sha256_effective":parent_sha256,"input_checkpoint_sha256":input_checkpoint_sha256};payload["identity_digest"]=canonical_sha(payload);return payload

def validate_checkpoint_identity(identity:Mapping[str,Any],expected:Mapping[str,Any]|None=None)->None:
    missing=[field for field in IDENTITY_FIELDS if field not in identity]
    if missing:raise BindingRefusal(f"checkpoint identity missing fields: {missing}")
    content={k:v for k,v in identity.items() if k!="identity_digest"}
    if identity.get("identity_digest")!=canonical_sha(content):raise BindingRefusal("checkpoint identity digest mismatch")
    if expected is not None and dict(identity)!=dict(expected):raise BindingRefusal("checkpoint identity differs from effective run identity")

class V7ProductionBackend(V6ProductionBackend):
    def __init__(self,*args,complete_identity:Mapping[str,Any],**kwargs):
        validate_checkpoint_identity(complete_identity);super().__init__(*args,**kwargs);self.spool=CandidateSpool(self.run_dir,run_digest=self.lineage.run_digest,rank=int(__import__("os").environ.get("RANK","0")),owned_root="/dev/shm/r16_dscp_v7");self.checkpoint_identity=dict(complete_identity)
    def checkpoint(self,progress:Mapping[str,Any],is_best:bool=True)->Mapping[str,Any]:
        validate_checkpoint_identity(self.checkpoint_identity);return super().checkpoint(progress,is_best)

def resume_v7_checkpoint(path,model:torch.nn.Module,optimizer:torch.optim.Optimizer,*,expected_identity:Mapping[str,Any])->tuple[Mapping[str,Any],LongResumeState]:
    validate_checkpoint_identity(expected_identity);payload=load_checkpoint(path,model,optimizer,expected_run_identity=expected_identity);validate_checkpoint_identity(payload["run_identity"],expected_identity);progress=payload.get("progress",{});missing=[key for key in LongResumeState.__dataclass_fields__ if key not in progress]
    if missing:raise BindingRefusal(f"resume progress missing: {missing}")
    state=LongResumeState(**{key:progress[key] for key in LongResumeState.__dataclass_fields__});return payload,state

def checkpoint_terminal_binding(saved:Mapping[str,Any],identity:Mapping[str,Any])->Mapping[str,Any]:
    validate_checkpoint_identity(identity);required=("best","last","size_bytes")
    if any(key not in saved for key in required):raise BindingRefusal("checkpoint terminal binding incomplete")
    return {"checkpoint":dict(saved),"checkpoint_identity_digest":identity["identity_digest"],"checkpoint_identity":dict(identity)}

__all__=["IDENTITY_FIELDS","V7ProductionBackend","checkpoint_terminal_binding","complete_checkpoint_identity","resume_v7_checkpoint","validate_checkpoint_identity"]
