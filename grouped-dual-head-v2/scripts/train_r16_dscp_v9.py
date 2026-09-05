#!/usr/bin/env python3
"""V9 sealed authorization constructors; production CLI follows tests."""
from __future__ import annotations
import argparse,json,platform,subprocess
from pathlib import Path
from typing import Any,Mapping
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v8 import V8ProductionBackend
import torch,yaml
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import R16DSCP
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v9 import SealedAuthorizationGuard,V9ProductionBackend,complete_identity_v9
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v9 import validate_test_chain,validate_validation_chain
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import sha256_file
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import BASIS_FILE_SHA256,PANELS_SHA256,PARENT_PATH,PARENT_SHA256,BindingRefusal,atomic_json_exclusive,checkpoint_payload,configure_determinism,load_checkpoint,make_optimizer,save_best_last
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v4 import CandidateSpool,LongResumeState,V4GuardedOnsetLoader,actual_coefficient_loss,dual_space_gate
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v7 import resume_v7_checkpoint,validate_checkpoint_identity
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v5 import model_state_digest,verify_loaded_state
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.eikonal import grid_eikonal_travel_time
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import canonical_sha
from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime

ROOT=Path(__file__).resolve().parents[1];CANDIDATE="r16_dscp_v9";SCRIPT=Path(__file__).resolve();ENGINE=ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v9.py";TEST=ROOT/"tests/saved_time_phase_operator_v4/test_r16_dscp_v9.py";CONFIG=ROOT/"configs/r16_dscp_v9.yaml";OUT=ROOT/"results/r16_dscp_v9";STATIC=OUT/"static_evidence.json";PREFLIGHT=OUT/"design_preflight.json";PREREG=ROOT/"results/r16_dscp_v9_preregistration_20260826.json";BASIS=ROOT/"results/r16_dscp_v1/basis_rank16.pt";PANELS=ROOT/"results/r16_dscp_v1/panels.json";TEST_LOG=Path("/tmp/r16_dscp_v9_tests.log")

class V9StageDispatcher:
    def __init__(self,backend:V9ProductionBackend):
        if not isinstance(backend,V9ProductionBackend):raise TypeError("v9 dispatcher requires V9ProductionBackend")
        self.backend=backend

class V9ProductionWiring:
    """Validate sealed chain before the V9 factory can touch data/CUDA."""
    def __init__(self,backend_factory):self.backend_factory=backend_factory
    def run(self,mode,authorization,*,effective,final_terminal_path=None,validation_terminal_path=None,lock_path=None,body=lambda backend,guard:None):
        guard=SealedAuthorizationGuard(authorization)
        if mode=="validation-once":
            final_path=Path(final_terminal_path);candidate_path=Path(lock_path);validate_validation_chain(authorization=authorization,final_terminal=json.loads(final_path.read_text()),final_terminal_path=final_path,lock=json.loads(candidate_path.read_text()),lock_path=candidate_path,expected_effective=effective)
        elif mode=="test-once":
            validation_path=Path(validation_terminal_path);candidate_path=Path(lock_path);validate_test_chain(authorization=authorization,validation_terminal=json.loads(validation_path.read_text()),validation_terminal_path=validation_path,lock=json.loads(candidate_path.read_text()),lock_path=candidate_path,expected_effective=effective)
        backend=self.backend_factory()
        if not isinstance(backend,V9ProductionBackend):raise TypeError("factory returned foreign backend")
        return body(backend,guard)

def build_v9_backend(mode,authorization,lineage,run_dir,identity):
    split="validation" if mode=="validation-once" else "test_id" if mode=="test-once" else "train";device=torch.device(f"cuda:{int(__import__('os').environ.get('LOCAL_RANK','0'))}");configure_determinism(372);artifact=basis_artifact();candidate=R16DSCP(artifact["basis"],artifact["coefficient_scales"]).to(device);optimizer=make_optimizer(list(candidate.parameters()))
    if lineage.input_checkpoint_sha256!="none":load_checkpoint(authorization["input_checkpoint_path"],candidate,optimizer,expected_run_identity=authorization["input_run_identity"])
    verify_loaded_state(candidate,authorization);parent,normalizer,manifest_payload,_=parent_runtime.load_model_context(device);manifest=parent_runtime.manifest_object(manifest_payload);panels=json.loads(PANELS.read_text())["records"];roles={"smoke":("smoke",),"pilot":("pilot_fit","pilot_confirm"),"scale-probe-1":("pilot_fit",),"scale-probe-4":("pilot_fit",),"long":("long_fit","long_calibration"),"final-train-confirm":("final_train_confirm",)}
    if split=="train":rows=[r for r in panels if r["role"] in roles[mode]];samples=[r["sample_id"] for r in rows];family={r["sample_id"]:r["family"] for r in rows};nontruth={r["sample_id"]:r["nontruth_input_sha256"] for r in rows}
    else:rows=sorted([r for r in manifest_payload["records"] if r["split"]==split],key=lambda r:(r["source_index"],r["sample_id"]));samples=[r["sample_id"] for r in rows];family={r["sample_id"]:r["medium_type"] for r in rows};nontruth={r["sample_id"]:r["sample_sha256"] for r in rows}
    loader=V4GuardedOnsetLoader(GuardedOnsetDataset(parent_runtime.SOURCE_H5_PATH,manifest,split=split,sample_ids=samples))
    @torch.inference_mode()
    def parent_predictor(public):
        v=public.velocity_mps[None,None].to(device);s=public.source_parameters[None].to(device);sm=public.source_map[None,None].to(device);medium=parent.encode_medium(v,normalizer);prepared=parent.prepare_sources(medium,s,sm,normalizer,record_to_medium=torch.zeros(1,dtype=torch.long,device=device));normalized=parent.dense_normalized(prepared,public.time_s.to(device),x_m=public.x_m.to(device),z_m=public.z_m.to(device),time_block=16);return normalizer.decode_pressure(normalized.float(),s[:,4])[0].cpu()
    travel=lambda v,s:grid_eikonal_travel_time(v,source_indices=[(round(float(s[1])/10),round(float(s[0])/10))],dx_m=10,dz_m=10)[0];return V9ProductionBackend(public_loader=loader,parent_predictor=parent_predictor,travel_builder=travel,candidate=candidate,optimizer=optimizer,authorization=authorization,lineage=lineage,source_h5=parent_runtime.SOURCE_H5_PATH,run_dir=run_dir,device=device,split=split,family_by_sample=family,once_token=run_dir/f"{split}.once.json" if split!="train" else None,cache_role=mode,parent_sha256=PARENT_SHA256,basis_sha256=BASIS_FILE_SHA256,feature_code_sha256=sha256_file(ENGINE),nontruth_by_sample=nontruth,complete_identity=identity)

def validation_authorization(*,final_terminal_path:str|Path,lock_path:str|Path,effective:Mapping[str,str])->dict[str,Any]:
    final_path=Path(final_terminal_path).resolve();candidate_path=Path(lock_path).resolve();terminal=json.loads(final_path.read_text());lock=json.loads(candidate_path.read_text());payload={"final_terminal_path":str(final_path),"final_terminal_sha256":sha256_file(final_path),"candidate_lock_path":str(candidate_path),"candidate_lock_sha256":sha256_file(candidate_path),"candidate_lock_digest":lock["lock_digest"],"candidate_checkpoint_path":lock["checkpoint_path"],"candidate_checkpoint_sha256":lock["checkpoint_sha256"],"threshold_digest":lock["threshold_digest"],"effective":dict(effective),"data_hashes":dict(lock["data_hashes"])};validate_validation_chain(authorization=payload,final_terminal=terminal,final_terminal_path=final_path,lock=lock,lock_path=candidate_path,expected_effective=effective);return payload

def test_authorization(*,validation_terminal_path:str|Path,lock_path:str|Path,effective:Mapping[str,str])->dict[str,Any]:
    validation_path=Path(validation_terminal_path).resolve();candidate_path=Path(lock_path).resolve();terminal=json.loads(validation_path.read_text());lock=json.loads(candidate_path.read_text());payload={"validation_terminal_path":str(validation_path),"validation_terminal_sha256":sha256_file(validation_path),"candidate_lock_path":str(candidate_path),"candidate_lock_sha256":sha256_file(candidate_path),"candidate_lock_digest":lock["lock_digest"],"candidate_checkpoint_path":lock["checkpoint_path"],"candidate_checkpoint_sha256":lock["checkpoint_sha256"],"threshold_digest":lock["threshold_digest"],"effective":dict(effective)};validate_test_chain(authorization=payload,validation_terminal=terminal,validation_terminal_path=validation_path,lock=lock,lock_path=candidate_path,expected_effective=effective);return payload

def smoke_authorization(preregistration_path:str|Path)->dict[str,Any]:
    path=Path(preregistration_path).resolve()
    if not path.is_file():raise FileNotFoundError(path)
    return {"mode":"smoke","preregistration_path":str(path),"preregistration_sha256":sha256_file(path)}

def bind(path):p=Path(path).resolve();return {"path":str(p),"sha256":sha256_file(p),"size_bytes":p.stat().st_size}
def effective_bindings():return {"engine":bind(ENGINE),"script":bind(SCRIPT),"test":bind(TEST),"config":bind(CONFIG),"model":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp.py"),"v9_transitive_v8":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v8.py"),"v9_transitive_v7":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v7.py"),"harness":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_training_v2.py"),"basis":bind(BASIS),"panels":bind(PANELS),"parent":bind(PARENT_PATH),"manifest":bind(parent_runtime.MANIFEST_PATH),"normalization":bind(parent_runtime.NORMALIZATION_PATH),"v2_replay":bind(ROOT/"results/r16_dscp_v2/replay_basis_verify/verification.json"),"v8_blocker":bind(ROOT/"results/r16_dscp_v8/design_preflight.json")}
def basis_artifact():return torch.load(BASIS,map_location="cpu",weights_only=False)
def metadata_seal():
 m=parent_runtime.load_manifest_payload();out={}
 for split in ("validation","test_id"):
  rows=sorted([{"source_index":r["source_index"],"sample_id":r["sample_id"],"group_id":r["group_id"],"sample_sha256":r["sample_sha256"],"family":r["medium_type"]} for r in m["records"] if r["split"]==split],key=lambda r:(r["source_index"],r["sample_id"]));out[split]={"count":len(rows),"ordered_digest":canonical_sha(rows),"by_family":{f:sum(r["family"]==f for r in rows) for f in ("uniform","layered","marmousi")},"wavefield_read":False}
 return out

def static_evidence():
 if not TEST_LOG.is_file() or "[100%]" not in TEST_LOG.read_text() or "failed" in TEST_LOG.read_text().lower():raise BindingRefusal("tests absent")
 v8=json.loads((ROOT/"results/r16_dscp_v8/static_evidence.json").read_text());return {"schema":"r16_dscp_v9_static_v1","parameters":1202,"model_macs":45451125,"host_cache_gates":v8["host_cache_gates"],"metadata":metadata_seal(),"tests":{"status":"passed","log_sha256":sha256_file(TEST_LOG),"tail":TEST_LOG.read_text().splitlines()[-1]},"auth_validators":["validate_validation_chain","validate_test_chain","SealedAuthorizationGuard"],"truth_read":False}

def prep():
 if OUT.exists() or PREREG.exists():raise FileExistsError("v9 exists")
 bindings=effective_bindings();static=static_evidence();OUT.mkdir();atomic_json_exclusive(static,STATIC);artifact=basis_artifact();model=R16DSCP(artifact["basis"],artifact["coefficient_scales"]);optimizer=make_optimizer(list(model.parameters()));h=w=201;args=(torch.ones(1,1,h,w)*2000,torch.zeros(1,1,h,w),torch.ones(1,1,h,w),torch.arange(w),torch.arange(h),torch.zeros(1,1,h,w),torch.zeros(1,1,h,w),torch.zeros(1,401,h,w),torch.tensor([1]),torch.tensor([2]));losses,adapted,_=actual_coefficient_loss(model,args,torch.ones(1,398,h,w));losses["total"].backward();optimizer.step();identity=complete_identity_v9(mode="prep",run_digest=canonical_sha(bindings),authorization_sha256="prep-none",engine_sha256=bindings["engine"]["sha256"],script_sha256=bindings["script"]["sha256"],config_sha256_effective=bindings["config"]["sha256"],panels_sha256_effective=PANELS_SHA256,basis_sha256_effective=BASIS_FILE_SHA256,parent_sha256_effective=PARENT_SHA256,input_checkpoint_sha256="none");state=LongResumeState(epoch=0,global_step=1,group_index=0,best=float("inf"),bad_epochs=0,best_epoch=-1);payload=checkpoint_payload(model,optimizer,run_identity=identity,sampler_order=list(range(12)),progress=state.__dict__);saved=save_best_last(payload,OUT/"actual_checkpoint",is_best=True);restored=R16DSCP(artifact["basis"],artifact["coefficient_scales"]);ro=make_optimizer(list(restored.parameters()));_,resume=resume_v7_checkpoint(saved["last"],restored,ro,expected_identity=identity);ledger=__import__("saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3",fromlist=["AccessLedger"]).AccessLedger("train");spool=CandidateSpool(OUT,run_digest=identity["run_digest"],rank=0,owned_root="/dev/shm/r16_dscp_v9");seal=spool.seal("size",adapted,ledger);spool_max=seal["serialized_bytes"];spool.cleanup(seal,ledger);gate1=dual_space_gate(workspace=ROOT,tmpfs="/dev/shm",checkpoint_bytes=saved["size_bytes"],spool_max_bytes=spool_max,world_size=1);gate4=dual_space_gate(workspace=ROOT,tmpfs="/dev/shm",checkpoint_bytes=saved["size_bytes"],spool_max_bytes=spool_max,world_size=4);pre={"schema":"r16_dscp_v9_preflight_v1","candidate":CANDIDATE,"status":"frozen","bindings":bindings,"run_identity":identity,"static":bind(STATIC),"checkpoint":{**saved,"sha256":sha256_file(saved["last"]),"resume_equal":resume==state,"identity_digest":identity["identity_digest"],"state_sha256":model_state_digest(model)},"host_cache_gates":static["host_cache_gates"],"spool":{"max_bytes":spool_max,"root":"/dev/shm/r16_dscp_v9/<run_digest>/rank<r>"},"space1":gate1,"space4":gate4,"metadata":static["metadata"],"truth_reads":0,"GPU_used":False,"auth_created":False};atomic_json_exclusive(pre,PREFLIGHT)

def verify_preflight():
 pre=json.loads(PREFLIGHT.read_text());current=effective_bindings()
 for key,value in pre["bindings"].items():
  if current[key]["sha256"]!=value["sha256"]:raise BindingRefusal(f"drift {key}")
 return pre

def freeze():
 if PREREG.exists():raise FileExistsError("v9 prereg exists")
 pre=verify_preflight();gpu=subprocess.run(["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.total","--format=csv,noheader"],check=True,capture_output=True,text=True).stdout.splitlines();payload={"schema":"r16_dscp_v9_prereg_v1","candidate":CANDIDATE,"status":"frozen","preflight":bind(PREFLIGHT),"effective_bindings":effective_bindings(),"v8_blocker":bind(ROOT/"results/r16_dscp_v8/design_preflight.json"),"only_change":"sealed evaluation authorization chain","config":yaml.safe_load(CONFIG.read_text()),"checkpoint":pre["checkpoint"],"host_cache_gates":pre["host_cache_gates"],"spool":pre["spool"],"space1":pre["space1"],"space4":pre["space4"],"metadata":pre["metadata"],"tests":json.loads(STATIC.read_text())["tests"],"identities":{"python":platform.python_version(),"torch":torch.__version__,"gpu":gpu,"cuda_used":False},"claim_boundary":"prep only, no stage/truth/GPU; immutable 10x unchanged","sealed":{"truth_read":False,"GPU_used":False,"auth_created":False}};atomic_json_exclusive(payload,PREREG)

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--mode",required=True,choices=("prep","freeze-prereg"));args=parser.parse_args();prep() if args.mode=="prep" else freeze()
if __name__=="__main__":main()

__all__=["V9ProductionWiring","V9StageDispatcher","smoke_authorization","test_authorization","validation_authorization"]
