#!/usr/bin/env python3
"""V4 production glue; this bounded revision intentionally has no stage CLI."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, subprocess, time
from pathlib import Path
from typing import Any, Mapping
import torch
import yaml

from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import R16DSCP
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import AccessLedger, Lineage
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import AUTH_SCHEMA, aggregate_scale_ranks, canonical_sha, run_evaluation_stage, run_pilot_stage, run_scale_probe, run_smoke_stage, scale_decision, validate_authorization
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.eikonal import grid_eikonal_travel_time
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v4 import (
    CandidateSpool, LongResumeState, V4GuardedOnsetLoader, V4LongRunner, V4ProductionBackend, actual_coefficient_loss, canonical_public, deployment_args, dual_space_gate,
    forward_actual_coefficients, open_truth_after_spool, score_same_sealed_candidate,
    long_command, tmpfs_constrained_world, validate_world_size,
)
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PANELS_SHA256,PARENT_PATH,PARENT_SHA256,BindingRefusal,atomic_json_exclusive,checkpoint_payload,configure_determinism,load_checkpoint,make_optimizer,save_best_last,sha256_file
from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime

ROOT=Path(__file__).resolve().parents[1];CANDIDATE="r16_dscp_v4";SCRIPT=Path(__file__).resolve();ENGINE=ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v4.py";TEST=ROOT/"tests/saved_time_phase_operator_v4/test_r16_dscp_v4.py";CONFIG=ROOT/"configs/r16_dscp_v4.yaml";OUT=ROOT/"results/r16_dscp_v4";PREFLIGHT=OUT/"design_preflight.json";STATIC=OUT/"static_evidence.json";PREREG=ROOT/"results/r16_dscp_v4_preregistration_20260826.json";BASIS=ROOT/"results/r16_dscp_v1/basis_rank16.pt";PANELS=ROOT/"results/r16_dscp_v1/panels.json";TEST_LOG=Path("/tmp/r16_dscp_v4_tests.log")

def bind(path):
 p=Path(path).resolve();return {"path":str(p),"sha256":sha256_file(p),"size_bytes":p.stat().st_size}
def effective_bindings():return {"engine":bind(ENGINE),"script":bind(SCRIPT),"test":bind(TEST),"config":bind(CONFIG),"model":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp.py"),"v4_transitive_v3":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v3.py"),"harness":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_training_v2.py"),"data_guard":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/data_guard.py"),"basis":bind(BASIS),"panels":bind(PANELS),"parent":bind(PARENT_PATH),"manifest":bind(parent_runtime.MANIFEST_PATH),"normalization":bind(parent_runtime.NORMALIZATION_PATH),"v2_replay":bind(ROOT/"results/r16_dscp_v2/replay_basis_verify/verification.json"),"v3_prereg_veto_provenance":bind(ROOT/"results/r16_dscp_v3_preregistration_20260826.json")}
def basis_artifact():
 if sha256_file(BASIS)!=BASIS_FILE_SHA256:raise BindingRefusal("basis drift")
 return torch.load(BASIS,map_location="cpu",weights_only=False)
def metadata_seal():
 manifest=parent_runtime.load_manifest_payload();out={};sets={}
 for split in ("validation","test_id"):
  rows=sorted([{"source_index":r["source_index"],"sample_id":r["sample_id"],"group_id":r["group_id"],"sample_sha256":r["sample_sha256"],"family":r["medium_type"]} for r in manifest["records"] if r["split"]==split],key=lambda r:(r["source_index"],r["sample_id"]));sets[split]={k:{r[k] for r in rows} for k in ("source_index","sample_id","group_id","sample_sha256")};out[split]={"count":len(rows),"ordered_digest":canonical_sha(rows),"by_family":{f:sum(r["family"]==f for r in rows) for f in ("uniform","layered","marmousi")},"unique_groups":len(sets[split]["group_id"]),"wavefield_read":False}
 if any(out[s]["count"]!=480 or out[s]["by_family"]!={"uniform":90,"layered":240,"marmousi":150} for s in out):raise BindingRefusal("sealed metadata census drift")
 out["cross_split_zero_overlap"]={k:not bool(sets["validation"][k]&sets["test_id"][k]) for k in sets["validation"]}
 if not all(out["cross_split_zero_overlap"].values()):raise BindingRefusal("sealed metadata overlap")
 return out

def prepare_and_seal(record:Any,parent:torch.Tensor,travel:torch.Tensor,model:R16DSCP,run_dir:str|Path,ledger:AccessLedger,device:torch.device)->tuple[tuple[Any,...],Mapping[str,Any],CandidateSpool]:
    public=canonical_public(record)
    if ledger.split not in {"train","validation","test_id"}:raise ValueError("ledger split is not registered")
    args=deployment_args(public,parent,travel,device);adapted,_=forward_actual_coefficients(model,args);spool=CandidateSpool(run_dir);seal=spool.seal(public.sample_id,adapted,ledger);return args,seal,spool

def open_bound_truth(spool:CandidateSpool,seal:Mapping[str,Any],*,source_h5:str|Path,source_index:int,k1:int,mode:str,ledger:AccessLedger,authorization:Mapping[str,Any],lineage:Lineage,once_token:Path|None=None)->torch.Tensor:
    return open_truth_after_spool(spool=spool,seal=seal,path=source_h5,index=source_index,k1=k1,mode=mode,split=ledger.split,ledger=ledger,authorization=authorization,lineage=lineage,once_token=once_token)

def load_sealed_for_score(spool:CandidateSpool,seal:Mapping[str,Any],ledger:AccessLedger)->torch.Tensor:
    return score_same_sealed_candidate(spool,seal,ledger)

class V4StageDispatcher:
    """All corrected production paths use V4ProductionBackend exclusively."""
    def __init__(self,backend:V4ProductionBackend):
        if not isinstance(backend,V4ProductionBackend):raise TypeError("v4 dispatcher refuses v3/foreign backend")
        self.backend=backend
    def dispatch(self,mode:str,index:int,*,phase:str="update",ridge:Any=None)->Mapping[str,Any]|None:
        if mode in {"smoke","pilot"} and phase=="update":return self.backend.update(self.backend.prepare(index))
        if mode=="pilot" and phase=="ridge_fit":
            prepared=self.backend.prepare(index,"ridge-fit");truth=self.backend.open_truth(prepared);self.backend.ridge_update(prepared,truth);return None
        if mode in {"scale-probe-1","scale-probe-4"}:return self.backend.measure_gradient(self.backend.prepare(index,"scale"))
        if mode=="pilot" and phase=="ridge_score":
            prepared=self.backend.prepare_ridge(index,ridge);truth=self.backend.open_truth(prepared);return self.backend.score(prepared,truth)
        if mode in {"pilot","final-train-confirm","validation-once","test-once"} and phase=="eval":
            prepared=self.backend.prepare(index,"eval");truth=self.backend.open_truth(prepared);return self.backend.score(prepared,truth)
        raise ValueError(f"unsupported v4 stage/phase: {mode}/{phase}")

class V4LongDispatcher:
    def __init__(self,runner:V4LongRunner,decision:Mapping[str,Any],cuda_visible:str):
        if not isinstance(runner,V4LongRunner):raise TypeError("v4 long dispatcher requires V4LongRunner")
        validate_world_size(decision,runner.world_size,cuda_visible);self.runner=runner
    def run(self,state:LongResumeState=LongResumeState())->Mapping[str,Any]:return self.runner.run(state)

def launch_long_command(decision:Mapping[str,Any],record_path:str|Path|None=None)->str:
    if decision.get("status")!="passed":raise ValueError("passed scale decision required")
    command=long_command(int(decision["selected_gpus"]))
    if record_path is not None:atomic_json_exclusive({"schema":"r16_dscp_v4_long_command_v1","selected_gpus":decision["selected_gpus"],"command":command,"launched":False},record_path)
    return command

def build_v4_backend(mode:str,authorization:Mapping[str,Any],lineage:Lineage,run_dir:Path):
    split="validation" if mode=="validation-once" else "test_id" if mode=="test-once" else "train";local_rank=int(os.environ.get("LOCAL_RANK","0"));device=torch.device(f"cuda:{local_rank}");configure_determinism(372);artifact=basis_artifact();candidate=R16DSCP(artifact["basis"],artifact["coefficient_scales"]).to(device);optimizer=make_optimizer(list(candidate.parameters()))
    if lineage.input_checkpoint_sha256!="none":load_checkpoint(authorization["input_checkpoint_path"],candidate,optimizer,expected_run_identity=authorization["input_run_identity"])
    parent,normalizer,manifest_payload,_=parent_runtime.load_model_context(device);manifest=parent_runtime.manifest_object(manifest_payload);panel_rows=json.loads(PANELS.read_text())["records"]
    role_map={"smoke":("smoke",),"pilot":("pilot_fit","pilot_confirm"),"scale-probe-1":("pilot_fit",),"scale-probe-4":("pilot_fit",),"long":("long_fit","long_calibration"),"final-train-confirm":("final_train_confirm",)}
    if split=="train":rows=[r for r in panel_rows if r["role"] in role_map[mode]];sample_ids=[r["sample_id"] for r in rows];family={r["sample_id"]:r["family"] for r in rows}
    else:rows=sorted([r for r in manifest_payload["records"] if r["split"]==split],key=lambda r:(r["source_index"],r["sample_id"]));sample_ids=[r["sample_id"] for r in rows];family={r["sample_id"]:r["medium_type"] for r in rows}
    raw=GuardedOnsetDataset(parent_runtime.SOURCE_H5_PATH,manifest,split=split,sample_ids=sample_ids);loader=V4GuardedOnsetLoader(raw)
    @torch.inference_mode()
    def parent_predictor(public):
        velocity=public.velocity_mps[None,None].to(device);source=public.source_parameters[None].to(device);source_map=public.source_map[None,None].to(device);medium=parent.encode_medium(velocity,normalizer);prepared=parent.prepare_sources(medium,source,source_map,normalizer,record_to_medium=torch.zeros(1,dtype=torch.long,device=device));normalized=parent.dense_normalized(prepared,public.time_s.to(device),x_m=public.x_m.to(device),z_m=public.z_m.to(device),time_block=16);return normalizer.decode_pressure(normalized.float(),source[:,4])[0].cpu()
    def travel(velocity,source):return grid_eikonal_travel_time(velocity,source_indices=[(round(float(source[1])/10),round(float(source[0])/10))],dx_m=10,dz_m=10)[0]
    token=run_dir/f"{split}.once.json" if split!="train" else None;backend=V4ProductionBackend(public_loader=loader,parent_predictor=parent_predictor,travel_builder=travel,candidate=candidate,optimizer=optimizer,authorization=authorization,lineage=lineage,source_h5=parent_runtime.SOURCE_H5_PATH,run_dir=run_dir,device=device,split=split,family_by_sample=family,once_token=token);backend.checkpoint_identity={**lineage.__dict__,"authorization_sha256":authorization.get("_file_sha256"),"engine_sha256":sha256_file(ENGINE),"script_sha256":sha256_file(SCRIPT),"config_sha256_effective":sha256_file(CONFIG),"panels_sha256_effective":PANELS_SHA256,"basis_sha256_effective":BASIS_FILE_SHA256,"parent_sha256_effective":PARENT_SHA256}
    index_by_role={role:[sample_ids.index(r["sample_id"]) for r in rows if r.get("role")==role] for role in role_map.get(mode,())};return backend,index_by_role,list(range(len(loader))),canonical_sha([{"source_index":r["source_index"],"sample_id":r["sample_id"],"group_id":r["group_id"],"sample_sha256":r.get("sample_sha256",r.get("manifest_sample_sha256"))} for r in rows])

def _score_indices(backend:V4ProductionBackend,indices:list[int],ridge=None):
    rows=[]
    for index in indices:
        prepared=backend.prepare_ridge(index,ridge) if ridge is not None else backend.prepare(index,"eval");truth=backend.open_truth(prepared);rows.append(backend.score(prepared,truth))
    return rows
def _update_index(backend,index):return backend.update(backend.prepare(index,"update"))
def _checkpoint(backend,mode,best=True):return backend.checkpoint({"mode":mode},best)

def dispatch_stage(mode:str,authorization:Mapping[str,Any],lineage:Lineage,run_dir:Path,resume_state:LongResumeState=LongResumeState()):
    backend,roles,all_indices,metadata_digest=build_v4_backend(mode,authorization,lineage,run_dir);dispatcher=V4StageDispatcher(backend);world=int(os.environ.get("WORLD_SIZE","1"));rank=int(os.environ.get("RANK","0"))
    try:
        if mode=="smoke":return run_smoke_stage(records=roles["smoke"],update=lambda i:dispatcher.dispatch("smoke",i),score=lambda:_score_indices(backend,roles["smoke"]),checkpoint=lambda:_checkpoint(backend,mode),lineage=lineage,run_dir=run_dir,resource_snapshot=lambda saved:backend.resources(saved,world))
        if mode=="pilot":
            def ridge_update(index):
                prepared=backend.prepare(index,"ridge-fit");truth=backend.open_truth(prepared);backend.ridge_update(prepared,truth)
            return run_pilot_stage(fit_records=roles["pilot_fit"],confirm_records=roles["pilot_confirm"],candidate_update=lambda i:dispatcher.dispatch("pilot",i),ridge_update=ridge_update,ridge_solve=backend.ridge_finalize,candidate_score=lambda rows:_score_indices(backend,list(rows)),ridge_score=lambda ridge,rows:_score_indices(backend,list(rows),ridge),checkpoint=lambda:_checkpoint(backend,mode),lineage=lineage,run_dir=run_dir,resource_snapshot=lambda saved:backend.resources(saved,world))
        if mode in {"scale-probe-1","scale-probe-4"}:
            records=roles["pilot_fit"][:12];report=run_scale_probe(mode=mode,records=records,measure=lambda i:dispatcher.dispatch(mode,i),model_hash=lambda:canonical_sha({k:sha256_file_tensor(v) for k,v in backend.candidate.state_dict().items()}),resource_snapshot=lambda:backend.resources({"size_bytes":Path(authorization["input_checkpoint_path"]).stat().st_size},world),lineage=lineage,run_dir=run_dir,world_size=world,rank=rank)
            return report
        if mode=="long":
            decision=json.loads(Path(authorization["scale_decision_path"]).read_text());validate_world_size(decision,world,os.environ.get("CUDA_VISIBLE_DEVICES",""));train=roles["long_fit"];cal=roles["long_calibration"]
            def gather(rows):
                if world==1:return rows
                gathered=[None]*world;torch.distributed.all_gather_object(gathered,rows);return [item for shard in gathered for item in shard]
            def synchronized_step():
                if world==4:
                    for parameter in backend.candidate.parameters():
                        if parameter.grad is not None:torch.distributed.all_reduce(parameter.grad);parameter.grad.div_(4.)
                backend.optimizer.step()
            runner=V4LongRunner(world_size=world,rank=rank,train_records=train,calibration_records=cal,zero_grad=lambda:backend.optimizer.zero_grad(set_to_none=True),backward_record=lambda i,s:backend.backward_only(backend.prepare(i,"long"),s),optimizer_step=synchronized_step,set_lr=lambda lr:[group.update(lr=lr) for group in backend.optimizer.param_groups],evaluate_record=lambda i:_score_indices(backend,[i])[0],gather=gather,checkpoint=lambda state,best:backend.checkpoint(state.__dict__,best),terminal=lambda payload:atomic_json_exclusive(payload,run_dir/"terminal.json"),barrier=(torch.distributed.barrier if world==4 else lambda:None));return runner.run(resume_state)
        if mode in {"final-train-confirm","validation-once","test-once"}:
            validation=json.loads((OUT/"validation-once/terminal.json").read_text()) if mode=="test-once" and (OUT/"validation-once/terminal.json").is_file() else None;lock={"schema":"r16_dscp_v4_candidate_lock_v1","checkpoint_sha256":lineage.input_checkpoint_sha256,"code_sha256":lineage.code_sha256,"config_sha256":lineage.config_sha256,"basis_sha256":lineage.basis_sha256,"metadata_digest":metadata_digest,"thresholds_sha256":authorization["gates_sha256"]}
            return run_evaluation_stage(mode=mode,records=all_indices,evaluate_one=lambda i:_score_indices(backend,[i])[0],resource_snapshot=lambda:backend.resources({"size_bytes":Path(authorization["input_checkpoint_path"]).stat().st_size},world),lineage=lineage,run_dir=run_dir,metadata_digest=metadata_digest,candidate_lock_payload=lock if mode=="final-train-confirm" else None,required_validation_terminal=validation)
        raise ValueError(f"unsupported v4 production mode: {mode}")
    except Exception as exc:
        backend.preserve_failure_spool()
        if rank==0 and not (run_dir/"terminal.json").exists():atomic_json_exclusive({"schema":"r16_dscp_v4_terminal_v1","status":"failed","mode":mode,"reason":str(exc),"lineage":lineage.__dict__},run_dir/"terminal.json")
        raise
    finally:backend.public_loader.close()

def sha256_file_tensor(value:torch.Tensor)->str:
    tensor=torch.as_tensor(value).detach().cpu().contiguous();return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

def static_evidence():
    cfg=yaml.safe_load(CONFIG.read_text());
    for name,command in cfg["commands"].items():
        if name not in {"prep","authorize_stage","scale_decide","launch_long_command"} and not command.startswith("env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="):raise BindingRefusal("CUDA command missing deterministic env")
    if not TEST_LOG.is_file() or "[100%]" not in TEST_LOG.read_text() or "failed" in TEST_LOG.read_text().lower():raise BindingRefusal("test evidence absent")
    v3=json.loads((ROOT/"results/r16_dscp_v3/static_evidence.json").read_text());return {"schema":"r16_dscp_v4_static_v1","parameters":1202,"ridge_parameters":480,"model_macs":45451125,"basis_conditions":v3["basis_conditions"],"router_census":v3["router_census"],"panels_census":v3["panels_census"],"metadata":metadata_seal(),"tests":{"status":"passed","log_sha256":sha256_file(TEST_LOG),"tail":TEST_LOG.read_text().splitlines()[-1]},"production_symbols":["V4GuardedOnsetLoader","V4ProductionBackend","V4StageDispatcher","V4LongRunner"],"foreign_backend_reachable":False,"truth_read":False}

def prep():
    if OUT.exists() or PREREG.exists():raise FileExistsError("v4 target exists")
    bindings=effective_bindings();static=static_evidence();OUT.mkdir();atomic_json_exclusive(static,STATIC);artifact=basis_artifact();candidate=R16DSCP(artifact["basis"],artifact["coefficient_scales"]);optimizer=make_optimizer(list(candidate.parameters()));h=w=201;parent=torch.zeros(1,401,h,w);args=(torch.ones(1,1,h,w)*2000,torch.zeros(1,1,h,w),torch.ones(1,1,h,w),torch.arange(w),torch.arange(h),torch.zeros(1,1,h,w),torch.zeros(1,1,h,w),parent,torch.tensor([1]),torch.tensor([2]));truth=torch.ones(1,398,h,w);optimizer.zero_grad();losses,adapted,_=actual_coefficient_loss(candidate,args,truth);losses["total"].backward();optimizer.step();identity={"candidate":CANDIDATE,"run_digest":canonical_sha(bindings),"code_sha256":bindings["engine"]["sha256"],"config_sha256":bindings["config"]["sha256"]};payload=checkpoint_payload(candidate,optimizer,run_identity=identity,sampler_order=list(range(12)),progress={"update":1});saved=save_best_last(payload,OUT/"actual_checkpoint",is_best=True);restored=R16DSCP(artifact["basis"],artifact["coefficient_scales"]);restored_optimizer=make_optimizer(list(restored.parameters()));loaded=load_checkpoint(saved["last"],restored,restored_optimizer,expected_run_identity=identity);resume_equal=all(torch.equal(a,b) for a,b in zip(candidate.parameters(),restored.parameters())) and loaded["progress"]=={"update":1}
    ledger=AccessLedger("train");spool=CandidateSpool(OUT,run_digest=identity["run_digest"],rank=0);seal=spool.seal("size-probe",adapted.detach(),ledger);spool_max=int(seal["serialized_bytes"]);spool.cleanup(seal,ledger);gate1=dual_space_gate(workspace=ROOT,tmpfs="/dev/shm",checkpoint_bytes=saved["size_bytes"],spool_max_bytes=spool_max,world_size=1);gate4=dual_space_gate(workspace=ROOT,tmpfs="/dev/shm",checkpoint_bytes=saved["size_bytes"],spool_max_bytes=spool_max,world_size=4)
    if not resume_equal or not saved["hardlinked"] or not gate1["passed"]:raise BindingRefusal("checkpoint/resume/space gate failed")
    pre={"schema":"r16_dscp_v4_preflight_v1","candidate":CANDIDATE,"status":"frozen","bindings":bindings,"run_identity":identity,"static":bind(STATIC),"checkpoint":{**saved,"sha256":sha256_file(saved["last"]),"resume_equal":resume_equal},"spool":{"max_bytes":spool_max,"root":"/dev/shm/r16_dscp_v4/<run_digest>/rank<r>"},"space_gate_1gpu":gate1,"space_gate_4gpu":gate4,"sealed_metadata":static["metadata"],"truth_reads":0,"authorizations_created":False};atomic_json_exclusive(pre,PREFLIGHT);print(json.dumps(pre["checkpoint"]))

def verify_preflight():
    pre=json.loads(PREFLIGHT.read_text());current=effective_bindings()
    for key,value in pre["bindings"].items():
        if current[key]["sha256"]!=value["sha256"]:raise BindingRefusal(f"binding drift: {key}")
    return pre

def freeze():
    if PREREG.exists():raise FileExistsError("v4 prereg exists")
    pre=verify_preflight();cfg=yaml.safe_load(CONFIG.read_text());v3=bind(ROOT/"results/r16_dscp_v3_preregistration_20260826.json");runtime=json.loads((ROOT/"results/r4e7_parent_e2e_runtime_train9_v1_20260825.json").read_text());gpu=subprocess.run(["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.total","--format=csv,noheader"],check=True,capture_output=True,text=True).stdout.splitlines();payload={"schema":"r16_dscp_v4_prereg_v1","candidate":CANDIDATE,"status":"frozen","preflight":bind(PREFLIGHT),"effective_bindings":effective_bindings(),"v3_veto_provenance":v3,"scientific_design":"unchanged; only canonical shapes, actual coefficient loss, serialized candidate, tmpfs spool, global-batch long fixes","config":cfg,"commands":cfg["commands"],"checkpoint":pre["checkpoint"],"spool":pre["spool"],"workspace_gate":pre["space_gate_1gpu"],"tmpfs_gate_1gpu":pre["space_gate_1gpu"],"tmpfs_gate_4gpu":pre["space_gate_4gpu"],"sealed_metadata":pre["sealed_metadata"],"tests":json.loads(STATIC.read_text())["tests"],"identities":{"python":platform.python_version(),"torch":torch.__version__,"torch_cuda":str(torch.version.cuda),"gpu":gpu,"cuda_used":False},"parent_runtime":{"mean_s":runtime["aggregate"]["arithmetic_mean_runtime_s"],"p95_s":runtime["aggregate"]["nearest_rank_p95_runtime_s"],"traditional_speedup":runtime["aggregate"]["reference_over_mean_speedup_x"]},"claim_boundary":"prep only; no train/validation/test. Parent is only 1.608x, so complete E2E cannot meet immutable 10x unless independently measured latency changes; gate is not weakened.","rollback":cfg["rollback"],"sealed":{"truth_read":False,"GPU_used":False,"authorizations_created":False,"fields_in_workspace":False}};atomic_json_exclusive(payload,PREREG);print(json.dumps(bind(PREREG)))

def authorize_stage(stage:str,output:str|Path,input_checkpoint:str|Path|None=None):
    if not PREREG.is_file():raise BindingRefusal("frozen prereg required")
    pre=verify_preflight();cfg=yaml.safe_load(CONFIG.read_text());chain={"smoke":[PREREG],"pilot":[OUT/"smoke/terminal.json"],"scale-probe-1":[OUT/"pilot/terminal.json"],"scale-probe-4":[OUT/"pilot/terminal.json"],"scale-decide":[OUT/"scale-probe-1/terminal.json",OUT/"scale-probe-4/terminal.json"],"long":[OUT/"pilot/terminal.json",OUT/"scale-decide/terminal.json"],"final-train-confirm":[OUT/"long/terminal.json"],"validation-once":[OUT/"final-train-confirm/terminal.json",OUT/"final-train-confirm/candidate_lock.json"],"test-once":[OUT/"validation-once/terminal.json",OUT/"final-train-confirm/candidate_lock.json"]}
    if stage not in chain:raise BindingRefusal("unknown stage")
    prereq={};locked=None;expected=[]
    for path in chain[stage]:
        if not path.is_file():raise BindingRefusal(f"prerequisite absent: {path}")
        obj=json.loads(path.read_text());status=obj.get("status")
        if "terminal" in path.name and status not in {"passed","completed_train"}:raise BindingRefusal("prerequisite failed")
        if path.name=="candidate_lock.json":locked=obj.get("checkpoint_sha256")
        if isinstance(obj.get("checkpoint"),dict) and obj["checkpoint"].get("best"):expected.append(str(Path(obj["checkpoint"]["best"]).resolve()))
        prereq[str(path.resolve())]=sha256_file(path)
    input_sha="none";input_identity=pre["run_identity"]
    if input_checkpoint is not None:
        path=Path(input_checkpoint).resolve();input_sha=sha256_file(path)
        if expected and str(path) not in expected:raise BindingRefusal("input is not prerequisite best")
        if locked and input_sha!=locked:raise BindingRefusal("input differs from lock")
        input_identity=torch.load(path,map_location="cpu",weights_only=False)["run_identity"]
    elif stage not in {"smoke","scale-decide"}:raise BindingRefusal("input checkpoint required")
    line=Lineage(CANDIDATE,stage,pre["run_identity"]["run_digest"],sha256_file(ENGINE),sha256_file(CONFIG),PANELS_SHA256,BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PARENT_SHA256,input_sha);payload={"schema":AUTH_SCHEMA,"status":"authorized",**line.__dict__,"input_checkpoint_path":None if input_checkpoint is None else str(Path(input_checkpoint).resolve()),"input_run_identity":input_identity,"prerequisite_bindings":prereq,"gates_sha256":canonical_sha(cfg["gates"]),"script_sha256":sha256_file(SCRIPT),"prereg_sha256":sha256_file(PREREG)}
    if stage.startswith("scale-probe"):payload["pilot_terminal_sha256"]=sha256_file(OUT/"pilot/terminal.json")
    if stage=="scale-decide":payload.update(scale1_terminal_sha256=sha256_file(OUT/"scale-probe-1/terminal.json"),scale4_terminal_sha256=sha256_file(OUT/"scale-probe-4/terminal.json"))
    if stage=="long":
        decision=json.loads((OUT/"scale-decide/terminal.json").read_text());payload.update(selected_world_size=decision["selected_gpus"],scale_decision_path=str((OUT/"scale-decide/terminal.json").resolve()),scale_decision_sha256=sha256_file(OUT/"scale-decide/terminal.json"))
    if stage in {"validation-once","test-once"}:payload["candidate_lock_sha256"]=sha256_file(OUT/"final-train-confirm/candidate_lock.json")
    if stage=="test-once":payload["validation_terminal_sha256"]=sha256_file(OUT/"validation-once/terminal.json")
    payload["authorization_digest"]=canonical_sha(payload);atomic_json_exclusive(payload,output)

def validate_auth(mode:str,path:str|Path):
    pre=verify_preflight();payload=json.loads(Path(path).read_text());line=Lineage(CANDIDATE,mode,pre["run_identity"]["run_digest"],sha256_file(ENGINE),sha256_file(CONFIG),PANELS_SHA256,BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PARENT_SHA256,str(payload.get("input_checkpoint_sha256","none")));validate_authorization(payload,line)
    if payload.get("script_sha256")!=sha256_file(SCRIPT) or payload.get("prereg_sha256")!=sha256_file(PREREG) or payload.get("gates_sha256")!=canonical_sha(yaml.safe_load(CONFIG.read_text())["gates"]):raise BindingRefusal("authorization drift")
    for p,d in payload["prerequisite_bindings"].items():
        if not Path(p).is_file() or sha256_file(p)!=d:raise BindingRefusal("prerequisite drift")
    return payload,line

def scale_decide(authorization:Mapping[str,Any],lineage:Lineage,run_dir:Path):
    one_path=OUT/"scale-probe-1/terminal.json";four_path=OUT/"scale-probe-4/terminal.json"
    if authorization["scale1_terminal_sha256"]!=sha256_file(one_path) or authorization["scale4_terminal_sha256"]!=sha256_file(four_path):raise BindingRefusal("scale terminal drift")
    one=json.loads(one_path.read_text());four=json.loads(four_path.read_text());decision=scale_decision(one,four);pre=verify_preflight();selected=tmpfs_constrained_world({"selected_gpus":decision["selected_gpus"]},pre["space_gate_1gpu"],pre["space_gate_4gpu"]);payload={"schema":"r16_dscp_v4_scale_decision_v1","status":"passed","selected_gpus":selected,"raw_gates":decision,"tmpfs_forced_fallback":selected!=decision["selected_gpus"],"long_launched":False,"lineage":lineage.__dict__};atomic_json_exclusive(payload,run_dir/"terminal.json");return payload

def run_authorized(mode:str,authorization_path:str|Path,*,resume:bool=False,world_size:int=1):
    if mode!="scale-decide":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise BindingRefusal("CUBLAS env mismatch")
    authorization,lineage=validate_auth(mode,authorization_path);authorization=dict(authorization);authorization["_file_sha256"]=sha256_file(authorization_path);rank=int(os.environ.get("RANK","0"));world=int(os.environ.get("WORLD_SIZE",str(world_size)));run_dir=OUT/mode;distributed=world==4 and mode in {"scale-probe-4","long"}
    if distributed:
        torch.distributed.init_process_group("nccl");torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    run_identity={**lineage.__dict__,"authorization_sha256":sha256_file(authorization_path),"engine_sha256":sha256_file(ENGINE),"script_sha256":sha256_file(SCRIPT),"config_sha256_effective":sha256_file(CONFIG),"panels_sha256_effective":PANELS_SHA256,"basis_sha256_effective":BASIS_FILE_SHA256,"parent_sha256_effective":PARENT_SHA256}
    resume_state=LongResumeState()
    try:
        if resume:
            if mode!="long":raise BindingRefusal("resume is long-only")
            payload=validate_resume_directory(run_dir,authorization_sha256=run_identity["authorization_sha256"],effective_bindings={k:run_identity[k] for k in ("engine_sha256","script_sha256","config_sha256_effective","panels_sha256_effective","basis_sha256_effective","parent_sha256_effective")},load_last=lambda p:torch.load(p,map_location="cpu",weights_only=False));progress=payload["progress"];resume_state=LongResumeState(**{k:progress[k] for k in LongResumeState.__dataclass_fields__});authorization=dict(authorization);authorization["input_checkpoint_path"]=str(run_dir/"last.pt");authorization["input_run_identity"]=payload["run_identity"]
        else:
            if rank==0:
                if run_dir.exists():raise FileExistsError("fresh stage exists")
                run_dir.mkdir();atomic_json_exclusive(run_identity,run_dir/"run_identity.json")
            if distributed:torch.distributed.barrier()
        if mode=="scale-decide":return scale_decide(authorization,lineage,run_dir)
        if mode=="long":
            decision=json.loads(Path(authorization["scale_decision_path"]).read_text());validate_world_size(decision,world,os.environ.get("CUDA_VISIBLE_DEVICES",""))
        if mode=="scale-probe-4":
            try:result=dispatch_stage(mode,authorization,lineage,run_dir,resume_state)
            except Exception as local_exc:result={"status":"failed","rank":rank,"world_size":world,"reason":str(local_exc)}
            gathered=[None]*world;torch.distributed.all_gather_object(gathered,result);failures=[report for report in gathered if report.get("status")!="passed"]
            if rank==0:
                if failures:result={"schema":"r16_dscp_v4_terminal_v1","status":"failed","mode":mode,"rank_failures":failures,"lineage":lineage.__dict__};atomic_json_exclusive(result,run_dir/"terminal.json")
                else:order=sorted({row["sample_id"] for report in gathered for row in report["rows"]});aggregate=aggregate_scale_ranks(gathered,order);aggregate.update(mode=mode,lineage=lineage.__dict__);atomic_json_exclusive(aggregate,run_dir/"terminal.json");result=aggregate
            torch.distributed.barrier()
            if failures:raise RuntimeError("one or more scale ranks failed")
        else:result=dispatch_stage(mode,authorization,lineage,run_dir,resume_state)
        return result
    except Exception as exc:
        if rank==0 and run_dir.is_dir() and not (run_dir/"terminal.json").exists():atomic_json_exclusive({"schema":"r16_dscp_v4_terminal_v1","status":"failed","mode":mode,"reason":str(exc),"lineage":run_identity},run_dir/"terminal.json")
        raise
    finally:
        if distributed and torch.distributed.is_initialized():torch.distributed.destroy_process_group()

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--mode",required=True,choices=("prep","freeze-prereg","authorize-stage","launch-long-command","smoke","pilot","scale-probe-1","scale-probe-4","scale-decide","long","final-train-confirm","validation-once","test-once"));parser.add_argument("--authorization");parser.add_argument("--stage");parser.add_argument("--output");parser.add_argument("--input-checkpoint");parser.add_argument("--resume",action="store_true");parser.add_argument("--world-size",type=int,default=1);args=parser.parse_args()
    if args.mode=="prep":prep()
    elif args.mode=="freeze-prereg":freeze()
    elif args.mode=="authorize-stage":authorize_stage(args.stage,args.output,args.input_checkpoint)
    elif args.mode=="launch-long-command":
        decision=json.loads((OUT/"scale-decide/terminal.json").read_text());print(launch_long_command(decision,args.output))
    else:
        if not args.authorization:raise BindingRefusal("authorization required")
        run_authorized(args.mode,args.authorization,resume=args.resume,world_size=args.world_size)

if __name__=="__main__":main()

__all__=["V4LongDispatcher","V4StageDispatcher","actual_coefficient_loss","launch_long_command","load_sealed_for_score","open_bound_truth","prepare_and_seal"]
