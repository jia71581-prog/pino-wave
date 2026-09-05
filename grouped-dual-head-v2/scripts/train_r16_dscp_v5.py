#!/usr/bin/env python3
"""V5 bounded production glue; no stage CLI in this revision."""
from __future__ import annotations
import argparse,hashlib,json,math,os,platform,subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch,yaml
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import R16DSCP
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v5 import BindingDriftRejection,V5LongRunner,V5ProductionBackend,run_v5_smoke,verify_before_data,verify_loaded_state
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v5 import long_terminal,model_state_digest,v5_long_command
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import AUTH_SCHEMA,Lineage,aggregate_scale_ranks,canonical_sha,run_evaluation_stage,run_pilot_stage,run_scale_probe,scale_decision,validate_authorization
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v4 import CandidateSpool,LongResumeState,V4GuardedOnsetLoader,actual_coefficient_loss,dual_space_gate,tmpfs_constrained_world,validate_resume_directory
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.eikonal import grid_eikonal_travel_time
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PANELS_SHA256,PARENT_PATH,PARENT_SHA256,BindingRefusal,atomic_json_exclusive,checkpoint_payload,configure_determinism,load_checkpoint,make_optimizer,save_best_last,sha256_file
from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime

ROOT=Path(__file__).resolve().parents[1];CANDIDATE="r16_dscp_v5";SCRIPT=Path(__file__).resolve();ENGINE=ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v5.py";TEST=ROOT/"tests/saved_time_phase_operator_v4/test_r16_dscp_v5.py";CONFIG=ROOT/"configs/r16_dscp_v5.yaml";OUT=ROOT/"results/r16_dscp_v5";STATIC=OUT/"static_evidence.json";PREFLIGHT=OUT/"design_preflight.json";PREREG=ROOT/"results/r16_dscp_v5_preregistration_20260826.json";BASIS=ROOT/"results/r16_dscp_v1/basis_rank16.pt";PANELS=ROOT/"results/r16_dscp_v1/panels.json";TEST_LOG=Path("/tmp/r16_dscp_v5_tests.log")

class V5StageDispatcher:
    def __init__(self,backend:V5ProductionBackend):
        if not isinstance(backend,V5ProductionBackend):raise TypeError("v5 dispatcher requires V5ProductionBackend")
        self.backend=backend
    def update(self,index:int)->Mapping[str,Any]:return self.backend.update(self.backend.prepare(index,"update"))
    def measure_gradient(self,index:int)->Mapping[str,Any]:return self.backend.measure_gradient(self.backend.prepare(index,"scale"))
    def backward_only(self,index:int,scale:float)->Mapping[str,Any]:return self.backend.backward_only(self.backend.prepare(index,"long"),scale)
    def score(self,index:int)->Mapping[str,Any]:
        prepared=self.backend.prepare(index,"eval");truth=self.backend.open_truth(prepared);return self.backend.score(prepared,truth)

def backend_after_strict_verification(authorization:Mapping[str,Any],*,input_checkpoint:str|None,parent:str,basis:str,config:str,code:str,backend_factory:Any)->V5ProductionBackend:
    verify_before_data(authorization,input_checkpoint=input_checkpoint,parent=parent,basis=basis,config=config,code=code);backend=backend_factory()
    if not isinstance(backend,V5ProductionBackend):raise TypeError("foreign backend refused")
    verify_loaded_state(backend.candidate,authorization);return backend

def validate_final_from_long(authorization:Mapping[str,Any],long_terminal:Mapping[str,Any])->None:
    best=long_terminal.get("best_checkpoint",{})
    if long_terminal.get("status")!="completed_train" or not best.get("path") or sha256_file(best["path"])!=best.get("sha256") or authorization.get("input_checkpoint_sha256")!=best.get("sha256"):raise BindingDriftRejection("final authorization must bind long best checkpoint")

class V5ProductionDispatcher(V5StageDispatcher):
    def pilot(self,**kwargs:Any):return run_pilot_stage(**kwargs)
    def scale(self,**kwargs:Any):return run_scale_probe(**kwargs)
    def long(self,runner:V5LongRunner,*args:Any,**kwargs:Any):
        if not isinstance(runner,V5LongRunner):raise TypeError("v5 long runner required")
        return runner.run(*args,**kwargs)
    def evaluate(self,**kwargs:Any):return run_evaluation_stage(**kwargs)

class V5ProductionWiring:
    """Authorization-first stage router; backend factory is invoked only after rehash."""
    def __init__(self,*,parent:str,basis:str,config:str,code:str,backend_factory:Any):self.parent=parent;self.basis=basis;self.config=config;self.code=code;self.backend_factory=backend_factory
    def backend(self,authorization:Mapping[str,Any],input_checkpoint:str|None)->V5ProductionBackend:
        return backend_after_strict_verification(authorization,input_checkpoint=input_checkpoint,parent=self.parent,basis=self.basis,config=self.config,code=self.code,backend_factory=self.backend_factory)
    def run(self,mode:str,authorization:Mapping[str,Any],input_checkpoint:str|None,**kwargs:Any):
        dispatcher=V5ProductionDispatcher(self.backend(authorization,input_checkpoint))
        if mode=="smoke":return smoke(dispatcher,**kwargs)
        if mode=="pilot":return dispatcher.pilot(**kwargs)
        if mode in {"scale-probe-1","scale-probe-4"}:return dispatcher.scale(**kwargs)
        if mode=="long":return dispatcher.long(**kwargs)
        if mode in {"final-train-confirm","validation-once","test-once"}:return dispatcher.evaluate(**kwargs)
        raise ValueError(f"unsupported v5 production mode: {mode}")

def bind(path):
    p=Path(path).resolve();return {"path":str(p),"sha256":sha256_file(p),"size_bytes":p.stat().st_size}
def effective_bindings():return {"engine":bind(ENGINE),"script":bind(SCRIPT),"test":bind(TEST),"config":bind(CONFIG),"model":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp.py"),"v5_transitive_v4":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v4.py"),"v5_transitive_v3":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v3.py"),"harness":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_training_v2.py"),"basis":bind(BASIS),"panels":bind(PANELS),"parent":bind(PARENT_PATH),"manifest":bind(parent_runtime.MANIFEST_PATH),"normalization":bind(parent_runtime.NORMALIZATION_PATH),"v2_replay":bind(ROOT/"results/r16_dscp_v2/replay_basis_verify/verification.json"),"v3_veto":bind(ROOT/"results/r16_dscp_v3_preregistration_20260826.json"),"v4_veto":bind(ROOT/"results/r16_dscp_v4_preregistration_20260826.json")}
def basis_artifact():
    if sha256_file(BASIS)!=BASIS_FILE_SHA256:raise BindingRefusal("basis drift")
    return torch.load(BASIS,map_location="cpu",weights_only=False)
def metadata_seal():
    manifest=parent_runtime.load_manifest_payload();out={};sets={}
    for split in ("validation","test_id"):
        rows=sorted([{"source_index":r["source_index"],"sample_id":r["sample_id"],"group_id":r["group_id"],"sample_sha256":r["sample_sha256"],"family":r["medium_type"]} for r in manifest["records"] if r["split"]==split],key=lambda r:(r["source_index"],r["sample_id"]));sets[split]={k:{r[k] for r in rows} for k in ("source_index","sample_id","group_id","sample_sha256")};out[split]={"count":len(rows),"ordered_digest":canonical_sha(rows),"by_family":{f:sum(r["family"]==f for r in rows) for f in ("uniform","layered","marmousi")},"wavefield_read":False}
    if any(out[s]["count"]!=480 or out[s]["by_family"]!={"uniform":90,"layered":240,"marmousi":150} for s in out):raise BindingRefusal("sealed metadata drift")
    out["zero_overlap"]={k:not bool(sets["validation"][k]&sets["test_id"][k]) for k in sets["validation"]};return out

def build_v5_backend(mode,authorization,lineage,run_dir):
    split="validation" if mode=="validation-once" else "test_id" if mode=="test-once" else "train";device=torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK','0'))}");configure_determinism(372);artifact=basis_artifact();candidate=R16DSCP(artifact["basis"],artifact["coefficient_scales"]).to(device);optimizer=make_optimizer(list(candidate.parameters()))
    if lineage.input_checkpoint_sha256!="none":load_checkpoint(authorization["input_checkpoint_path"],candidate,optimizer,expected_run_identity=authorization["input_run_identity"])
    verify_loaded_state(candidate,authorization);parent,normalizer,manifest_payload,_=parent_runtime.load_model_context(device);manifest=parent_runtime.manifest_object(manifest_payload);panels=json.loads(PANELS.read_text())["records"];roles={"smoke":("smoke",),"pilot":("pilot_fit","pilot_confirm"),"scale-probe-1":("pilot_fit",),"scale-probe-4":("pilot_fit",),"long":("long_fit","long_calibration"),"final-train-confirm":("final_train_confirm",)}
    if split=="train":rows=[r for r in panels if r["role"] in roles[mode]];samples=[r["sample_id"] for r in rows];family={r["sample_id"]:r["family"] for r in rows}
    else:rows=sorted([r for r in manifest_payload["records"] if r["split"]==split],key=lambda r:(r["source_index"],r["sample_id"]));samples=[r["sample_id"] for r in rows];family={r["sample_id"]:r["medium_type"] for r in rows}
    loader=V4GuardedOnsetLoader(GuardedOnsetDataset(parent_runtime.SOURCE_H5_PATH,manifest,split=split,sample_ids=samples))
    @torch.inference_mode()
    def parent_predictor(public):
        velocity=public.velocity_mps[None,None].to(device);source=public.source_parameters[None].to(device);source_map=public.source_map[None,None].to(device);medium=parent.encode_medium(velocity,normalizer);prepared=parent.prepare_sources(medium,source,source_map,normalizer,record_to_medium=torch.zeros(1,dtype=torch.long,device=device));normalized=parent.dense_normalized(prepared,public.time_s.to(device),x_m=public.x_m.to(device),z_m=public.z_m.to(device),time_block=16);return normalizer.decode_pressure(normalized.float(),source[:,4])[0].cpu()
    travel=lambda velocity,source:grid_eikonal_travel_time(velocity,source_indices=[(round(float(source[1])/10),round(float(source[0])/10))],dx_m=10,dz_m=10)[0];token=run_dir/f"{split}.once.json" if split!="train" else None;backend=V5ProductionBackend(public_loader=loader,parent_predictor=parent_predictor,travel_builder=travel,candidate=candidate,optimizer=optimizer,authorization=authorization,lineage=lineage,source_h5=parent_runtime.SOURCE_H5_PATH,run_dir=run_dir,device=device,split=split,family_by_sample=family,once_token=token);backend.checkpoint_identity={**lineage.__dict__,"authorization_sha256":authorization.get("_file_sha256"),"engine_sha256":sha256_file(ENGINE),"script_sha256":sha256_file(SCRIPT),"config_sha256_effective":sha256_file(CONFIG),"panels_sha256_effective":PANELS_SHA256,"basis_sha256_effective":BASIS_FILE_SHA256,"parent_sha256_effective":PARENT_SHA256};index_roles={role:[samples.index(r["sample_id"]) for r in rows if r.get("role")==role] for role in roles.get(mode,())};return backend,index_roles,list(range(len(loader))),canonical_sha(rows)

def score_indices(backend,indices,ridge=None):
    rows=[]
    for i in indices:
        prepared=backend.prepare_ridge(i,ridge) if ridge is not None else backend.prepare(i,"eval");truth=backend.open_truth(prepared);rows.append(backend.score(prepared,truth))
    return rows
def checkpoint_binding(saved):
    path=Path(saved["best"] or saved["last"]);return {"path":str(path),"sha256":sha256_file(path),"size_bytes":path.stat().st_size}

def dispatch_v5(mode,authorization,lineage,run_dir,resume_state=LongResumeState()):
    backend,roles,all_indices,metadata_digest=build_v5_backend(mode,authorization,lineage,run_dir);dispatcher=V5StageDispatcher(backend);world=int(os.environ.get("WORLD_SIZE","1"));rank=int(os.environ.get("RANK","0"));ledgers=[]
    try:
        if mode=="smoke":
            def final_score():return score_indices(backend,roles["smoke"])
            return run_v5_smoke(records=roles["smoke"],update=dispatcher.update,quick_score=lambda:(lambda rows:(rows,all(r["aggregate_rel_l2"]<=r["parent_rel_l2"] for r in rows)))(final_score()),final_score=final_score,checkpoint=lambda:backend.checkpoint({"mode":mode}),resource_snapshot=lambda saved:backend.resources(saved,world),terminal=lambda payload:atomic_json_exclusive(payload,run_dir/"terminal.json"))
        if mode=="pilot":
            def ridge_update(i):
                p=backend.prepare(i,"ridge");truth=backend.open_truth(p);backend.ridge_update(p,truth)
            return run_pilot_stage(fit_records=roles["pilot_fit"],confirm_records=roles["pilot_confirm"],candidate_update=dispatcher.update,ridge_update=ridge_update,ridge_solve=backend.ridge_finalize,candidate_score=lambda rows:score_indices(backend,list(rows)),ridge_score=lambda ridge,rows:score_indices(backend,list(rows),ridge),checkpoint=lambda:backend.checkpoint({"mode":mode}),lineage=lineage,run_dir=run_dir,resource_snapshot=lambda saved:backend.resources(saved,world))
        if mode in {"scale-probe-1","scale-probe-4"}:return run_scale_probe(mode=mode,records=roles["pilot_fit"][:12],measure=dispatcher.measure_gradient,model_hash=lambda:model_state_digest(backend.candidate),resource_snapshot=lambda:backend.resources({"size_bytes":Path(authorization["input_checkpoint_path"]).stat().st_size},world),lineage=lineage,run_dir=run_dir,world_size=world,rank=rank)
        if mode=="long":
            decision=json.loads(Path(authorization["scale_decision_path"]).read_text());train=roles["long_fit"];cal=roles["long_calibration"];saved={"best":None,"last":None};access=[]
            def gather(rows):
                if world==1:return rows
                shards=[None]*world;torch.distributed.all_gather_object(shards,rows);return [x for shard in shards for x in shard]
            def step():
                try:
                    if world==4:
                        for p in backend.candidate.parameters():
                            if p.grad is not None:torch.distributed.all_reduce(p.grad);p.grad.div_(4.)
                    gradients=[p.grad for p in backend.candidate.parameters() if p.grad is not None]
                    if not gradients or any(not torch.isfinite(g).all() for g in gradients):raise RuntimeError("nonfinite global gradient")
                    unclipped=math.sqrt(sum(float(g.double().square().sum()) for g in gradients));torch.nn.utils.clip_grad_norm_(backend.candidate.parameters(),1.);clipped=math.sqrt(sum(float(g.double().square().sum()) for g in gradients))
                    if not math.isfinite(unclipped) or not math.isfinite(clipped):raise RuntimeError("nonfinite global grad norm")
                    backend.optimizer.step()
                except Exception:backend.optimizer.zero_grad(set_to_none=True);raise
            def checkpoint(state,best):
                item=backend.checkpoint(state.__dict__,best);saved["last"]=checkpoint_binding({**item,"best":None});
                if best:saved["best"]=checkpoint_binding(item)
            def terminal(payload):
                if saved["last"] is None:saved["last"]={"path":authorization["input_checkpoint_path"],"sha256":lineage.input_checkpoint_sha256,"size_bytes":Path(authorization["input_checkpoint_path"]).stat().st_size}
                best=saved["best"] or saved["last"];last=saved["last"];resources=backend.resources({"size_bytes":last["size_bytes"]},world);full=long_terminal(status=payload["status"],decision=decision,run_digest=lineage.run_digest,effective={"code":lineage.code_sha256,"config":lineage.config_sha256,"panels":lineage.panels_sha256,"basis":lineage.basis_sha256,"parent":lineage.parent_sha256},input_checkpoint={"path":authorization["input_checkpoint_path"],"sha256":lineage.input_checkpoint_sha256,"size_bytes":Path(authorization["input_checkpoint_path"]).stat().st_size},best=best,last=last,world_size=world,wall_s=payload["wall_s"],gpu_seconds=payload["gpu_seconds"],epoch=payload.get("best_epoch",-1),best_metric=payload.get("best_metric",math.inf),access_digest=canonical_sha(access),latency=resources["protocol_timing"],peak_vram=resources["peak_bytes"],gates={"budget_passed":payload["status"]=="completed_train"});atomic_json_exclusive(full,run_dir/"terminal.json")
            runner=V5LongRunner(world_size=world,rank=rank,train_records=train,calibration_records=cal,zero_grad=lambda:backend.optimizer.zero_grad(set_to_none=True),backward_record=lambda i,s:(lambda r:access.append(r["ledger_digest"]))(dispatcher.backward_only(i,s)),optimizer_step=step,set_lr=lambda lr:[g.update(lr=lr) for g in backend.optimizer.param_groups],evaluate=lambda i:score_indices(backend,[i])[0],gather=gather,checkpoint=checkpoint,terminal=terminal);return runner.run(resume_state)
        if mode in {"final-train-confirm","validation-once","test-once"}:
            validation=json.loads((OUT/"validation-once/terminal.json").read_text()) if mode=="test-once" else None;lock={"schema":"r16_dscp_v5_candidate_lock_v1","checkpoint_sha256":lineage.input_checkpoint_sha256,"code_sha256":lineage.code_sha256,"config_sha256":lineage.config_sha256,"basis_sha256":lineage.basis_sha256,"metadata_digest":metadata_digest,"thresholds_sha256":authorization["gates_sha256"]};return run_evaluation_stage(mode=mode,records=all_indices,evaluate_one=lambda i:score_indices(backend,[i])[0],resource_snapshot=lambda:backend.resources({"size_bytes":Path(authorization["input_checkpoint_path"]).stat().st_size},world),lineage=lineage,run_dir=run_dir,metadata_digest=metadata_digest,candidate_lock_payload=lock if mode=="final-train-confirm" else None,required_validation_terminal=validation)
        raise ValueError(mode)
    finally:backend.public_loader.close()

def static_evidence():
    cfg=yaml.safe_load(CONFIG.read_text())
    for name,command in cfg["commands"].items():
        if name not in {"prep","authorize","scale_decide"} and not command.startswith("env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="):raise BindingRefusal("CUDA command env drift")
    if not TEST_LOG.is_file() or "[100%]" not in TEST_LOG.read_text() or "failed" in TEST_LOG.read_text().lower():raise BindingRefusal("test evidence absent")
    v4=json.loads((ROOT/"results/r16_dscp_v4/static_evidence.json").read_text());return {"schema":"r16_dscp_v5_static_v1","parameters":1202,"ridge_parameters":480,"model_macs":45451125,"basis_conditions":v4["basis_conditions"],"router_census":v4["router_census"],"panels_census":v4["panels_census"],"metadata":metadata_seal(),"tests":{"status":"passed","log_sha256":sha256_file(TEST_LOG),"tail":TEST_LOG.read_text().splitlines()[-1]},"production_symbols":["V5ProductionBackend","V5ProductionDispatcher","V5ProductionWiring","V5LongRunner"],"v4_backend_reachable":False,"autocast":"cuda_bfloat16","finite_gates":True,"protocol_spool_io_excluded":True,"truth_read":False}

def prep():
    if OUT.exists() or PREREG.exists():raise FileExistsError("v5 target exists")
    bindings=effective_bindings();static=static_evidence();OUT.mkdir();atomic_json_exclusive(static,STATIC);artifact=basis_artifact();candidate=R16DSCP(artifact["basis"],artifact["coefficient_scales"]);optimizer=make_optimizer(list(candidate.parameters()));h=w=201;args=(torch.ones(1,1,h,w)*2000,torch.zeros(1,1,h,w),torch.ones(1,1,h,w),torch.arange(w),torch.arange(h),torch.zeros(1,1,h,w),torch.zeros(1,1,h,w),torch.zeros(1,401,h,w),torch.tensor([1]),torch.tensor([2]));truth=torch.ones(1,398,h,w);losses,adapted,_=actual_coefficient_loss(candidate,args,truth);bare=object.__new__(V5ProductionBackend);bare.candidate=candidate;bare.optimizer=optimizer;bare.device=torch.device("cpu");norms=bare._finite_backward(losses["total"],reset=True,step=True);identity={"candidate":CANDIDATE,"run_digest":canonical_sha(bindings),"code_sha256":bindings["engine"]["sha256"],"config_sha256":bindings["config"]["sha256"]};payload=checkpoint_payload(candidate,optimizer,run_identity=identity,sampler_order=list(range(12)),progress={"update":1});saved=save_best_last(payload,OUT/"actual_checkpoint",is_best=True);restored=R16DSCP(artifact["basis"],artifact["coefficient_scales"]);restored_optimizer=make_optimizer(list(restored.parameters()));loaded=load_checkpoint(saved["last"],restored,restored_optimizer,expected_run_identity=identity);resume=all(torch.equal(a,b) for a,b in zip(candidate.parameters(),restored.parameters())) and loaded["progress"]=={"update":1};ledger=__import__("saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3",fromlist=["AccessLedger"]).AccessLedger("train");spool=CandidateSpool(OUT,run_digest=identity["run_digest"],rank=0,owned_root="/dev/shm/r16_dscp_v5");seal=spool.seal("size",adapted,ledger);spool_max=seal["serialized_bytes"];spool.cleanup(seal,ledger);gate1=dual_space_gate(workspace=ROOT,tmpfs="/dev/shm",checkpoint_bytes=saved["size_bytes"],spool_max_bytes=spool_max,world_size=1);gate4=dual_space_gate(workspace=ROOT,tmpfs="/dev/shm",checkpoint_bytes=saved["size_bytes"],spool_max_bytes=spool_max,world_size=4)
    if not resume or not saved["hardlinked"] or not gate1["passed"]:raise BindingRefusal("prep checkpoint/space gate")
    pre={"schema":"r16_dscp_v5_preflight_v1","candidate":CANDIDATE,"status":"frozen","bindings":bindings,"run_identity":identity,"static":bind(STATIC),"checkpoint":{**saved,"sha256":sha256_file(saved["last"]),"resume_equal":resume,"finite_norms":norms,"state_sha256":model_state_digest(candidate)},"spool":{"root":"/dev/shm/r16_dscp_v5/<run_digest>/rank<r>","max_bytes":spool_max},"space1":gate1,"space4":gate4,"metadata":static["metadata"],"truth_reads":0,"GPU_used":False,"authorizations_created":False};atomic_json_exclusive(pre,PREFLIGHT);print(json.dumps(pre["checkpoint"]))

def verify_preflight():
    pre=json.loads(PREFLIGHT.read_text());current=effective_bindings()
    for key,value in pre["bindings"].items():
        if current[key]["sha256"]!=value["sha256"]:raise BindingRefusal(f"preflight drift {key}")
    return pre

def freeze():
    if PREREG.exists():raise FileExistsError("v5 prereg exists")
    pre=verify_preflight();cfg=yaml.safe_load(CONFIG.read_text());runtime=json.loads((ROOT/"results/r4e7_parent_e2e_runtime_train9_v1_20260825.json").read_text());gpu=subprocess.run(["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.total","--format=csv,noheader"],check=True,capture_output=True,text=True).stdout.splitlines();payload={"schema":"r16_dscp_v5_prereg_v1","candidate":CANDIDATE,"status":"frozen","preflight":bind(PREFLIGHT),"effective_bindings":effective_bindings(),"v3_veto":bind(ROOT/"results/r16_dscp_v3_preregistration_20260826.json"),"v4_veto":bind(ROOT/"results/r16_dscp_v4_preregistration_20260826.json"),"scientific_design":"unchanged; only BF16 finite timing smoke deadline and lineage execution fixes","config":cfg,"commands":cfg["commands"],"checkpoint":pre["checkpoint"],"spool":pre["spool"],"space1":pre["space1"],"space4":pre["space4"],"metadata":pre["metadata"],"tests":json.loads(STATIC.read_text())["tests"],"identities":{"python":platform.python_version(),"torch":torch.__version__,"torch_cuda":str(torch.version.cuda),"gpu":gpu,"cuda_used":False},"parent_runtime":{"mean_s":runtime["aggregate"]["arithmetic_mean_runtime_s"],"p95_s":runtime["aggregate"]["nearest_rank_p95_runtime_s"],"traditional_speedup":runtime["aggregate"]["reference_over_mean_speedup_x"]},"claim_boundary":"prep only, no stage/truth/GPU. Parent only 1.608x; immutable 10x gate remains and is currently impossible without independently measured latency change.","rollback":cfg["rollback"],"sealed":{"truth_read":False,"GPU_used":False,"authorizations_created":False,"workspace_fields":False}};atomic_json_exclusive(payload,PREREG);print(json.dumps(bind(PREREG)))

def authorize_stage(stage,output,input_checkpoint=None):
    if not PREREG.is_file():raise BindingRefusal("frozen prereg required")
    pre=verify_preflight();cfg=yaml.safe_load(CONFIG.read_text());chain={"smoke":[PREREG],"pilot":[OUT/"smoke/terminal.json"],"scale-probe-1":[OUT/"pilot/terminal.json"],"scale-probe-4":[OUT/"pilot/terminal.json"],"scale-decide":[OUT/"scale-probe-1/terminal.json",OUT/"scale-probe-4/terminal.json"],"long":[OUT/"pilot/terminal.json",OUT/"scale-decide/terminal.json"],"final-train-confirm":[OUT/"long/terminal.json"],"validation-once":[OUT/"final-train-confirm/terminal.json",OUT/"final-train-confirm/candidate_lock.json"],"test-once":[OUT/"validation-once/terminal.json",OUT/"final-train-confirm/candidate_lock.json"]};prereq={};expected=[];locked=None
    for path in chain[stage]:
        if not path.is_file():raise BindingRefusal("prerequisite absent")
        obj=json.loads(path.read_text());
        if "terminal" in path.name and obj.get("status") not in {"passed","completed_train"}:raise BindingRefusal("prerequisite failed")
        if isinstance(obj.get("best_checkpoint"),dict):
            best=obj["best_checkpoint"]
            if not best.get("path") or sha256_file(best["path"])!=best.get("sha256"):raise BindingRefusal("prerequisite best checkpoint drift")
            expected.append(best["path"])
        if isinstance(obj.get("checkpoint"),dict) and obj["checkpoint"].get("best"):expected.append(obj["checkpoint"]["best"])
        if path.name=="candidate_lock.json":locked=obj.get("checkpoint_sha256")
        prereq[str(path.resolve())]=sha256_file(path)
    input_sha="none";input_identity=pre["run_identity"];input_state=None
    if input_checkpoint:
        path=Path(input_checkpoint).resolve();input_sha=sha256_file(path)
        if expected and str(path) not in [str(Path(x).resolve()) for x in expected]:raise BindingRefusal("input is not prerequisite best")
        if locked and input_sha!=locked:raise BindingRefusal("input differs from lock")
        checkpoint=torch.load(path,map_location="cpu",weights_only=False);input_identity=checkpoint["run_identity"];artifact=basis_artifact();candidate=R16DSCP(artifact["basis"],artifact["coefficient_scales"]);optimizer=make_optimizer(list(candidate.parameters()));load_checkpoint(path,candidate,optimizer,expected_run_identity=input_identity);input_state=model_state_digest(candidate)
    elif stage not in {"smoke","scale-decide"}:raise BindingRefusal("input checkpoint required")
    line=Lineage(CANDIDATE,stage,pre["run_identity"]["run_digest"],sha256_file(ENGINE),sha256_file(CONFIG),PANELS_SHA256,BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PARENT_SHA256,input_sha);payload={"schema":AUTH_SCHEMA,"status":"authorized",**line.__dict__,"input_checkpoint_path":None if not input_checkpoint else str(Path(input_checkpoint).resolve()),"input_run_identity":input_identity,"input_state_sha256":input_state,"parent_sha256":PARENT_SHA256,"basis_sha256":BASIS_FILE_SHA256,"config_sha256":sha256_file(CONFIG),"code_sha256":sha256_file(ENGINE),"prerequisite_bindings":prereq,"gates_sha256":canonical_sha(cfg["gates"]),"script_sha256":sha256_file(SCRIPT),"prereg_sha256":sha256_file(PREREG)}
    if stage=="long":
        decision=OUT/"scale-decide/terminal.json";payload.update(scale_decision_path=str(decision.resolve()),scale_decision_sha256=sha256_file(decision),selected_world_size=json.loads(decision.read_text())["selected_gpus"])
    if stage in {"validation-once","test-once"}:payload["candidate_lock_sha256"]=sha256_file(OUT/"final-train-confirm/candidate_lock.json")
    if stage=="test-once":payload["validation_terminal_sha256"]=sha256_file(OUT/"validation-once/terminal.json")
    payload["authorization_digest"]=canonical_sha(payload);atomic_json_exclusive(payload,output)

def validate_auth(mode,path):
    pre=verify_preflight();auth=json.loads(Path(path).read_text());line=Lineage(CANDIDATE,mode,pre["run_identity"]["run_digest"],sha256_file(ENGINE),sha256_file(CONFIG),PANELS_SHA256,BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PARENT_SHA256,str(auth.get("input_checkpoint_sha256","none")));validate_authorization(auth,line)
    if auth["script_sha256"]!=sha256_file(SCRIPT) or auth["prereg_sha256"]!=sha256_file(PREREG):raise BindingRefusal("auth drift")
    verify_before_data(auth,input_checkpoint=auth.get("input_checkpoint_path"),parent=PARENT_PATH,basis=BASIS,config=CONFIG,code=ENGINE)
    for p,d in auth["prerequisite_bindings"].items():
        if not Path(p).is_file() or sha256_file(p)!=d:raise BindingRefusal("prerequisite drift")
    return auth,line

def checkpoint_loaded_state(path,run_identity):
    artifact=basis_artifact();candidate=R16DSCP(artifact["basis"],artifact["coefficient_scales"]);optimizer=make_optimizer(list(candidate.parameters()));load_checkpoint(path,candidate,optimizer,expected_run_identity=run_identity);return model_state_digest(candidate)

def decide_scale(auth,line,run_dir):
    one_path=OUT/"scale-probe-1/terminal.json";four_path=OUT/"scale-probe-4/terminal.json";one=json.loads(one_path.read_text());four=json.loads(four_path.read_text());decision=scale_decision(one,four);pre=verify_preflight();selected=tmpfs_constrained_world({"selected_gpus":decision["selected_gpus"]},pre["space1"],pre["space4"]);payload={"schema":"r16_dscp_v5_scale_decision_v1","status":"passed","selected_gpus":selected,"raw_gates":decision,"tmpfs_fallback":selected!=decision["selected_gpus"],"long_launched":False,"lineage":line.__dict__};atomic_json_exclusive(payload,run_dir/"terminal.json");return payload

def run_authorized(mode,auth_path,*,world_size=1,resume=False,preregistration=None,config=None,scale_path=None,input_checkpoint=None):
    auth,line=validate_auth(mode,auth_path);auth=dict(auth);auth["_file_sha256"]=sha256_file(auth_path);rank=int(os.environ.get("RANK","0"));world=int(os.environ.get("WORLD_SIZE",str(world_size)));distributed=world==4 and mode in {"scale-probe-4","long"};run_dir=OUT/mode
    if mode=="long":
        for observed,expected in ((preregistration,PREREG),(config,CONFIG),(scale_path,Path(auth["scale_decision_path"])),(input_checkpoint,Path(auth["input_checkpoint_path"]))):
            if observed is None or Path(observed).resolve()!=Path(expected).resolve():raise BindingRefusal("long CLI binding mismatch")
        if int(auth["selected_world_size"])!=world:raise BindingRefusal("long world mismatch")
    resume_state=LongResumeState()
    if resume:
        if mode!="long":raise BindingRefusal("resume long-only")
        effective={"engine_sha256":sha256_file(ENGINE),"script_sha256":sha256_file(SCRIPT),"config_sha256_effective":sha256_file(CONFIG),"panels_sha256_effective":PANELS_SHA256,"basis_sha256_effective":BASIS_FILE_SHA256,"parent_sha256_effective":PARENT_SHA256};payload=validate_resume_directory(run_dir,authorization_sha256=sha256_file(auth_path),effective_bindings=effective,load_last=lambda p:torch.load(p,map_location="cpu",weights_only=False));auth=dict(auth);auth["input_checkpoint_path"]=str(run_dir/"last.pt");auth["input_run_identity"]=payload["run_identity"];auth["input_state_sha256"]=checkpoint_loaded_state(run_dir/"last.pt",payload["run_identity"]);resume_state=LongResumeState(**{key:payload["progress"][key] for key in LongResumeState.__dataclass_fields__})
    if mode!="scale-decide" and os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise BindingRefusal("CUBLAS env mismatch")
    if distributed:torch.distributed.init_process_group("nccl");torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    try:
        if rank==0:
            if run_dir.exists() and not (resume and mode=="long"):raise FileExistsError("stage exists")
            if not run_dir.exists():run_dir.mkdir();atomic_json_exclusive({**line.__dict__,"authorization_sha256":sha256_file(auth_path),"engine_sha256":sha256_file(ENGINE),"config_sha256_effective":sha256_file(CONFIG)},run_dir/"run_identity.json")
        if distributed:torch.distributed.barrier()
        if mode=="scale-decide":return decide_scale(auth,line,run_dir)
        if mode=="scale-probe-4":
            try:result=dispatch_v5(mode,auth,line,run_dir,resume_state)
            except Exception as local_exc:result={"status":"failed","rank":rank,"world_size":world,"reason":str(local_exc)}
            gathered=[None]*world;torch.distributed.all_gather_object(gathered,result);fail=[r for r in gathered if r.get("status")!="passed"]
            if rank==0:
                if fail:result={"status":"failed","rank_failures":fail};atomic_json_exclusive(result,run_dir/"terminal.json")
                else:order=sorted({row["sample_id"] for report in gathered for row in report["rows"]});result=aggregate_scale_ranks(gathered,order);atomic_json_exclusive(result,run_dir/"terminal.json")
            torch.distributed.barrier()
            if fail:raise RuntimeError("scale rank failure")
        else:result=dispatch_v5(mode,auth,line,run_dir,resume_state)
        return result
    except Exception as exc:
        if rank==0 and run_dir.is_dir() and not (run_dir/"terminal.json").exists():atomic_json_exclusive({"schema":"r16_dscp_v5_terminal_v1","status":"failed","mode":mode,"reason":str(exc),"lineage":line.__dict__},run_dir/"terminal.json")
        raise
    finally:
        if distributed and torch.distributed.is_initialized():torch.distributed.destroy_process_group()

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--mode",required=True,choices=("prep","freeze-prereg","authorize-stage","smoke","pilot","scale-probe-1","scale-probe-4","scale-decide","long","final-train-confirm","validation-once","test-once"));parser.add_argument("--authorization");parser.add_argument("--stage");parser.add_argument("--output");parser.add_argument("--input-checkpoint");parser.add_argument("--world-size",type=int,default=1);parser.add_argument("--resume",action="store_true");parser.add_argument("--preregistration");parser.add_argument("--config");parser.add_argument("--scale-decision");args=parser.parse_args()
    if args.mode=="prep":prep()
    elif args.mode=="freeze-prereg":freeze()
    elif args.mode=="authorize-stage":authorize_stage(args.stage,args.output,args.input_checkpoint)
    else:
        if not args.authorization:raise BindingRefusal("authorization required")
        run_authorized(args.mode,args.authorization,world_size=args.world_size,resume=args.resume,preregistration=args.preregistration,config=args.config,scale_path=args.scale_decision,input_checkpoint=args.input_checkpoint)

if __name__=="__main__":main()

def smoke(dispatcher:V5StageDispatcher,records:Sequence[int],**callbacks:Any)->Mapping[str,Any]:
    return run_v5_smoke(records=records,update=dispatcher.update,**callbacks)

__all__=["V5ProductionDispatcher","V5ProductionWiring","V5StageDispatcher","backend_after_strict_verification","smoke","validate_final_from_long"]
