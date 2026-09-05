#!/usr/bin/env python3
"""V8 exception boundary and terminal-safe stage glue."""
from __future__ import annotations
import argparse,json,os,platform,subprocess,sys,time
from pathlib import Path
from typing import Any,Callable,Mapping
import torch,yaml
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import R16DSCP
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v7 import V7ProductionBackend
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v8 import V8ProductionBackend,failure_terminal
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v8 import complete_identity_v8,long_terminal_v8,run_v8_smoke
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v7 import checkpoint_terminal_binding,complete_checkpoint_identity,resume_v7_checkpoint,validate_checkpoint_identity
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v6 import run_v6_smoke,stage_memory_gate
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v5 import V5LongRunner,model_state_digest,verify_before_data,verify_loaded_state
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import AUTH_SCHEMA,Lineage,aggregate_scale_ranks,canonical_sha,run_evaluation_stage,run_pilot_stage,run_scale_probe,scale_decision,validate_authorization
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v4 import CandidateSpool,LongResumeState,V4GuardedOnsetLoader,actual_coefficient_loss,dual_space_gate,tmpfs_constrained_world
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.eikonal import grid_eikonal_travel_time
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PANELS_SHA256,PARENT_PATH,PARENT_SHA256,BindingRefusal,atomic_json_exclusive,checkpoint_payload,configure_determinism,load_checkpoint,make_optimizer,save_best_last,sha256_file
from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime

ROOT=Path(__file__).resolve().parents[1];CANDIDATE="r16_dscp_v8";SCRIPT=Path(__file__).resolve();ENGINE=ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v8.py";TEST=ROOT/"tests/saved_time_phase_operator_v4/test_r16_dscp_v8.py";CONFIG=ROOT/"configs/r16_dscp_v8.yaml";OUT=ROOT/"results/r16_dscp_v8";STATIC=OUT/"static_evidence.json";PREFLIGHT=OUT/"design_preflight.json";PREREG=ROOT/"results/r16_dscp_v8_preregistration_20260826.json";BASIS=ROOT/"results/r16_dscp_v1/basis_rank16.pt";PANELS=ROOT/"results/r16_dscp_v1/panels.json";TEST_LOG=Path("/tmp/r16_dscp_v8_tests.log")

class V8StageDispatcher:
    def __init__(self,backend:V8ProductionBackend):
        if not isinstance(backend,V8ProductionBackend):raise TypeError("v8 dispatcher requires V8ProductionBackend")
        self.backend=backend
    def preload(self,indices):
        for i in indices:self.backend.preload(i)
    def update(self,i):return self.backend.update_cached(i)
    def measure(self,i):return self.backend.measure_cached(i)
    def backward(self,i,s):return self.backend.backward_cached(i,s)
    def score(self,i,release=False):return self.backend.score_cached(i,release)
    def close(self):self.backend.cache.clear();self.backend.public_loader.close()

def execute_with_boundary(*,mode:str,run_dir:str|Path,authorization:Mapping[str,Any],lineage:Mapping[str,Any],effective:Mapping[str,Any],parent:Mapping[str,Any],factory:Callable[[],V8ProductionBackend],body:Callable[[V8ProductionBackend],Any],terminal_writer:Callable[[Mapping[str,Any]],None]|None=None,checkpoint_provider:Callable[[],Mapping[str,Any]|None]=lambda:None):
    root=Path(run_dir);writer=terminal_writer or (lambda payload:atomic_json_exclusive(payload,root/"terminal.json"));backend=None
    try:
        backend=factory()
        if not isinstance(backend,V8ProductionBackend):raise TypeError("factory returned foreign backend")
        return body(backend)
    except Exception as exc:
        if backend is not None:
            try:backend.optimizer.zero_grad(set_to_none=True);backend.preserve_failure_spool();backend.cache.clear()
            except Exception:pass
        payload=failure_terminal(mode=mode,error=exc,lineage=lineage,authorization=authorization,effective=effective,parent=parent,checkpoint=checkpoint_provider())
        try:writer(payload)
        except Exception as terminal_exc:
            sys.stderr.write(f"v8 terminal write failed: {type(terminal_exc).__name__}: {terminal_exc}\n")
        raise
    finally:
        if backend is not None:
            try:backend.cache.clear();backend.public_loader.close()
            except Exception:pass

def validate_long_cli(*,arg_world_size:int,actual_world_size:int,cuda_visible:str,authorization:Mapping[str,Any],scale_decision:Mapping[str,Any],resume:bool,terminal_exists:bool)->None:
    selected=int(scale_decision.get("selected_gpus",0))
    if int(arg_world_size) not in {1,4} or int(arg_world_size)!=int(actual_world_size) or selected!=int(actual_world_size) or int(authorization.get("selected_world_size",0))!=selected:raise ValueError("long world-size/decision/authorization mismatch")
    if len([x for x in str(cuda_visible).split(",") if x])!=selected:raise ValueError("CUDA_VISIBLE_DEVICES mismatch")
    if resume and terminal_exists:raise ValueError("resume refuses any existing terminal")

def bind(path):p=Path(path).resolve();return {"path":str(p),"sha256":sha256_file(p),"size_bytes":p.stat().st_size}
def effective_bindings():return {"engine":bind(ENGINE),"script":bind(SCRIPT),"test":bind(TEST),"config":bind(CONFIG),"model":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp.py"),"v8_transitive_v7":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v7.py"),"v8_transitive_v6":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v6.py"),"harness":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_training_v2.py"),"basis":bind(BASIS),"panels":bind(PANELS),"parent":bind(PARENT_PATH),"manifest":bind(parent_runtime.MANIFEST_PATH),"normalization":bind(parent_runtime.NORMALIZATION_PATH),"v2_replay":bind(ROOT/"results/r16_dscp_v2/replay_basis_verify/verification.json"),"v3_veto":bind(ROOT/"results/r16_dscp_v3_preregistration_20260826.json"),"v4_veto":bind(ROOT/"results/r16_dscp_v4_preregistration_20260826.json"),"v5_veto":bind(ROOT/"results/r16_dscp_v5_preregistration_20260826.json"),"v6_blocker":bind(ROOT/"results/r16_dscp_v6/design_preflight.json"),"v7_prereg":bind(ROOT/"results/r16_dscp_v7_preregistration_20260826.json")}
def basis_artifact():
    if sha256_file(BASIS)!=BASIS_FILE_SHA256:raise BindingRefusal("basis drift")
    return torch.load(BASIS,map_location="cpu",weights_only=False)
def metadata_seal():
    m=parent_runtime.load_manifest_payload();out={};sets={}
    for split in ("validation","test_id"):
        rows=sorted([{"source_index":r["source_index"],"sample_id":r["sample_id"],"group_id":r["group_id"],"sample_sha256":r["sample_sha256"],"family":r["medium_type"]} for r in m["records"] if r["split"]==split],key=lambda r:(r["source_index"],r["sample_id"]));sets[split]={k:{r[k] for r in rows} for k in ("source_index","sample_id","group_id","sample_sha256")};out[split]={"count":len(rows),"ordered_digest":canonical_sha(rows),"by_family":{f:sum(r["family"]==f for r in rows) for f in ("uniform","layered","marmousi")},"wavefield_read":False}
    out["zero_overlap"]={k:not bool(sets["validation"][k]&sets["test_id"][k]) for k in sets["validation"]};return out

def build_v8_backend(mode,authorization,lineage,run_dir,identity,resume_checkpoint=None,expected_resume_state_sha=None):
    split="validation" if mode=="validation-once" else "test_id" if mode=="test-once" else "train";device=torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK','0'))}");configure_determinism(372);artifact=basis_artifact();candidate=R16DSCP(artifact["basis"],artifact["coefficient_scales"]).to(device);optimizer=make_optimizer(list(candidate.parameters()))
    if resume_checkpoint is not None:load_checkpoint(resume_checkpoint,candidate,optimizer,expected_run_identity=identity)
    elif lineage.input_checkpoint_sha256!="none":load_checkpoint(authorization["input_checkpoint_path"],candidate,optimizer,expected_run_identity=authorization["input_run_identity"])
    if expected_resume_state_sha is not None:authorization={**authorization,"input_state_sha256":expected_resume_state_sha}
    verify_loaded_state(candidate,authorization);parent,normalizer,manifest_payload,_=parent_runtime.load_model_context(device);manifest=parent_runtime.manifest_object(manifest_payload);panels=json.loads(PANELS.read_text())["records"];roles={"smoke":("smoke",),"pilot":("pilot_fit","pilot_confirm"),"scale-probe-1":("pilot_fit",),"scale-probe-4":("pilot_fit",),"long":("long_fit","long_calibration"),"final-train-confirm":("final_train_confirm",)}
    if split=="train":rows=[r for r in panels if r["role"] in roles[mode]];samples=[r["sample_id"] for r in rows];family={r["sample_id"]:r["family"] for r in rows};nontruth={r["sample_id"]:r["nontruth_input_sha256"] for r in rows}
    else:rows=sorted([r for r in manifest_payload["records"] if r["split"]==split],key=lambda r:(r["source_index"],r["sample_id"]));samples=[r["sample_id"] for r in rows];family={r["sample_id"]:r["medium_type"] for r in rows};nontruth={r["sample_id"]:r["sample_sha256"] for r in rows}
    loader=V4GuardedOnsetLoader(GuardedOnsetDataset(parent_runtime.SOURCE_H5_PATH,manifest,split=split,sample_ids=samples))
    @torch.inference_mode()
    def parent_predictor(public):
        v=public.velocity_mps[None,None].to(device);s=public.source_parameters[None].to(device);sm=public.source_map[None,None].to(device);medium=parent.encode_medium(v,normalizer);prepared=parent.prepare_sources(medium,s,sm,normalizer,record_to_medium=torch.zeros(1,dtype=torch.long,device=device));normalized=parent.dense_normalized(prepared,public.time_s.to(device),x_m=public.x_m.to(device),z_m=public.z_m.to(device),time_block=16);return normalizer.decode_pressure(normalized.float(),s[:,4])[0].cpu()
    travel=lambda v,s:grid_eikonal_travel_time(v,source_indices=[(round(float(s[1])/10),round(float(s[0])/10))],dx_m=10,dz_m=10)[0];backend=V8ProductionBackend(public_loader=loader,parent_predictor=parent_predictor,travel_builder=travel,candidate=candidate,optimizer=optimizer,authorization=authorization,lineage=lineage,source_h5=parent_runtime.SOURCE_H5_PATH,run_dir=run_dir,device=device,split=split,family_by_sample=family,once_token=run_dir/f"{split}.once.json" if split!="train" else None,cache_role=mode,parent_sha256=PARENT_SHA256,basis_sha256=BASIS_FILE_SHA256,feature_code_sha256=sha256_file(ENGINE),nontruth_by_sample=nontruth,complete_identity=identity);indices={role:[samples.index(r["sample_id"]) for r in rows if r.get("role")==role] for role in roles.get(mode,())};return backend,indices,list(range(len(loader))),canonical_sha(rows)

def dispatch_v8(mode,authorization,lineage,run_dir,identity,state):
    backend,roles,all_indices,metadata=build_v8_backend(mode,authorization,lineage,run_dir,identity);dispatcher=V8StageDispatcher(backend);world=int(os.environ.get("WORLD_SIZE","1"));rank=int(os.environ.get("RANK","0"));effective={"engine":identity["engine_sha256"],"script":identity["script_sha256"],"config":identity["config_sha256_effective"],"panels":identity["panels_sha256_effective"],"basis":identity["basis_sha256_effective"],"parent":identity["parent_sha256_effective"]};parent={"sha256":PARENT_SHA256}
    if mode=="smoke":return run_v8_smoke(backend=backend,records=roles["smoke"],quick_gate=lambda rows:all(r["aggregate_rel_l2"]<=r["parent_rel_l2"] for r in rows),checkpoint=lambda:backend.checkpoint({"mode":mode}),resource_snapshot=lambda saved:backend.resources(saved,world),terminal=lambda payload:atomic_json_exclusive(payload,run_dir/"terminal.json"),lineage=identity,effective=effective,parent=parent)
    if mode=="pilot":
        dispatcher.preload= lambda indices:[backend.preload(i) for i in indices];dispatcher.update=lambda i:backend.update_cached(i);dispatcher.preload(roles["pilot_fit"]+roles["pilot_confirm"])
        def ridge_update(i):p=backend.cached_prepare(i,"ridge");backend.ridge_update(p,backend._cached_truth(p))
        return run_pilot_stage(fit_records=roles["pilot_fit"],confirm_records=roles["pilot_confirm"],candidate_update=dispatcher.update,ridge_update=ridge_update,ridge_solve=backend.ridge_finalize,candidate_score=lambda rows:[backend.score_cached(i) for i in rows],ridge_score=lambda ridge,rows:[backend.ridge_score_cached(i,ridge) for i in rows],checkpoint=lambda:backend.checkpoint({"mode":mode}),lineage=lineage,run_dir=run_dir,resource_snapshot=lambda saved:backend.resources(saved,world),terminal_extra=lambda:{"checkpoint_identity_digest":identity["identity_digest"]})
    if mode in {"scale-probe-1","scale-probe-4"}:
        records=roles["pilot_fit"][:12];[backend.preload(i) for i in records[rank::world]];return run_scale_probe(mode=mode,records=records,measure=lambda i:backend.measure_cached(i),model_hash=lambda:model_state_digest(backend.candidate),resource_snapshot=lambda:backend.resources({"size_bytes":Path(authorization["input_checkpoint_path"]).stat().st_size},world),lineage=lineage,run_dir=run_dir,world_size=world,rank=rank)
    if mode=="long":
        train=roles["long_fit"];cal=roles["long_calibration"];started=time.monotonic();[backend.preload(i) for i in train+cal];preload=time.monotonic()-started;saved={"best":None,"last":None}
        def gather(rows):
            if world==1:return rows
            shards=[None]*world;torch.distributed.all_gather_object(shards,rows);return [x for s in shards for x in s]
        def step():
            if world==4:
                for p in backend.candidate.parameters():
                    if p.grad is not None:torch.distributed.all_reduce(p.grad);p.grad.div_(4.)
            torch.nn.utils.clip_grad_norm_(backend.candidate.parameters(),1.);backend.optimizer.step()
        def cp(progress,best):item=backend.checkpoint(progress.__dict__,best);saved["last"]=item; saved.__setitem__("best",item if best else saved["best"])
        def terminal(payload):
            item=saved["best"] or saved["last"];best=None if item is None else {"path":item["best"] or item["last"],"sha256":sha256_file(item["best"] or item["last"]),"size_bytes":item["size_bytes"]};last=None if saved["last"] is None else {"path":saved["last"]["last"],"sha256":sha256_file(saved["last"]["last"]),"size_bytes":saved["last"]["size_bytes"]};full=long_terminal_v8(status=payload["status"],lineage=identity,effective=effective,parent=parent,decision=json.loads(Path(authorization["scale_decision_path"]).read_text()),input_checkpoint={"path":authorization["input_checkpoint_path"],"sha256":lineage.input_checkpoint_sha256},best=best,last=last,world_size=world,wall_s=payload["wall_s"],gpu_seconds=payload["gpu_seconds"],epoch=payload.get("best_epoch",-1),best_metric=payload.get("best_metric",float("inf")),access={},cache=backend.cache.payload(),latency=backend.protocol.payload(),peak_vram=backend.peak_bytes,gates={"budget":payload["status"]=="completed_train"});atomic_json_exclusive(full,run_dir/"terminal.json")
        runner=V5LongRunner(world_size=world,rank=rank,train_records=train,calibration_records=cal,zero_grad=lambda:backend.optimizer.zero_grad(set_to_none=True),backward_record=lambda i,s:backend.backward_cached(i,s),optimizer_step=step,set_lr=lambda lr:[g.update(lr=lr) for g in backend.optimizer.param_groups],evaluate=lambda i:backend.score_cached(i),gather=gather,checkpoint=cp,terminal=terminal,clock=lambda:time.monotonic()-preload);return runner.run(state)
    if mode in {"final-train-confirm","validation-once","test-once"}:
        def evaluate(i):backend.preload(i);return backend.score_cached(i,True)
        validation=json.loads((OUT/"validation-once/terminal.json").read_text()) if mode=="test-once" else None;lock={"schema":"r16_dscp_v8_lock_v1","checkpoint_sha256":lineage.input_checkpoint_sha256,"metadata_digest":metadata};return run_evaluation_stage(mode=mode,records=all_indices,evaluate_one=evaluate,resource_snapshot=lambda:backend.resources({"size_bytes":Path(authorization["input_checkpoint_path"]).stat().st_size},world),lineage=lineage,run_dir=run_dir,metadata_digest=metadata,candidate_lock_payload=lock if mode=="final-train-confirm" else None,required_validation_terminal=validation)
    raise ValueError(mode)

def static_evidence():
    cfg=yaml.safe_load(CONFIG.read_text());
    if not TEST_LOG.is_file() or "[100%]" not in TEST_LOG.read_text() or "failed" in TEST_LOG.read_text().lower():raise BindingRefusal("tests absent")
    for name,command in cfg["commands"].items():
        if name not in {"prep","authorize","scale_decide"} and not command.startswith("env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="):raise BindingRefusal("command drift")
    v7=json.loads((ROOT/"results/r16_dscp_v7/static_evidence.json").read_text());return {"schema":"r16_dscp_v8_static_v1","parameters":1202,"ridge_parameters":480,"model_macs":45451125,"basis_conditions":v7["basis_conditions"],"router_census":v7["router_census"],"panels_census":v7["panels_census"],"host_cache_gates":v7["host_cache_gates"],"metadata":metadata_seal(),"tests":{"status":"passed","log_sha256":sha256_file(TEST_LOG),"tail":TEST_LOG.read_text().splitlines()[-1]},"production_backend":"V8ProductionBackend","terminal_schemas":["r16_dscp_v8_smoke_terminal_v1","r16_dscp_v8_long_terminal_v1"],"outer_exception_boundary":True,"truth_read":False}

def prep():
    if OUT.exists() or PREREG.exists():raise FileExistsError("v8 exists")
    bindings=effective_bindings();static=static_evidence();OUT.mkdir();atomic_json_exclusive(static,STATIC);artifact=basis_artifact();model=R16DSCP(artifact["basis"],artifact["coefficient_scales"]);optimizer=make_optimizer(list(model.parameters()));h=w=201;args=(torch.ones(1,1,h,w)*2000,torch.zeros(1,1,h,w),torch.ones(1,1,h,w),torch.arange(w),torch.arange(h),torch.zeros(1,1,h,w),torch.zeros(1,1,h,w),torch.zeros(1,401,h,w),torch.tensor([1]),torch.tensor([2]));losses,adapted,_=actual_coefficient_loss(model,args,torch.ones(1,398,h,w));losses["total"].backward();optimizer.step();identity=complete_identity_v8(mode="prep",run_digest=canonical_sha(bindings),authorization_sha256="prep-none",engine_sha256=bindings["engine"]["sha256"],script_sha256=bindings["script"]["sha256"],config_sha256_effective=bindings["config"]["sha256"],panels_sha256_effective=PANELS_SHA256,basis_sha256_effective=BASIS_FILE_SHA256,parent_sha256_effective=PARENT_SHA256,input_checkpoint_sha256="none");state=LongResumeState(epoch=0,global_step=1,group_index=0,best=float("inf"),bad_epochs=0,best_epoch=-1);payload=checkpoint_payload(model,optimizer,run_identity=identity,sampler_order=list(range(12)),progress=state.__dict__);saved=save_best_last(payload,OUT/"actual_checkpoint",is_best=True);restored=R16DSCP(artifact["basis"],artifact["coefficient_scales"]);ro=make_optimizer(list(restored.parameters()));_,resume_state=resume_v7_checkpoint(saved["last"],restored,ro,expected_identity=identity);resume=resume_state==state and all(torch.equal(a,b) for a,b in zip(model.parameters(),restored.parameters()));ledger=__import__("saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3",fromlist=["AccessLedger"]).AccessLedger("train");spool=CandidateSpool(OUT,run_digest=identity["run_digest"],rank=0,owned_root="/dev/shm/r16_dscp_v8");seal=spool.seal("size",adapted,ledger);spool_max=seal["serialized_bytes"];spool.cleanup(seal,ledger);gate1=dual_space_gate(workspace=ROOT,tmpfs="/dev/shm",checkpoint_bytes=saved["size_bytes"],spool_max_bytes=spool_max,world_size=1);gate4=dual_space_gate(workspace=ROOT,tmpfs="/dev/shm",checkpoint_bytes=saved["size_bytes"],spool_max_bytes=spool_max,world_size=4)
    if not resume or not saved["hardlinked"] or not gate1["passed"]:raise BindingRefusal("prep gate")
    pre={"schema":"r16_dscp_v8_preflight_v1","candidate":CANDIDATE,"status":"frozen","bindings":bindings,"run_identity":identity,"static":bind(STATIC),"checkpoint":{**saved,"sha256":sha256_file(saved["last"]),"resume_equal":resume,"identity_digest":identity["identity_digest"],"state_sha256":model_state_digest(model)},"host_cache_gates":static["host_cache_gates"],"spool":{"max_bytes":spool_max,"root":"/dev/shm/r16_dscp_v8/<run_digest>/rank<r>"},"space1":gate1,"space4":gate4,"metadata":static["metadata"],"truth_reads":0,"GPU_used":False,"auth_created":False};atomic_json_exclusive(pre,PREFLIGHT);print(json.dumps(pre["checkpoint"]))

def verify_preflight():
    pre=json.loads(PREFLIGHT.read_text());current=effective_bindings()
    for key,value in pre["bindings"].items():
        if current[key]["sha256"]!=value["sha256"]:raise BindingRefusal(f"drift {key}")
    return pre

def freeze():
    if PREREG.exists():raise FileExistsError("v8 prereg exists")
    pre=verify_preflight();cfg=yaml.safe_load(CONFIG.read_text());runtime=json.loads((ROOT/"results/r4e7_parent_e2e_runtime_train9_v1_20260825.json").read_text());gpu=subprocess.run(["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.total","--format=csv,noheader"],check=True,capture_output=True,text=True).stdout.splitlines();payload={"schema":"r16_dscp_v8_prereg_v1","candidate":CANDIDATE,"status":"frozen","preflight":bind(PREFLIGHT),"effective_bindings":effective_bindings(),"v7_prereg":bind(ROOT/"results/r16_dscp_v7_preregistration_20260826.json"),"only_change":"terminal exception long CLI contract","config":cfg,"commands":cfg["commands"],"checkpoint":pre["checkpoint"],"host_cache_gates":pre["host_cache_gates"],"spool":pre["spool"],"space1":pre["space1"],"space4":pre["space4"],"metadata":pre["metadata"],"tests":json.loads(STATIC.read_text())["tests"],"identities":{"python":platform.python_version(),"torch":torch.__version__,"torch_cuda":str(torch.version.cuda),"gpu":gpu,"cuda_used":False},"parent_runtime":{"mean_s":runtime["aggregate"]["arithmetic_mean_runtime_s"],"traditional_speedup":runtime["aggregate"]["reference_over_mean_speedup_x"]},"claim_boundary":"prep only; no stage/truth/GPU; immutable 10x remains unmet","rollback":"v8 atomic failure boundary","sealed":{"truth_read":False,"GPU_used":False,"auth_created":False}};atomic_json_exclusive(payload,PREREG);print(json.dumps(bind(PREREG)))

def authorize_stage(stage,output,input_checkpoint=None):
    if not PREREG.is_file():raise BindingRefusal("prereg required")
    pre=verify_preflight();chain={"smoke":[PREREG],"pilot":[OUT/"smoke/terminal.json"],"scale-probe-1":[OUT/"pilot/terminal.json"],"scale-probe-4":[OUT/"pilot/terminal.json"],"scale-decide":[OUT/"scale-probe-1/terminal.json",OUT/"scale-probe-4/terminal.json"],"long":[OUT/"pilot/terminal.json",OUT/"scale-decide/terminal.json"],"final-train-confirm":[OUT/"long/terminal.json"],"validation-once":[OUT/"final-train-confirm/terminal.json"],"test-once":[OUT/"validation-once/terminal.json"]};prereq={}
    for path in chain[stage]:
        if not path.is_file():raise BindingRefusal("prerequisite absent")
        obj=json.loads(path.read_text());
        if "terminal" in path.name and obj.get("status") not in {"passed","completed_train"}:raise BindingRefusal("prerequisite failed")
        prereq[str(path.resolve())]=sha256_file(path)
    input_sha="none";input_identity=pre["run_identity"]
    if input_checkpoint:path=Path(input_checkpoint).resolve();input_sha=sha256_file(path);input_identity=torch.load(path,map_location="cpu",weights_only=False)["run_identity"];validate_checkpoint_identity(input_identity)
    elif stage not in {"smoke","scale-decide"}:raise BindingRefusal("input required")
    line=Lineage(CANDIDATE,stage,pre["run_identity"]["run_digest"],sha256_file(ENGINE),sha256_file(CONFIG),PANELS_SHA256,BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PARENT_SHA256,input_sha);auth={"schema":AUTH_SCHEMA,"status":"authorized",**line.__dict__,"input_checkpoint_path":None if not input_checkpoint else str(path),"input_run_identity":input_identity,"parent_sha256":PARENT_SHA256,"basis_sha256":BASIS_FILE_SHA256,"config_sha256":sha256_file(CONFIG),"code_sha256":sha256_file(ENGINE),"prerequisite_bindings":prereq,"gates_sha256":canonical_sha(yaml.safe_load(CONFIG.read_text())),"script_sha256":sha256_file(SCRIPT),"prereg_sha256":sha256_file(PREREG)};auth["authorization_digest"]=canonical_sha(auth);atomic_json_exclusive(auth,output)

def run_authorized(mode,auth_path,world_size=1,resume=False):
    pre=verify_preflight();auth=json.loads(Path(auth_path).read_text());line=Lineage(CANDIDATE,mode,pre["run_identity"]["run_digest"],sha256_file(ENGINE),sha256_file(CONFIG),PANELS_SHA256,BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PARENT_SHA256,str(auth.get("input_checkpoint_sha256","none")));validate_authorization(auth,line);verify_before_data(auth,input_checkpoint=auth.get("input_checkpoint_path"),parent=PARENT_PATH,basis=BASIS,config=CONFIG,code=ENGINE);run=OUT/mode;actual=int(os.environ.get("WORLD_SIZE",str(world_size)));decision=json.loads((OUT/"scale-decide/terminal.json").read_text()) if mode=="long" else {"selected_gpus":actual}
    if mode=="long":validate_long_cli(arg_world_size=world_size,actual_world_size=actual,cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES",""),authorization=auth,scale_decision=decision,resume=resume,terminal_exists=(run/"terminal.json").exists())
    identity=complete_identity_v8(mode=mode,run_digest=line.run_digest,authorization_sha256=sha256_file(auth_path),engine_sha256=sha256_file(ENGINE),script_sha256=sha256_file(SCRIPT),config_sha256_effective=sha256_file(CONFIG),panels_sha256_effective=PANELS_SHA256,basis_sha256_effective=BASIS_FILE_SHA256,parent_sha256_effective=PARENT_SHA256,input_checkpoint_sha256=line.input_checkpoint_sha256);state=LongResumeState();resume_path=None;resume_state_sha=None
    if resume:
        resume_path=run/"last.pt";raw=torch.load(resume_path,map_location="cpu",weights_only=False);validate_checkpoint_identity(raw["run_identity"],identity);artifact=basis_artifact();model=R16DSCP(artifact["basis"],artifact["coefficient_scales"]);optimizer=make_optimizer(list(model.parameters()));_,state=resume_v7_checkpoint(resume_path,model,optimizer,expected_identity=identity);resume_state_sha=model_state_digest(model)
    if run.exists() and not resume:raise FileExistsError("stage exists")
    if not run.exists():run.mkdir();atomic_json_exclusive(identity,run/"run_identity.json")
    if mode=="scale-decide":
        one=json.loads((OUT/"scale-probe-1/terminal.json").read_text());four=json.loads((OUT/"scale-probe-4/terminal.json").read_text());result=scale_decision(one,four);payload={"schema":"r16_dscp_v8_scale_decision_v1","candidate":CANDIDATE,"mode":mode,"status":"passed","selected_gpus":result["selected_gpus"],"gates":result,"long_launched":False,"lineage":identity};atomic_json_exclusive(payload,run/"terminal.json");return payload
    effective={"engine":identity["engine_sha256"],"script":identity["script_sha256"],"config":identity["config_sha256_effective"],"panels":identity["panels_sha256_effective"],"basis":identity["basis_sha256_effective"],"parent":identity["parent_sha256_effective"]};parent={"sha256":PARENT_SHA256};holder={}
    return execute_with_boundary(mode=mode,run_dir=run,authorization=auth,lineage=identity,effective=effective,parent=parent,factory=lambda:(holder.setdefault("backend",build_v8_backend(mode,auth,line,run,identity,resume_checkpoint=resume_path,expected_resume_state_sha=resume_state_sha))),body=lambda backend:dispatch_v8(mode,auth,line,run,identity,state),checkpoint_provider=lambda:None)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--mode",required=True,choices=("prep","freeze-prereg","authorize-stage","smoke","pilot","scale-probe-1","scale-probe-4","scale-decide","long","final-train-confirm","validation-once","test-once"));parser.add_argument("--stage");parser.add_argument("--output");parser.add_argument("--input-checkpoint");parser.add_argument("--authorization");parser.add_argument("--world-size",type=int,choices=(1,4),default=1);parser.add_argument("--resume",action="store_true");args=parser.parse_args()
    if args.mode=="prep":prep()
    elif args.mode=="freeze-prereg":freeze()
    elif args.mode=="authorize-stage":authorize_stage(args.stage,args.output,args.input_checkpoint)
    else:run_authorized(args.mode,args.authorization,args.world_size,args.resume)
if __name__=="__main__":main()

__all__=["V8StageDispatcher","execute_with_boundary","validate_long_cli"]
