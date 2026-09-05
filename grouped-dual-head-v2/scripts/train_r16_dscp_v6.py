#!/usr/bin/env python3
"""V6 bounded cache/identity production glue; no stage CLI yet."""
from __future__ import annotations
import argparse,json,os,platform,subprocess
from pathlib import Path
from typing import Any,Mapping,Sequence
import torch,yaml
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import R16DSCP
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v6 import V6ProductionBackend,complete_run_identity,run_v6_smoke,stage_memory_gate,tensor_bytes
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v5 import V5LongRunner,long_terminal,model_state_digest,verify_before_data,verify_loaded_state
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import AUTH_SCHEMA,Lineage,aggregate_scale_ranks,canonical_sha,run_evaluation_stage,run_pilot_stage,run_scale_probe,scale_decision,validate_authorization
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v4 import CandidateSpool,LongResumeState,V4GuardedOnsetLoader,actual_coefficient_loss,dual_space_gate,tmpfs_constrained_world,validate_resume_directory
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.eikonal import grid_eikonal_travel_time
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PANELS_SHA256,PARENT_PATH,PARENT_SHA256,BindingRefusal,atomic_json_exclusive,checkpoint_payload,configure_determinism,load_checkpoint,make_optimizer,save_best_last,sha256_file
from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime

ROOT=Path(__file__).resolve().parents[1];CANDIDATE="r16_dscp_v6";SCRIPT=Path(__file__).resolve();ENGINE=ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v6.py";TEST=ROOT/"tests/saved_time_phase_operator_v4/test_r16_dscp_v6.py";CONFIG=ROOT/"configs/r16_dscp_v6.yaml";OUT=ROOT/"results/r16_dscp_v6";STATIC=OUT/"static_evidence.json";PREFLIGHT=OUT/"design_preflight.json";PREREG=ROOT/"results/r16_dscp_v6_preregistration_20260826.json";BASIS=ROOT/"results/r16_dscp_v1/basis_rank16.pt";PANELS=ROOT/"results/r16_dscp_v1/panels.json";TEST_LOG=Path("/tmp/r16_dscp_v6_tests.log")

def bind(path):p=Path(path).resolve();return {"path":str(p),"sha256":sha256_file(p),"size_bytes":p.stat().st_size}
def effective_bindings():return {"engine":bind(ENGINE),"script":bind(SCRIPT),"test":bind(TEST),"config":bind(CONFIG),"model":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp.py"),"v6_transitive_v5":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v5.py"),"v6_transitive_v4":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v4.py"),"harness":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_training_v2.py"),"basis":bind(BASIS),"panels":bind(PANELS),"parent":bind(PARENT_PATH),"manifest":bind(parent_runtime.MANIFEST_PATH),"normalization":bind(parent_runtime.NORMALIZATION_PATH),"v2_replay":bind(ROOT/"results/r16_dscp_v2/replay_basis_verify/verification.json"),"v3_veto":bind(ROOT/"results/r16_dscp_v3_preregistration_20260826.json"),"v4_veto":bind(ROOT/"results/r16_dscp_v4_preregistration_20260826.json"),"v5_veto":bind(ROOT/"results/r16_dscp_v5_preregistration_20260826.json")}
def basis_artifact():
 if sha256_file(BASIS)!=BASIS_FILE_SHA256:raise BindingRefusal("basis drift")
 return torch.load(BASIS,map_location="cpu",weights_only=False)
def metadata_seal():
 m=parent_runtime.load_manifest_payload();out={};sets={}
 for split in ("validation","test_id"):
  rows=sorted([{"source_index":r["source_index"],"sample_id":r["sample_id"],"group_id":r["group_id"],"sample_sha256":r["sample_sha256"],"family":r["medium_type"]} for r in m["records"] if r["split"]==split],key=lambda r:(r["source_index"],r["sample_id"]));sets[split]={k:{r[k] for r in rows} for k in ("source_index","sample_id","group_id","sample_sha256")};out[split]={"count":len(rows),"ordered_digest":canonical_sha(rows),"by_family":{f:sum(r["family"]==f for r in rows) for f in ("uniform","layered","marmousi")},"wavefield_read":False}
 out["zero_overlap"]={k:not bool(sets["validation"][k]&sets["test_id"][k]) for k in sets["validation"]};return out

def build_v6_backend(mode,authorization,lineage,run_dir):
 split="validation" if mode=="validation-once" else "test_id" if mode=="test-once" else "train";device=torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK','0'))}");configure_determinism(372);artifact=basis_artifact();candidate=R16DSCP(artifact["basis"],artifact["coefficient_scales"]).to(device);optimizer=make_optimizer(list(candidate.parameters()))
 if lineage.input_checkpoint_sha256!="none":load_checkpoint(authorization["input_checkpoint_path"],candidate,optimizer,expected_run_identity=authorization["input_run_identity"])
 verify_loaded_state(candidate,authorization);parent,normalizer,manifest_payload,_=parent_runtime.load_model_context(device);manifest=parent_runtime.manifest_object(manifest_payload);panels=json.loads(PANELS.read_text())["records"];roles={"smoke":("smoke",),"pilot":("pilot_fit","pilot_confirm"),"scale-probe-1":("pilot_fit",),"scale-probe-4":("pilot_fit",),"long":("long_fit","long_calibration"),"final-train-confirm":("final_train_confirm",)}
 if split=="train":rows=[r for r in panels if r["role"] in roles[mode]];samples=[r["sample_id"] for r in rows];family={r["sample_id"]:r["family"] for r in rows};nontruth={r["sample_id"]:r["nontruth_input_sha256"] for r in rows}
 else:rows=sorted([r for r in manifest_payload["records"] if r["split"]==split],key=lambda r:(r["source_index"],r["sample_id"]));samples=[r["sample_id"] for r in rows];family={r["sample_id"]:r["medium_type"] for r in rows};nontruth={r["sample_id"]:r["sample_sha256"] for r in rows}
 loader=V4GuardedOnsetLoader(GuardedOnsetDataset(parent_runtime.SOURCE_H5_PATH,manifest,split=split,sample_ids=samples))
 @torch.inference_mode()
 def parent_predictor(public):
  v=public.velocity_mps[None,None].to(device);s=public.source_parameters[None].to(device);sm=public.source_map[None,None].to(device);medium=parent.encode_medium(v,normalizer);prepared=parent.prepare_sources(medium,s,sm,normalizer,record_to_medium=torch.zeros(1,dtype=torch.long,device=device));normalized=parent.dense_normalized(prepared,public.time_s.to(device),x_m=public.x_m.to(device),z_m=public.z_m.to(device),time_block=16);return normalizer.decode_pressure(normalized.float(),s[:,4])[0].cpu()
 travel=lambda v,s:grid_eikonal_travel_time(v,source_indices=[(round(float(s[1])/10),round(float(s[0])/10))],dx_m=10,dz_m=10)[0];backend=V6ProductionBackend(public_loader=loader,parent_predictor=parent_predictor,travel_builder=travel,candidate=candidate,optimizer=optimizer,authorization=authorization,lineage=lineage,source_h5=parent_runtime.SOURCE_H5_PATH,run_dir=run_dir,device=device,split=split,family_by_sample=family,once_token=run_dir/f"{split}.once.json" if split!="train" else None,cache_role=mode,parent_sha256=PARENT_SHA256,basis_sha256=BASIS_FILE_SHA256,feature_code_sha256=sha256_file(ENGINE),nontruth_by_sample=nontruth);indices={role:[samples.index(r["sample_id"]) for r in rows if r.get("role")==role] for role in roles.get(mode,())};return backend,indices,list(range(len(loader))),canonical_sha(rows)

class V6StageDispatcher:
    def __init__(self,backend:V6ProductionBackend):
        if not isinstance(backend,V6ProductionBackend):raise TypeError("v6 dispatcher requires V6ProductionBackend")
        self.backend=backend
    def preload(self,indices:Sequence[int])->None:
        for index in indices:self.backend.preload(index)
    def update(self,index:int)->Mapping[str,Any]:return self.backend.update_cached(index)
    def measure(self,index:int)->Mapping[str,Any]:return self.backend.measure_cached(index)
    def backward(self,index:int,scale:float)->Mapping[str,Any]:return self.backend.backward_cached(index,scale)
    def score(self,index:int,release:bool=False)->Mapping[str,Any]:return self.backend.score_cached(index,release)
    def close(self)->None:self.backend.cache.clear();self.backend.public_loader.close()

def smoke(dispatcher:V6StageDispatcher,records:Sequence[int],**kwargs:Any)->Mapping[str,Any]:return run_v6_smoke(backend=dispatcher.backend,records=records,**kwargs)

def run_identity(**kwargs:Any)->Mapping[str,Any]:return complete_run_identity(**kwargs)

def _scores(backend,indices,release=False):return [backend.score_cached(i,release) for i in indices]
def dispatch_v6(mode,authorization,lineage,run_dir,resume_state=LongResumeState()):
 backend,roles,all_indices,metadata=build_v6_backend(mode,authorization,lineage,run_dir);world=int(os.environ.get("WORLD_SIZE","1"));rank=int(os.environ.get("RANK","0"));dispatcher=V6StageDispatcher(backend)
 try:
  if mode=="smoke":return run_v6_smoke(backend=backend,records=roles["smoke"],quick_gate=lambda rows:all(r["aggregate_rel_l2"]<=r["parent_rel_l2"] for r in rows),checkpoint=lambda:backend.checkpoint({"mode":mode}),resource_snapshot=lambda saved:backend.resources(saved,world),terminal=lambda payload:atomic_json_exclusive(payload,run_dir/"terminal.json"))
  if mode=="pilot":
   dispatcher.preload(roles["pilot_fit"]+roles["pilot_confirm"])
   def ridge_update(i):p=backend.cached_prepare(i,"ridge");backend.ridge_update(p,backend._cached_truth(p))
   return run_pilot_stage(fit_records=roles["pilot_fit"],confirm_records=roles["pilot_confirm"],candidate_update=dispatcher.update,ridge_update=ridge_update,ridge_solve=backend.ridge_finalize,candidate_score=lambda rows:_scores(backend,list(rows)),ridge_score=lambda ridge,rows:[backend.ridge_score_cached(i,ridge) for i in rows],checkpoint=lambda:backend.checkpoint({"mode":mode}),lineage=lineage,run_dir=run_dir,resource_snapshot=lambda saved:backend.resources(saved,world))
  if mode in {"scale-probe-1","scale-probe-4"}:
   records=roles["pilot_fit"][:12];assigned=records[rank::world];dispatcher.preload(assigned);return run_scale_probe(mode=mode,records=records,measure=dispatcher.measure,model_hash=lambda:model_state_digest(backend.candidate),resource_snapshot=lambda:backend.resources({"size_bytes":Path(authorization["input_checkpoint_path"]).stat().st_size},world),lineage=lineage,run_dir=run_dir,world_size=world,rank=rank)
  if mode=="long":
   train=roles["long_fit"];cal=roles["long_calibration"];preload_started=__import__("time").monotonic();dispatcher.preload(train+cal);preload_elapsed=__import__("time").monotonic()-preload_started;saved={"best":None,"last":None};access=[]
   def gather(rows):
    if world==1:return rows
    shards=[None]*world;torch.distributed.all_gather_object(shards,rows);return [x for shard in shards for x in shard]
   def step():
    if world==4:
     for p in backend.candidate.parameters():
      if p.grad is not None:torch.distributed.all_reduce(p.grad);p.grad.div_(4.)
    torch.nn.utils.clip_grad_norm_(backend.candidate.parameters(),1.);backend.optimizer.step()
   def binding(item):
    path=Path(item["best"] or item["last"]);return {"path":str(path),"sha256":sha256_file(path),"size_bytes":path.stat().st_size}
   def cp(state,best):item=backend.checkpoint(state.__dict__,best);saved["last"]=binding({**item,"best":None}); saved.__setitem__("best",binding(item) if best else saved["best"])
   def terminal(payload):
    fallback={"path":authorization["input_checkpoint_path"],"sha256":lineage.input_checkpoint_sha256,"size_bytes":Path(authorization["input_checkpoint_path"]).stat().st_size};last=saved["last"] or fallback;best=saved["best"] or last;resources=backend.resources({"size_bytes":last["size_bytes"]},world);full={"schema":"r16_dscp_v6_long_terminal_v1","candidate":CANDIDATE,"mode":"long","status":payload["status"],"decision":json.loads(Path(authorization["scale_decision_path"]).read_text()),"run_digest":lineage.run_digest,"effective":{"engine":lineage.code_sha256,"script":sha256_file(SCRIPT),"config":lineage.config_sha256,"panels":lineage.panels_sha256,"basis":lineage.basis_sha256,"parent":lineage.parent_sha256},"input_checkpoint":{"path":authorization["input_checkpoint_path"],"sha256":lineage.input_checkpoint_sha256},"best_checkpoint":best,"last_checkpoint":last,"world_size":world,"wall_s":payload["wall_s"],"gpu_seconds":payload["gpu_seconds"],"gpu_hours":payload["gpu_seconds"]/3600.,"global_step":payload.get("global_step",0),"best_metric":payload.get("best_metric",float("inf")),"access_ledger_digest":canonical_sha(access),"cache":backend.cache.payload(),"latency":resources["protocol_timing"],"peak_vram_bytes":resources["peak_bytes"],"gates":{"budget_passed":payload["status"]=="completed_train"}};atomic_json_exclusive(full,run_dir/"terminal.json")
   runner=V5LongRunner(world_size=world,rank=rank,train_records=train,calibration_records=cal,zero_grad=lambda:backend.optimizer.zero_grad(set_to_none=True),backward_record=lambda i,s:(lambda r:access.append(r["ledger_digest"]))(dispatcher.backward(i,s)),optimizer_step=step,set_lr=lambda lr:[g.update(lr=lr) for g in backend.optimizer.param_groups],evaluate=lambda i:dispatcher.score(i),gather=gather,checkpoint=cp,terminal=terminal,clock=lambda:__import__("time").monotonic()-preload_elapsed);return runner.run(resume_state)
  if mode in {"final-train-confirm","validation-once","test-once"}:
   def evaluate(i):backend.preload(i);return dispatcher.score(i,True)
   validation=json.loads((OUT/"validation-once/terminal.json").read_text()) if mode=="test-once" else None;lock={"schema":"r16_dscp_v6_lock_v1","checkpoint_sha256":lineage.input_checkpoint_sha256,"metadata_digest":metadata};return run_evaluation_stage(mode=mode,records=all_indices,evaluate_one=evaluate,resource_snapshot=lambda:backend.resources({"size_bytes":Path(authorization["input_checkpoint_path"]).stat().st_size},world),lineage=lineage,run_dir=run_dir,metadata_digest=metadata,candidate_lock_payload=lock if mode=="final-train-confirm" else None,required_validation_terminal=validation)
  raise ValueError(mode)
 finally:dispatcher.close()

def estimated_record_bytes()->int:return 1241*201*201*4+5*4
def static_evidence():
 cfg=yaml.safe_load(CONFIG.read_text())
 for name,command in cfg["commands"].items():
  if name not in {"prep","authorize","scale_decide"} and not command.startswith("env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="):raise BindingRefusal("CUDA command drift")
 if not TEST_LOG.is_file() or "[100%]" not in TEST_LOG.read_text() or "failed" in TEST_LOG.read_text().lower():raise BindingRefusal("tests absent")
 roles={role:stage_memory_gate(role,estimated_record_bytes()) for role in ("smoke","pilot","scale-probe-1","scale-probe-4","long","final-train-confirm","validation-once","test-once")}
 if not all(g["passed"] for g in roles.values()):raise BindingRefusal("host cache memory gate")
 v5=json.loads((ROOT/"results/r16_dscp_v5/static_evidence.json").read_text());return {"schema":"r16_dscp_v6_static_v1","parameters":1202,"ridge_parameters":480,"model_macs":45451125,"basis_conditions":v5["basis_conditions"],"router_census":v5["router_census"],"panels_census":v5["panels_census"],"metadata":metadata_seal(),"record_cache_bytes_estimate":estimated_record_bytes(),"host_cache_gates":roles,"tests":{"status":"passed","log_sha256":sha256_file(TEST_LOG),"tail":TEST_LOG.read_text().splitlines()[-1]},"production_symbols":["V6RecordCache","V6ProductionBackend","V6StageDispatcher"],"v5_backend_reachable":False,"truth_read":False}

def prep():
 if OUT.exists() or PREREG.exists():raise FileExistsError("v6 exists")
 bindings=effective_bindings();static=static_evidence();OUT.mkdir();atomic_json_exclusive(static,STATIC);artifact=basis_artifact();candidate=R16DSCP(artifact["basis"],artifact["coefficient_scales"]);optimizer=make_optimizer(list(candidate.parameters()));h=w=201;args=(torch.ones(1,1,h,w)*2000,torch.zeros(1,1,h,w),torch.ones(1,1,h,w),torch.arange(w),torch.arange(h),torch.zeros(1,1,h,w),torch.zeros(1,1,h,w),torch.zeros(1,401,h,w),torch.tensor([1]),torch.tensor([2]));losses,adapted,_=actual_coefficient_loss(candidate,args,torch.ones(1,398,h,w));losses["total"].backward();optimizer.step();identity=complete_run_identity({"candidate":CANDIDATE,"run_digest":canonical_sha(bindings)},authorization_sha256="prep-none",engine_sha256=bindings["engine"]["sha256"],script_sha256=bindings["script"]["sha256"],config_sha256=bindings["config"]["sha256"],panels_sha256=PANELS_SHA256,basis_sha256=BASIS_FILE_SHA256,parent_sha256=PARENT_SHA256,input_checkpoint_sha256="none");payload=checkpoint_payload(candidate,optimizer,run_identity=identity,sampler_order=list(range(12)),progress={"update":1});saved=save_best_last(payload,OUT/"actual_checkpoint",is_best=True);restored=R16DSCP(artifact["basis"],artifact["coefficient_scales"]);ro=make_optimizer(list(restored.parameters()));loaded=load_checkpoint(saved["last"],restored,ro,expected_run_identity=identity);resume=all(torch.equal(a,b) for a,b in zip(candidate.parameters(),restored.parameters()));ledger=__import__("saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3",fromlist=["AccessLedger"]).AccessLedger("train");spool=CandidateSpool(OUT,run_digest=identity["run_digest"],rank=0,owned_root="/dev/shm/r16_dscp_v6");seal=spool.seal("size",adapted,ledger);spool_max=seal["serialized_bytes"];spool.cleanup(seal,ledger);gate1=dual_space_gate(workspace=ROOT,tmpfs="/dev/shm",checkpoint_bytes=saved["size_bytes"],spool_max_bytes=spool_max,world_size=1);gate4=dual_space_gate(workspace=ROOT,tmpfs="/dev/shm",checkpoint_bytes=saved["size_bytes"],spool_max_bytes=spool_max,world_size=4)
 if not resume or not saved["hardlinked"] or not gate1["passed"]:raise BindingRefusal("prep gate")
 pre={"schema":"r16_dscp_v6_preflight_v1","candidate":CANDIDATE,"status":"frozen","bindings":bindings,"run_identity":identity,"static":bind(STATIC),"checkpoint":{**saved,"sha256":sha256_file(saved["last"]),"resume_equal":resume,"state_sha256":model_state_digest(candidate)},"record_cache_bytes":estimated_record_bytes(),"host_cache_gates":static["host_cache_gates"],"spool":{"max_bytes":spool_max,"root":"/dev/shm/r16_dscp_v6/<run_digest>/rank<r>"},"space1":gate1,"space4":gate4,"metadata":static["metadata"],"truth_reads":0,"GPU_used":False,"auth_created":False};atomic_json_exclusive(pre,PREFLIGHT);print(json.dumps(pre["checkpoint"]))

def verify_preflight():
 pre=json.loads(PREFLIGHT.read_text());current=effective_bindings()
 for key,value in pre["bindings"].items():
  if current[key]["sha256"]!=value["sha256"]:raise BindingRefusal(f"drift {key}")
 return pre

def freeze():
 if PREREG.exists():raise FileExistsError("v6 prereg exists")
 pre=verify_preflight();cfg=yaml.safe_load(CONFIG.read_text());runtime=json.loads((ROOT/"results/r4e7_parent_e2e_runtime_train9_v1_20260825.json").read_text());gpu=subprocess.run(["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.total","--format=csv,noheader"],check=True,capture_output=True,text=True).stdout.splitlines();payload={"schema":"r16_dscp_v6_prereg_v1","candidate":CANDIDATE,"status":"frozen","preflight":bind(PREFLIGHT),"effective_bindings":effective_bindings(),"v3_veto":bind(ROOT/"results/r16_dscp_v3_preregistration_20260826.json"),"v4_veto":bind(ROOT/"results/r16_dscp_v4_preregistration_20260826.json"),"v5_veto":bind(ROOT/"results/r16_dscp_v5_preregistration_20260826.json"),"scientific_design":"unchanged; only RAM cache complete identity and terminal fixes","config":cfg,"commands":cfg["commands"],"checkpoint":pre["checkpoint"],"record_cache_bytes":pre["record_cache_bytes"],"host_cache_gates":pre["host_cache_gates"],"spool":pre["spool"],"space1":pre["space1"],"space4":pre["space4"],"metadata":pre["metadata"],"tests":json.loads(STATIC.read_text())["tests"],"identities":{"python":platform.python_version(),"torch":torch.__version__,"torch_cuda":str(torch.version.cuda),"gpu":gpu,"cuda_used":False},"parent_runtime":{"mean_s":runtime["aggregate"]["arithmetic_mean_runtime_s"],"p95_s":runtime["aggregate"]["nearest_rank_p95_runtime_s"],"traditional_speedup":runtime["aggregate"]["reference_over_mean_speedup_x"]},"claim_boundary":"prep only; no stage/truth/GPU. Parent 1.608x; immutable 10x remains unmet unless independent latency changes.","rollback":cfg["rollback"],"sealed":{"truth_read":False,"GPU_used":False,"auth_created":False,"cache_serialized":False}};atomic_json_exclusive(payload,PREREG);print(json.dumps(bind(PREREG)))

def authorize_stage(stage,output,input_checkpoint=None):
 if not PREREG.is_file():raise BindingRefusal("prereg required")
 pre=verify_preflight();cfg=yaml.safe_load(CONFIG.read_text());chain={"smoke":[PREREG],"pilot":[OUT/"smoke/terminal.json"],"scale-probe-1":[OUT/"pilot/terminal.json"],"scale-probe-4":[OUT/"pilot/terminal.json"],"scale-decide":[OUT/"scale-probe-1/terminal.json",OUT/"scale-probe-4/terminal.json"],"long":[OUT/"pilot/terminal.json",OUT/"scale-decide/terminal.json"],"final-train-confirm":[OUT/"long/terminal.json"],"validation-once":[OUT/"final-train-confirm/terminal.json",OUT/"final-train-confirm/candidate_lock.json"],"test-once":[OUT/"validation-once/terminal.json",OUT/"final-train-confirm/candidate_lock.json"]};prereq={};expected=[];locked=None
 for path in chain[stage]:
  if not path.is_file():raise BindingRefusal("prerequisite absent")
  obj=json.loads(path.read_text());
  if "terminal" in path.name and obj.get("status") not in {"passed","completed_train"}:raise BindingRefusal("prerequisite failed")
  if isinstance(obj.get("best_checkpoint"),dict):expected.append(obj["best_checkpoint"]["path"])
  checkpoint=obj.get("checkpoint",{})
  if isinstance(checkpoint,dict) and checkpoint.get("best"):
   if not Path(checkpoint["best"]).is_file():raise BindingRefusal("best checkpoint absent")
   expected.append(checkpoint["best"])
  if path.name=="candidate_lock.json":locked=obj.get("checkpoint_sha256")
  prereq[str(path.resolve())]=sha256_file(path)
 input_sha="none";input_identity=pre["run_identity"];input_state=None
 if input_checkpoint:
  path=Path(input_checkpoint).resolve();input_sha=sha256_file(path)
  if expected and str(path) not in [str(Path(x).resolve()) for x in expected]:raise BindingRefusal("not best")
  if locked and input_sha!=locked:raise BindingRefusal("input differs from lock")
  payload=torch.load(path,map_location="cpu",weights_only=False);input_identity=payload["run_identity"];artifact=basis_artifact();model=R16DSCP(artifact["basis"],artifact["coefficient_scales"]);opt=make_optimizer(list(model.parameters()));load_checkpoint(path,model,opt,expected_run_identity=input_identity);input_state=model_state_digest(model)
 elif stage not in {"smoke","scale-decide"}:raise BindingRefusal("input required")
 line=Lineage(CANDIDATE,stage,pre["run_identity"]["run_digest"],sha256_file(ENGINE),sha256_file(CONFIG),PANELS_SHA256,BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PARENT_SHA256,input_sha);auth={"schema":AUTH_SCHEMA,"status":"authorized",**line.__dict__,"input_checkpoint_path":None if not input_checkpoint else str(Path(input_checkpoint).resolve()),"input_run_identity":input_identity,"input_state_sha256":input_state,"parent_sha256":PARENT_SHA256,"basis_sha256":BASIS_FILE_SHA256,"config_sha256":sha256_file(CONFIG),"code_sha256":sha256_file(ENGINE),"prerequisite_bindings":prereq,"gates_sha256":canonical_sha(cfg["gates"]),"script_sha256":sha256_file(SCRIPT),"prereg_sha256":sha256_file(PREREG),"record_cache_bytes":pre["record_cache_bytes"]}
 if stage.startswith("scale-probe"):auth["pilot_terminal_sha256"]=sha256_file(OUT/"pilot/terminal.json")
 if stage=="scale-decide":auth.update(scale1_terminal_sha256=sha256_file(OUT/"scale-probe-1/terminal.json"),scale4_terminal_sha256=sha256_file(OUT/"scale-probe-4/terminal.json"))
 if stage=="long":
  decision=OUT/"scale-decide/terminal.json";auth.update(scale_decision_path=str(decision.resolve()),scale_decision_sha256=sha256_file(decision),selected_world_size=json.loads(decision.read_text())["selected_gpus"])
 if stage in {"validation-once","test-once"}:auth["candidate_lock_sha256"]=sha256_file(OUT/"final-train-confirm/candidate_lock.json")
 if stage=="test-once":auth["validation_terminal_sha256"]=sha256_file(OUT/"validation-once/terminal.json")
 auth["authorization_digest"]=canonical_sha(auth);atomic_json_exclusive(auth,output)

def run_authorized(mode,auth_path,world_size=1,resume=False):
 pre=verify_preflight();auth=json.loads(Path(auth_path).read_text());line=Lineage(CANDIDATE,mode,pre["run_identity"]["run_digest"],sha256_file(ENGINE),sha256_file(CONFIG),PANELS_SHA256,BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PARENT_SHA256,str(auth.get("input_checkpoint_sha256","none")));validate_authorization(auth,line)
 if auth.get("script_sha256")!=sha256_file(SCRIPT) or auth.get("prereg_sha256")!=sha256_file(PREREG) or auth.get("gates_sha256")!=canonical_sha(yaml.safe_load(CONFIG.read_text())["gates"]):raise BindingRefusal("authorization effective drift")
 for path,digest in auth.get("prerequisite_bindings",{}).items():
  if not Path(path).is_file() or sha256_file(path)!=digest:raise BindingRefusal("prerequisite drift")
 verify_before_data(auth,input_checkpoint=auth.get("input_checkpoint_path"),parent=PARENT_PATH,basis=BASIS,config=CONFIG,code=ENGINE);run=OUT/mode;rank=int(os.environ.get("RANK","0"));world=int(os.environ.get("WORLD_SIZE",str(world_size)));distributed=world==4 and mode in {"scale-probe-4","long"}
 if mode=="scale-decide":
  if run.exists():raise FileExistsError("stage exists")
  run.mkdir();one=json.loads((OUT/"scale-probe-1/terminal.json").read_text());four=json.loads((OUT/"scale-probe-4/terminal.json").read_text());decision=scale_decision(one,four);selected=tmpfs_constrained_world({"selected_gpus":decision["selected_gpus"]},pre["space1"],pre["space4"]);payload={"schema":"r16_dscp_v6_scale_decision_v1","status":"passed","selected_gpus":selected,"raw_gates":decision,"long_launched":False};atomic_json_exclusive(payload,run/"terminal.json");return payload
 gate=stage_memory_gate(mode,int(auth["record_cache_bytes"]));
 if not gate["passed"]:raise BindingRefusal("host cache gate")
 if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise BindingRefusal("CUBLAS")
 if distributed:torch.distributed.init_process_group("nccl");torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
 identity=complete_run_identity(line.__dict__,authorization_sha256=sha256_file(auth_path),engine_sha256=sha256_file(ENGINE),script_sha256=sha256_file(SCRIPT),config_sha256=sha256_file(CONFIG),panels_sha256=PANELS_SHA256,basis_sha256=BASIS_FILE_SHA256,parent_sha256=PARENT_SHA256,input_checkpoint_sha256=line.input_checkpoint_sha256);resume_state=LongResumeState()
 if resume:
  if mode!="long":raise BindingRefusal("resume long-only")
  effective={key:identity[key] for key in ("engine_sha256","script_sha256","config_sha256_effective","panels_sha256_effective","basis_sha256_effective","parent_sha256_effective")};payload=validate_resume_directory(run,authorization_sha256=sha256_file(auth_path),effective_bindings=effective,load_last=lambda p:torch.load(p,map_location="cpu",weights_only=False));auth=dict(auth);auth["input_checkpoint_path"]=str(run/"last.pt");auth["input_run_identity"]=payload["run_identity"];artifact=basis_artifact();candidate=R16DSCP(artifact["basis"],artifact["coefficient_scales"]);optimizer=make_optimizer(list(candidate.parameters()));load_checkpoint(run/"last.pt",candidate,optimizer,expected_run_identity=payload["run_identity"]);auth["input_state_sha256"]=model_state_digest(candidate);resume_state=LongResumeState(**{key:payload["progress"][key] for key in LongResumeState.__dataclass_fields__})
 try:
  if rank==0:
   if run.exists() and not (resume and mode=="long"):raise FileExistsError("stage exists")
   if not run.exists():run.mkdir();atomic_json_exclusive(identity,run/"run_identity.json")
  if distributed:torch.distributed.barrier()
  result=dispatch_v6(mode,auth,line,run,resume_state)
  if mode=="scale-probe-4":
   gathered=[None]*world;torch.distributed.all_gather_object(gathered,result);failures=[x for x in gathered if x.get("status")!="passed"]
   if rank==0:
    if failures:result={"status":"failed","rank_failures":failures};atomic_json_exclusive(result,run/"terminal.json")
    else:order=sorted({row["sample_id"] for report in gathered for row in report["rows"]});result=aggregate_scale_ranks(gathered,order);atomic_json_exclusive(result,run/"terminal.json")
   torch.distributed.barrier()
   if failures:raise RuntimeError("scale rank failure")
  return result
 except Exception as exc:
  if rank==0 and run.is_dir() and not (run/"terminal.json").exists():atomic_json_exclusive({"schema":"r16_dscp_v6_terminal_v1","status":"failed","mode":mode,"reason":str(exc),"run_identity":identity},run/"terminal.json")
  raise
 finally:
  if distributed and torch.distributed.is_initialized():torch.distributed.destroy_process_group()

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--mode",required=True,choices=("prep","freeze-prereg","authorize-stage","smoke","pilot","scale-probe-1","scale-probe-4","scale-decide","long","final-train-confirm","validation-once","test-once"));parser.add_argument("--authorization");parser.add_argument("--stage");parser.add_argument("--output");parser.add_argument("--input-checkpoint");parser.add_argument("--world-size",type=int,default=1);parser.add_argument("--resume",action="store_true");args=parser.parse_args()
 if args.mode=="prep":prep()
 elif args.mode=="freeze-prereg":freeze()
 elif args.mode=="authorize-stage":authorize_stage(args.stage,args.output,args.input_checkpoint)
 else:run_authorized(args.mode,args.authorization,args.world_size,args.resume)
if __name__=="__main__":main()

__all__=["V6StageDispatcher","run_identity","smoke"]
