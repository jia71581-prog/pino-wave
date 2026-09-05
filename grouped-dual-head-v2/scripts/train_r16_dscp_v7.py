#!/usr/bin/env python3
"""V7 complete-identity production glue; final CLI is added after tests."""
from __future__ import annotations
import argparse,json,os,platform,subprocess,time
from pathlib import Path
from typing import Any,Mapping
import torch,yaml
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import R16DSCP
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v7 import V7ProductionBackend,validate_checkpoint_identity
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v7 import complete_checkpoint_identity,resume_v7_checkpoint,checkpoint_terminal_binding
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v6 import complete_run_identity,run_v6_smoke,stage_memory_gate
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v5 import V5LongRunner,model_state_digest,verify_before_data,verify_loaded_state
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import AUTH_SCHEMA,Lineage,aggregate_scale_ranks,canonical_sha,run_evaluation_stage,run_pilot_stage,run_scale_probe,scale_decision,validate_authorization
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v4 import CandidateSpool,LongResumeState,V4GuardedOnsetLoader,actual_coefficient_loss,dual_space_gate,tmpfs_constrained_world,validate_resume_directory
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.eikonal import grid_eikonal_travel_time
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PANELS_SHA256,PARENT_PATH,PARENT_SHA256,BindingRefusal,atomic_json_exclusive,checkpoint_payload,configure_determinism,load_checkpoint,make_optimizer,save_best_last,sha256_file
from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime

ROOT=Path(__file__).resolve().parents[1];CANDIDATE="r16_dscp_v7";SCRIPT=Path(__file__).resolve();ENGINE=ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v7.py";TEST=ROOT/"tests/saved_time_phase_operator_v4/test_r16_dscp_v7.py";CONFIG=ROOT/"configs/r16_dscp_v7.yaml";OUT=ROOT/"results/r16_dscp_v7";STATIC=OUT/"static_evidence.json";PREFLIGHT=OUT/"design_preflight.json";PREREG=ROOT/"results/r16_dscp_v7_preregistration_20260826.json";BASIS=ROOT/"results/r16_dscp_v1/basis_rank16.pt";PANELS=ROOT/"results/r16_dscp_v1/panels.json";TEST_LOG=Path("/tmp/r16_dscp_v7_tests.log")

def bind(path):p=Path(path).resolve();return {"path":str(p),"sha256":sha256_file(p),"size_bytes":p.stat().st_size}
def effective_bindings():return {"engine":bind(ENGINE),"script":bind(SCRIPT),"test":bind(TEST),"config":bind(CONFIG),"model":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp.py"),"v7_transitive_v6":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v6.py"),"v7_transitive_v5":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v5.py"),"harness":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_training_v2.py"),"basis":bind(BASIS),"panels":bind(PANELS),"parent":bind(PARENT_PATH),"manifest":bind(parent_runtime.MANIFEST_PATH),"normalization":bind(parent_runtime.NORMALIZATION_PATH),"v2_replay":bind(ROOT/"results/r16_dscp_v2/replay_basis_verify/verification.json"),"v3_veto":bind(ROOT/"results/r16_dscp_v3_preregistration_20260826.json"),"v4_veto":bind(ROOT/"results/r16_dscp_v4_preregistration_20260826.json"),"v5_veto":bind(ROOT/"results/r16_dscp_v5_preregistration_20260826.json"),"v6_blocker":bind(ROOT/"results/r16_dscp_v6/design_preflight.json")}
def basis_artifact():
 if sha256_file(BASIS)!=BASIS_FILE_SHA256:raise BindingRefusal("basis drift")
 return torch.load(BASIS,map_location="cpu",weights_only=False)
def metadata_seal():
 m=parent_runtime.load_manifest_payload();out={};sets={}
 for split in ("validation","test_id"):
  rows=sorted([{"source_index":r["source_index"],"sample_id":r["sample_id"],"group_id":r["group_id"],"sample_sha256":r["sample_sha256"],"family":r["medium_type"]} for r in m["records"] if r["split"]==split],key=lambda r:(r["source_index"],r["sample_id"]));sets[split]={k:{r[k] for r in rows} for k in ("source_index","sample_id","group_id","sample_sha256")};out[split]={"count":len(rows),"ordered_digest":canonical_sha(rows),"by_family":{f:sum(r["family"]==f for r in rows) for f in ("uniform","layered","marmousi")},"wavefield_read":False}
 out["zero_overlap"]={k:not bool(sets["validation"][k]&sets["test_id"][k]) for k in sets["validation"]};return out

def build_v7_backend(mode,authorization,lineage,run_dir,identity):
 split="validation" if mode=="validation-once" else "test_id" if mode=="test-once" else "train";device=torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK','0'))}");configure_determinism(372);artifact=basis_artifact();candidate=R16DSCP(artifact["basis"],artifact["coefficient_scales"]).to(device);optimizer=make_optimizer(list(candidate.parameters()))
 if lineage.input_checkpoint_sha256!="none":load_checkpoint(authorization["input_checkpoint_path"],candidate,optimizer,expected_run_identity=authorization["input_run_identity"])
 verify_loaded_state(candidate,authorization);parent,normalizer,manifest_payload,_=parent_runtime.load_model_context(device);manifest=parent_runtime.manifest_object(manifest_payload);panels=json.loads(PANELS.read_text())["records"];roles={"smoke":("smoke",),"pilot":("pilot_fit","pilot_confirm"),"scale-probe-1":("pilot_fit",),"scale-probe-4":("pilot_fit",),"long":("long_fit","long_calibration"),"final-train-confirm":("final_train_confirm",)}
 if split=="train":rows=[r for r in panels if r["role"] in roles[mode]];samples=[r["sample_id"] for r in rows];family={r["sample_id"]:r["family"] for r in rows};nontruth={r["sample_id"]:r["nontruth_input_sha256"] for r in rows}
 else:rows=sorted([r for r in manifest_payload["records"] if r["split"]==split],key=lambda r:(r["source_index"],r["sample_id"]));samples=[r["sample_id"] for r in rows];family={r["sample_id"]:r["medium_type"] for r in rows};nontruth={r["sample_id"]:r["sample_sha256"] for r in rows}
 loader=V4GuardedOnsetLoader(GuardedOnsetDataset(parent_runtime.SOURCE_H5_PATH,manifest,split=split,sample_ids=samples))
 @torch.inference_mode()
 def parent_predictor(public):
  v=public.velocity_mps[None,None].to(device);s=public.source_parameters[None].to(device);sm=public.source_map[None,None].to(device);medium=parent.encode_medium(v,normalizer);prepared=parent.prepare_sources(medium,s,sm,normalizer,record_to_medium=torch.zeros(1,dtype=torch.long,device=device));normalized=parent.dense_normalized(prepared,public.time_s.to(device),x_m=public.x_m.to(device),z_m=public.z_m.to(device),time_block=16);return normalizer.decode_pressure(normalized.float(),s[:,4])[0].cpu()
 travel=lambda v,s:grid_eikonal_travel_time(v,source_indices=[(round(float(s[1])/10),round(float(s[0])/10))],dx_m=10,dz_m=10)[0];backend=V7ProductionBackend(public_loader=loader,parent_predictor=parent_predictor,travel_builder=travel,candidate=candidate,optimizer=optimizer,authorization=authorization,lineage=lineage,source_h5=parent_runtime.SOURCE_H5_PATH,run_dir=run_dir,device=device,split=split,family_by_sample=family,once_token=run_dir/f"{split}.once.json" if split!="train" else None,cache_role=mode,parent_sha256=PARENT_SHA256,basis_sha256=BASIS_FILE_SHA256,feature_code_sha256=sha256_file(ENGINE),nontruth_by_sample=nontruth,complete_identity=identity);indices={role:[samples.index(r["sample_id"]) for r in rows if r.get("role")==role] for role in roles.get(mode,())};return backend,indices,list(range(len(loader))),canonical_sha(rows)

class V7StageDispatcher:
    def __init__(self,backend:V7ProductionBackend):
        if not isinstance(backend,V7ProductionBackend):raise TypeError("v7 dispatcher requires V7ProductionBackend")
        validate_checkpoint_identity(backend.checkpoint_identity);self.backend=backend
    def checkpoint(self,progress:Mapping[str,Any],best:bool=True):return self.backend.checkpoint(progress,best)
    def preload(self,indices):
        for index in indices:self.backend.preload(index)
    def update(self,index):return self.backend.update_cached(index)
    def measure(self,index):return self.backend.measure_cached(index)
    def backward(self,index,scale):return self.backend.backward_cached(index,scale)
    def score(self,index,release=False):return self.backend.score_cached(index,release)
    def close(self):self.backend.cache.clear();self.backend.public_loader.close()

def scores(backend,indices,release=False):return [backend.score_cached(i,release) for i in indices]
def dispatch_v7(mode,authorization,lineage,run_dir,identity,resume_state=LongResumeState()):
 backend,roles,all_indices,metadata=build_v7_backend(mode,authorization,lineage,run_dir,identity);dispatcher=V7StageDispatcher(backend);world=int(os.environ.get("WORLD_SIZE","1"));rank=int(os.environ.get("RANK","0"))
 try:
  if mode=="smoke":return run_v6_smoke(backend=backend,records=roles["smoke"],quick_gate=lambda rows:all(r["aggregate_rel_l2"]<=r["parent_rel_l2"] for r in rows),checkpoint=lambda:backend.checkpoint({"mode":mode}),resource_snapshot=lambda saved:backend.resources(saved,world),terminal=lambda payload:atomic_json_exclusive({**payload,**checkpoint_terminal_binding(payload["checkpoint"],identity)},run_dir/"terminal.json"))
  if mode=="pilot":
   dispatcher.preload(roles["pilot_fit"]+roles["pilot_confirm"])
   def ridge_update(i):p=backend.cached_prepare(i,"ridge");backend.ridge_update(p,backend._cached_truth(p))
   return run_pilot_stage(fit_records=roles["pilot_fit"],confirm_records=roles["pilot_confirm"],candidate_update=dispatcher.update,ridge_update=ridge_update,ridge_solve=backend.ridge_finalize,candidate_score=lambda rows:scores(backend,list(rows)),ridge_score=lambda ridge,rows:[backend.ridge_score_cached(i,ridge) for i in rows],checkpoint=lambda:backend.checkpoint({"mode":mode}),lineage=lineage,run_dir=run_dir,resource_snapshot=lambda saved:backend.resources(saved,world),terminal_extra=lambda:{"checkpoint_identity_digest":identity["identity_digest"]})
  if mode in {"scale-probe-1","scale-probe-4"}:
   records=roles["pilot_fit"][:12];dispatcher.preload(records[rank::world]);return run_scale_probe(mode=mode,records=records,measure=dispatcher.measure,model_hash=lambda:model_state_digest(backend.candidate),resource_snapshot=lambda:backend.resources({"size_bytes":Path(authorization["input_checkpoint_path"]).stat().st_size},world),lineage=lineage,run_dir=run_dir,world_size=world,rank=rank)
  if mode=="long":
   train=roles["long_fit"];cal=roles["long_calibration"];started=time.monotonic();dispatcher.preload(train+cal);preload_elapsed=time.monotonic()-started;saved={"best":None,"last":None}
   def gather(rows):
    if world==1:return rows
    shards=[None]*world;torch.distributed.all_gather_object(shards,rows);return [x for shard in shards for x in shard]
   def step():
    if world==4:
     for p in backend.candidate.parameters():
      if p.grad is not None:torch.distributed.all_reduce(p.grad);p.grad.div_(4.)
    torch.nn.utils.clip_grad_norm_(backend.candidate.parameters(),1.);backend.optimizer.step()
   def cp(state,best):item=backend.checkpoint(state.__dict__,best);saved["last"]=item; saved.__setitem__("best",item if best else saved["best"])
   def terminal(payload):
    item=saved["best"] or saved["last"] or {"best":authorization["input_checkpoint_path"],"last":authorization["input_checkpoint_path"],"size_bytes":Path(authorization["input_checkpoint_path"]).stat().st_size};atomic_json_exclusive({**payload,**checkpoint_terminal_binding(item,identity)},run_dir/"terminal.json")
   runner=V5LongRunner(world_size=world,rank=rank,train_records=train,calibration_records=cal,zero_grad=lambda:backend.optimizer.zero_grad(set_to_none=True),backward_record=lambda i,s:dispatcher.backward(i,s),optimizer_step=step,set_lr=lambda lr:[g.update(lr=lr) for g in backend.optimizer.param_groups],evaluate=lambda i:dispatcher.score(i),gather=gather,checkpoint=cp,terminal=terminal,clock=lambda:time.monotonic()-preload_elapsed);return runner.run(resume_state)
  if mode in {"final-train-confirm","validation-once","test-once"}:
   def evaluate(i):backend.preload(i);return dispatcher.score(i,True)
   validation=json.loads((OUT/"validation-once/terminal.json").read_text()) if mode=="test-once" else None;lock={"schema":"r16_dscp_v7_lock_v1","checkpoint_sha256":lineage.input_checkpoint_sha256,"metadata_digest":metadata};return run_evaluation_stage(mode=mode,records=all_indices,evaluate_one=evaluate,resource_snapshot=lambda:backend.resources({"size_bytes":Path(authorization["input_checkpoint_path"]).stat().st_size},world),lineage=lineage,run_dir=run_dir,metadata_digest=metadata,candidate_lock_payload=lock if mode=="final-train-confirm" else None,required_validation_terminal=validation)
  raise ValueError(mode)
 finally:dispatcher.close()

def static_evidence():
 cfg=yaml.safe_load(CONFIG.read_text())
 if not TEST_LOG.is_file() or "[100%]" not in TEST_LOG.read_text() or "failed" in TEST_LOG.read_text().lower():raise BindingRefusal("tests absent")
 for name,command in cfg["commands"].items():
  if name not in {"prep","authorize","scale_decide"} and not command.startswith("env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="):raise BindingRefusal("command env drift")
 v6=json.loads((ROOT/"results/r16_dscp_v6/static_evidence.json").read_text());return {"schema":"r16_dscp_v7_static_v1","parameters":1202,"ridge_parameters":480,"model_macs":45451125,"basis_conditions":v6["basis_conditions"],"router_census":v6["router_census"],"panels_census":v6["panels_census"],"host_cache_gates":v6["host_cache_gates"],"metadata":metadata_seal(),"tests":{"status":"passed","log_sha256":sha256_file(TEST_LOG),"tail":TEST_LOG.read_text().splitlines()[-1]},"identity_fields":list(__import__("saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v7",fromlist=["IDENTITY_FIELDS"]).IDENTITY_FIELDS),"production_backend":"V7ProductionBackend","checkpoint_identity_assignment_source":canonical_sha(build_v7_backend.__code__.co_code.hex()),"truth_read":False}

def prep():
 if OUT.exists() or PREREG.exists():raise FileExistsError("v7 exists")
 bindings=effective_bindings();static=static_evidence();OUT.mkdir();atomic_json_exclusive(static,STATIC);artifact=basis_artifact();candidate=R16DSCP(artifact["basis"],artifact["coefficient_scales"]);optimizer=make_optimizer(list(candidate.parameters()));h=w=201;args=(torch.ones(1,1,h,w)*2000,torch.zeros(1,1,h,w),torch.ones(1,1,h,w),torch.arange(w),torch.arange(h),torch.zeros(1,1,h,w),torch.zeros(1,1,h,w),torch.zeros(1,401,h,w),torch.tensor([1]),torch.tensor([2]));losses,adapted,_=actual_coefficient_loss(candidate,args,torch.ones(1,398,h,w));losses["total"].backward();optimizer.step();identity=complete_checkpoint_identity(mode="prep",run_digest=canonical_sha(bindings),authorization_sha256="prep-none",engine_sha256=bindings["engine"]["sha256"],script_sha256=bindings["script"]["sha256"],config_sha256=bindings["config"]["sha256"],panels_sha256=PANELS_SHA256,basis_sha256=BASIS_FILE_SHA256,parent_sha256=PARENT_SHA256,input_checkpoint_sha256="none");state=LongResumeState(epoch=0,global_step=1,group_index=0,best=float("inf"),bad_epochs=0,best_epoch=-1);payload=checkpoint_payload(candidate,optimizer,run_identity=identity,sampler_order=list(range(12)),progress=state.__dict__);saved=save_best_last(payload,OUT/"actual_checkpoint",is_best=True);restored=R16DSCP(artifact["basis"],artifact["coefficient_scales"]);ro=make_optimizer(list(restored.parameters()));loaded,resume_state=resume_v7_checkpoint(saved["last"],restored,ro,expected_identity=identity);resume=all(torch.equal(a,b) for a,b in zip(candidate.parameters(),restored.parameters())) and resume_state==state;ledger=__import__("saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3",fromlist=["AccessLedger"]).AccessLedger("train");spool=CandidateSpool(OUT,run_digest=identity["run_digest"],rank=0,owned_root="/dev/shm/r16_dscp_v7");seal=spool.seal("size",adapted,ledger);spool_max=seal["serialized_bytes"];spool.cleanup(seal,ledger);gate1=dual_space_gate(workspace=ROOT,tmpfs="/dev/shm",checkpoint_bytes=saved["size_bytes"],spool_max_bytes=spool_max,world_size=1);gate4=dual_space_gate(workspace=ROOT,tmpfs="/dev/shm",checkpoint_bytes=saved["size_bytes"],spool_max_bytes=spool_max,world_size=4)
 if not resume or not saved["hardlinked"] or not gate1["passed"]:raise BindingRefusal("prep gate")
 pre={"schema":"r16_dscp_v7_preflight_v1","candidate":CANDIDATE,"status":"frozen","bindings":bindings,"run_identity":identity,"static":bind(STATIC),"checkpoint":{**saved,"sha256":sha256_file(saved["last"]),"resume_equal":resume,"identity_digest":identity["identity_digest"],"state_sha256":model_state_digest(candidate)},"host_cache_gates":static["host_cache_gates"],"spool":{"max_bytes":spool_max,"root":"/dev/shm/r16_dscp_v7/<run_digest>/rank<r>"},"space1":gate1,"space4":gate4,"metadata":static["metadata"],"truth_reads":0,"GPU_used":False,"auth_created":False};atomic_json_exclusive(pre,PREFLIGHT);print(json.dumps(pre["checkpoint"]))

def verify_preflight():
 pre=json.loads(PREFLIGHT.read_text());current=effective_bindings()
 for key,value in pre["bindings"].items():
  if current[key]["sha256"]!=value["sha256"]:raise BindingRefusal(f"drift {key}")
 return pre

def freeze():
 if PREREG.exists():raise FileExistsError("v7 prereg exists")
 pre=verify_preflight();cfg=yaml.safe_load(CONFIG.read_text());runtime=json.loads((ROOT/"results/r4e7_parent_e2e_runtime_train9_v1_20260825.json").read_text());gpu=subprocess.run(["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.total","--format=csv,noheader"],check=True,capture_output=True,text=True).stdout.splitlines();payload={"schema":"r16_dscp_v7_prereg_v1","candidate":CANDIDATE,"status":"frozen","preflight":bind(PREFLIGHT),"effective_bindings":effective_bindings(),"v3_veto":bind(ROOT/"results/r16_dscp_v3_preregistration_20260826.json"),"v4_veto":bind(ROOT/"results/r16_dscp_v4_preregistration_20260826.json"),"v5_veto":bind(ROOT/"results/r16_dscp_v5_preregistration_20260826.json"),"v6_blocker":bind(ROOT/"results/r16_dscp_v6/design_preflight.json"),"only_change":"complete checkpoint identity wiring","config":cfg,"commands":cfg["commands"],"checkpoint":pre["checkpoint"],"host_cache_gates":pre["host_cache_gates"],"spool":pre["spool"],"space1":pre["space1"],"space4":pre["space4"],"metadata":pre["metadata"],"tests":json.loads(STATIC.read_text())["tests"],"identities":{"python":platform.python_version(),"torch":torch.__version__,"torch_cuda":str(torch.version.cuda),"gpu":gpu,"cuda_used":False},"parent_runtime":{"mean_s":runtime["aggregate"]["arithmetic_mean_runtime_s"],"traditional_speedup":runtime["aggregate"]["reference_over_mean_speedup_x"]},"claim_boundary":"prep only; no stage/truth/GPU; immutable 10x remains unmet with parent 1.608x","rollback":cfg["rollback"],"sealed":{"truth_read":False,"GPU_used":False,"auth_created":False,"cache_serialized":False}};atomic_json_exclusive(payload,PREREG);print(json.dumps(bind(PREREG)))

def authorize_stage(stage,output,input_checkpoint=None):
 if not PREREG.is_file():raise BindingRefusal("prereg required")
 pre=verify_preflight();cfg=yaml.safe_load(CONFIG.read_text());chain={"smoke":[PREREG],"pilot":[OUT/"smoke/terminal.json"],"scale-probe-1":[OUT/"pilot/terminal.json"],"scale-probe-4":[OUT/"pilot/terminal.json"],"scale-decide":[OUT/"scale-probe-1/terminal.json",OUT/"scale-probe-4/terminal.json"],"long":[OUT/"pilot/terminal.json",OUT/"scale-decide/terminal.json"],"final-train-confirm":[OUT/"long/terminal.json"],"validation-once":[OUT/"final-train-confirm/terminal.json",OUT/"final-train-confirm/candidate_lock.json"],"test-once":[OUT/"validation-once/terminal.json",OUT/"final-train-confirm/candidate_lock.json"]};prereq={};expected=[]
 for path in chain[stage]:
  if not path.is_file():raise BindingRefusal("prerequisite absent")
  obj=json.loads(path.read_text());
  if "terminal" in path.name and obj.get("status") not in {"passed","completed_train"}:raise BindingRefusal("prerequisite failed")
  if isinstance(obj.get("checkpoint_identity"),dict):validate_checkpoint_identity(obj["checkpoint_identity"])
  if isinstance(obj.get("best_checkpoint"),dict):expected.append(obj["best_checkpoint"]["path"])
  if isinstance(obj.get("checkpoint"),dict) and obj["checkpoint"].get("best"):expected.append(obj["checkpoint"]["best"])
  prereq[str(path.resolve())]=sha256_file(path)
 input_sha="none";input_identity=pre["run_identity"];input_state=None
 if input_checkpoint:
  path=Path(input_checkpoint).resolve();input_sha=sha256_file(path)
  if expected and str(path) not in [str(Path(x).resolve()) for x in expected]:raise BindingRefusal("not best")
  raw=torch.load(path,map_location="cpu",weights_only=False);input_identity=raw["run_identity"];validate_checkpoint_identity(input_identity);artifact=basis_artifact();model=R16DSCP(artifact["basis"],artifact["coefficient_scales"]);opt=make_optimizer(list(model.parameters()));resume_v7_checkpoint(path,model,opt,expected_identity=input_identity);input_state=model_state_digest(model)
 elif stage not in {"smoke","scale-decide"}:raise BindingRefusal("input required")
 line=Lineage(CANDIDATE,stage,pre["run_identity"]["run_digest"],sha256_file(ENGINE),sha256_file(CONFIG),PANELS_SHA256,BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PARENT_SHA256,input_sha);auth={"schema":AUTH_SCHEMA,"status":"authorized",**line.__dict__,"input_checkpoint_path":None if not input_checkpoint else str(Path(input_checkpoint).resolve()),"input_run_identity":input_identity,"input_state_sha256":input_state,"parent_sha256":PARENT_SHA256,"basis_sha256":BASIS_FILE_SHA256,"config_sha256":sha256_file(CONFIG),"code_sha256":sha256_file(ENGINE),"prerequisite_bindings":prereq,"gates_sha256":canonical_sha(cfg),"script_sha256":sha256_file(SCRIPT),"prereg_sha256":sha256_file(PREREG)};auth["authorization_digest"]=canonical_sha(auth);atomic_json_exclusive(auth,output)

def run_authorized(mode,auth_path,resume=False):
 pre=verify_preflight();auth=json.loads(Path(auth_path).read_text());line=Lineage(CANDIDATE,mode,pre["run_identity"]["run_digest"],sha256_file(ENGINE),sha256_file(CONFIG),PANELS_SHA256,BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PARENT_SHA256,str(auth.get("input_checkpoint_sha256","none")));validate_authorization(auth,line);verify_before_data(auth,input_checkpoint=auth.get("input_checkpoint_path"),parent=PARENT_PATH,basis=BASIS,config=CONFIG,code=ENGINE);run=OUT/mode;rank=int(os.environ.get("RANK","0"));world=int(os.environ.get("WORLD_SIZE","1"));distributed=world==4 and mode in {"scale-probe-4","long"};identity=complete_checkpoint_identity(mode=mode,run_digest=line.run_digest,authorization_sha256=sha256_file(auth_path),engine_sha256=sha256_file(ENGINE),script_sha256=sha256_file(SCRIPT),config_sha256=sha256_file(CONFIG),panels_sha256=PANELS_SHA256,basis_sha256=BASIS_FILE_SHA256,parent_sha256=PARENT_SHA256,input_checkpoint_sha256=line.input_checkpoint_sha256);state=LongResumeState()
 if mode=="scale-decide":
  if run.exists():raise FileExistsError("stage exists")
  run.mkdir();one=json.loads((OUT/"scale-probe-1/terminal.json").read_text());four=json.loads((OUT/"scale-probe-4/terminal.json").read_text());decision=scale_decision(one,four);payload={"schema":"r16_dscp_v7_scale_decision_v1","status":"passed","selected_gpus":decision["selected_gpus"],"raw_gates":decision,"long_launched":False};atomic_json_exclusive(payload,run/"terminal.json");return payload
 if resume:
  raw=torch.load(run/"last.pt",map_location="cpu",weights_only=False);expected=raw["run_identity"];validate_checkpoint_identity(expected,identity);artifact=basis_artifact();model=R16DSCP(artifact["basis"],artifact["coefficient_scales"]);opt=make_optimizer(list(model.parameters()));_,state=resume_v7_checkpoint(run/"last.pt",model,opt,expected_identity=identity);auth=dict(auth);auth["input_checkpoint_path"]=str(run/"last.pt");auth["input_run_identity"]=identity;auth["input_state_sha256"]=model_state_digest(model)
 if distributed:torch.distributed.init_process_group("nccl");torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
 try:
  if rank==0:
   if run.exists() and not resume:raise FileExistsError("stage exists")
   if not run.exists():run.mkdir();atomic_json_exclusive(identity,run/"run_identity.json")
  if distributed:torch.distributed.barrier()
  if mode=="scale-probe-4":
   try:result=dispatch_v7(mode,auth,line,run,identity,state)
   except Exception as exc:result={"status":"failed","rank":rank,"world_size":world,"reason":str(exc)}
   gathered=[None]*world;torch.distributed.all_gather_object(gathered,result);failures=[x for x in gathered if x.get("status")!="passed"]
   if rank==0:
    if failures:result={"status":"failed","rank_failures":failures};atomic_json_exclusive(result,run/"terminal.json")
    else:order=sorted({row["sample_id"] for report in gathered for row in report["rows"]});result=aggregate_scale_ranks(gathered,order);atomic_json_exclusive(result,run/"terminal.json")
   torch.distributed.barrier()
   if failures:raise RuntimeError("scale rank failure")
  else:result=dispatch_v7(mode,auth,line,run,identity,state)
  return result
 finally:
  if distributed and torch.distributed.is_initialized():torch.distributed.destroy_process_group()

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--mode",required=True,choices=("prep","freeze-prereg","authorize-stage","smoke","pilot","scale-probe-1","scale-probe-4","scale-decide","long","final-train-confirm","validation-once","test-once"));parser.add_argument("--stage");parser.add_argument("--output");parser.add_argument("--input-checkpoint");parser.add_argument("--authorization");parser.add_argument("--resume",action="store_true");args=parser.parse_args()
 if args.mode=="prep":prep()
 elif args.mode=="freeze-prereg":freeze()
 elif args.mode=="authorize-stage":authorize_stage(args.stage,args.output,args.input_checkpoint)
 else:run_authorized(args.mode,args.authorization,args.resume)
if __name__=="__main__":main()

__all__=["V7StageDispatcher"]
