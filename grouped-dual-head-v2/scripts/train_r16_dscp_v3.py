#!/usr/bin/env python3
"""Preparation and authorized real-stage entrypoint for R16-DSCP v3."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, shutil, subprocess, sys, time
from pathlib import Path
from typing import Any, Mapping
import torch, yaml

ROOT=Path(__file__).resolve().parents[1]
for p in (str(ROOT),str(ROOT/"src")):
 if p not in sys.path:sys.path.insert(0,p)
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import R16DSCP, predictor_parameter_count, analytic_model_macs
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import *
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import *
from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime
from scripts import probe_r4_family_temporal_pod_capacity as pod
from saved_time_phase_operator_v4.instance_adaptation.contracts import onset_indices
from saved_time_phase_operator_v4.eikonal import grid_eikonal_travel_time

CAND="r16_dscp_v3";SCRIPT=Path(__file__).resolve();ENGINE=ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v3.py";TEST=ROOT/"tests/saved_time_phase_operator_v4/test_r16_dscp_v3.py";CONFIG=ROOT/"configs/r16_dscp_v3.yaml"
OUT=ROOT/"results/r16_dscp_v3";PREF=OUT/"design_preflight.json";STATIC=OUT/"static_evidence.json";PREREG=ROOT/"results/r16_dscp_v3_preregistration_20260826.json"
BASIS=ROOT/"results/r16_dscp_v1/basis_rank16.pt";PANELS=ROOT/"results/r16_dscp_v1/panels.json";V2=ROOT/"results/r16_dscp_v2_preregistration_20260826.json"
TEST_LOG=Path("/tmp/r16_dscp_v3_tests.log")

def bind(p):
 p=Path(p).resolve();s=p.stat();return {"path":str(p),"sha256":sha256_file(p),"size_bytes":s.st_size}
def bindings():return {"engine":bind(ENGINE),"script":bind(SCRIPT),"test":bind(TEST),"config":bind(CONFIG),"model":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp.py"),"v2_harness":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_training_v2.py"),"data_guard":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/data_guard.py"),"contracts":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/contracts.py"),"eikonal":bind(ROOT/"saved_time_phase_operator_v4/eikonal.py"),"parent_helper":bind(ROOT/"scripts/benchmark_r4_parent_e2e_trainonly.py"),"pod_helper":bind(ROOT/"scripts/probe_r4_family_temporal_pod_capacity.py"),"basis":bind(BASIS),"panels":bind(PANELS),"parent":bind(PARENT_PATH),"v2_prereg":bind(V2),"manifest":bind(parent_runtime.MANIFEST_PATH),"normalization":bind(parent_runtime.NORMALIZATION_PATH),"parent_run_identity":bind(parent_runtime.RUN_IDENTITY_PATH),"oracle_result":bind(ROOT/"results/r4e7_raw_weighted_pod_fresh3_confirmation_v1_20260826.json"),"oracle_prereg":bind(ROOT/"results/r4e7_raw_weighted_pod_fresh3_confirmation_v1_preregistration_20260826.json"),"parent_runtime_result":bind(ROOT/"results/r4e7_parent_e2e_runtime_train9_v1_20260825.json")}
def identities():
 query=subprocess.run(["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.total","--format=csv,noheader"],check=True,capture_output=True,text=True).stdout.splitlines()
 return {"software":{"python":platform.python_version(),"torch":torch.__version__,"torch_cuda":str(torch.version.cuda),"numpy":__import__("numpy").__version__},"gpu_inventory":[line.strip() for line in query],"cuda_used":False}
def basis_artifact():
 if sha256_file(BASIS)!=BASIS_FILE_SHA256:raise BindingRefusal("basis drift")
 return torch.load(BASIS,map_location="cpu",weights_only=False)
def metadata_seal():
 m=parent_runtime.load_manifest_payload();out={};sets={}
 for split in ("validation","test_id"):
  rows=[{"source_index":int(r["source_index"]),"sample_id":r["sample_id"],"group_id":r["group_id"],"sample_sha256":r["sample_sha256"],"family":r["medium_type"]} for r in m["records"] if r["split"]==split]
  rows.sort(key=lambda r:(r["source_index"],r["sample_id"]));sets[split]={key:{r[key] for r in rows} for key in ("source_index","sample_id","group_id","sample_sha256")};out[split]={"count":len(rows),"ordered_digest":canonical_sha(rows),"by_family":{f:sum(r["family"]==f for r in rows) for f in ("uniform","layered","marmousi")},"unique_source_indices":len(sets[split]["source_index"]),"unique_sample_ids":len(sets[split]["sample_id"]),"unique_sample_hashes":len(sets[split]["sample_sha256"]),"unique_groups":len(sets[split]["group_id"]),"wavefield_read":False}
 if out["validation"]["count"]!=480 or out["test_id"]["count"]!=480:raise BindingRefusal("sealed split census drift")
 for key in sets["validation"]:
  if sets["validation"][key]&sets["test_id"][key]:raise BindingRefusal(f"sealed split metadata overlap: {key}")
 for split in out:
  if out[split]["unique_source_indices"]!=480 or out[split]["unique_sample_ids"]!=480 or out[split]["unique_sample_hashes"]!=480:raise BindingRefusal(f"sealed metadata uniqueness drift: {split}")
 out["cross_split_zero_overlap"]={key:True for key in sets["validation"]}
 return out
def static():
 a=basis_artifact();m=R16DSCP(a["basis"],a["coefficient_scales"]);cfg=yaml.safe_load(CONFIG.read_text())
 for k,v in cfg["commands"].items():
  if k not in {"prep","static","authorize_stage","scale_decide"} and not v.startswith("env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="):raise BindingRefusal("CUDA command drift")
 if not TEST_LOG.is_file() or "[100%]" not in TEST_LOG.read_text() or "failed" in TEST_LOG.read_text().lower():raise BindingRefusal("focused/full test evidence absent")
 v1s=json.loads((ROOT/"results/r16_dscp_v1/static_evidence.json").read_text());v1p=json.loads((ROOT/"results/r16_dscp_v1/design_freeze_preflight.json").read_text());panels=json.loads(PANELS.read_text())
 return {"schema":"r16_dscp_v3_static_v1","parameters":predictor_parameter_count(m),"ridge_parameters":480,"macs":analytic_model_macs(),"projection_macs":259212816,"materialization_macs":259212816,"basis_conditions":v1s["basis_conditions_on_all_291_frozen_train_panel_records"],"router_census":v1p["router_train_census"],"panels_census":panels["census"],"engine_symbols":sorted(__import__("saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3",fromlist=["__all__"]).__all__),"engine_modes":list(cfg["commands"]),"tests":{"status":"passed","log_sha256":sha256_file(TEST_LOG),"log_tail":TEST_LOG.read_text().splitlines()[-1]},"metadata":metadata_seal(),"validation_truth_read":False,"test_truth_read":False}
def prep():
 if OUT.exists() or PREREG.exists():raise FileExistsError("v3 exists")
 b=bindings();s=static();OUT.mkdir();atomic_json_exclusive(s,STATIC);a=basis_artifact();m=R16DSCP(a["basis"],a["coefficient_scales"]);o=make_optimizer(list(m.parameters()));o.zero_grad()
 x=torch.randn(1,29,9,11);loss=m.coefficient_head(x).square().mean();loss.backward();o.step();ident={"candidate":CAND,"run_digest":canonical_sha(b),"code_sha256":b["engine"]["sha256"],"config_sha256":b["config"]["sha256"]}
 cp=checkpoint_payload(m,o,run_identity=ident,sampler_order=list(range(12)),progress={"update":1});rec=save_best_last(cp,OUT/"actual_checkpoint",is_best=True);restored=R16DSCP(a["basis"],a["coefficient_scales"]);restored_optimizer=make_optimizer(list(restored.parameters()));loaded=load_checkpoint(rec["last"],restored,restored_optimizer,expected_run_identity=ident);resume_equal=all(torch.equal(x,y) for x,y in zip(m.parameters(),restored.parameters())) and loaded["progress"]=={"update":1};gate=space_gate(rec["size_bytes"],ROOT)
 if not gate["passed"] or rec["size_bytes"]>2*1024**2 or not resume_equal or not rec["hardlinked"]:raise BindingRefusal("checkpoint space/resume/hardlink gate")
 p={"schema":"r16_dscp_v3_preflight_v1","candidate":CAND,"status":"frozen","bindings":b,"run_identity":ident,"static":bind(STATIC),"checkpoint":{**rec,"sha256":sha256_file(rec["last"]),"resume_equal":resume_equal},"space_gate":gate,"sealed_metadata":s["metadata"],"scientific_design":"identical v2; only stage engine/lineage added","truth_reads":{"train_future":0,"validation":0,"test_id":0}}
 atomic_json_exclusive(p,PREF);print(json.dumps(p["checkpoint"]))
def verify():
 p=json.loads(PREF.read_text());cur=bindings()
 for k,v in p["bindings"].items():
  if cur[k]["sha256"]!=v["sha256"]:raise BindingRefusal(f"binding drift {k}")
 return p
def authorize(mode,auth_path):
 pre=verify();payload=json.loads(Path(auth_path).read_text());input_sha=str(payload.get("input_checkpoint_sha256","none"));line=Lineage(CAND,mode,pre["run_identity"]["run_digest"],pre["bindings"]["engine"]["sha256"],pre["bindings"]["config"]["sha256"],PANELS_SHA256,BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PARENT_SHA256,input_sha);validate_authorization(payload,line)
 if payload.get("effective_script_sha256")!=sha256_file(SCRIPT) or payload.get("preregistration_sha256")!=sha256_file(PREREG) or payload.get("gates_sha256")!=canonical_sha(yaml.safe_load(CONFIG.read_text())["gates"]):raise BindingRefusal("authorization effective script/prereg/gates drift")
 for path,digest in payload.get("prerequisite_bindings",{}).items():
  if not Path(path).is_file() or sha256_file(path)!=digest:raise BindingRefusal("authorization prerequisite drift")
 input_path=payload.get("input_checkpoint_path")
 if input_sha!="none" and (not input_path or sha256_file(input_path)!=input_sha):raise BindingRefusal("input checkpoint binding mismatch")
 return pre,payload,line
def build_production_smoke_pilot_backend(mode,authorization,line,run):
 device=torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK','0'))}");configure_determinism(372);artifact=basis_artifact();candidate=R16DSCP(artifact["basis"],artifact["coefficient_scales"]).to(device);optimizer=make_optimizer(list(candidate.parameters()))
 resume_payload=None
 if line.input_checkpoint_sha256!="none":resume_payload=load_checkpoint(authorization["input_checkpoint_path"],candidate,optimizer,expected_run_identity=authorization["input_run_identity"])
 parent,normalizer,manifest_payload,_=parent_runtime.load_model_context(device);manifest=parent_runtime.manifest_object(manifest_payload);panel_rows=json.loads(PANELS.read_text())["records"]
 roles={"smoke":("smoke",),"pilot":("pilot_fit","pilot_confirm"),"scale-probe-1":("pilot_fit",),"scale-probe-4":("pilot_fit",),"long":("long_fit","long_calibration"),"final-train-confirm":("final_train_confirm",)}[mode];selected=[r for r in panel_rows if r["role"] in roles];sample_ids=[r["sample_id"] for r in selected];row_by_sample={r["sample_id"]:r for r in selected};family_by_sample={r["sample_id"]:r["family"] for r in selected}
 loader=GuardedPublicLoader(parent_runtime.SOURCE_H5_PATH,manifest,split="train",sample_ids=sample_ids)
 def parent_predictor(public):
  row=row_by_sample[public.sample_id];record=parent_runtime.SelectedRecord(public.source_index,public.sample_id,public.group_id,row["family"],"train",int(row["split_id"]),row["manifest_sample_sha256"]);loaded=parent_runtime.RecordInput(record,public.velocity_mps.numpy().astype("float32"),public.source_parameters.numpy().astype("float32"),public.source_map.numpy().astype("float32"),public.input_digest)
  return pod.generate_parent_full401(parent,normalizer,loaded,manifest_payload,device=device,time_block=16).cpu()
 def travel_builder(public):
  return torch.from_numpy(grid_eikonal_travel_time(public.velocity_mps.numpy(),source_indices=[(round(float(public.source_parameters[1])/10),round(float(public.source_parameters[0])/10))],dx_m=10,dz_m=10)[0])
 backend=ProductionSmokePilotBackend(public_loader=loader,parent_predictor=parent_predictor,travel_builder=travel_builder,candidate=candidate,optimizer=optimizer,authorization=authorization,lineage=line,source_h5=parent_runtime.SOURCE_H5_PATH,run_dir=run,device=device,family_by_sample=family_by_sample);backend.resume_payload=resume_payload
 index_by_role={role:[sample_ids.index(r["sample_id"]) for r in selected if r["role"]==role] for role in roles};return backend,index_by_role
def _dispatch_smoke_pilot_authorized(mode,authorization,line,run,*,backend_factory=build_production_smoke_pilot_backend,smoke_runner=run_smoke_stage,pilot_runner=run_pilot_stage):
 try:backend,indices=backend_factory(mode,authorization,line,run)
 except Exception as exc:
  if not (run/"terminal.json").exists():write_failure_terminal(run,mode=mode,reason=str(exc),run_identity=line.__dict__)
  raise
 parent_before=sha256_file(PARENT_PATH);ledgers=[]
 def update(index):
  prepared=backend.prepare(index)
  try: result=dict(backend.update(prepared));ledgers.append(prepared.ledger.digest());return result
  finally: backend.release(prepared)
 def score(indexes,model=None,ridge=None):
  rows=[]
  for index in indexes:
   prepared=backend.prepare(index)
   try:
    truth=backend.open_train_truth(prepared);row=dict(backend.ridge_score(ridge,prepared,truth) if ridge is not None else backend.score(model or backend.candidate,prepared,truth));row.update(sample_id=prepared.public.sample_id,ledger_digest=prepared.ledger.digest());rows.append(row)
   finally:backend.release(prepared)
  return rows
 try:
  if mode=="smoke":
   terminal=smoke_runner(records=indices["smoke"],update=update,score=lambda:score(indices["smoke"]),checkpoint=lambda:backend.checkpoint({"mode":mode}),lineage=line,run_dir=run,max_updates=500,max_seconds=600,resource_snapshot=backend.resources,terminal_extra=lambda:{"access_ledger_digest":canonical_sha(ledgers),"parent_sha256_before":parent_before,"parent_sha256_expected":PARENT_SHA256})
  else:
   def ridge_update(index):
    prepared=backend.prepare(index)
    try:truth=backend.open_train_truth(prepared);backend.ridge_update(prepared,truth);ledgers.append(prepared.ledger.digest())
    finally:backend.release(prepared)
   terminal=pilot_runner(fit_records=indices["pilot_fit"],confirm_records=indices["pilot_confirm"],candidate_update=update,ridge_update=ridge_update,ridge_solve=backend.ridge_finalize,candidate_score=lambda rows:score(rows),ridge_score=lambda ridge,rows:score(rows,ridge=ridge),checkpoint=lambda:backend.checkpoint({"mode":mode}),lineage=line,run_dir=run,max_updates=1000,max_seconds=1800,resource_snapshot=backend.resources,terminal_extra=lambda:{"access_ledger_digest":canonical_sha(ledgers),"parent_sha256_before":parent_before,"parent_sha256_expected":PARENT_SHA256})
  if sha256_file(PARENT_PATH)!=parent_before:raise BindingRefusal("parent changed during stage")
  return terminal
 except Exception as exc:
  if not (run/"terminal.json").exists():write_failure_terminal(run,mode=mode,reason=str(exc),run_identity={**line.__dict__,"access_ledger_digest":canonical_sha(ledgers)})
  raise
 finally:backend.public_loader.close()
def build_production_evaluation_backend(mode,authorization,line,run):
 split="train" if mode=="final-train-confirm" else "validation" if mode=="validation-once" else "test_id";device=torch.device("cuda:0");configure_determinism(372);artifact=basis_artifact();candidate=R16DSCP(artifact["basis"],artifact["coefficient_scales"]).to(device);optimizer=make_optimizer(list(candidate.parameters()))
 if line.input_checkpoint_sha256=="none":raise BindingRefusal("evaluation requires locked input checkpoint")
 load_checkpoint(authorization["input_checkpoint_path"],candidate,optimizer,expected_run_identity=authorization["input_run_identity"]);parent,normalizer,manifest_payload,_=parent_runtime.load_model_context(device);manifest=parent_runtime.manifest_object(manifest_payload)
 if split=="train":rows=[r for r in json.loads(PANELS.read_text())["records"] if r["role"]=="final_train_confirm"];sample_ids=[r["sample_id"] for r in rows];family={r["sample_id"]:r["family"] for r in rows}
 else:rows=sorted([r for r in manifest_payload["records"] if r["split"]==split],key=lambda r:(int(r["source_index"]),r["sample_id"]));sample_ids=[r["sample_id"] for r in rows];family={r["sample_id"]:r["medium_type"] for r in rows}
 loader=GuardedPublicLoader(parent_runtime.SOURCE_H5_PATH,manifest,split=split,sample_ids=sample_ids)
 @torch.inference_mode()
 def parent_predictor(public):
  velocity=public.velocity_mps[None,None].to(device);source=public.source_parameters[None].to(device);source_map=public.source_map[None,None].to(device);medium=parent.encode_medium(velocity,normalizer);prepared=parent.prepare_sources(medium,source,source_map,normalizer,record_to_medium=torch.zeros(1,dtype=torch.long,device=device));normalized=parent.dense_normalized(prepared,public.time_s.to(device),x_m=public.x_m.to(device),z_m=public.z_m.to(device),time_block=16);return normalizer.decode_pressure(normalized.float(),source[:,4])[0].cpu()
 def travel(public):return torch.from_numpy(grid_eikonal_travel_time(public.velocity_mps.numpy(),source_indices=[(round(float(public.source_parameters[1])/10),round(float(public.source_parameters[0])/10))],dx_m=10,dz_m=10)[0])
 token=run/f"{split}.once.json" if split!="train" else None;backend=ProductionSmokePilotBackend(public_loader=loader,parent_predictor=parent_predictor,travel_builder=travel,candidate=candidate,optimizer=optimizer,authorization=authorization,lineage=line,source_h5=parent_runtime.SOURCE_H5_PATH,run_dir=run,device=device,family_by_sample=family,split=split,once_token=token);return backend,list(range(len(loader))),canonical_sha([{"source_index":int(r["source_index"]),"sample_id":r["sample_id"],"group_id":r["group_id"],"sample_sha256":r.get("sample_sha256",r.get("manifest_sample_sha256"))} for r in rows])
def _dispatch_long_authorized(authorization,line,run):
 backend,roles=build_production_smoke_pilot_backend("long",authorization,line,run);ledgers=[];parent_before=sha256_file(PARENT_PATH)
 def update(i):
  p=backend.prepare(i)
  try:r=backend.update(p);ledgers.append(p.ledger.digest());return r
  finally:backend.release(p)
 def evaluate(indexes):
  out=[]
  for i in indexes:
   p=backend.prepare(i)
   try:t=backend.open_train_truth(p);r=dict(backend.score(backend.candidate,p,t));r.update(sample_id=p.public.sample_id,ledger_digest=p.ledger.digest());out.append(r)
   finally:backend.release(p)
  return out
 try:
  progress=(backend.resume_payload or {}).get("progress",{});resume={k:progress[k] for k in ("best","bad_epochs","best_epoch") if k in progress};resume_state={"best":resume.get("best",float("inf")),"bad_epochs":resume.get("bad_epochs",0),"best_epoch":resume.get("best_epoch",-1)}
  result=run_long_stage(fit_records=roles["long_fit"],calibration_records=roles["long_calibration"],update=update,evaluate=evaluate,set_lr=lambda lr:[group.update(lr=lr) for group in backend.optimizer.param_groups],checkpoint=lambda state,best:backend.checkpoint(state,best),resource_snapshot=backend.resources,lineage=line,run_dir=run,start_epoch=int(progress.get("epoch",-1))+1,resume_state=resume_state,terminal_extra=lambda:{"access_ledger_digest":canonical_sha(ledgers),"parent_sha256_before":parent_before});
  if sha256_file(PARENT_PATH)!=parent_before:raise BindingRefusal("parent changed during long")
  return result
 except Exception as exc:
  if not (run/"terminal.json").exists():write_failure_terminal(run,mode="long",reason=str(exc),run_identity=line.__dict__)
  raise
 finally:backend.public_loader.close()
def _dispatch_evaluation_authorized(mode,authorization,line,run):
 if mode in {"validation-once","test-once"}:
  lock_path=OUT/"final-train-confirm/candidate_lock.json"
  if not lock_path.is_file() or authorization.get("candidate_lock_sha256")!=sha256_file(lock_path):raise BindingRefusal("sealed evaluation requires bound candidate lock")
 backend,indices,metadata_digest=build_production_evaluation_backend(mode,authorization,line,run);ledgers=[];parent_before=sha256_file(PARENT_PATH)
 def evaluate(i):
  p=backend.prepare(i)
  try:t=backend.open_truth(p);r=dict(backend.score(backend.candidate,p,t));r.update(sample_id=p.public.sample_id,ledger_digest=p.ledger.digest());ledgers.append(p.ledger.digest());return r
  finally:backend.release(p)
 resources=lambda:backend.resources({"size_bytes":Path(authorization["input_checkpoint_path"]).stat().st_size})
 lock={"schema":"r16_dscp_v3_candidate_lock_v1","checkpoint_sha256":line.input_checkpoint_sha256,"code_sha256":line.code_sha256,"config_sha256":line.config_sha256,"basis_sha256":line.basis_sha256,"panels_sha256":line.panels_sha256,"parent_sha256":line.parent_sha256,"thresholds_sha256":authorization["gates_sha256"],"metadata_digest":metadata_digest}
 validation=json.loads((OUT/"validation-once/terminal.json").read_text()) if mode=="test-once" and (OUT/"validation-once/terminal.json").is_file() else None
 if mode=="test-once" and (validation is None or authorization.get("validation_terminal_sha256")!=sha256_file(OUT/"validation-once/terminal.json")):raise BindingRefusal("test requires bound passed validation terminal")
 try:
  result=run_evaluation_stage(mode=mode,records=indices,evaluate_one=evaluate,resource_snapshot=resources,lineage=line,run_dir=run,metadata_digest=metadata_digest,candidate_lock_payload=lock if mode=="final-train-confirm" else None,required_validation_terminal=validation,terminal_extra=lambda:{"access_ledger_digest":canonical_sha(ledgers),"parent_sha256_before":parent_before});
  if sha256_file(PARENT_PATH)!=parent_before:raise BindingRefusal("parent changed during evaluation")
  return result
 except Exception as exc:
  if not (run/"terminal.json").exists():write_failure_terminal(run,mode=mode,reason=str(exc),run_identity=line.__dict__)
  raise
 finally:backend.public_loader.close()
def _require_pilot_for_scale(authorization):
 terminal=OUT/"pilot/terminal.json"
 if not terminal.is_file() or json.loads(terminal.read_text()).get("status")!="passed" or authorization.get("pilot_terminal_sha256")!=sha256_file(terminal):raise BindingRefusal("scale requires bound passed pilot terminal")
def _dispatch_scale_probe_authorized(mode,authorization,line,run):
 _require_pilot_for_scale(authorization);distributed=mode=="scale-probe-4";rank=int(os.environ.get("RANK","0"));world=int(os.environ.get("WORLD_SIZE","1"))
 if distributed:
  if world!=4:raise BindingRefusal("scale-4 requires world size 4")
  torch.distributed.init_process_group("nccl");torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
 backend,roles=build_production_smoke_pilot_backend(mode,authorization,line,run);records=roles["pilot_fit"][:12];order=[backend.public_loader[i].sample_id for i in records]
 def measure(i):
  prepared=backend.prepare(i)
  try:return backend.measure_gradient(prepared)
  finally:backend.release(prepared)
 try:
  try:report=run_scale_probe(mode=mode,records=records,measure=measure,model_hash=lambda:model_digest(backend.candidate),resource_snapshot=lambda:backend.resources({"size_bytes":Path(authorization["input_checkpoint_path"]).stat().st_size}),lineage=line,run_dir=run,world_size=world,rank=rank)
  except Exception as local_exc:
   if not distributed:raise
   report={"schema":"r16_dscp_v3_scale_rank_v1","status":"failed","rank":rank,"world_size":world,"reason":str(local_exc)}
  if distributed:
   gathered=[None]*world;torch.distributed.all_gather_object(gathered,report)
   failed=[r for r in gathered if r.get("status")!="passed"]
   if rank==0:
    if failed:write_failure_terminal(run,mode=mode,reason=canonical_sha(failed),run_identity={**line.__dict__,"rank_failures":failed});result={"status":"failed"}
    else:result=aggregate_scale_ranks(gathered,order);result.update(mode=mode,parent_sha256=PARENT_SHA256,schedule_sha256=report["schedule_sha256"]);atomic_json_exclusive(result,run/"terminal.json")
   else:result={"status":"failed" if failed else "passed_nonroot"}
   torch.distributed.barrier()
   if failed:raise GateFailure("one or more scale ranks failed")
   return result
  return report
 except Exception as exc:
  if rank==0 and not (run/"terminal.json").exists():write_failure_terminal(run,mode=mode,reason=str(exc),run_identity=line.__dict__)
  raise
 finally:
  backend.public_loader.close()
  if distributed and torch.distributed.is_initialized():torch.distributed.destroy_process_group()
def _dispatch_scale_decide_authorized(authorization,line,run):
 _require_pilot_for_scale(authorization);one_path=OUT/"scale-probe-1/terminal.json";four_path=OUT/"scale-probe-4/terminal.json"
 for name,path in (("scale1_terminal_sha256",one_path),("scale4_terminal_sha256",four_path)):
  if not path.is_file() or authorization.get(name)!=sha256_file(path) or json.loads(path.read_text()).get("status")!="passed":raise BindingRefusal("scale decision terminal binding mismatch")
 one=json.loads(one_path.read_text());four=json.loads(four_path.read_text());decision=scale_decision(one,four);payload={"schema":"r16_dscp_v3_scale_decision_v1","candidate":CAND,"status":"passed","selected_gpus":decision["selected_gpus"],"gates":decision,"scale1_sha256":sha256_file(one_path),"scale4_sha256":sha256_file(four_path),"long_launched":False,"lineage":line.__dict__};atomic_json_exclusive(payload,run/"terminal.json");return payload
def run_authorized(mode,devices,auth):
 if mode!="scale-decide":require_cuda_environment(visible_devices=devices)
 pre,authorization,line=authorize(mode,auth);split="validation" if mode=="validation-once" else "test_id" if mode=="test-once" else "train"
 if mode!="scale-decide":assert_stage_split(mode,split)
 run=OUT/mode
 if mode=="scale-probe-4":
  rank=int(os.environ.get("RANK","0"))
  if rank==0:
   if run.exists():raise FileExistsError("stage already exists")
   run.mkdir();atomic_json_exclusive({**line.__dict__,"authorization_sha256":sha256_file(auth)},run/"run_identity.json")
  else:
   for _ in range(300):
    if (run/"run_identity.json").is_file():break
    time.sleep(.1)
   else:raise BindingRefusal("rank0 run identity unavailable")
 else:
  if run.exists():raise FileExistsError("stage already exists")
  run.mkdir();atomic_json_exclusive({**line.__dict__,"authorization_sha256":sha256_file(auth)},run/"run_identity.json")
 if mode in {"smoke","pilot"}:return _dispatch_smoke_pilot_authorized(mode,authorization,line,run)
 if mode=="long":return _dispatch_long_authorized(authorization,line,run)
 if mode in {"scale-probe-1","scale-probe-4"}:return _dispatch_scale_probe_authorized(mode,authorization,line,run)
 if mode=="scale-decide":return _dispatch_scale_decide_authorized(authorization,line,run)
 if mode in {"final-train-confirm","validation-once","test-once"}:return _dispatch_evaluation_authorized(mode,authorization,line,run)
 raise AssertionError(f"unhandled mode after specialized dispatch: {mode}")
def freeze():
 if PREREG.exists():raise FileExistsError("prereg exists")
 p=verify();cfg=yaml.safe_load(CONFIG.read_text());static_payload=json.loads(STATIC.read_text());runtime=json.loads((ROOT/"results/r4e7_parent_e2e_runtime_train9_v1_20260825.json").read_text());payload={"schema":"r16_dscp_v3_prereg_v1","candidate":CAND,"status":"frozen","created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"preflight":bind(PREF),"effective_bindings":bindings(),"scientific_design":"identical r16_dscp_v2 rank16/29ch/router/basis/panels/loss/gates; only real stage engine and lineage added","checkpoint":p["checkpoint"],"space_gate":p["space_gate"],"sealed_metadata":p["sealed_metadata"],"config":cfg,"commands":cfg["commands"],"authorization_command":"env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python scripts/train_r16_dscp_v3.py --mode authorize-stage --stage STAGE --output results/r16_dscp_v3/authorizations/STAGE.json --input-checkpoint BEST_PT","engine_symbols":static_payload["engine_symbols"],"static":bind(STATIC),"tests":static_payload["tests"],"identities":identities(),"parent_runtime":{"mean_s":runtime["aggregate"]["arithmetic_mean_runtime_s"],"p95_s":runtime["aggregate"]["nearest_rank_p95_runtime_s"],"traditional_speedup":runtime["aggregate"]["reference_over_mean_speedup_x"]},"claim_boundary":"Preparation and executable engine only; no train, validation, or test result. Parent protocol speedup is only 1.608x. Complete-candidate E2E includes the parent, so the immutable 10x traditional gate is currently impossible unless independently measured latency changes; this does not weaken or skip the gate.","rollback":cfg["rollback"],"authorization_chain":cfg["authorization_chain"],"artifacts":{"fields_serialized":False,"coefficient_maps_serialized":False,"maximum_new_bytes":64*1024**2},"sealed":{"train_future_read":False,"validation_read":False,"test_id_read":False,"validation_test_metadata_only":True,"authorizations_created_by_prep":False}}
 atomic_json_exclusive(payload,PREREG);print(json.dumps(bind(PREREG)))
def authorize_stage(stage,output,input_checkpoint=None):
 if not PREREG.is_file() or json.loads(PREREG.read_text()).get("status")!="frozen":raise BindingRefusal("frozen prereg required")
 pre=verify();cfg=yaml.safe_load(CONFIG.read_text());paths={
  "smoke":[PREREG],"pilot":[OUT/"smoke/terminal.json"],"scale-probe-1":[OUT/"pilot/terminal.json"],"scale-probe-4":[OUT/"pilot/terminal.json"],"scale-decide":[OUT/"scale-probe-1/terminal.json",OUT/"scale-probe-4/terminal.json"],"long":[OUT/"pilot/terminal.json",OUT/"scale-decide/terminal.json"],"final-train-confirm":[OUT/"long/terminal.json"],"validation-once":[OUT/"final-train-confirm/terminal.json",OUT/"final-train-confirm/candidate_lock.json"],"test-once":[OUT/"validation-once/terminal.json",OUT/"final-train-confirm/candidate_lock.json"]}
 if stage not in paths:raise BindingRefusal("unknown authorization stage")
 prereq={};allowed_status={"completed_train","passed","frozen"};expected_checkpoints=[];locked_checkpoint_sha=None
 for path in paths[stage]:
  if not path.is_file():raise BindingRefusal(f"prerequisite absent: {path}")
  observed=json.loads(path.read_text())
  if "terminal" in path.name and observed.get("status") not in allowed_status:raise BindingRefusal(f"prerequisite not passed: {path}")
  checkpoint=observed.get("checkpoint",{})
  if isinstance(checkpoint,dict) and checkpoint.get("best"):expected_checkpoints.append(str(Path(checkpoint["best"]).resolve()))
  if path.name=="candidate_lock.json":locked_checkpoint_sha=observed.get("checkpoint_sha256")
  prereq[str(path.resolve())]=sha256_file(path)
 input_sha="none";input_identity=pre["run_identity"]
 if input_checkpoint is not None:
  input_path=Path(input_checkpoint).resolve()
  if not input_path.is_file():raise BindingRefusal("input checkpoint absent")
  if expected_checkpoints and str(input_path) not in expected_checkpoints:raise BindingRefusal("input checkpoint is not prerequisite best")
  input_sha=sha256_file(input_path)
  if locked_checkpoint_sha is not None and input_sha!=locked_checkpoint_sha:raise BindingRefusal("input checkpoint differs from candidate lock")
  input_identity=torch.load(input_path,map_location="cpu",weights_only=False).get("run_identity",{})
 elif stage!="smoke" and stage!="scale-decide":raise BindingRefusal("stage requires input checkpoint")
 line=Lineage(CAND,stage,pre["run_identity"]["run_digest"],sha256_file(ENGINE),sha256_file(CONFIG),PANELS_SHA256,BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PARENT_SHA256,input_sha)
 payload={"schema":AUTH_SCHEMA,"status":"authorized",**line.__dict__,"input_checkpoint_path":None if input_checkpoint is None else str(Path(input_checkpoint).resolve()),"input_run_identity":input_identity,"prerequisite_bindings":prereq,"gates_sha256":canonical_sha(cfg["gates"]),"effective_script_sha256":sha256_file(SCRIPT),"preregistration_sha256":sha256_file(PREREG),"created_by":"explicit_authorize-stage_mode"}
 if stage.startswith("scale-probe"):payload["pilot_terminal_sha256"]=sha256_file(OUT/"pilot/terminal.json")
 if stage=="scale-decide":payload.update(scale1_terminal_sha256=sha256_file(OUT/"scale-probe-1/terminal.json"),scale4_terminal_sha256=sha256_file(OUT/"scale-probe-4/terminal.json"))
 if stage in {"validation-once","test-once"}:payload["candidate_lock_sha256"]=sha256_file(OUT/"final-train-confirm/candidate_lock.json")
 if stage=="test-once":payload["validation_terminal_sha256"]=sha256_file(OUT/"validation-once/terminal.json")
 payload["authorization_digest"]=canonical_sha(payload);atomic_json_exclusive(payload,Path(output));print(json.dumps(bind(output)))
def main():
 q=argparse.ArgumentParser();q.add_argument("--mode",required=True,choices=("prep","static","smoke","pilot","scale-probe-1","scale-probe-4","scale-decide","long","final-train-confirm","validation-once","test-once","freeze-prereg","authorize-stage"));q.add_argument("--devices",default="0");q.add_argument("--authorization");q.add_argument("--stage");q.add_argument("--output");q.add_argument("--input-checkpoint");a=q.parse_args()
 if a.mode=="prep":prep()
 elif a.mode=="static":print(json.dumps(static()))
 elif a.mode=="freeze-prereg":freeze()
 elif a.mode=="authorize-stage":
  if not a.stage or not a.output:raise BindingRefusal("authorize-stage requires --stage and --output")
  authorize_stage(a.stage,a.output,a.input_checkpoint)
 else:
  if not a.authorization:raise BindingRefusal("authorization required")
  run_authorized(a.mode,a.devices,a.authorization)
if __name__=="__main__":main()
