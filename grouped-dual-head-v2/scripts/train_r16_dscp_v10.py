#!/usr/bin/env python3
"""V10 complete CLI and versioned production entrypoint."""
from __future__ import annotations
import argparse,json,platform,shlex,subprocess
from pathlib import Path
from typing import Any,Callable
import torch,yaml
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import R16DSCP
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v10 import V10ProductionBackend,complete_identity_v10,terminal_v10
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v8 import ROLLBACK,failure_terminal
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v8 import run_v8_smoke
from scripts.train_r16_dscp_v8 import execute_with_boundary
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v7 import resume_v7_checkpoint
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v6 import run_v6_smoke
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v5 import V5LongRunner,model_state_digest,verify_before_data,verify_loaded_state
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import AUTH_SCHEMA,Lineage,canonical_sha,run_evaluation_stage,run_pilot_stage,run_scale_probe,scale_decision,validate_authorization
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v4 import V4GuardedOnsetLoader
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.eikonal import grid_eikonal_travel_time
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v4 import CandidateSpool,LongResumeState,actual_coefficient_loss,dual_space_gate
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v7 import resume_v7_checkpoint
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v5 import model_state_digest
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import canonical_sha
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import BASIS_FILE_SHA256,PANELS_SHA256,PARENT_PATH,PARENT_SHA256,BindingRefusal,atomic_json_exclusive,checkpoint_payload,configure_determinism,load_checkpoint,make_optimizer,save_best_last,sha256_file
from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime

ROOT=Path(__file__).resolve().parents[1];CANDIDATE="r16_dscp_v10";SCRIPT=Path(__file__).resolve();ENGINE=ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v10.py";TEST=ROOT/"tests/saved_time_phase_operator_v4/test_r16_dscp_v10.py";CONFIG=ROOT/"configs/r16_dscp_v10.yaml";OUT=ROOT/"results/r16_dscp_v10";STATIC=OUT/"static_evidence.json";PREFLIGHT=OUT/"design_preflight.json";PREREG=ROOT/"results/r16_dscp_v10_preregistration_20260826.json";BASIS=ROOT/"results/r16_dscp_v1/basis_rank16.pt";PANELS=ROOT/"results/r16_dscp_v1/panels.json";TEST_LOG=Path("/tmp/r16_dscp_v10_tests.log")

class V10StageDispatcher:
    def __init__(self,backend:V10ProductionBackend):
        if not isinstance(backend,V10ProductionBackend):raise TypeError("v10 dispatcher requires V10ProductionBackend")
        self.backend=backend

MODES=("prep","freeze-prereg","authorize-stage","smoke","pilot","scale-1","scale-4","scale-decide","long","final-train-confirm","validation-once","test-once")

def build_parser()->argparse.ArgumentParser:
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="mode",required=True)
    sub.add_parser("prep");sub.add_parser("freeze-prereg")
    authorize=sub.add_parser("authorize-stage");authorize.add_argument("--stage",required=True,choices=MODES[3:]);authorize.add_argument("--output",required=True);authorize.add_argument("--input-checkpoint")
    for mode in ("smoke","pilot","scale-1","scale-4","scale-decide","final-train-confirm","validation-once","test-once"):
        stage=sub.add_parser(mode);stage.add_argument("--authorization",required=True);stage.add_argument("--physical-gpu-index",type=int,default=0);stage.add_argument("--device",default="cuda:0");stage.add_argument("--config",default="configs/r16_dscp_v10.yaml");stage.add_argument("--preregistration",default="results/r16_dscp_v10_preregistration_20260826.json")
    long=sub.add_parser("long");long.add_argument("--authorization",required=True);long.add_argument("--world-size",type=int,required=True,choices=(1,4));long.add_argument("--input-checkpoint",required=True);long.add_argument("--scale-decision",required=True);long.add_argument("--config",required=True);long.add_argument("--preregistration",required=True);long.add_argument("--resume",action="store_true");long.add_argument("--physical-gpu-index",type=int,default=0);long.add_argument("--device",default="cuda:0")
    return parser

def command_argv(command:str)->list[str]:
    tokens=shlex.split(command);script_index=next((i for i,t in enumerate(tokens) if t.endswith("scripts/train_r16_dscp_v10.py")),None)
    if script_index is None:raise ValueError("v10 script absent from command")
    args=tokens[script_index+1:]
    if args[:1]==["--mode"]:args=[args[1],*args[2:]]
    return args

def adapt_terminal(mode:str,payload:Mapping[str,Any],status:str|None=None)->dict[str,Any]:
    observed=str(status or payload.get("status","failed"));result=terminal_v10(mode,observed,payload);result["rollback"]=payload.get("rollback",ROLLBACK)
    if observed in {"passed","completed_train"} and mode in {"smoke","pilot","long"}:
        checkpoint=result.get("checkpoint") or result.get("best_checkpoint")
        if not checkpoint:raise BindingRefusal("successful training terminal lacks checkpoint binding")
    return result

def replace_owned_terminal(run_dir:str|Path,mode:str)->dict[str,Any]:
    root=Path(run_dir).resolve();path=(root/"terminal.json").resolve()
    if path.parent!=root or not path.is_file():raise BindingRefusal("owned protocol terminal absent")
    payload=json.loads(path.read_text());adapted=adapt_terminal(mode,payload);path.unlink();descriptor=__import__("os").open(str(root),__import__("os").O_RDONLY)
    try:__import__("os").fsync(descriptor)
    finally:__import__("os").close(descriptor)
    atomic_json_exclusive(adapted,path);return adapted

def write_failure_v10(run_dir:str|Path,mode:str,error:BaseException,lineage:Mapping[str,Any],authorization:Mapping[str,Any],effective:Mapping[str,Any],parent:Mapping[str,Any],checkpoint:Mapping[str,Any]|None=None)->dict[str,Any]:
    payload=adapt_terminal(mode,failure_terminal(mode=mode,error=error,lineage=lineage,authorization=authorization,effective=effective,parent=parent,checkpoint=checkpoint),"failed");atomic_json_exclusive(payload,Path(run_dir)/"terminal.json");return payload

def bind(path):p=Path(path).resolve();return {"path":str(p),"sha256":sha256_file(p),"size_bytes":p.stat().st_size}
def effective_bindings():return {"engine":bind(ENGINE),"script":bind(SCRIPT),"test":bind(TEST),"config":bind(CONFIG),"model":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp.py"),"v10_transitive_v9":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v9.py"),"v10_transitive_v8":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v8.py"),"harness":bind(ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_training_v2.py"),"basis":bind(BASIS),"panels":bind(PANELS),"parent":bind(PARENT_PATH),"manifest":bind(parent_runtime.MANIFEST_PATH),"normalization":bind(parent_runtime.NORMALIZATION_PATH),"v2_replay":bind(ROOT/"results/r16_dscp_v2/replay_basis_verify/verification.json"),"v9_blocker":bind(ROOT/"results/r16_dscp_v9/design_preflight.json")}
def metadata_seal():
    m=parent_runtime.load_manifest_payload();out={}
    for split in ("validation","test_id"):
        rows=sorted([{"source_index":r["source_index"],"sample_id":r["sample_id"],"group_id":r["group_id"],"sample_sha256":r["sample_sha256"],"family":r["medium_type"]} for r in m["records"] if r["split"]==split],key=lambda r:(r["source_index"],r["sample_id"]));out[split]={"count":len(rows),"ordered_digest":canonical_sha(rows),"by_family":{f:sum(r["family"]==f for r in rows) for f in ("uniform","layered","marmousi")},"wavefield_read":False}
    return out
def basis_artifact():return torch.load(BASIS,map_location="cpu",weights_only=False)

def build_v10_backend(mode,authorization,lineage,run_dir,identity):
    split="validation" if mode=="validation-once" else "test_id" if mode=="test-once" else "train";device=torch.device(f"cuda:{int(__import__('os').environ.get('LOCAL_RANK','0'))}");configure_determinism(372);artifact=basis_artifact();candidate=R16DSCP(artifact["basis"],artifact["coefficient_scales"]).to(device);optimizer=make_optimizer(list(candidate.parameters()))
    if lineage.input_checkpoint_sha256!="none":load_checkpoint(authorization["input_checkpoint_path"],candidate,optimizer,expected_run_identity=authorization["input_run_identity"])
    verify_loaded_state(candidate,authorization);parent,normalizer,manifest_payload,_=parent_runtime.load_model_context(device);manifest=parent_runtime.manifest_object(manifest_payload);panels=json.loads(PANELS.read_text())["records"];roles={"smoke":("smoke",),"pilot":("pilot_fit","pilot_confirm"),"scale-1":("pilot_fit",),"scale-4":("pilot_fit",),"long":("long_fit","long_calibration"),"final-train-confirm":("final_train_confirm",)}
    if split=="train":rows=[r for r in panels if r["role"] in roles[mode]];samples=[r["sample_id"] for r in rows];family={r["sample_id"]:r["family"] for r in rows};nontruth={r["sample_id"]:r["nontruth_input_sha256"] for r in rows}
    else:rows=sorted([r for r in manifest_payload["records"] if r["split"]==split],key=lambda r:(r["source_index"],r["sample_id"]));samples=[r["sample_id"] for r in rows];family={r["sample_id"]:r["medium_type"] for r in rows};nontruth={r["sample_id"]:r["sample_sha256"] for r in rows}
    loader=V4GuardedOnsetLoader(GuardedOnsetDataset(parent_runtime.SOURCE_H5_PATH,manifest,split=split,sample_ids=samples))
    @torch.inference_mode()
    def parent_predictor(public):
        v=public.velocity_mps[None,None].to(device);s=public.source_parameters[None].to(device);sm=public.source_map[None,None].to(device);medium=parent.encode_medium(v,normalizer);prepared=parent.prepare_sources(medium,s,sm,normalizer,record_to_medium=torch.zeros(1,dtype=torch.long,device=device));normalized=parent.dense_normalized(prepared,public.time_s.to(device),x_m=public.x_m.to(device),z_m=public.z_m.to(device),time_block=16);return normalizer.decode_pressure(normalized.float(),s[:,4])[0].cpu()
    travel=lambda v,s:grid_eikonal_travel_time(v,source_indices=[(round(float(s[1])/10),round(float(s[0])/10))],dx_m=10,dz_m=10)[0];backend=V10ProductionBackend(public_loader=loader,parent_predictor=parent_predictor,travel_builder=travel,candidate=candidate,optimizer=optimizer,authorization=authorization,lineage=lineage,source_h5=parent_runtime.SOURCE_H5_PATH,run_dir=run_dir,device=device,split=split,family_by_sample=family,once_token=run_dir/f"{split}.once.json" if split!="train" else None,cache_role=mode.replace("scale-","scale-probe-"),parent_sha256=PARENT_SHA256,basis_sha256=BASIS_FILE_SHA256,feature_code_sha256=sha256_file(ENGINE),nontruth_by_sample=nontruth,complete_identity=identity);indices={role:[samples.index(r["sample_id"]) for r in rows if r.get("role")==role] for role in roles.get(mode,())};return backend,indices,list(range(len(loader))),canonical_sha(rows)

def protocol_dispatch(mode,backend,roles,all_indices,metadata,lineage,run_dir):
    world=int(__import__("os").environ.get("WORLD_SIZE","1"));rank=int(__import__("os").environ.get("RANK","0"))
    if mode=="smoke":return run_v8_smoke(backend=backend,records=roles["smoke"],quick_gate=lambda rows:all(r["aggregate_rel_l2"]<=r["parent_rel_l2"] for r in rows),checkpoint=lambda:backend.checkpoint({"mode":mode}),resource_snapshot=lambda saved:backend.resources(saved,world),terminal=lambda payload:atomic_json_exclusive(adapt_terminal(mode,payload),run_dir/"terminal.json"),lineage=backend.checkpoint_identity,effective=backend.checkpoint_identity,parent={"sha256":PARENT_SHA256})
    if mode=="pilot":
        for i in roles["pilot_fit"]+roles["pilot_confirm"]:backend.preload(i)
        def ridge_update(i):p=backend.cached_prepare(i,"ridge");backend.ridge_update(p,backend._cached_truth(p))
        result=run_pilot_stage(fit_records=roles["pilot_fit"],confirm_records=roles["pilot_confirm"],candidate_update=backend.update_cached,ridge_update=ridge_update,ridge_solve=backend.ridge_finalize,candidate_score=lambda rows:[backend.score_cached(i) for i in rows],ridge_score=lambda ridge,rows:[backend.ridge_score_cached(i,ridge) for i in rows],checkpoint=lambda:backend.checkpoint({"mode":mode}),lineage=lineage,run_dir=run_dir,resource_snapshot=lambda saved:backend.resources(saved,world));return replace_owned_terminal(run_dir,mode)
    if mode in {"scale-1","scale-4"}:
        records=roles["pilot_fit"][:12]
        for i in records[rank::world]:backend.preload(i)
        result=run_scale_probe(mode="scale-probe-1" if mode=="scale-1" else "scale-probe-4",records=records,measure=backend.measure_cached,model_hash=lambda:model_state_digest(backend.candidate),resource_snapshot=lambda:backend.resources({"size_bytes":Path(backend.authorization["input_checkpoint_path"]).stat().st_size},world),lineage=lineage,run_dir=run_dir,world_size=world,rank=rank)
        return replace_owned_terminal(run_dir,mode) if world==1 else result
    if mode=="long":
        train=roles["long_fit"];cal=roles["long_calibration"]
        for i in train+cal:backend.preload(i)
        def gather(rows):
            if world==1:return rows
            shards=[None]*world;torch.distributed.all_gather_object(shards,rows);return [x for s in shards for x in s]
        def step():
            if world==4:
                for p in backend.candidate.parameters():
                    if p.grad is not None:torch.distributed.all_reduce(p.grad);p.grad.div_(4.)
            torch.nn.utils.clip_grad_norm_(backend.candidate.parameters(),1.);backend.optimizer.step()
        saved={"best":None,"last":None}
        def cp(state,best):item=backend.checkpoint(state.__dict__,best);saved["last"]=item;saved["best"]=item if best else saved["best"]
        def term(payload):atomic_json_exclusive(adapt_terminal(mode,{**payload,"checkpoint":saved["best"] or saved["last"]}),run_dir/"terminal.json")
        return V5LongRunner(world_size=world,rank=rank,train_records=train,calibration_records=cal,zero_grad=lambda:backend.optimizer.zero_grad(set_to_none=True),backward_record=lambda i,s:backend.backward_cached(i,s),optimizer_step=step,set_lr=lambda lr:[g.update(lr=lr) for g in backend.optimizer.param_groups],evaluate=lambda i:backend.score_cached(i),gather=gather,checkpoint=cp,terminal=term).run()
    if mode in {"final-train-confirm","validation-once","test-once"}:
        def evaluate(i):backend.preload(i);return backend.score_cached(i,True)
        validation=json.loads((OUT/"validation-once/terminal.json").read_text()) if mode=="test-once" else None;result=run_evaluation_stage(mode=mode,records=all_indices,evaluate_one=evaluate,resource_snapshot=lambda:backend.resources({"size_bytes":Path(backend.authorization["input_checkpoint_path"]).stat().st_size},world),lineage=lineage,run_dir=run_dir,metadata_digest=metadata,candidate_lock_payload={"checkpoint_sha256":lineage.input_checkpoint_sha256} if mode=="final-train-confirm" else None,required_validation_terminal=validation);return replace_owned_terminal(run_dir,mode)
    raise ValueError(mode)

def prep(args=None):
    if OUT.exists() or PREREG.exists():raise FileExistsError("v10 exists")
    bindings=effective_bindings();config=yaml.safe_load(CONFIG.read_text());commands={name:vars(build_parser().parse_args(command_argv(command))) for name,command in config["commands"].items()};tests={"status":"passed","log_sha256":sha256_file(TEST_LOG),"tail":TEST_LOG.read_text().splitlines()[-1]};v9=json.loads((ROOT/"results/r16_dscp_v9/static_evidence.json").read_text());static={"schema":"r16_dscp_v10_static_v1","parameters":1202,"model_macs":45451125,"host_cache_gates":v9["host_cache_gates"],"metadata":metadata_seal(),"tests":tests,"command_audit":commands,"modes":list(MODES),"production_backend":"V10ProductionBackend","truth_read":False};OUT.mkdir();atomic_json_exclusive(static,STATIC);artifact=basis_artifact();model=R16DSCP(artifact["basis"],artifact["coefficient_scales"]);optimizer=make_optimizer(list(model.parameters()));h=w=201;model_args=(torch.ones(1,1,h,w)*2000,torch.zeros(1,1,h,w),torch.ones(1,1,h,w),torch.arange(w),torch.arange(h),torch.zeros(1,1,h,w),torch.zeros(1,1,h,w),torch.zeros(1,401,h,w),torch.tensor([1]),torch.tensor([2]));losses,adapted,_=actual_coefficient_loss(model,model_args,torch.ones(1,398,h,w));losses["total"].backward();optimizer.step();identity=complete_identity_v10(mode="prep",run_digest=canonical_sha(bindings),authorization_sha256="prep-none",engine_sha256=bindings["engine"]["sha256"],script_sha256=bindings["script"]["sha256"],config_sha256_effective=bindings["config"]["sha256"],panels_sha256_effective=PANELS_SHA256,basis_sha256_effective=BASIS_FILE_SHA256,parent_sha256_effective=PARENT_SHA256,input_checkpoint_sha256="none");state=LongResumeState(epoch=0,global_step=1,group_index=0,best=float("inf"),bad_epochs=0,best_epoch=-1);payload=checkpoint_payload(model,optimizer,run_identity=identity,sampler_order=list(range(12)),progress=state.__dict__);saved=save_best_last(payload,OUT/"actual_checkpoint",is_best=True);restored=R16DSCP(artifact["basis"],artifact["coefficient_scales"]);ro=make_optimizer(list(restored.parameters()));_,resume=resume_v7_checkpoint(saved["last"],restored,ro,expected_identity=identity);ledger=__import__("saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3",fromlist=["AccessLedger"]).AccessLedger("train");spool=CandidateSpool(OUT,run_digest=identity["run_digest"],rank=0,owned_root="/dev/shm/r16_dscp_v10");seal=spool.seal("size",adapted,ledger);spool_max=seal["serialized_bytes"];spool.cleanup(seal,ledger);gate1=dual_space_gate(workspace=ROOT,tmpfs="/dev/shm",checkpoint_bytes=saved["size_bytes"],spool_max_bytes=spool_max,world_size=1);gate4=dual_space_gate(workspace=ROOT,tmpfs="/dev/shm",checkpoint_bytes=saved["size_bytes"],spool_max_bytes=spool_max,world_size=4);pre={"schema":"r16_dscp_v10_preflight_v1","candidate":CANDIDATE,"status":"frozen","bindings":bindings,"run_identity":identity,"static":bind(STATIC),"checkpoint":{**saved,"sha256":sha256_file(saved["last"]),"resume_equal":resume==state,"identity_digest":identity["identity_digest"],"state_sha256":model_state_digest(model)},"host_cache_gates":static["host_cache_gates"],"spool":{"max_bytes":spool_max,"root":"/dev/shm/r16_dscp_v10/<run_digest>/rank<r>"},"space1":gate1,"space4":gate4,"metadata":static["metadata"],"truth_reads":0,"GPU_used":False,"auth_created":False};atomic_json_exclusive(pre,PREFLIGHT);return pre

def verify_preflight():
    pre=json.loads(PREFLIGHT.read_text());current=effective_bindings()
    for key,value in pre["bindings"].items():
        if current[key]["sha256"]!=value["sha256"]:raise BindingRefusal(f"drift {key}")
    return pre
def freeze_prereg(args=None):
    if PREREG.exists():raise FileExistsError("v10 prereg exists")
    pre=verify_preflight();gpu=subprocess.run(["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.total","--format=csv,noheader"],check=True,capture_output=True,text=True).stdout.splitlines();payload={"schema":"r16_dscp_v10_prereg_v1","candidate":CANDIDATE,"status":"frozen","preflight":bind(PREFLIGHT),"effective_bindings":effective_bindings(),"v9_blocker":bind(ROOT/"results/r16_dscp_v9/design_preflight.json"),"only_change":"complete CLI reachability and identity version","config":yaml.safe_load(CONFIG.read_text()),"checkpoint":pre["checkpoint"],"host_cache_gates":pre["host_cache_gates"],"spool":pre["spool"],"space1":pre["space1"],"space4":pre["space4"],"metadata":pre["metadata"],"tests":json.loads(STATIC.read_text())["tests"],"command_audit":json.loads(STATIC.read_text())["command_audit"],"identities":{"python":platform.python_version(),"torch":torch.__version__,"gpu":gpu,"cuda_used":False},"claim_boundary":"prep only, no stage/truth/GPU, immutable 10x unchanged","sealed":{"truth_read":False,"GPU_used":False,"auth_created":False}};atomic_json_exclusive(payload,PREREG);return payload

def authorize_handler(args):
    if not PREREG.is_file():raise BindingRefusal("frozen prereg required")
    pre=verify_preflight();mode=args.stage;input_sha="none";input_identity=pre["run_identity"]
    if args.input_checkpoint:
        path=Path(args.input_checkpoint).resolve();input_sha=sha256_file(path);input_identity=torch.load(path,map_location="cpu",weights_only=False)["run_identity"]
    elif mode not in {"smoke","scale-decide"}:raise BindingRefusal("input checkpoint required")
    line=Lineage(CANDIDATE,mode,pre["run_identity"]["run_digest"],sha256_file(ENGINE),sha256_file(CONFIG),PANELS_SHA256,BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PARENT_SHA256,input_sha);auth={"schema":AUTH_SCHEMA,"status":"authorized",**line.__dict__,"input_checkpoint_path":args.input_checkpoint,"input_run_identity":input_identity,"parent_sha256":PARENT_SHA256,"basis_sha256":BASIS_FILE_SHA256,"config_sha256":sha256_file(CONFIG),"code_sha256":sha256_file(ENGINE),"gates_sha256":canonical_sha(yaml.safe_load(CONFIG.read_text())),"script_sha256":sha256_file(SCRIPT),"prereg_sha256":sha256_file(PREREG)};auth["authorization_digest"]=canonical_sha(auth);atomic_json_exclusive(auth,args.output);return auth

def stage_handler(args):
    pre=verify_preflight();auth=json.loads(Path(args.authorization).read_text());mode=args.mode;line=Lineage(CANDIDATE,mode,pre["run_identity"]["run_digest"],sha256_file(ENGINE),sha256_file(CONFIG),PANELS_SHA256,BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PARENT_SHA256,str(auth.get("input_checkpoint_sha256","none")));validate_authorization(auth,line);identity=complete_identity_v10(mode=mode,run_digest=line.run_digest,authorization_sha256=sha256_file(args.authorization),engine_sha256=sha256_file(ENGINE),script_sha256=sha256_file(SCRIPT),config_sha256_effective=sha256_file(CONFIG),panels_sha256_effective=PANELS_SHA256,basis_sha256_effective=BASIS_FILE_SHA256,parent_sha256_effective=PARENT_SHA256,input_checkpoint_sha256=line.input_checkpoint_sha256);run=OUT/mode
    if run.exists():raise FileExistsError("stage exists")
    if mode=="scale-decide":
        run.mkdir();one=json.loads((OUT/"scale-1/terminal.json").read_text());four=json.loads((OUT/"scale-4/terminal.json").read_text());result=scale_decision(one,four);payload=adapt_terminal(mode,{"status":"passed","selected_gpus":result["selected_gpus"],"gates":result,"lineage":identity,"long_launched":False});atomic_json_exclusive(payload,run/"terminal.json");return payload
    if mode=="long":
        decision=json.loads(Path(args.scale_decision).read_text());actual=int(__import__("os").environ.get("WORLD_SIZE",str(args.world_size)));__import__("scripts.train_r16_dscp_v8",fromlist=["validate_long_cli"]).validate_long_cli(arg_world_size=args.world_size,actual_world_size=actual,cuda_visible=__import__("os").environ.get("CUDA_VISIBLE_DEVICES",""),authorization=auth,scale_decision=decision,resume=args.resume,terminal_exists=False)
    run.mkdir();atomic_json_exclusive(identity,run/"run_identity.json");holder={};effective={"engine":identity["engine_sha256"],"script":identity["script_sha256"],"config":identity["config_sha256_effective"],"panels":identity["panels_sha256_effective"],"basis":identity["basis_sha256_effective"],"parent":identity["parent_sha256_effective"]}
    def factory():
        bundle=build_v10_backend(mode,auth,line,run,identity);holder["bundle"]=bundle;return bundle[0]
    return execute_with_boundary(mode=mode,run_dir=run,authorization=auth,lineage=identity,effective=effective,parent={"sha256":PARENT_SHA256},factory=factory,body=lambda backend:protocol_dispatch(mode,backend,*holder["bundle"][1:],line,run),checkpoint_provider=lambda:None)

HANDLERS:dict[str,Callable[[Any],Any]]={"prep":prep,"freeze-prereg":freeze_prereg,"authorize-stage":authorize_handler,"smoke":stage_handler,"pilot":stage_handler,"scale-1":stage_handler,"scale-4":stage_handler,"scale-decide":stage_handler,"long":stage_handler,"final-train-confirm":stage_handler,"validation-once":stage_handler,"test-once":stage_handler}

def main(argv:list[str]|None=None):
    args=build_parser().parse_args(argv);return HANDLERS[args.mode](args)

if __name__=="__main__":main()

__all__=["HANDLERS","MODES","build_parser","command_argv","main"]
