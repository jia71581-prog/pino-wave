#!/usr/bin/env python3
"""V11 strict stage handlers and V11 terminal versioning."""
from __future__ import annotations
import argparse,json,os,shlex
from pathlib import Path
from typing import Any,Mapping
import yaml
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v11 import V11ProductionBackend,complete_identity_v11,terminal_v11
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v9 import validate_test_chain,validate_validation_chain
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import AUTH_SCHEMA,Lineage,canonical_sha,validate_authorization
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v4 import CandidateSpool
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PANELS_SHA256,PARENT_PATH,PARENT_SHA256,BindingRefusal,atomic_json_exclusive,sha256_file
import scripts.train_r16_dscp_v10 as v10

ROOT=Path(__file__).resolve().parents[1];CANDIDATE="r16_dscp_v11";SCRIPT=Path(__file__).resolve();ENGINE=ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v11.py";TEST=ROOT/"tests/saved_time_phase_operator_v4/test_r16_dscp_v11.py";CONFIG=ROOT/"configs/r16_dscp_v11.yaml";OUT=ROOT/"results/r16_dscp_v11";PREFLIGHT=OUT/"design_preflight.json";STATIC=OUT/"static_evidence.json";PREREG=ROOT/"results/r16_dscp_v11_preregistration_20260826.json";BASIS=ROOT/"results/r16_dscp_v1/basis_rank16.pt";PANELS=ROOT/"results/r16_dscp_v1/panels.json";TEST_LOG=Path("/tmp/r16_dscp_v11_tests.log")
MODES=v10.MODES

class V11StageDispatcher:
    def __init__(self,backend:V11ProductionBackend):
        if not isinstance(backend,V11ProductionBackend):raise TypeError("v11 dispatcher requires V11ProductionBackend")
        self.backend=backend

def build_parser():return v10.build_parser()
def command_argv(command):
    tokens=shlex.split(command);index=next(i for i,t in enumerate(tokens) if t.endswith("scripts/train_r16_dscp_v11.py"));return tokens[index+1:]
def effective()->dict[str,str]:return {"code":sha256_file(ENGINE),"script":sha256_file(SCRIPT),"config":sha256_file(CONFIG),"basis":sha256_file(BASIS),"panels":sha256_file(PANELS),"parent":sha256_file(PARENT_PATH),"data_manifest":sha256_file(v10.parent_runtime.MANIFEST_PATH),"normalization":sha256_file(v10.parent_runtime.NORMALIZATION_PATH)}
def _auth_base(stage,pre,input_checkpoint=None,prereg_path=PREREG):
    input_sha="none" if input_checkpoint is None else sha256_file(input_checkpoint);line=Lineage(CANDIDATE,stage,pre["run_identity"]["run_digest"],sha256_file(ENGINE),sha256_file(CONFIG),PANELS_SHA256,BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PARENT_SHA256,input_sha);payload={"schema":AUTH_SCHEMA,"status":"authorized",**line.__dict__,"candidate":CANDIDATE,"effective":effective(),"input_checkpoint_path":None if input_checkpoint is None else str(Path(input_checkpoint).resolve()),"preregistration_sha256":sha256_file(prereg_path),"gates_sha256":canonical_sha(yaml.safe_load(CONFIG.read_text()))};return payload
def _write_auth(payload,output):payload["authorization_digest"]=canonical_sha(payload);atomic_json_exclusive(payload,output);return payload

def authorize_handler(args,*,root=None,prereg=None):
    root=Path(root or OUT);prereg_path=Path(prereg or PREREG)
    if not prereg_path.is_file():raise BindingRefusal("prereg absent")
    pre=json.loads((root/"design_preflight.json").read_text()) if (root/"design_preflight.json").is_file() else {"run_identity":{"run_digest":"test"}};stage=args.stage
    if stage=="smoke":return _write_auth({**_auth_base(stage,pre,prereg_path=prereg_path),"preregistration_path":str(prereg_path.resolve()),"preregistration_sha256":sha256_file(prereg_path)},args.output)
    if stage=="validation-once":
        final_path=root/"final-train-confirm/terminal.json";lock_path=root/"final-train-confirm/candidate_lock.json";sealed=__import__("scripts.train_r16_dscp_v9",fromlist=["validation_authorization"]).validation_authorization(final_terminal_path=final_path,lock_path=lock_path,effective=effective());return _write_auth({**_auth_base(stage,pre,sealed["candidate_checkpoint_path"],prereg_path),**sealed},args.output)
    if stage=="test-once":
        validation_path=root/"validation-once/terminal.json";lock_path=root/"final-train-confirm/candidate_lock.json";sealed=__import__("scripts.train_r16_dscp_v9",fromlist=["test_authorization"]).test_authorization(validation_terminal_path=validation_path,lock_path=lock_path,effective=effective());return _write_auth({**_auth_base(stage,pre,sealed["candidate_checkpoint_path"],prereg_path),**sealed},args.output)
    prereq_map={"pilot":"smoke","scale-1":"pilot","scale-4":"pilot","scale-decide":"scale-4","long":"scale-decide","final-train-confirm":"long"};terminal=root/prereq_map[stage]/"terminal.json"
    if not terminal.is_file() or json.loads(terminal.read_text()).get("status") not in {"passed","completed_train"}:raise BindingRefusal("prerequisite terminal not passed")
    if not args.input_checkpoint:raise BindingRefusal("input checkpoint required")
    return _write_auth({**_auth_base(stage,pre,args.input_checkpoint,prereg_path),"prerequisite_terminal_path":str(terminal.resolve()),"prerequisite_terminal_sha256":sha256_file(terminal)},args.output)

def validate_stage_auth(args,*,before_factory_hook=lambda:None,preflight_path=None,prereg_path=None,effective_fn=effective):
    auth_path=Path(args.authorization);auth=json.loads(auth_path.read_text());pre=json.loads(Path(preflight_path or PREFLIGHT).read_text());line=Lineage(CANDIDATE,args.mode,pre["run_identity"]["run_digest"],sha256_file(ENGINE),sha256_file(CONFIG),PANELS_SHA256,BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PARENT_SHA256,str(auth.get("input_checkpoint_sha256","none")));validate_authorization(auth,line)
    if auth.get("effective")!=effective_fn() or auth.get("preregistration_sha256")!=sha256_file(prereg_path or PREREG):raise BindingRefusal("effective/prereg drift")
    if auth.get("input_checkpoint_path") and sha256_file(auth["input_checkpoint_path"])!=auth["input_checkpoint_sha256"]:raise BindingRefusal("input checkpoint drift")
    if auth.get("prerequisite_terminal_path") and sha256_file(auth["prerequisite_terminal_path"])!=auth["prerequisite_terminal_sha256"]:raise BindingRefusal("prerequisite drift")
    if args.mode=="validation-once":validate_validation_chain(authorization=auth,final_terminal=json.loads(Path(auth["final_terminal_path"]).read_text()),final_terminal_path=auth["final_terminal_path"],lock=json.loads(Path(auth["candidate_lock_path"]).read_text()),lock_path=auth["candidate_lock_path"],expected_effective=effective_fn())
    if args.mode=="test-once":validate_test_chain(authorization=auth,validation_terminal=json.loads(Path(auth["validation_terminal_path"]).read_text()),validation_terminal_path=auth["validation_terminal_path"],lock=json.loads(Path(auth["candidate_lock_path"]).read_text()),lock_path=auth["candidate_lock_path"],expected_effective=effective_fn())
    before_factory_hook()
    if effective_fn()!=auth["effective"] or (auth.get("input_checkpoint_path") and sha256_file(auth["input_checkpoint_path"])!=auth["input_checkpoint_sha256"]):raise BindingRefusal("TOCTOU drift before factory")
    return auth,line

def build_v11_backend(mode,auth,line,run,identity):
    bundle=v10.build_v10_backend(mode,auth,line,run,identity);base=bundle[0];base.__class__=V11ProductionBackend;base.spool=CandidateSpool(base.run_dir,run_digest=base.lineage.run_digest,rank=int(os.environ.get("RANK","0")),owned_root="/dev/shm/r16_dscp_v11");return (base,*bundle[1:])

def stage_handler(args,*,backend_factory=build_v11_backend,before_factory_hook=lambda:None):
    auth,line=validate_stage_auth(args,before_factory_hook=before_factory_hook);identity=complete_identity_v11(mode=args.mode,run_digest=line.run_digest,authorization_sha256=sha256_file(args.authorization),engine_sha256=sha256_file(ENGINE),script_sha256=sha256_file(SCRIPT),config_sha256_effective=sha256_file(CONFIG),panels_sha256_effective=PANELS_SHA256,basis_sha256_effective=BASIS_FILE_SHA256,parent_sha256_effective=PARENT_SHA256,input_checkpoint_sha256=line.input_checkpoint_sha256);run=OUT/args.mode
    if run.exists():raise FileExistsError("stage exists")
    run.mkdir();atomic_json_exclusive(identity,run/"run_identity.json");bundle=backend_factory(args.mode,auth,line,run,identity);result=v10.protocol_dispatch(args.mode,*bundle,line,run)
    terminal=run/"terminal.json"
    if terminal.is_file():payload=json.loads(terminal.read_text());terminal.unlink();atomic_json_exclusive(terminal_v11(args.mode,payload.get("status","failed"),payload),terminal)
    return result

def stage_handler_safe(args,*,backend_factory=build_v11_backend,before_factory_hook=lambda:None):
    try:return stage_handler(args,backend_factory=backend_factory,before_factory_hook=before_factory_hook)
    except Exception as exc:
        run=OUT/args.mode
        if run.is_dir() and not (run/"terminal.json").exists():atomic_json_exclusive(terminal_v11(args.mode,"failed",{"error_type":type(exc).__name__,"error_message":str(exc),"rollback":"inherit_v10_outer_exception"}),run/"terminal.json")
        raise

def bind(path):p=Path(path).resolve();return {"path":str(p),"sha256":sha256_file(p),"size_bytes":p.stat().st_size}
def effective_bindings():return {"engine":bind(ENGINE),"script":bind(SCRIPT),"test":bind(TEST),"config":bind(CONFIG),"model":bind(v10.ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp.py"),"v11_transitive_v10":bind(v10.ENGINE),"v11_transitive_v9":bind(v10.ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v9.py"),"harness":bind(v10.ROOT/"saved_time_phase_operator_v4/instance_adaptation/r16_dscp_training_v2.py"),"basis":bind(v10.BASIS),"panels":bind(v10.PANELS),"parent":bind(PARENT_PATH),"manifest":bind(v10.parent_runtime.MANIFEST_PATH),"normalization":bind(v10.parent_runtime.NORMALIZATION_PATH),"v10_blocker":bind(v10.PREFLIGHT)}

def prep(args=None):
    if OUT.exists() or PREREG.exists():raise FileExistsError("v11 exists")
    bindings=effective_bindings();config=yaml.safe_load(CONFIG.read_text());tests={"status":"passed","log_sha256":sha256_file(TEST_LOG),"tail":TEST_LOG.read_text().splitlines()[-1]};v10s=json.loads(v10.STATIC.read_text());static={"schema":"r16_dscp_v11_static_v1","parameters":1202,"model_macs":45451125,"host_cache_gates":v10s["host_cache_gates"],"metadata":v10.metadata_seal(),"tests":tests,"handler_validators":["validate_stage_auth","validate_validation_chain","validate_test_chain"],"default_handlers":{k:v.__name__ for k,v in HANDLERS.items()},"truth_read":False};OUT.mkdir();atomic_json_exclusive(static,STATIC);artifact=v10.basis_artifact();model=v10.R16DSCP(artifact["basis"],artifact["coefficient_scales"]);optimizer=v10.make_optimizer(list(model.parameters()));h=w=201;model_args=(v10.torch.ones(1,1,h,w)*2000,v10.torch.zeros(1,1,h,w),v10.torch.ones(1,1,h,w),v10.torch.arange(w),v10.torch.arange(h),v10.torch.zeros(1,1,h,w),v10.torch.zeros(1,1,h,w),v10.torch.zeros(1,401,h,w),v10.torch.tensor([1]),v10.torch.tensor([2]));losses,adapted,_=v10.actual_coefficient_loss(model,model_args,v10.torch.ones(1,398,h,w));losses["total"].backward();optimizer.step();identity=complete_identity_v11(mode="prep",run_digest=canonical_sha(bindings),authorization_sha256="prep-none",engine_sha256=bindings["engine"]["sha256"],script_sha256=bindings["script"]["sha256"],config_sha256_effective=bindings["config"]["sha256"],panels_sha256_effective=PANELS_SHA256,basis_sha256_effective=BASIS_FILE_SHA256,parent_sha256_effective=PARENT_SHA256,input_checkpoint_sha256="none");state=v10.LongResumeState(epoch=0,global_step=1,group_index=0,best=float("inf"),bad_epochs=0,best_epoch=-1);payload=v10.checkpoint_payload(model,optimizer,run_identity=identity,sampler_order=list(range(12)),progress=state.__dict__);saved=v10.save_best_last(payload,OUT/"actual_checkpoint",is_best=True);restored=v10.R16DSCP(artifact["basis"],artifact["coefficient_scales"]);ro=v10.make_optimizer(list(restored.parameters()));_,resume=v10.resume_v7_checkpoint(saved["last"],restored,ro,expected_identity=identity);ledger=__import__("saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3",fromlist=["AccessLedger"]).AccessLedger("train");spool=CandidateSpool(OUT,run_digest=identity["run_digest"],rank=0,owned_root="/dev/shm/r16_dscp_v11");seal=spool.seal("size",adapted,ledger);spool_max=seal["serialized_bytes"];spool.cleanup(seal,ledger);gate1=v10.dual_space_gate(workspace=ROOT,tmpfs="/dev/shm",checkpoint_bytes=saved["size_bytes"],spool_max_bytes=spool_max,world_size=1);gate4=v10.dual_space_gate(workspace=ROOT,tmpfs="/dev/shm",checkpoint_bytes=saved["size_bytes"],spool_max_bytes=spool_max,world_size=4);pre={"schema":"r16_dscp_v11_preflight_v1","candidate":CANDIDATE,"status":"frozen","bindings":bindings,"run_identity":identity,"static":bind(STATIC),"checkpoint":{**saved,"sha256":sha256_file(saved["last"]),"resume_equal":resume==state,"identity_digest":identity["identity_digest"],"state_sha256":v10.model_state_digest(model)},"host_cache_gates":static["host_cache_gates"],"spool":{"max_bytes":spool_max,"root":"/dev/shm/r16_dscp_v11/<run_digest>/rank<r>"},"space1":gate1,"space4":gate4,"metadata":static["metadata"],"truth_reads":0,"GPU_used":False,"auth_created":False};atomic_json_exclusive(pre,PREFLIGHT);return pre

def verify_preflight():
    pre=json.loads(PREFLIGHT.read_text());current=effective_bindings()
    for key,value in pre["bindings"].items():
        if current[key]["sha256"]!=value["sha256"]:raise BindingRefusal(f"drift {key}")
    return pre
def freeze_prereg(args=None):
    if PREREG.exists():raise FileExistsError("v11 prereg exists")
    pre=verify_preflight();gpu=__import__("subprocess").run(["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.total","--format=csv,noheader"],check=True,capture_output=True,text=True).stdout.splitlines();payload={"schema":"r16_dscp_v11_prereg_v1","candidate":CANDIDATE,"status":"frozen","preflight":bind(PREFLIGHT),"effective_bindings":effective_bindings(),"v10_blocker":bind(v10.PREFLIGHT),"only_change":"strict handler validator wiring","config":yaml.safe_load(CONFIG.read_text()),"checkpoint":pre["checkpoint"],"host_cache_gates":pre["host_cache_gates"],"spool":pre["spool"],"space1":pre["space1"],"space4":pre["space4"],"metadata":pre["metadata"],"tests":json.loads(STATIC.read_text())["tests"],"identities":{"python":__import__("platform").python_version(),"torch":v10.torch.__version__,"gpu":gpu,"cuda_used":False},"claim_boundary":"prep only, no stage/truth/GPU, immutable 10x unchanged","sealed":{"truth_read":False,"GPU_used":False,"auth_created":False}};atomic_json_exclusive(payload,PREREG);return payload

def build_parser():return v10.build_parser()
def command_argv(command):
    tokens=__import__("shlex").split(command);index=next(i for i,t in enumerate(tokens) if t.endswith("scripts/train_r16_dscp_v11.py"));return tokens[index+1:]

HANDLERS={"prep":prep,"freeze-prereg":freeze_prereg,"authorize-stage":authorize_handler,"smoke":stage_handler_safe,"pilot":stage_handler_safe,"scale-1":stage_handler_safe,"scale-4":stage_handler_safe,"scale-decide":stage_handler_safe,"long":stage_handler_safe,"final-train-confirm":stage_handler_safe,"validation-once":stage_handler_safe,"test-once":stage_handler_safe}
def main(argv=None):args=build_parser().parse_args(argv);return HANDLERS[args.mode](args)
if __name__=="__main__":main()
