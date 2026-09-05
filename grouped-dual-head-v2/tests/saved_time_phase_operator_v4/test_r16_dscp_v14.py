from __future__ import annotations
import inspect,json,os,subprocess,sys,textwrap
from pathlib import Path
from types import SimpleNamespace
import pytest,torch,yaml
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v11 import V11ProductionBackend
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import AUTH_SCHEMA,Lineage,canonical_sha
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import BASIS_FILE_SHA256,BASIS_TENSOR_SHA256,PANELS_SHA256,PARENT_SHA256,checkpoint_payload,make_optimizer,save_best_last,sha256_file
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v14 import InclusiveBudgetClock,LongResumeRefusal,RuntimeClosureRefusal,V14ChainRefusal,V14ProductionBackend,build_runtime_import_closure,build_test_terminal,complete_identity_v14,inspect_long_resume,verify_runtime_import_closure
import scripts.train_r16_dscp_v14 as cli
from scripts.train_r16_dscp_v14 import V14StageDispatcher,authorize_test,authorize_validation,final_handler,test_handler as run_test_handler,validation_handler

EFFECTIVE={"code":"c","script":"s","config":"cfg","panels":"p","basis":"b","parent":"par","input":"i","data":"d"};DATA={"validation":"v","test":"t"};THRESHOLDS={"accuracy":.05,"speed":10.}

def test_end_to_end_production_handler_chain(tmp_path):
    checkpoint=tmp_path/"best.pt";checkpoint.write_bytes(b"best");final_dir=tmp_path/"final-train-confirm";terminal,lock=final_handler(run_dir=final_dir,checkpoint_path=checkpoint,thresholds=THRESHOLDS,effective=EFFECTIVE,data_hashes=DATA,evaluate=lambda:("passed",{"all":{"passed":True}}));assert lock and lock["created_before_validation"] and lock["checkpoint_path"]==str(checkpoint.resolve())
    validation_auth=authorize_validation(run_root=tmp_path,effective=EFFECTIVE);validation_path=tmp_path/"validation-once/terminal.json";validation_path.parent.mkdir();validation=validation_handler(terminal_path=validation_path,authorization=validation_auth,effective=EFFECTIVE,evaluate=lambda:("passed",{"all":{"passed":True}}));assert validation["candidate_lock_digest"]==lock["lock_digest"]
    test_auth=authorize_test(validation_terminal_path=validation_path,lock_path=final_dir/"candidate_lock.json",effective=EFFECTIVE);test_path=tmp_path/"test-once/terminal.json";test_path.parent.mkdir();test_terminal=run_test_handler(authorization=test_auth,effective=EFFECTIVE,evaluate=lambda:("passed",{"all":{"passed":True}}),terminal_path=test_path);assert test_terminal["candidate"]=="r16_dscp_v14" and test_terminal["candidate_lock_digest"]==lock["lock_digest"]

@pytest.mark.parametrize("mutation",["checkpoint","lock","final","validation","effective"])
def test_chain_tamper_and_failed_terminals_rejected(tmp_path,mutation):
    checkpoint=tmp_path/"best.pt";checkpoint.write_bytes(b"best");final_dir=tmp_path/"final-train-confirm";terminal,lock=final_handler(run_dir=final_dir,checkpoint_path=checkpoint,thresholds=THRESHOLDS,effective=EFFECTIVE,data_hashes=DATA,evaluate=lambda:("passed",{"all":{"passed":True}}));validation_auth=authorize_validation(run_root=tmp_path,effective=EFFECTIVE);validation_path=tmp_path/"validation.json";validation=validation_handler(terminal_path=validation_path,authorization=validation_auth,effective=EFFECTIVE,evaluate=lambda:("passed",{"all":{"passed":True}}))
    if mutation=="checkpoint":checkpoint.write_bytes(b"drift")
    elif mutation=="lock":(final_dir/"candidate_lock.json").write_text("{}")
    elif mutation=="final":(final_dir/"terminal.json").write_text("{}")
    elif mutation=="validation":validation["status"]="fail_gate";validation_path.write_text(json.dumps(validation))
    else:
        with pytest.raises(V14ChainRefusal):authorize_validation(run_root=tmp_path,effective={**EFFECTIVE,"code":"drift"})
        return
    with pytest.raises(V14ChainRefusal):authorize_test(validation_terminal_path=validation_path,lock_path=final_dir/"candidate_lock.json",effective=EFFECTIVE)

def test_final_fail_writes_no_lock(tmp_path):
    terminal,lock=final_handler(run_dir=tmp_path,checkpoint_path=None,thresholds=THRESHOLDS,effective=EFFECTIVE,data_hashes=DATA,evaluate=lambda:("fail_gate",{"all":{"passed":False}}));assert lock is None and terminal["status"]=="fail_gate" and not (tmp_path/"candidate_lock.json").exists()

def test_dispatcher_refuses_v11_backend():
    with pytest.raises(TypeError):V14StageDispatcher(object.__new__(V11ProductionBackend))

def test_candidate_lock_and_validation_terminal_are_complete(tmp_path):
    checkpoint=tmp_path/"best.pt";checkpoint.write_bytes(b"complete")
    final_dir=tmp_path/"final-train-confirm";terminal,lock=final_handler(run_dir=final_dir,checkpoint_path=checkpoint,thresholds=THRESHOLDS,effective=EFFECTIVE,data_hashes=DATA,evaluate=lambda:("passed",{"accuracy":{"passed":True},"speed":{"passed":True}}))
    required={"schema","candidate","checkpoint_path","checkpoint_sha256","checkpoint_size_bytes","thresholds","threshold_digest","effective","data_hashes","final_terminal_path","final_terminal_sha256","created_before_validation","validation_opened","test_opened","lock_digest"}
    assert required<=set(lock) and lock["validation_opened"] is False and lock["test_opened"] is False
    assert terminal["best_checkpoint"]["sha256"]==lock["checkpoint_sha256"] and terminal["gates"]["speed"]["passed"]
    auth=authorize_validation(run_root=tmp_path,effective=EFFECTIVE);path=tmp_path/"validation.json";row=validation_handler(terminal_path=path,authorization=auth,effective=EFFECTIVE,evaluate=lambda:("passed",{"accuracy":{"passed":True},"speed":{"passed":True}}))
    assert {"candidate_lock_path","candidate_lock_sha256","candidate_lock_digest","candidate_checkpoint_path","candidate_checkpoint_sha256","effective","data_hashes","gates"}<=set(row)

def test_validation_failure_cannot_authorize_test(tmp_path):
    checkpoint=tmp_path/"best.pt";checkpoint.write_bytes(b"best");final_dir=tmp_path/"final-train-confirm"
    final_handler(run_dir=final_dir,checkpoint_path=checkpoint,thresholds=THRESHOLDS,effective=EFFECTIVE,data_hashes=DATA,evaluate=lambda:("passed",{"all":{"passed":True}}))
    auth=authorize_validation(run_root=tmp_path,effective=EFFECTIVE);path=tmp_path/"validation.json";validation_handler(terminal_path=path,authorization=auth,effective=EFFECTIVE,evaluate=lambda:("fail_gate",{"all":{"passed":False}}))
    with pytest.raises(V14ChainRefusal):authorize_test(validation_terminal_path=path,lock_path=final_dir/"candidate_lock.json",effective=EFFECTIVE)

def test_all_frozen_commands_parse_and_handlers_are_real():
    config=yaml.safe_load(cli.CONFIG.read_text())
    parsed={name:cli.build_parser().parse_args(cli.command_argv(command)).mode for name,command in config["commands"].items()}
    assert len(parsed)==15 and set(parsed.values())==set(cli.MODES)
    assert set(cli.HANDLERS)==set(cli.MODES)
    assert all(callable(handler) and "unwired" not in handler.__name__ for handler in cli.HANDLERS.values())
    assert "_resume-selftest" not in inspect.getsource(cli) and "_resume-selftest" not in cli.build_parser().format_help()

def test_production_eval_dispatch_uses_v14_chain_not_v10_lock():
    source=inspect.getsource(cli._eval_protocol)
    assert "build_final_chain" in source and "build_validation_terminal" in source and "build_test_terminal" in source
    assert "run_evaluation_stage" not in source and "candidate_lock_payload" not in source


def test_runtime_import_closure_is_complete_and_deterministic():
    first=cli.runtime_import_closure();second=cli.runtime_import_closure()
    assert first==second and first["entries"]==sorted(first["entries"],key=lambda row:row["path"])
    paths={row["path"] for row in first["entries"]}
    assert {"scripts/train_r16_dscp_v14.py","saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v14.py","saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v12.py","scripts/train_r16_dscp_v10.py"}<=paths
    assert first["dynamic_imports"] and len(paths)>20


def test_runtime_closure_rejects_drift_addition_and_removal(tmp_path):
    entry=tmp_path/"entry.py";helper=tmp_path/"helper.py"
    entry.write_text("import helper\n");helper.write_text("VALUE=1\n")
    frozen=build_runtime_import_closure(root=tmp_path,entrypoints=[entry])
    helper.write_text("VALUE=2\n")
    with pytest.raises(RuntimeClosureRefusal,match="drift=.*helper.py"):
        verify_runtime_import_closure(frozen,root=tmp_path)
    helper.write_text("VALUE=1\n");extra=tmp_path/"extra.py";extra.write_text("VALUE=3\n");entry.write_text("import helper\nimport extra\n")
    with pytest.raises(RuntimeClosureRefusal,match="added=.*extra.py"):
        verify_runtime_import_closure(frozen,root=tmp_path)
    entry.write_text("import helper\n");extra.unlink();helper.unlink()
    with pytest.raises(RuntimeClosureRefusal,match="removed=.*helper.py"):
        verify_runtime_import_closure(frozen,root=tmp_path)


def test_runtime_closure_rejects_symlink_escape_and_unbound_dynamic_import(tmp_path):
    outside=tmp_path.parent/(tmp_path.name+"-outside.py");outside.write_text("VALUE=1\n")
    link=tmp_path/"link.py";link.symlink_to(outside)
    with pytest.raises(RuntimeClosureRefusal,match="symlink|escape"):
        build_runtime_import_closure(root=tmp_path,entrypoints=[link])
    entry=tmp_path/"entry.py";entry.write_text("name='helper'\n__import__(name)\n")
    with pytest.raises(RuntimeClosureRefusal,match="unbound dynamic import"):
        build_runtime_import_closure(root=tmp_path,entrypoints=[entry])
    with pytest.raises(RuntimeClosureRefusal,match="escape"):
        build_runtime_import_closure(root=tmp_path,entrypoints=[outside])


def test_runtime_closure_binds_literal_dynamic_import(tmp_path):
    entry=tmp_path/"entry.py";helper=tmp_path/"helper.py"
    entry.write_text("__import__('helper')\n");helper.write_text("VALUE=1\n")
    closure=build_runtime_import_closure(root=tmp_path,entrypoints=[entry])
    assert [row["path"] for row in closure["entries"]]==["entry.py","helper.py"]
    assert closure["dynamic_imports"]==[{"importer":"entry.py","line":1,"module":"helper"}]


def test_factory_closure_checks_precede_data_cuda_and_distributed(monkeypatch,tmp_path):
    events=[];auth_path=tmp_path/"auth.json";auth_path.write_text("{}")
    prereg=tmp_path/"prereg.json";prereg.write_text("{}")
    line=Lineage(cli.CANDIDATE,"smoke","run","c","cfg","p","bf","bt","par","none")
    def fake_validate(args,**kwargs):kwargs["event_hook"]("closure_after_authorization");return {"effective":{}},line
    monkeypatch.setattr(cli,"validate_stage_auth",fake_validate)
    monkeypatch.setattr(cli,"complete_identity_v14",lambda **kwargs:{"candidate":cli.CANDIDATE,"identity_digest":"id"})
    monkeypatch.setattr(cli,"verify_frozen_runtime_import_closure",lambda path:events.append("closure_call") or {"closure_sha256":"x"})
    monkeypatch.setattr(cli,"OUT",tmp_path/"out")
    class Session:
        initialized=False;is_writer=True;rank=0;rank_contract=None
        def __init__(self,**kwargs):self.backend=kwargs.get("backend","gloo")
        def initialize(self):self.initialized=False;events.append("distributed_init");return self
        def synchronize_failure(self,error):return []
        def close(self):events.append("distributed_destroy")
    monkeypatch.setattr(cli,"DistributedSession",Session)
    monkeypatch.setattr(cli,"protocol_dispatch",lambda *args,**kwargs:(events.append("protocol") or {}))
    args=SimpleNamespace(mode="smoke",authorization=str(auth_path),preregistration=str(prereg),physical_gpu_index=0,device="cpu")
    def builder(*args,**kwargs):events.append("data_cuda_factory_build");return (object(),{},[],"meta")
    cli.stage_handler(args,backend_factory=builder,event_hook=events.append)
    assert events.index("closure_after_authorization")<events.index("closure_immediately_before_factory_construction")
    assert events.index("closure_immediately_before_factory_construction")<events.index("factory_constructed_without_data_cuda")<events.index("closure_immediately_after_factory_construction")
    assert events.index("closure_immediately_after_factory_construction")<events.index("distributed_init")<events.index("data_cuda_factory_build")
    assert events[-2:]==["distributed_destroy","distributed_destroyed"]


def _production_long_fixture(root:Path,run:Path,*,world_size:int,**controls):
    root.mkdir(parents=True)
    prereg=root/"prereg.json";closure=cli.runtime_import_closure();prereg.write_text(json.dumps({"runtime_import_closure":closure}))
    run_digest=canonical_sha({"run":str(run),"world_size":world_size})
    pre_identity=complete_identity_v14(mode="prep",run_digest=run_digest,authorization_sha256="prep-none",engine_sha256=sha256_file(cli.ENGINE),script_sha256=sha256_file(cli.SCRIPT),config_sha256_effective=sha256_file(cli.CONFIG),panels_sha256_effective=PANELS_SHA256,basis_sha256_effective=BASIS_FILE_SHA256,parent_sha256_effective=PARENT_SHA256,input_checkpoint_sha256="none")
    preflight=root/"design_preflight.json";preflight.write_text(json.dumps({"candidate":cli.CANDIDATE,"status":"frozen","run_identity":pre_identity}))
    pilot=root/"pilot/terminal.json";pilot.parent.mkdir();pilot.write_text(json.dumps({"status":"passed"}))
    decision=root/"scale-decide/terminal.json";decision.parent.mkdir();decision.write_text(json.dumps({"status":"passed","selected_gpus":world_size}))
    torch.manual_seed(11);model=torch.nn.Linear(1,1,bias=False);optimizer=make_optimizer(list(model.parameters()))
    input_identity=complete_identity_v14(mode="pilot",run_digest=run_digest,authorization_sha256="fixture-input",engine_sha256=sha256_file(cli.ENGINE),script_sha256=sha256_file(cli.SCRIPT),config_sha256_effective=sha256_file(cli.CONFIG),panels_sha256_effective=PANELS_SHA256,basis_sha256_effective=BASIS_FILE_SHA256,parent_sha256_effective=PARENT_SHA256,input_checkpoint_sha256="none")
    payload=checkpoint_payload(model,optimizer,run_identity=input_identity,sampler_order=list(range(9)),progress={"epoch":0,"global_step":0,"group_index":0,"best":float("inf"),"bad_epochs":0,"best_epoch":-1})
    payload["rng"]["torch_cuda"]=[]
    input_checkpoint=save_best_last(payload,root/"input-checkpoint",is_best=True)["best"]
    auth_path=root/"authorization.json";auth_command=[sys.executable,"scripts/train_r16_dscp_v14.py","authorize-stage","--stage","long","--output",str(auth_path),"--input-checkpoint",str(input_checkpoint)]
    auth_env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","PYTHONPATH":"src:.","R16_DSCP_V14_TEST_MODE":"1","R16_DSCP_V14_TEST_AUTH_ROOT":str(root),"R16_DSCP_V14_TEST_AUTH_PREREG":str(prereg),"R16_DSCP_V14_TEST_AUTH_CONTROLS":json.dumps({"test_only_production_backend":"deterministic_v14","test_max_epochs":2,"test_cache_delay_s":.002,**controls})}
    authorized=subprocess.run(auth_command,cwd=cli.ROOT,env=auth_env,text=True,capture_output=True,timeout=90);assert authorized.returncode==0,authorized.stdout+authorized.stderr
    command=[sys.executable,"scripts/train_r16_dscp_v14.py","long","--authorization",str(auth_path),"--world-size",str(world_size),"--input-checkpoint",str(input_checkpoint),"--scale-decision",str(decision),"--config",str(cli.CONFIG),"--preregistration",str(prereg)]
    return command,auth_path


def _events(path:Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_production_long_cli_interrupt_resume_and_reference_equivalence(tmp_path):
    fixture=tmp_path/"fixture";run=fixture/"long";events=tmp_path/"events"
    command,auth_path=_production_long_fixture(fixture,run,world_size=1,test_interrupt_after_checkpoint=1)
    base={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","PYTHONPATH":"src:.","CUDA_VISIBLE_DEVICES":"0","OMP_NUM_THREADS":"1","R16_DSCP_V14_TEST_MODE":"1","R16_DSCP_V14_EVENT_DIR":str(events)}
    first=subprocess.run(command,cwd=cli.ROOT,env={**base,"R16_DSCP_V14_TEST_INTERRUPT":"1"},text=True,capture_output=True,timeout=90)
    assert first.returncode!=0 and "ControlledLongInterruption" in first.stderr
    assert (run/"last.pt").is_file() and (run/"run_identity.json").is_file() and not (run/"terminal.json").exists()
    checkpoint_sha=sha256_file(run/"last.pt");identity=json.loads((run/"run_identity.json").read_text())
    second=subprocess.run([*command,"--resume"],cwd=cli.ROOT,env=base,text=True,capture_output=True,timeout=90)
    assert second.returncode==0,second.stdout+second.stderr
    terminal=json.loads((run/"terminal.json").read_text());assert terminal["status"]=="completed_train" and terminal["global_step"]==4
    rows=_events(events/"rank0.jsonl");resume_pid=next(row["pid"] for row in rows if row["event"]=="resume_checkpoint_inspected_before_factory_cache")
    resumed=[row for row in rows if row["pid"]==resume_pid];names=[row["event"] for row in resumed]
    assert names.index("authorization_and_initial_closure_verified")<names.index("resume_checkpoint_inspected_before_factory_cache")<names.index("closure_rechecked_before_factory")<names.index("resume_checkpoint_loaded_before_cache")<names.index("cache_rebuild_started")<names.index("cache_entry_materialized")
    assert names.count("closure_rechecked_before_factory")==1 and names.count("closure_rechecked_after_factory")==1
    inspected=next(row for row in resumed if row["event"]=="resume_checkpoint_inspected_before_factory_cache");loaded=next(row for row in resumed if row["event"]=="resume_checkpoint_loaded_before_cache")
    assert inspected["checkpoint_sha256"]==loaded["checkpoint_sha256"]==checkpoint_sha
    assert inspected["identity_digest"]==loaded["identity_digest"]==identity["identity_digest"] and inspected["progress"]==loaded["progress"]
    assert loaded["cache_entries"]==0 and loaded["rng_fields"]==["numpy","python","torch_cpu","torch_cuda"]
    rebuild=next(row for row in resumed if row["event"]=="cache_rebuild_completed");budget=next(row for row in resumed if row["event"]=="original_budget_started")
    assert rebuild["entries"]==9 and rebuild["elapsed_s"]>0 and terminal["cache_rebuild_s"]==pytest.approx(rebuild["elapsed_s"]) and terminal["wall_s"]>=rebuild["elapsed_s"]
    assert budget["monotonic_s"]<=rebuild["monotonic_s"]
    ref_fixture=tmp_path/"reference-fixture";ref_run=ref_fixture/"long";ref_events=tmp_path/"reference-events";ref_command,_=_production_long_fixture(ref_fixture,ref_run,world_size=1)
    reference=subprocess.run(ref_command,cwd=cli.ROOT,env={**base,"R16_DSCP_V14_EVENT_DIR":str(ref_events)},text=True,capture_output=True,timeout=90)
    assert reference.returncode==0,reference.stdout+reference.stderr
    ref_terminal=json.loads((ref_run/"terminal.json").read_text());assert terminal["model_state_sha256"]==ref_terminal["model_state_sha256"] and terminal["global_step"]==ref_terminal["global_step"]
    resumed_steps=[row["state_sha256"] for row in resumed if row["event"]=="optimizer_step_completed"]
    reference_rows=_events(ref_events/"rank0.jsonl");reference_steps=[row["state_sha256"] for row in reference_rows if row["event"]=="optimizer_step_completed"]
    assert resumed_steps==reference_steps[1:]
    resumed_random=[row["random_scale"] for row in resumed if row["event"]=="backward_record"]
    reference_random=[row["random_scale"] for row in reference_rows if row["event"]=="backward_record"]
    assert resumed_random==reference_random[4:]
    with pytest.raises(LongResumeRefusal,match="exact authorized"):
        inspect_long_resume(run_dir=run,authorized_run_dir=tmp_path/"foreign",expected_identity=identity,authorization_sha256=sha256_file(auth_path))


def test_torchrun_production_fault_synchronizes_all_ranks_and_external_record(tmp_path):
    fixture=tmp_path/"fixture";run=fixture/"long";events=tmp_path/"events";external=tmp_path/"external-failure.json";claims=tmp_path/"claims"
    command,_=_production_long_fixture(fixture,run,world_size=4,test_fail_rank0_persistence=True,test_distributed_backend="gloo")
    env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","PYTHONPATH":"src:.","CUDA_VISIBLE_DEVICES":"0,1,2,3","OMP_NUM_THREADS":"1","R16_DSCP_V14_TEST_MODE":"1","R16_DSCP_V14_EVENT_DIR":str(events),"R16_DSCP_V14_EXTERNAL_FAILURE_RECORD":str(external),"R16_DSCP_V14_RANK_CONTRACT_ROOT":str(claims)}
    completed=subprocess.run(["torchrun","--standalone","--nproc-per-node=4",*command[1:]],cwd=cli.ROOT,env=env,text=True,capture_output=True,timeout=90)
    assert completed.returncode!=0 and external.is_file() and not (run/"terminal.json").exists()
    record=json.loads(external.read_text());assert record["schema"]=="r16_dscp_v14_external_failure_record_v1" and record["status"]=="failed" and record["rank0_persistence_error"]["error_type"]=="OSError"
    assert list(tmp_path.glob("external-failure*.json"))==[external]
    rank_rows={rank:_events(events/f"rank{rank}.jsonl") for rank in range(4)}
    protocols=[]
    for rank,rows in rank_rows.items():
        assert any(row["event"]=="rank_exiting_nonzero" for row in rows)
        assert rows[-1]["event"]=="process_group_destroyed" and rows[-1]["destroyed"] is True
        sync=[row for row in rows if row["event"]=="synchronized_failure_protocol"]
        assert sync;protocols.append(sync[0]["failures"])
    assert all(protocol==protocols[0] for protocol in protocols) and protocols[0]==[{"rank":0,"error_type":"OSError","error_message":"injected rank0 persistent checkpoint write failure"}]
    failure_time=next(row["monotonic_s"] for row in rank_rows[0] if row["event"]=="rank0_persistence_failure_recorded")
    assert not any(row["event"]=="training_collective_entered" and row["monotonic_s"]>failure_time for rows in rank_rows.values() for row in rows)
    assert sorted(json.loads(path.read_text())["local_rank"] for path in claims.glob("rank.*.json"))==[0,1,2,3]


@pytest.mark.parametrize("kind",["rank_local_mismatch","duplicate_rank","duplicate_visible"])
def test_torchrun_production_preflight_rejects_rank_and_cuda_mapping_before_factory(tmp_path,kind):
    root=tmp_path/kind;fixture=root/"fixture";run=fixture/"long";events=root/"events";claims=root/"claims"
    command,_=_production_long_fixture(fixture,run,world_size=4,test_distributed_backend="gloo")
    env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","PYTHONPATH":"src:.","CUDA_VISIBLE_DEVICES":"0,1,2,3","OMP_NUM_THREADS":"1","R16_DSCP_V14_TEST_MODE":"1","R16_DSCP_V14_EVENT_DIR":str(events),"R16_DSCP_V14_RANK_CONTRACT_ROOT":str(claims)}
    if kind=="rank_local_mismatch":env["R16_DSCP_V14_TEST_RANK_OVERRIDES"]=json.dumps({"3":[3,2]})
    elif kind=="duplicate_rank":env["R16_DSCP_V14_TEST_RANK_OVERRIDES"]=json.dumps({"3":[2,2]})
    else:env["CUDA_VISIBLE_DEVICES"]="0,1,2,2"
    completed=subprocess.run(["torchrun","--standalone","--nproc-per-node=4",*command[1:]],cwd=cli.ROOT,env=env,text=True,capture_output=True,timeout=90)
    assert completed.returncode!=0 and not run.exists()
    rows=[row for path in events.glob("rank*.jsonl") for row in _events(path)]
    assert rows and not any(row["event"] in {"distributed_initialized","cache_rebuild_started","cache_entry_materialized","training_collective_entered"} for row in rows)
    assert any(path.name.startswith("abort.") for path in claims.iterdir())


def test_resume_budget_clock_includes_pre_cache_elapsed_time():
    values=iter([15.0,16.0]);clock=InclusiveBudgetClock(10.0,clock=lambda:next(values))
    assert clock()==10.0 and clock()==15.0 and clock()==16.0
