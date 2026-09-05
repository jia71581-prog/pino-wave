from __future__ import annotations
import inspect,json,os,subprocess,sys,textwrap
from pathlib import Path
from types import SimpleNamespace
import pytest,yaml
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v11 import V11ProductionBackend
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import Lineage
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v13 import InclusiveBudgetClock,LongResumeRefusal,RuntimeClosureRefusal,V13ChainRefusal,V13ProductionBackend,build_runtime_import_closure,build_test_terminal,inspect_long_resume,verify_runtime_import_closure
import scripts.train_r16_dscp_v13 as cli
from scripts.train_r16_dscp_v13 import V13StageDispatcher,authorize_test,authorize_validation,final_handler,test_handler as run_test_handler,validation_handler

EFFECTIVE={"code":"c","script":"s","config":"cfg","panels":"p","basis":"b","parent":"par","input":"i","data":"d"};DATA={"validation":"v","test":"t"};THRESHOLDS={"accuracy":.05,"speed":10.}

def test_end_to_end_production_handler_chain(tmp_path):
    checkpoint=tmp_path/"best.pt";checkpoint.write_bytes(b"best");final_dir=tmp_path/"final-train-confirm";terminal,lock=final_handler(run_dir=final_dir,checkpoint_path=checkpoint,thresholds=THRESHOLDS,effective=EFFECTIVE,data_hashes=DATA,evaluate=lambda:("passed",{"all":{"passed":True}}));assert lock and lock["created_before_validation"] and lock["checkpoint_path"]==str(checkpoint.resolve())
    validation_auth=authorize_validation(run_root=tmp_path,effective=EFFECTIVE);validation_path=tmp_path/"validation-once/terminal.json";validation_path.parent.mkdir();validation=validation_handler(terminal_path=validation_path,authorization=validation_auth,effective=EFFECTIVE,evaluate=lambda:("passed",{"all":{"passed":True}}));assert validation["candidate_lock_digest"]==lock["lock_digest"]
    test_auth=authorize_test(validation_terminal_path=validation_path,lock_path=final_dir/"candidate_lock.json",effective=EFFECTIVE);test_path=tmp_path/"test-once/terminal.json";test_path.parent.mkdir();test_terminal=run_test_handler(authorization=test_auth,effective=EFFECTIVE,evaluate=lambda:("passed",{"all":{"passed":True}}),terminal_path=test_path);assert test_terminal["candidate"]=="r16_dscp_v13" and test_terminal["candidate_lock_digest"]==lock["lock_digest"]

@pytest.mark.parametrize("mutation",["checkpoint","lock","final","validation","effective"])
def test_chain_tamper_and_failed_terminals_rejected(tmp_path,mutation):
    checkpoint=tmp_path/"best.pt";checkpoint.write_bytes(b"best");final_dir=tmp_path/"final-train-confirm";terminal,lock=final_handler(run_dir=final_dir,checkpoint_path=checkpoint,thresholds=THRESHOLDS,effective=EFFECTIVE,data_hashes=DATA,evaluate=lambda:("passed",{"all":{"passed":True}}));validation_auth=authorize_validation(run_root=tmp_path,effective=EFFECTIVE);validation_path=tmp_path/"validation.json";validation=validation_handler(terminal_path=validation_path,authorization=validation_auth,effective=EFFECTIVE,evaluate=lambda:("passed",{"all":{"passed":True}}))
    if mutation=="checkpoint":checkpoint.write_bytes(b"drift")
    elif mutation=="lock":(final_dir/"candidate_lock.json").write_text("{}")
    elif mutation=="final":(final_dir/"terminal.json").write_text("{}")
    elif mutation=="validation":validation["status"]="fail_gate";validation_path.write_text(json.dumps(validation))
    else:
        with pytest.raises(V13ChainRefusal):authorize_validation(run_root=tmp_path,effective={**EFFECTIVE,"code":"drift"})
        return
    with pytest.raises(V13ChainRefusal):authorize_test(validation_terminal_path=validation_path,lock_path=final_dir/"candidate_lock.json",effective=EFFECTIVE)

def test_final_fail_writes_no_lock(tmp_path):
    terminal,lock=final_handler(run_dir=tmp_path,checkpoint_path=None,thresholds=THRESHOLDS,effective=EFFECTIVE,data_hashes=DATA,evaluate=lambda:("fail_gate",{"all":{"passed":False}}));assert lock is None and terminal["status"]=="fail_gate" and not (tmp_path/"candidate_lock.json").exists()

def test_dispatcher_refuses_v11_backend():
    with pytest.raises(TypeError):V13StageDispatcher(object.__new__(V11ProductionBackend))

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
    with pytest.raises(V13ChainRefusal):authorize_test(validation_terminal_path=path,lock_path=final_dir/"candidate_lock.json",effective=EFFECTIVE)

def test_all_frozen_commands_parse_and_handlers_are_real():
    config=yaml.safe_load(cli.CONFIG.read_text())
    parsed={name:cli.build_parser().parse_args(cli.command_argv(command)).mode for name,command in config["commands"].items()}
    assert len(parsed)==15 and set(parsed.values())==set(cli.MODES)
    assert set(cli.HANDLERS)==set(cli.MODES)
    assert all(callable(handler) and "unwired" not in handler.__name__ for handler in cli.HANDLERS.values())

def test_production_eval_dispatch_uses_v13_chain_not_v10_lock():
    source=inspect.getsource(cli._eval_protocol)
    assert "build_final_chain" in source and "build_validation_terminal" in source and "build_test_terminal" in source
    assert "run_evaluation_stage" not in source and "candidate_lock_payload" not in source


def test_runtime_import_closure_is_complete_and_deterministic():
    first=cli.runtime_import_closure();second=cli.runtime_import_closure()
    assert first==second and first["entries"]==sorted(first["entries"],key=lambda row:row["path"])
    paths={row["path"] for row in first["entries"]}
    assert {"scripts/train_r16_dscp_v13.py","saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v13.py","saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v12.py","scripts/train_r16_dscp_v10.py"}<=paths
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
    monkeypatch.setattr(cli,"complete_identity_v13",lambda **kwargs:{"candidate":cli.CANDIDATE,"identity_digest":"id"})
    monkeypatch.setattr(cli,"verify_frozen_runtime_import_closure",lambda path:events.append("closure_call") or {"closure_sha256":"x"})
    monkeypatch.setattr(cli,"OUT",tmp_path/"out")
    class Session:
        initialized=False;is_writer=True;rank=0
        def __init__(self,**kwargs):pass
        def initialize(self):self.initialized=False;events.append("distributed_init");return self
        def synchronize_failure(self,error):return []
        def close(self):events.append("distributed_destroy")
    monkeypatch.setattr(cli,"DistributedSession",Session)
    monkeypatch.setattr(cli,"protocol_dispatch",lambda *args:(events.append("protocol") or {}))
    args=SimpleNamespace(mode="smoke",authorization=str(auth_path),preregistration=str(prereg),physical_gpu_index=0,device="cpu")
    def builder(*args,**kwargs):events.append("data_cuda_factory_build");return (object(),{},[],"meta")
    cli.stage_handler(args,backend_factory=builder,event_hook=events.append)
    assert events.index("closure_after_authorization")<events.index("closure_immediately_before_factory_construction")
    assert events.index("closure_immediately_before_factory_construction")<events.index("factory_constructed_without_data_cuda")<events.index("closure_immediately_after_factory_construction")
    assert events.index("closure_immediately_after_factory_construction")<events.index("distributed_init")<events.index("data_cuda_factory_build")
    assert events[-2:]==["distributed_destroy","distributed_destroyed"]


def test_real_local_torchrun_initializes_syncs_failure_destroys_and_rank0_writes(tmp_path):
    helper=tmp_path/"torchrun_v13.py";report=tmp_path/"report.json"
    helper.write_text(textwrap.dedent('''
        import json,os,sys,torch
        from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v13 import DistributedSession
        session=DistributedSession(world_size=4,backend="gloo");session.initialize()
        assert torch.distributed.is_initialized()
        value=torch.tensor([session.rank],dtype=torch.int64);torch.distributed.all_reduce(value)
        error=RuntimeError("rank1-integration-failure") if session.rank==1 else None
        failures=session.synchronize_failure(error);writer=session.is_writer;rank=session.rank
        session.close();destroyed=not torch.distributed.is_initialized()
        if writer:
            with open(sys.argv[1],"x",encoding="utf8") as handle:
                json.dump({"rank":rank,"sum":int(value),"failures":failures,"destroyed":destroyed},handle)
    '''))
    env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","PYTHONPATH":"src:.","CUDA_VISIBLE_DEVICES":"0,1,2,3","OMP_NUM_THREADS":"1"}
    completed=subprocess.run(["torchrun","--standalone","--nproc-per-node=4",str(helper),str(report)],cwd=cli.ROOT,env=env,text=True,capture_output=True,timeout=90)
    assert completed.returncode==0,completed.stdout+completed.stderr
    payload=json.loads(report.read_text())
    assert payload=={"rank":0,"sum":6,"failures":[{"rank":1,"error_type":"RuntimeError","error_message":"rank1-integration-failure"}],"destroyed":True}
    assert sorted(path.name for path in tmp_path.iterdir())==["report.json","torchrun_v13.py"]


def test_production_cli_fresh_process_resume_next_step_equivalence(tmp_path):
    run=tmp_path/"authorized-long"
    env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","PYTHONPATH":"src:.","CUDA_VISIBLE_DEVICES":""}
    common=[sys.executable,"scripts/train_r16_dscp_v13.py","_resume-selftest","--run-dir",str(run),"--authorized-run-dir",str(run)]
    first=subprocess.run([*common,"--phase","interrupt"],cwd=cli.ROOT,env=env,text=True,capture_output=True,timeout=90)
    assert first.returncode==0,first.stdout+first.stderr
    assert not (run/"terminal.json").exists()
    second=subprocess.run([*common,"--phase","resume"],cwd=cli.ROOT,env=env,text=True,capture_output=True,timeout=90)
    assert second.returncode==0,second.stdout+second.stderr
    result=json.loads((run/"resume_result.json").read_text())
    assert all(result[key] for key in ("fresh_process","fresh_objects","checkpoint_loaded_before_cache","next_random_equal","next_state_equal","progress_equal","next_step_equivalent"))
    identity=json.loads((run/"run_identity.json").read_text())
    with pytest.raises(LongResumeRefusal,match="exact authorized"):
        inspect_long_resume(run_dir=run,authorized_run_dir=tmp_path/"foreign",expected_identity=identity,authorization_sha256="a"*64)


def test_resume_budget_clock_includes_pre_cache_elapsed_time():
    values=iter([15.0,16.0]);clock=InclusiveBudgetClock(10.0,clock=lambda:next(values))
    assert clock()==10.0 and clock()==15.0 and clock()==16.0
