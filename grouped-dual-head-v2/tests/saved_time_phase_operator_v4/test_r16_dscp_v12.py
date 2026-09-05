from __future__ import annotations
import inspect,json,pytest,yaml
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v11 import V11ProductionBackend
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v12 import V12ChainRefusal,V12ProductionBackend,build_test_terminal
import scripts.train_r16_dscp_v12 as cli
from scripts.train_r16_dscp_v12 import V12StageDispatcher,authorize_test,authorize_validation,final_handler,test_handler as run_test_handler,validation_handler

EFFECTIVE={"code":"c","script":"s","config":"cfg","panels":"p","basis":"b","parent":"par","input":"i","data":"d"};DATA={"validation":"v","test":"t"};THRESHOLDS={"accuracy":.05,"speed":10.}

def test_end_to_end_production_handler_chain(tmp_path):
    checkpoint=tmp_path/"best.pt";checkpoint.write_bytes(b"best");final_dir=tmp_path/"final-train-confirm";terminal,lock=final_handler(run_dir=final_dir,checkpoint_path=checkpoint,thresholds=THRESHOLDS,effective=EFFECTIVE,data_hashes=DATA,evaluate=lambda:("passed",{"all":{"passed":True}}));assert lock and lock["created_before_validation"] and lock["checkpoint_path"]==str(checkpoint.resolve())
    validation_auth=authorize_validation(run_root=tmp_path,effective=EFFECTIVE);validation_path=tmp_path/"validation-once/terminal.json";validation_path.parent.mkdir();validation=validation_handler(terminal_path=validation_path,authorization=validation_auth,effective=EFFECTIVE,evaluate=lambda:("passed",{"all":{"passed":True}}));assert validation["candidate_lock_digest"]==lock["lock_digest"]
    test_auth=authorize_test(validation_terminal_path=validation_path,lock_path=final_dir/"candidate_lock.json",effective=EFFECTIVE);test_path=tmp_path/"test-once/terminal.json";test_path.parent.mkdir();test_terminal=run_test_handler(authorization=test_auth,effective=EFFECTIVE,evaluate=lambda:("passed",{"all":{"passed":True}}),terminal_path=test_path);assert test_terminal["candidate"]=="r16_dscp_v12" and test_terminal["candidate_lock_digest"]==lock["lock_digest"]

@pytest.mark.parametrize("mutation",["checkpoint","lock","final","validation","effective"])
def test_chain_tamper_and_failed_terminals_rejected(tmp_path,mutation):
    checkpoint=tmp_path/"best.pt";checkpoint.write_bytes(b"best");final_dir=tmp_path/"final-train-confirm";terminal,lock=final_handler(run_dir=final_dir,checkpoint_path=checkpoint,thresholds=THRESHOLDS,effective=EFFECTIVE,data_hashes=DATA,evaluate=lambda:("passed",{"all":{"passed":True}}));validation_auth=authorize_validation(run_root=tmp_path,effective=EFFECTIVE);validation_path=tmp_path/"validation.json";validation=validation_handler(terminal_path=validation_path,authorization=validation_auth,effective=EFFECTIVE,evaluate=lambda:("passed",{"all":{"passed":True}}))
    if mutation=="checkpoint":checkpoint.write_bytes(b"drift")
    elif mutation=="lock":(final_dir/"candidate_lock.json").write_text("{}")
    elif mutation=="final":(final_dir/"terminal.json").write_text("{}")
    elif mutation=="validation":validation["status"]="fail_gate";validation_path.write_text(json.dumps(validation))
    else:
        with pytest.raises(V12ChainRefusal):authorize_validation(run_root=tmp_path,effective={**EFFECTIVE,"code":"drift"})
        return
    with pytest.raises(V12ChainRefusal):authorize_test(validation_terminal_path=validation_path,lock_path=final_dir/"candidate_lock.json",effective=EFFECTIVE)

def test_final_fail_writes_no_lock(tmp_path):
    terminal,lock=final_handler(run_dir=tmp_path,checkpoint_path=None,thresholds=THRESHOLDS,effective=EFFECTIVE,data_hashes=DATA,evaluate=lambda:("fail_gate",{"all":{"passed":False}}));assert lock is None and terminal["status"]=="fail_gate" and not (tmp_path/"candidate_lock.json").exists()

def test_dispatcher_refuses_v11_backend():
    with pytest.raises(TypeError):V12StageDispatcher(object.__new__(V11ProductionBackend))

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
    with pytest.raises(V12ChainRefusal):authorize_test(validation_terminal_path=path,lock_path=final_dir/"candidate_lock.json",effective=EFFECTIVE)

def test_all_frozen_commands_parse_and_handlers_are_real():
    config=yaml.safe_load(cli.CONFIG.read_text())
    parsed={name:cli.build_parser().parse_args(cli.command_argv(command)).mode for name,command in config["commands"].items()}
    assert len(parsed)==13 and set(parsed.values())==set(cli.MODES)
    assert set(cli.HANDLERS)==set(cli.MODES)
    assert all(callable(handler) and "unwired" not in handler.__name__ for handler in cli.HANDLERS.values())

def test_production_eval_dispatch_uses_v12_chain_not_v10_lock():
    source=inspect.getsource(cli._eval_protocol)
    assert "build_final_chain" in source and "build_validation_terminal" in source and "build_test_terminal" in source
    assert "run_evaluation_stage" not in source and "candidate_lock_payload" not in source
