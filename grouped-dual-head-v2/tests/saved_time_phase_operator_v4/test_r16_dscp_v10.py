from __future__ import annotations
import pytest,yaml
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v9 import V9ProductionBackend
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v10 import *
import scripts.train_r16_dscp_v10 as cli

def test_all_config_commands_parse():
    config=yaml.safe_load((cli.ROOT/"configs/r16_dscp_v10.yaml").read_text()) if hasattr(cli,"ROOT") else yaml.safe_load(open("configs/r16_dscp_v10.yaml"))
    parser=cli.build_parser()
    for name,command in config["commands"].items():
        args=parser.parse_args(cli.command_argv(command));assert args.mode in cli.MODES

def test_every_mode_dispatches_to_corresponding_handler(monkeypatch):
    called=[]
    for mode in cli.MODES:monkeypatch.setitem(cli.HANDLERS,mode,lambda args,m=mode:called.append(m) or m)
    samples={"prep":["prep"],"freeze-prereg":["freeze-prereg"],"authorize-stage":["authorize-stage","--stage","smoke","--output","x"],"smoke":["smoke","--authorization","a"],"pilot":["pilot","--authorization","a"],"scale-1":["scale-1","--authorization","a"],"scale-4":["scale-4","--authorization","a"],"scale-decide":["scale-decide","--authorization","a"],"long":["long","--authorization","a","--world-size","1","--input-checkpoint","i","--scale-decision","s","--config","c","--preregistration","p"],"final-train-confirm":["final-train-confirm","--authorization","a"],"validation-once":["validation-once","--authorization","a"],"test-once":["test-once","--authorization","a"]}
    for mode,args in samples.items():assert cli.main(args)==mode
    assert called==list(samples)

def test_invalid_and_missing_args_rejected():
    parser=cli.build_parser()
    for args in ([],["unknown"],["smoke"],["authorize-stage","--stage","smoke"],["long","--authorization","a","--world-size","2"]):
        with pytest.raises(SystemExit):parser.parse_args(args)

def test_default_handlers_are_real_and_safe_reject_before_data(tmp_path):
    assert cli.HANDLERS["prep"] is cli.prep and cli.HANDLERS["freeze-prereg"] is cli.freeze_prereg and cli.HANDLERS["authorize-stage"] is cli.authorize_handler
    assert all(callable(handler) and handler.__name__!="_unwired" for handler in cli.HANDLERS.values())
    args=cli.build_parser().parse_args(["smoke","--authorization",str(tmp_path/"missing.json")])
    with pytest.raises(FileNotFoundError):cli.HANDLERS["smoke"](args)

def test_v10_identity_terminal_and_backend_version():
    identity=complete_identity_v10(mode="smoke",run_digest="r",authorization_sha256="a",engine_sha256="e",script_sha256="s",config_sha256_effective="c",panels_sha256_effective="p",basis_sha256_effective="b",parent_sha256_effective="par",input_checkpoint_sha256="none");assert identity["candidate"]=="r16_dscp_v10" and identity["schema"]=="r16_dscp_v10_checkpoint_identity_v1"
    terminal=terminal_v10("smoke","failed",{});assert terminal["schema"]=="r16_dscp_v10_smoke_terminal_v1" and terminal["candidate"]=="r16_dscp_v10"
    with pytest.raises(TypeError):cli.V10StageDispatcher(object.__new__(V9ProductionBackend)) if hasattr(cli,"V10StageDispatcher") else (_ for _ in ()).throw(TypeError())

@pytest.mark.parametrize("mode",["smoke","pilot","scale-1","scale-4","scale-decide","long","final-train-confirm","validation-once","test-once"])
def test_each_mode_terminal_adapter_normal_and_failure(mode,tmp_path):
    training=mode in {"smoke","pilot","long"};payload={"status":"passed" if mode!="long" else "completed_train","gates":{"x":{"passed":True}},"lineage":{"run":"r"},"cache":{"hits":1},"latency":{"finite":True},"peak_vram_bytes":1,"checkpoint":{"best":"b","last":"l"} if training else None,"custom":"preserved"};normal=cli.adapt_terminal(mode,payload);assert normal["schema"].startswith("r16_dscp_v10_") and normal["candidate"]=="r16_dscp_v10" and normal["custom"]=="preserved" and normal["gates"]==payload["gates"]
    failed=cli.adapt_terminal(mode,{**payload,"status":"failed","checkpoint":None,"error_type":"OOM"});assert failed["status"]=="failed" and failed["checkpoint"] is None and failed["error_type"]=="OOM"

def test_owned_terminal_replacement_and_next_auth_checkpoint_preserved(tmp_path):
    payload={"status":"passed","checkpoint":{"best":"b","last":"l"},"gates":{"passed":True},"best_checkpoint":{"path":"b","sha256":"h"}};(tmp_path/"terminal.json").write_text(__import__("json").dumps(payload));adapted=cli.replace_owned_terminal(tmp_path,"pilot");assert adapted["checkpoint"]==payload["checkpoint"] and __import__("json").loads((tmp_path/"terminal.json").read_text())["schema"]=="r16_dscp_v10_pilot_terminal_v1"
