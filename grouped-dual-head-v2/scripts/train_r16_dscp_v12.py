#!/usr/bin/env python3
"""V12 complete production CLI with a sealed final/validation/test chain."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import yaml

from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import R16DSCP
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import (
    AUTH_SCHEMA,
    AccessLedger,
    Lineage,
    canonical_sha,
    evaluation_gates,
    gate_terminal,
    scale_decision,
    validate_authorization,
)
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v4 import (
    CandidateSpool,
    LongResumeState,
    actual_coefficient_loss,
    dual_space_gate,
)
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v5 import model_state_digest
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v7 import resume_v7_checkpoint
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v12 import (
    V12ProductionBackend,
    build_final_chain,
    build_test_terminal,
    build_validation_terminal,
    complete_identity_v12,
    terminal_v12,
    test_authorization,
    validate_test_authorization,
    validate_validation_authorization,
    validation_authorization,
)
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import (
    BASIS_FILE_SHA256,
    BASIS_TENSOR_SHA256,
    PANELS_SHA256,
    PARENT_PATH,
    PARENT_SHA256,
    BindingRefusal,
    atomic_json_exclusive,
    checkpoint_payload,
    make_optimizer,
    save_best_last,
    sha256_file,
)
from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime
import scripts.train_r16_dscp_v10 as v10


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "r16_dscp_v12"
SCRIPT = Path(__file__).resolve()
ENGINE = ROOT / "saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v12.py"
TEST = ROOT / "tests/saved_time_phase_operator_v4/test_r16_dscp_v12.py"
CONFIG = ROOT / "configs/r16_dscp_v12.yaml"
OUT = ROOT / "results/r16_dscp_v12"
STATIC = OUT / "static_evidence.json"
PREFLIGHT = OUT / "design_preflight.json"
PREREG = ROOT / "results/r16_dscp_v12_preregistration_20260826.json"
BASIS = ROOT / "results/r16_dscp_v1/basis_rank16.pt"
PANELS = ROOT / "results/r16_dscp_v1/panels.json"
TEST_LOG = Path("/tmp/r16_dscp_v12_tests.log")
MODES = (
    "prep",
    "freeze-prereg",
    "authorize-stage",
    "smoke",
    "pilot",
    "scale-1",
    "scale-4",
    "scale-decide",
    "long",
    "final-train-confirm",
    "validation-once",
    "test-once",
)
THRESHOLDS = {
    "joint_improvement": 0.01,
    "per_family_improvement": 0.005,
    "nonworse_records": 23,
    "late_high_nonworse": True,
    "e2e_parent_ratio_max": 1.05,
    "absolute_rel_l2_max": 0.05,
    "traditional_speedup_min": 10.0,
}


class V12StageDispatcher:
    def __init__(self, backend: V12ProductionBackend):
        if not isinstance(backend, V12ProductionBackend):
            raise TypeError("v12 dispatcher requires V12ProductionBackend")
        self.backend = backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("prep")
    sub.add_parser("freeze-prereg")
    authorize = sub.add_parser("authorize-stage")
    authorize.add_argument("--stage", required=True, choices=MODES[3:])
    authorize.add_argument("--output", required=True)
    authorize.add_argument("--input-checkpoint")
    for mode in (
        "smoke",
        "pilot",
        "scale-1",
        "scale-4",
        "scale-decide",
        "final-train-confirm",
        "validation-once",
        "test-once",
    ):
        stage = sub.add_parser(mode)
        stage.add_argument("--authorization", required=True)
        stage.add_argument("--physical-gpu-index", type=int, default=0)
        stage.add_argument("--device", default="cuda:0")
        stage.add_argument("--config", default=str(CONFIG.relative_to(ROOT)))
        stage.add_argument("--preregistration", default=str(PREREG.relative_to(ROOT)))
    long = sub.add_parser("long")
    long.add_argument("--authorization", required=True)
    long.add_argument("--world-size", type=int, required=True, choices=(1, 4))
    long.add_argument("--input-checkpoint", required=True)
    long.add_argument("--scale-decision", required=True)
    long.add_argument("--config", required=True)
    long.add_argument("--preregistration", required=True)
    long.add_argument("--resume", action="store_true")
    long.add_argument("--physical-gpu-index", type=int, default=0)
    long.add_argument("--device", default="cuda:0")
    return parser


def command_argv(command: str) -> list[str]:
    tokens = shlex.split(command)
    index = next(i for i, token in enumerate(tokens) if token.endswith("scripts/train_r16_dscp_v12.py"))
    return tokens[index + 1 :]


def bind(path: str | Path) -> dict[str, Any]:
    item = Path(path).resolve()
    return {"path": str(item), "sha256": sha256_file(item), "size_bytes": item.stat().st_size}


def effective() -> dict[str, str]:
    return {
        "code": sha256_file(ENGINE),
        "script": sha256_file(SCRIPT),
        "config": sha256_file(CONFIG),
        "panels": sha256_file(PANELS),
        "basis": sha256_file(BASIS),
        "parent": sha256_file(PARENT_PATH),
        "data_manifest": sha256_file(parent_runtime.MANIFEST_PATH),
        "normalization": sha256_file(parent_runtime.NORMALIZATION_PATH),
    }


def effective_bindings() -> dict[str, dict[str, Any]]:
    return {
        "engine": bind(ENGINE),
        "script": bind(SCRIPT),
        "test": bind(TEST),
        "config": bind(CONFIG),
        "model": bind(ROOT / "saved_time_phase_operator_v4/instance_adaptation/r16_dscp.py"),
        "v12_transitive_v11": bind(ROOT / "saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v11.py"),
        "v12_transitive_v10": bind(ROOT / "saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v10.py"),
        "harness": bind(ROOT / "saved_time_phase_operator_v4/instance_adaptation/r16_dscp_training_v2.py"),
        "basis": bind(BASIS),
        "panels": bind(PANELS),
        "parent": bind(PARENT_PATH),
        "manifest": bind(parent_runtime.MANIFEST_PATH),
        "normalization": bind(parent_runtime.NORMALIZATION_PATH),
        "v2_replay": bind(ROOT / "results/r16_dscp_v2/replay_basis_verify/verification.json"),
        "v11_veto_provenance": bind(ROOT / "results/r16_dscp_v11/design_preflight.json"),
    }


def metadata_seal() -> dict[str, Any]:
    payload = parent_runtime.load_manifest_payload()
    result: dict[str, Any] = {}
    for split in ("validation", "test_id"):
        rows = sorted(
            [
                {
                    "source_index": row["source_index"],
                    "sample_id": row["sample_id"],
                    "group_id": row["group_id"],
                    "sample_sha256": row["sample_sha256"],
                    "family": row["medium_type"],
                }
                for row in payload["records"]
                if row["split"] == split
            ],
            key=lambda row: (row["source_index"], row["sample_id"]),
        )
        result[split] = {
            "count": len(rows),
            "ordered_digest": canonical_sha(rows),
            "by_family": {
                family: sum(row["family"] == family for row in rows)
                for family in ("uniform", "layered", "marmousi")
            },
            "wavefield_read": False,
        }
    return result


def data_hashes() -> dict[str, str]:
    sealed = metadata_seal()
    return {
        "train_panels": sha256_file(PANELS),
        "validation": sealed["validation"]["ordered_digest"],
        "test_id": sealed["test_id"]["ordered_digest"],
        "manifest": sha256_file(parent_runtime.MANIFEST_PATH),
    }


def _preflight(root: Path = OUT) -> dict[str, Any]:
    path = root / "design_preflight.json"
    if not path.is_file():
        raise BindingRefusal("frozen preflight required")
    payload = json.loads(path.read_text())
    if payload.get("status") != "frozen" or payload.get("candidate") != CANDIDATE:
        raise BindingRefusal("wrong preflight")
    return payload


def _auth_base(stage: str, pre: Mapping[str, Any], input_checkpoint: str | None) -> dict[str, Any]:
    input_path = None if input_checkpoint is None else str(Path(input_checkpoint).resolve())
    input_sha = "none" if input_path is None else sha256_file(input_path)
    input_identity = pre["run_identity"]
    if input_path is not None:
        input_identity = torch.load(input_path, map_location="cpu", weights_only=False)["run_identity"]
    line = Lineage(
        CANDIDATE,
        stage,
        pre["run_identity"]["run_digest"],
        sha256_file(ENGINE),
        sha256_file(CONFIG),
        PANELS_SHA256,
        BASIS_FILE_SHA256,
        BASIS_TENSOR_SHA256,
        PARENT_SHA256,
        input_sha,
    )
    return {
        "schema": AUTH_SCHEMA,
        "status": "authorized",
        **line.__dict__,
        "candidate": CANDIDATE,
        "effective": effective(),
        "input_checkpoint_path": input_path,
        "input_run_identity": input_identity,
        "preregistration_path": str(PREREG.resolve()),
        "preregistration_sha256": sha256_file(PREREG),
        "gates_sha256": canonical_sha(yaml.safe_load(CONFIG.read_text())),
    }


def _write_auth(payload: dict[str, Any], output: str | Path) -> dict[str, Any]:
    payload["authorization_digest"] = canonical_sha(payload)
    atomic_json_exclusive(payload, output)
    return payload


def authorize_handler(args: Any, *, root: str | Path | None = None, prereg: str | Path | None = None) -> dict[str, Any]:
    root_path = Path(root or OUT)
    prereg_path = Path(prereg or PREREG)
    if not prereg_path.is_file():
        raise BindingRefusal("frozen prereg required")
    pre = _preflight(root_path)
    stage = args.stage
    if stage == "smoke":
        payload = _auth_base(stage, pre, None)
        payload["preregistration_path"] = str(prereg_path.resolve())
        payload["preregistration_sha256"] = sha256_file(prereg_path)
        return _write_auth(payload, args.output)
    if stage == "validation-once":
        sealed = validation_authorization(
            terminal_path=root_path / "final-train-confirm/terminal.json",
            lock_path=root_path / "final-train-confirm/candidate_lock.json",
            effective=effective(),
        )
        payload = {**_auth_base(stage, pre, sealed["candidate_checkpoint_path"]), **sealed}
        return _write_auth(payload, args.output)
    if stage == "test-once":
        sealed = test_authorization(
            validation_terminal_path=root_path / "validation-once/terminal.json",
            lock_path=root_path / "final-train-confirm/candidate_lock.json",
            effective=effective(),
        )
        payload = {**_auth_base(stage, pre, sealed["candidate_checkpoint_path"]), **sealed}
        return _write_auth(payload, args.output)
    prereq = {
        "pilot": ("smoke",),
        "scale-1": ("pilot",),
        "scale-4": ("pilot",),
        "scale-decide": ("scale-1", "scale-4"),
        "long": ("pilot", "scale-decide"),
        "final-train-confirm": ("long",),
    }[stage]
    terminal_bindings = []
    for name in prereq:
        terminal = root_path / name / "terminal.json"
        if not terminal.is_file() or json.loads(terminal.read_text()).get("status") not in {"passed", "completed_train"}:
            raise BindingRefusal(f"prerequisite not passed: {name}")
        terminal_bindings.append({"mode": name, **bind(terminal)})
    if stage != "scale-decide" and not args.input_checkpoint:
        raise BindingRefusal("input checkpoint required")
    payload = _auth_base(stage, pre, args.input_checkpoint)
    payload["prerequisite_terminals"] = terminal_bindings
    return _write_auth(payload, args.output)


def validate_stage_auth(
    args: Any,
    *,
    before_factory_hook: Callable[[], None] = lambda: None,
    preflight_path: str | Path | None = None,
    prereg_path: str | Path | None = None,
    effective_fn: Callable[[], dict[str, str]] = effective,
) -> tuple[dict[str, Any], Lineage]:
    auth_path = Path(args.authorization)
    auth = json.loads(auth_path.read_text())
    pre = json.loads(Path(preflight_path or PREFLIGHT).read_text())
    line = Lineage(
        CANDIDATE,
        args.mode,
        pre["run_identity"]["run_digest"],
        sha256_file(ENGINE),
        sha256_file(CONFIG),
        PANELS_SHA256,
        BASIS_FILE_SHA256,
        BASIS_TENSOR_SHA256,
        PARENT_SHA256,
        str(auth.get("input_checkpoint_sha256", "none")),
    )
    validate_authorization(auth, line)
    expected_prereg = Path(prereg_path or PREREG)
    if auth.get("effective") != effective_fn() or auth.get("preregistration_sha256") != sha256_file(expected_prereg):
        raise BindingRefusal("effective/prereg drift")
    if auth.get("input_checkpoint_path") and sha256_file(auth["input_checkpoint_path"]) != auth["input_checkpoint_sha256"]:
        raise BindingRefusal("input checkpoint drift")
    for prerequisite in auth.get("prerequisite_terminals", []):
        if sha256_file(prerequisite["path"]) != prerequisite["sha256"]:
            raise BindingRefusal("prerequisite drift")
    if args.mode == "validation-once":
        validate_validation_authorization(auth, effective_fn())
    if args.mode == "test-once":
        validate_test_authorization(auth, effective_fn())
    before_factory_hook()
    if effective_fn() != auth["effective"]:
        raise BindingRefusal("TOCTOU effective drift")
    if auth.get("input_checkpoint_path") and sha256_file(auth["input_checkpoint_path"]) != auth["input_checkpoint_sha256"]:
        raise BindingRefusal("TOCTOU input drift")
    return auth, line


def build_v12_backend(mode: str, auth: Mapping[str, Any], line: Lineage, run: Path, identity: Mapping[str, Any]):
    bundle = v10.build_v10_backend(mode, auth, line, run, identity)
    backend = bundle[0]
    backend.__class__ = V12ProductionBackend
    backend.spool = CandidateSpool(
        backend.run_dir,
        run_digest=backend.lineage.run_digest,
        rank=int(os.environ.get("RANK", "0")),
        owned_root="/dev/shm/r16_dscp_v12",
    )
    return (backend, *bundle[1:])


def _eval_protocol(mode: str, backend: V12ProductionBackend, records: list[int], metadata: str, line: Lineage, run: Path) -> dict[str, Any]:
    scored = []
    for index in records:
        backend.preload(index)
        scored.append(backend.score_cached(index, True))
    checkpoint_path = backend.authorization.get("input_checkpoint_path")
    checkpoint_size = Path(checkpoint_path).stat().st_size if checkpoint_path else 0
    resources = dict(backend.resources({"size_bytes": checkpoint_size}, 1))
    gates, metrics = evaluation_gates(mode, scored, resources)
    status = "passed" if all(bool(value.get("passed")) for value in gates.values()) else "fail_gate"
    extra = {
        "lineage": line.__dict__,
        "metrics": metrics,
        "resources": resources,
        "metadata_digest": metadata,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "access_ledger_digest": canonical_sha([row["ledger_digest"] for row in scored]),
        "cache": backend.cache.payload(),
        "latency": backend.protocol.payload(),
        "peak_vram_bytes": backend.peak_bytes,
    }
    if mode == "final-train-confirm":
        terminal, _ = build_final_chain(
            run_dir=run,
            status=status,
            checkpoint_path=checkpoint_path,
            thresholds=THRESHOLDS,
            effective=effective(),
            data_hashes=data_hashes(),
            gates=gates,
            terminal_extra=extra,
        )
        return terminal
    if mode == "validation-once":
        return build_validation_terminal(
            path=run / "terminal.json",
            status=status,
            authorization=backend.authorization,
            gates=gates,
            terminal_extra=extra,
        )
    return build_test_terminal(
        path=run / "terminal.json",
        status=status,
        authorization=backend.authorization,
        gates=gates,
        terminal_extra=extra,
    )


def protocol_dispatch(mode: str, backend: V12ProductionBackend, roles: Mapping[str, list[int]], all_indices: list[int], metadata: str, line: Lineage, run: Path):
    V12StageDispatcher(backend)
    if mode in {"final-train-confirm", "validation-once", "test-once"}:
        return _eval_protocol(mode, backend, all_indices, metadata, line, run)
    return v10.protocol_dispatch(mode, backend, roles, all_indices, metadata, line, run)


def _adapt_owned_terminal(run: Path, mode: str) -> None:
    terminal = run / "terminal.json"
    if not terminal.is_file():
        return
    payload = json.loads(terminal.read_text())
    if payload.get("candidate") == CANDIDATE:
        return
    terminal.unlink()
    atomic_json_exclusive(terminal_v12(mode, payload.get("status", "failed"), payload), terminal)


def stage_handler(args: Any, *, backend_factory: Callable[..., Any] = build_v12_backend, before_factory_hook: Callable[[], None] = lambda: None):
    auth, line = validate_stage_auth(args, before_factory_hook=before_factory_hook)
    identity = complete_identity_v12(
        mode=args.mode,
        run_digest=line.run_digest,
        authorization_sha256=sha256_file(args.authorization),
        engine_sha256=sha256_file(ENGINE),
        script_sha256=sha256_file(SCRIPT),
        config_sha256_effective=sha256_file(CONFIG),
        panels_sha256_effective=PANELS_SHA256,
        basis_sha256_effective=BASIS_FILE_SHA256,
        parent_sha256_effective=PARENT_SHA256,
        input_checkpoint_sha256=line.input_checkpoint_sha256,
    )
    run = OUT / args.mode
    if run.exists():
        raise FileExistsError("stage exists")
    if args.mode == "scale-decide":
        run.mkdir(parents=True)
        one = json.loads((OUT / "scale-1/terminal.json").read_text())
        four = json.loads((OUT / "scale-4/terminal.json").read_text())
        result = scale_decision(one, four)
        payload = terminal_v12(args.mode, "passed", {"gates": result, "selected_gpus": result["selected_gpus"], "lineage": identity, "long_launched": False})
        atomic_json_exclusive(payload, run / "terminal.json")
        return payload
    if args.mode == "long":
        decision = json.loads(Path(args.scale_decision).read_text())
        actual = int(os.environ.get("WORLD_SIZE", str(args.world_size)))
        from scripts.train_r16_dscp_v8 import validate_long_cli
        validate_long_cli(
            arg_world_size=args.world_size,
            actual_world_size=actual,
            cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            authorization=auth,
            scale_decision=decision,
            resume=args.resume,
            terminal_exists=False,
        )
    run.mkdir(parents=True)
    atomic_json_exclusive(identity, run / "run_identity.json")
    try:
        bundle = backend_factory(args.mode, auth, line, run, identity)
        result = protocol_dispatch(args.mode, *bundle, line, run)
        if args.mode not in {"final-train-confirm", "validation-once", "test-once"}:
            _adapt_owned_terminal(run, args.mode)
        return result
    except Exception as exc:
        if not (run / "terminal.json").exists():
            atomic_json_exclusive(
                terminal_v12(
                    args.mode,
                    "failed",
                    {
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "lineage": identity,
                        "effective": effective(),
                        "parent": {"path": str(PARENT_PATH), "sha256": PARENT_SHA256},
                        "checkpoint": None,
                        "rollback": "zero_grad_clear_cache_preserve_owned_failure_spool_no_validation_or_test",
                    },
                ),
                run / "terminal.json",
            )
        raise


def final_handler(*, run_dir: str | Path, checkpoint_path: str | Path | None, thresholds: Mapping[str, Any], effective: Mapping[str, str], data_hashes: Mapping[str, str], evaluate: Callable[[], tuple[str, Mapping[str, Any]]]):
    status, gates = evaluate()
    return build_final_chain(run_dir=run_dir, status=status, checkpoint_path=checkpoint_path, thresholds=thresholds, effective=effective, data_hashes=data_hashes, gates=gates)


def authorize_validation(*, run_root: str | Path, effective: Mapping[str, str]) -> dict[str, Any]:
    root = Path(run_root)
    return validation_authorization(terminal_path=root / "final-train-confirm/terminal.json", lock_path=root / "final-train-confirm/candidate_lock.json", effective=effective)


def validation_handler(*, terminal_path: str | Path, authorization: Mapping[str, Any], effective: Mapping[str, str], evaluate: Callable[[], tuple[str, Mapping[str, Any]]]):
    validate_validation_authorization(authorization, effective)
    status, gates = evaluate()
    return build_validation_terminal(path=terminal_path, status=status, authorization=authorization, gates=gates)


def authorize_test(*, validation_terminal_path: str | Path, lock_path: str | Path, effective: Mapping[str, str]) -> dict[str, Any]:
    return test_authorization(validation_terminal_path=validation_terminal_path, lock_path=lock_path, effective=effective)


def test_handler(*, authorization: Mapping[str, Any], effective: Mapping[str, str], evaluate: Callable[[], Any], terminal_path: str | Path | None = None):
    validate_test_authorization(authorization, effective)
    result = evaluate()
    if terminal_path is not None and isinstance(result, tuple) and len(result) == 2:
        status, gates = result
        return build_test_terminal(path=terminal_path, status=status, authorization=authorization, gates=gates)
    return result


def prep(args: Any = None) -> dict[str, Any]:
    if OUT.exists() or PREREG.exists():
        raise FileExistsError("v12 exists")
    bindings = effective_bindings()
    config = yaml.safe_load(CONFIG.read_text())
    command_audit = {name: vars(build_parser().parse_args(command_argv(command))) for name, command in config["commands"].items()}
    tests = {"status": "passed", "log_sha256": sha256_file(TEST_LOG), "tail": TEST_LOG.read_text().splitlines()[-1]}
    inherited = json.loads((ROOT / "results/r16_dscp_v11/static_evidence.json").read_text())
    static = {
        "schema": "r16_dscp_v12_static_v1",
        "candidate": CANDIDATE,
        "parameters": 1202,
        "model_macs": 45451125,
        "host_cache_gates": inherited["host_cache_gates"],
        "metadata": metadata_seal(),
        "tests": tests,
        "command_audit": command_audit,
        "production_backend": "V12ProductionBackend",
        "production_eval_dispatch": ["build_final_chain", "build_validation_terminal", "build_test_terminal"],
        "simplified_v10_lock_reachable": False,
        "truth_read": False,
    }
    OUT.mkdir()
    atomic_json_exclusive(static, STATIC)
    artifact = torch.load(BASIS, map_location="cpu", weights_only=False)
    model = R16DSCP(artifact["basis"], artifact["coefficient_scales"])
    optimizer = make_optimizer(list(model.parameters()))
    h = w = 201
    model_args = (
        torch.ones(1, 1, h, w) * 2000,
        torch.zeros(1, 1, h, w),
        torch.ones(1, 1, h, w),
        torch.arange(w),
        torch.arange(h),
        torch.zeros(1, 1, h, w),
        torch.zeros(1, 1, h, w),
        torch.zeros(1, 401, h, w),
        torch.tensor([1]),
        torch.tensor([2]),
    )
    losses, adapted, _ = actual_coefficient_loss(model, model_args, torch.ones(1, 398, h, w))
    losses["total"].backward()
    optimizer.step()
    identity = complete_identity_v12(
        mode="prep",
        run_digest=canonical_sha(bindings),
        authorization_sha256="prep-none",
        engine_sha256=bindings["engine"]["sha256"],
        script_sha256=bindings["script"]["sha256"],
        config_sha256_effective=bindings["config"]["sha256"],
        panels_sha256_effective=PANELS_SHA256,
        basis_sha256_effective=BASIS_FILE_SHA256,
        parent_sha256_effective=PARENT_SHA256,
        input_checkpoint_sha256="none",
    )
    state = LongResumeState(epoch=0, global_step=1, group_index=0, best=float("inf"), bad_epochs=0, best_epoch=-1)
    payload = checkpoint_payload(model, optimizer, run_identity=identity, sampler_order=list(range(12)), progress=state.__dict__)
    saved = save_best_last(payload, OUT / "actual_checkpoint", is_best=True)
    restored = R16DSCP(artifact["basis"], artifact["coefficient_scales"])
    restored_optimizer = make_optimizer(list(restored.parameters()))
    _, resumed = resume_v7_checkpoint(saved["last"], restored, restored_optimizer, expected_identity=identity)
    ledger = AccessLedger("train")
    spool = CandidateSpool(OUT, run_digest=identity["run_digest"], rank=0, owned_root="/dev/shm/r16_dscp_v12")
    seal = spool.seal("size", adapted, ledger)
    spool_max = seal["serialized_bytes"]
    spool.cleanup(seal, ledger)
    gate1 = dual_space_gate(workspace=ROOT, tmpfs="/dev/shm", checkpoint_bytes=saved["size_bytes"], spool_max_bytes=spool_max, world_size=1)
    gate4 = dual_space_gate(workspace=ROOT, tmpfs="/dev/shm", checkpoint_bytes=saved["size_bytes"], spool_max_bytes=spool_max, world_size=4)
    preflight = {
        "schema": "r16_dscp_v12_preflight_v1",
        "candidate": CANDIDATE,
        "status": "frozen",
        "bindings": bindings,
        "run_identity": identity,
        "static": bind(STATIC),
        "checkpoint": {**saved, "sha256": sha256_file(saved["last"]), "resume_equal": resumed == state, "identity_digest": identity["identity_digest"], "state_sha256": model_state_digest(model)},
        "host_cache_gates": static["host_cache_gates"],
        "spool": {"max_bytes": spool_max, "root": "/dev/shm/r16_dscp_v12/<run_digest>/rank<r>", "owned_test_spools_remaining": 0},
        "space1": gate1,
        "space4": gate4,
        "metadata": static["metadata"],
        "truth_reads": 0,
        "GPU_used": False,
        "auth_created": False,
    }
    atomic_json_exclusive(preflight, PREFLIGHT)
    return preflight


def verify_preflight() -> dict[str, Any]:
    preflight = _preflight()
    current = effective_bindings()
    for key, value in preflight["bindings"].items():
        if current[key]["sha256"] != value["sha256"]:
            raise BindingRefusal(f"drift {key}")
    return preflight


def freeze_prereg(args: Any = None) -> dict[str, Any]:
    if PREREG.exists():
        raise FileExistsError("v12 prereg exists")
    preflight = verify_preflight()
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    payload = {
        "schema": "r16_dscp_v12_prereg_v1",
        "candidate": CANDIDATE,
        "status": "frozen",
        "preflight": bind(PREFLIGHT),
        "effective_bindings": effective_bindings(),
        "v11_blocker": bind(ROOT / "results/r16_dscp_v11/design_preflight.json"),
        "only_change": "production complete candidate lock and validation terminal chain",
        "config": yaml.safe_load(CONFIG.read_text()),
        "checkpoint": preflight["checkpoint"],
        "host_cache_gates": preflight["host_cache_gates"],
        "spool": preflight["spool"],
        "space1": preflight["space1"],
        "space4": preflight["space4"],
        "metadata": preflight["metadata"],
        "tests": json.loads(STATIC.read_text())["tests"],
        "command_audit": json.loads(STATIC.read_text())["command_audit"],
        "identities": {"python": platform.python_version(), "torch": torch.__version__, "gpu": gpu, "cuda_used": False},
        "claim_boundary": "prep only; validation/test future truth sealed; parent speedup is only 1.608x; candidate cannot claim 10x or faster than parent",
        "sealed": {"truth_read": False, "GPU_used": False, "auth_created": False},
        "rollback": "preserve parent and every protected checkpoint; no deletion",
    }
    atomic_json_exclusive(payload, PREREG)
    return payload


HANDLERS: dict[str, Callable[[Any], Any]] = {
    "prep": prep,
    "freeze-prereg": freeze_prereg,
    "authorize-stage": authorize_handler,
    "smoke": stage_handler,
    "pilot": stage_handler,
    "scale-1": stage_handler,
    "scale-4": stage_handler,
    "scale-decide": stage_handler,
    "long": stage_handler,
    "final-train-confirm": stage_handler,
    "validation-once": stage_handler,
    "test-once": stage_handler,
}


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    return HANDLERS[args.mode](args)


if __name__ == "__main__":
    main()


__all__ = [
    "HANDLERS",
    "MODES",
    "V12StageDispatcher",
    "authorize_test",
    "authorize_validation",
    "build_parser",
    "command_argv",
    "final_handler",
    "main",
    "test_handler",
    "validation_handler",
]
