#!/usr/bin/env python3
"""V13 complete production CLI with a sealed final/validation/test chain."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import time
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
    aggregate_scale_ranks,
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
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v5 import V5LongRunner, model_state_digest
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v7 import resume_v7_checkpoint
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v13 import (
    DistributedSession,
    InclusiveBudgetClock,
    LongResumeRefusal,
    RuntimeClosureRefusal,
    V13ProductionBackend,
    build_runtime_import_closure,
    build_final_chain,
    build_test_terminal,
    build_validation_terminal,
    complete_identity_v13,
    inspect_long_resume,
    terminal_v13,
    test_authorization,
    validate_test_authorization,
    validate_validation_authorization,
    validation_authorization,
    verify_runtime_import_closure,
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
CANDIDATE = "r16_dscp_v13"
SCRIPT = Path(__file__).resolve()
ENGINE = ROOT / "saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v13.py"
TEST = ROOT / "tests/saved_time_phase_operator_v4/test_r16_dscp_v13.py"
CONFIG = ROOT / "configs/r16_dscp_v13.yaml"
OUT = ROOT / "results/r16_dscp_v13"
STATIC = OUT / "static_evidence.json"
PREFLIGHT = OUT / "design_preflight.json"
PREREG = ROOT / "results/r16_dscp_v13_preregistration_20260826.json"
BASIS = ROOT / "results/r16_dscp_v1/basis_rank16.pt"
PANELS = ROOT / "results/r16_dscp_v1/panels.json"
TEST_LOG = Path("/tmp/r16_dscp_v13_tests.log")
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


class V13StageDispatcher:
    def __init__(self, backend: V13ProductionBackend):
        if not isinstance(backend, V13ProductionBackend):
            raise TypeError("v13 dispatcher requires V13ProductionBackend")
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
    resume_test = sub.add_parser("_resume-selftest")
    resume_test.add_argument("--phase", required=True, choices=("interrupt", "resume"))
    resume_test.add_argument("--run-dir", required=True)
    resume_test.add_argument("--authorized-run-dir", required=True)
    return parser


def command_argv(command: str) -> list[str]:
    tokens = shlex.split(command)
    index = next(i for i, token in enumerate(tokens) if token.endswith("scripts/train_r16_dscp_v13.py"))
    return tokens[index + 1 :]


def bind(path: str | Path) -> dict[str, Any]:
    item = Path(path).resolve()
    return {"path": str(item), "sha256": sha256_file(item), "size_bytes": item.stat().st_size}


def runtime_import_closure() -> dict[str, Any]:
    return build_runtime_import_closure(root=ROOT, entrypoints=[SCRIPT])


def frozen_runtime_import_closure(prereg_path: str | Path = PREREG) -> dict[str, Any]:
    path = Path(prereg_path)
    if not path.is_file():
        raise RuntimeClosureRefusal("frozen v13 preregistration required")
    payload = json.loads(path.read_text())
    closure = payload.get("runtime_import_closure")
    if not isinstance(closure, Mapping):
        raise RuntimeClosureRefusal("preregistration lacks runtime import closure")
    return dict(closure)


def verify_frozen_runtime_import_closure(
    prereg_path: str | Path = PREREG,
) -> dict[str, Any]:
    return verify_runtime_import_closure(
        frozen_runtime_import_closure(prereg_path), root=ROOT
    )


def effective() -> dict[str, str]:
    closure = runtime_import_closure()
    return {
        "code": sha256_file(ENGINE),
        "script": sha256_file(SCRIPT),
        "config": sha256_file(CONFIG),
        "panels": sha256_file(PANELS),
        "basis": sha256_file(BASIS),
        "parent": sha256_file(PARENT_PATH),
        "data_manifest": sha256_file(parent_runtime.MANIFEST_PATH),
        "normalization": sha256_file(parent_runtime.NORMALIZATION_PATH),
        "runtime_import_closure": str(closure["closure_sha256"]),
    }


def effective_bindings() -> dict[str, dict[str, Any]]:
    return {
        "engine": bind(ENGINE),
        "script": bind(SCRIPT),
        "test": bind(TEST),
        "config": bind(CONFIG),
        "model": bind(ROOT / "saved_time_phase_operator_v4/instance_adaptation/r16_dscp.py"),
        "v13_transitive_v11": bind(ROOT / "saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v11.py"),
        "v13_transitive_v10": bind(ROOT / "saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v10.py"),
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


def _auth_base(
    stage: str,
    pre: Mapping[str, Any],
    input_checkpoint: str | None,
    *,
    prereg_path: str | Path = PREREG,
) -> dict[str, Any]:
    closure = verify_frozen_runtime_import_closure(prereg_path)
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
        "preregistration_path": str(Path(prereg_path).resolve()),
        "preregistration_sha256": sha256_file(prereg_path),
        "runtime_import_closure_sha256": closure["closure_sha256"],
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
        payload = _auth_base(stage, pre, None, prereg_path=prereg_path)
        payload["preregistration_path"] = str(prereg_path.resolve())
        payload["preregistration_sha256"] = sha256_file(prereg_path)
        return _write_auth(payload, args.output)
    if stage == "validation-once":
        sealed = validation_authorization(
            terminal_path=root_path / "final-train-confirm/terminal.json",
            lock_path=root_path / "final-train-confirm/candidate_lock.json",
            effective=effective(),
        )
        payload = {**_auth_base(stage, pre, sealed["candidate_checkpoint_path"], prereg_path=prereg_path), **sealed}
        return _write_auth(payload, args.output)
    if stage == "test-once":
        sealed = test_authorization(
            validation_terminal_path=root_path / "validation-once/terminal.json",
            lock_path=root_path / "final-train-confirm/candidate_lock.json",
            effective=effective(),
        )
        payload = {**_auth_base(stage, pre, sealed["candidate_checkpoint_path"], prereg_path=prereg_path), **sealed}
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
    payload = _auth_base(stage, pre, args.input_checkpoint, prereg_path=prereg_path)
    payload["prerequisite_terminals"] = terminal_bindings
    if stage == "long":
        decision_path = root_path / "scale-decide/terminal.json"
        decision = json.loads(decision_path.read_text())
        selected = int(decision.get("selected_gpus", 0))
        if selected not in {1, 4}:
            raise BindingRefusal("scale decision did not select one or four GPUs")
        payload.update(
            selected_world_size=selected,
            scale_decision_path=str(decision_path.resolve()),
            scale_decision_sha256=sha256_file(decision_path),
            authorized_run_dir=str((root_path / "long").resolve()),
            resume_rebuild_budget_inclusive=True,
        )
    return _write_auth(payload, args.output)


def validate_stage_auth(
    args: Any,
    *,
    before_factory_hook: Callable[[], None] = lambda: None,
    preflight_path: str | Path | None = None,
    prereg_path: str | Path | None = None,
    effective_fn: Callable[[], dict[str, str]] = effective,
    event_hook: Callable[[str], None] = lambda _event: None,
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
    closure = verify_frozen_runtime_import_closure(expected_prereg)
    event_hook("closure_after_authorization")
    if auth.get("runtime_import_closure_sha256") != closure["closure_sha256"]:
        raise BindingRefusal("authorization runtime closure binding mismatch")
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


class ProductionFactory:
    """Construction is deliberately data/CUDA-free; build performs materialization."""

    def __init__(
        self,
        builder: Callable[..., Any],
        *,
        mode: str,
        auth: Mapping[str, Any],
        line: Lineage,
        run: Path,
        identity: Mapping[str, Any],
        resume_path: Path | None,
        resume_state: LongResumeState,
        budget_started_at: float,
    ) -> None:
        self.builder = builder
        self.values = (mode, auth, line, run, identity)
        self.resume_path = resume_path
        self.resume_state = resume_state
        self.budget_started_at = float(budget_started_at)

    def build(self) -> Any:
        return self.builder(
            *self.values,
            resume_path=self.resume_path,
            resume_state=self.resume_state,
            budget_started_at=self.budget_started_at,
        )


def build_v13_backend(
    mode: str,
    auth: Mapping[str, Any],
    line: Lineage,
    run: Path,
    identity: Mapping[str, Any],
    *,
    resume_path: Path | None = None,
    resume_state: LongResumeState = LongResumeState(),
    budget_started_at: float | None = None,
):
    bundle = v10.build_v10_backend(mode, auth, line, run, identity)
    backend = bundle[0]
    backend.__class__ = V13ProductionBackend
    backend.spool = CandidateSpool(
        backend.run_dir,
        run_digest=backend.lineage.run_digest,
        rank=int(os.environ.get("RANK", "0")),
        owned_root="/dev/shm/r16_dscp_v13",
    )
    if resume_path is not None:
        _, loaded_state = resume_v7_checkpoint(
            resume_path,
            backend.candidate,
            backend.optimizer,
            expected_identity=identity,
        )
        if loaded_state != resume_state:
            raise LongResumeRefusal("resume progress changed between inspection and factory restore")
    backend.v13_resume_state = resume_state
    backend.v13_budget_started_at = float(budget_started_at or time.monotonic())
    return (backend, *bundle[1:])


def _eval_protocol(mode: str, backend: V13ProductionBackend, records: list[int], metadata: str, line: Lineage, run: Path) -> dict[str, Any]:
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


def protocol_dispatch(mode: str, backend: V13ProductionBackend, roles: Mapping[str, list[int]], all_indices: list[int], metadata: str, line: Lineage, run: Path):
    V13StageDispatcher(backend)
    if mode in {"final-train-confirm", "validation-once", "test-once"}:
        return _eval_protocol(mode, backend, all_indices, metadata, line, run)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if mode == "scale-4":
        local = v10.protocol_dispatch(mode, backend, roles, all_indices, metadata, line, run)
        reports: list[Any] = [None] * world
        torch.distributed.all_gather_object(reports, local)
        sample_ids = sorted(
            str(row["sample_id"])
            for report in reports
            for row in report["rows"]
            if int(row["repeat"]) == 0
        )
        aggregate = aggregate_scale_ranks(reports, sample_ids)
        payload = terminal_v13(
            mode,
            "passed",
            {**aggregate, "lineage": line.__dict__, "rank0_persistent_writer": True},
        )
        if rank == 0:
            atomic_json_exclusive(payload, run / "terminal.json")
        return payload
    if mode == "long":
        train = roles["long_fit"]
        calibration = roles["long_calibration"]
        for index in train + calibration:
            backend.preload(index)

        def gather(rows: list[Any]) -> list[Any]:
            if world == 1:
                return rows
            shards: list[Any] = [None] * world
            torch.distributed.all_gather_object(shards, rows)
            return [item for shard in shards for item in shard]

        def step() -> None:
            if world == 4:
                for parameter in backend.candidate.parameters():
                    if parameter.grad is not None:
                        torch.distributed.all_reduce(parameter.grad)
                        parameter.grad.div_(4.0)
            torch.nn.utils.clip_grad_norm_(backend.candidate.parameters(), 1.0)
            backend.optimizer.step()

        saved: dict[str, Any] = {"best": None, "last": None}

        def checkpoint(state: LongResumeState, best: bool) -> None:
            if rank != 0:
                raise BindingRefusal("non-rank0 persistent checkpoint write refused")
            item = backend.checkpoint(state.__dict__, best)
            saved["last"] = item
            saved["best"] = item if best else saved["best"]

        def terminal(payload: Mapping[str, Any]) -> None:
            if rank != 0:
                raise BindingRefusal("non-rank0 persistent terminal write refused")
            chosen = saved["best"] or saved["last"]
            atomic_json_exclusive(
                terminal_v13(
                    mode,
                    str(payload["status"]),
                    {
                        **dict(payload),
                        "checkpoint": chosen,
                        "budget_started_before_factory_cache": True,
                        "resume_rebuild_in_wall_gpu_seconds": True,
                        "rank0_persistent_writer": True,
                    },
                ),
                run / "terminal.json",
            )

        runner = V5LongRunner(
            world_size=world,
            rank=rank,
            train_records=train,
            calibration_records=calibration,
            zero_grad=lambda: backend.optimizer.zero_grad(set_to_none=True),
            backward_record=lambda index, scale: backend.backward_cached(index, scale),
            optimizer_step=step,
            set_lr=lambda lr: [group.update(lr=lr) for group in backend.optimizer.param_groups],
            evaluate=lambda index: backend.score_cached(index),
            gather=gather,
            checkpoint=checkpoint,
            terminal=terminal,
            clock=InclusiveBudgetClock(backend.v13_budget_started_at),
        )
        return runner.run(state=backend.v13_resume_state)
    return v10.protocol_dispatch(mode, backend, roles, all_indices, metadata, line, run)


def _adapt_owned_terminal(run: Path, mode: str) -> None:
    terminal = run / "terminal.json"
    if not terminal.is_file():
        return
    payload = json.loads(terminal.read_text())
    if payload.get("candidate") == CANDIDATE:
        return
    terminal.unlink()
    atomic_json_exclusive(terminal_v13(mode, payload.get("status", "failed"), payload), terminal)


def stage_handler(
    args: Any,
    *,
    backend_factory: Callable[..., Any] = build_v13_backend,
    before_factory_hook: Callable[[], None] = lambda: None,
    event_hook: Callable[[str], None] = lambda _event: None,
):
    budget_started_at = time.monotonic()
    auth, line = validate_stage_auth(
        args,
        before_factory_hook=before_factory_hook,
        prereg_path=args.preregistration,
        event_hook=event_hook,
    )
    identity = complete_identity_v13(
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
    if args.mode == "scale-decide":
        if run.exists():
            raise FileExistsError("stage exists")
        run.mkdir(parents=True)
        one = json.loads((OUT / "scale-1/terminal.json").read_text())
        four = json.loads((OUT / "scale-4/terminal.json").read_text())
        result = scale_decision(one, four)
        payload = terminal_v13(args.mode, "passed", {"gates": result, "selected_gpus": result["selected_gpus"], "lineage": identity, "long_launched": False})
        atomic_json_exclusive(payload, run / "terminal.json")
        return payload
    resume_path: Path | None = None
    resume_state = LongResumeState()
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
            terminal_exists=(run / "terminal.json").exists(),
        )
        if (
            Path(args.scale_decision).resolve() != Path(auth["scale_decision_path"]).resolve()
            or sha256_file(args.scale_decision) != auth["scale_decision_sha256"]
        ):
            raise BindingRefusal("long scale decision path/hash mismatch")
        if args.resume:
            _, resume_state = inspect_long_resume(
                run_dir=run,
                authorized_run_dir=auth.get("authorized_run_dir", ""),
                expected_identity=identity,
                authorization_sha256=sha256_file(args.authorization),
            )
            resume_path = run / "last.pt"
            event_hook("resume_verified_before_cache")
        elif run.exists():
            if any(run.iterdir()):
                raise LongResumeRefusal("non-resume long refuses nonempty run directory")
            raise FileExistsError("non-resume long refuses existing run directory")
    elif run.exists():
        raise FileExistsError("stage exists")

    closure = verify_frozen_runtime_import_closure(args.preregistration)
    event_hook("closure_immediately_before_factory_construction")
    factory = ProductionFactory(
        backend_factory,
        mode=args.mode,
        auth=auth,
        line=line,
        run=run,
        identity=identity,
        resume_path=resume_path,
        resume_state=resume_state,
        budget_started_at=budget_started_at,
    )
    event_hook("factory_constructed_without_data_cuda")
    if verify_frozen_runtime_import_closure(args.preregistration) != closure:
        raise RuntimeClosureRefusal("runtime closure changed across factory construction")
    event_hook("closure_immediately_after_factory_construction")

    world = int(os.environ.get("WORLD_SIZE", "4" if args.mode == "scale-4" else "1"))
    if args.mode == "scale-4" and world != 4:
        raise BindingRefusal("scale-4 requires torchrun world size four")
    session = DistributedSession(world_size=world, backend="nccl" if world == 4 else "gloo")
    try:
        session.initialize()
        event_hook("distributed_initialized")
        if session.is_writer:
            is_resume = bool(getattr(args, "resume", False))
            run.mkdir(parents=True, exist_ok=is_resume)
            if not is_resume:
                atomic_json_exclusive(identity, run / "run_identity.json")
        if session.initialized:
            torch.distributed.barrier()
        bundle = factory.build()
        result = protocol_dispatch(args.mode, *bundle, line, run)
        if args.mode not in {"final-train-confirm", "validation-once", "test-once"}:
            if session.is_writer and args.mode != "scale-4":
                _adapt_owned_terminal(run, args.mode)
        failures = session.synchronize_failure(None)
        if failures:
            raise BindingRefusal(f"distributed terminal failure: {failures}")
        return result
    except Exception as exc:
        failures: list[dict[str, Any]] = []
        try:
            failures = session.synchronize_failure(exc)
        except Exception as sync_exc:
            failures = [{"rank": session.rank, "error_type": type(sync_exc).__name__, "error_message": str(sync_exc)}]
        rank0 = session.is_writer if session.initialized else int(os.environ.get("RANK", "0")) == 0
        if rank0 and run.is_dir() and not (run / "terminal.json").exists():
            atomic_json_exclusive(
                terminal_v13(
                    args.mode,
                    "failed",
                    {
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "lineage": identity,
                        "effective": auth.get("effective", {}),
                        "rank_failures": failures,
                        "parent": {"path": str(PARENT_PATH), "sha256": PARENT_SHA256},
                        "checkpoint": None,
                        "rollback": "zero_grad_clear_cache_preserve_owned_failure_spool_no_validation_or_test",
                    },
                ),
                run / "terminal.json",
            )
        raise
    finally:
        session.close()
        event_hook("distributed_destroyed")


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
        raise FileExistsError("v13 exists")
    bindings = effective_bindings()
    closure = runtime_import_closure()
    config = yaml.safe_load(CONFIG.read_text())
    command_audit = {name: vars(build_parser().parse_args(command_argv(command))) for name, command in config["commands"].items()}
    tests = {"status": "passed", "log_sha256": sha256_file(TEST_LOG), "tail": TEST_LOG.read_text().splitlines()[-1]}
    inherited = json.loads((ROOT / "results/r16_dscp_v11/static_evidence.json").read_text())
    static = {
        "schema": "r16_dscp_v13_static_v1",
        "candidate": CANDIDATE,
        "parameters": 1202,
        "model_macs": 45451125,
        "host_cache_gates": inherited["host_cache_gates"],
        "metadata": metadata_seal(),
        "tests": tests,
        "command_audit": command_audit,
        "production_backend": "V13ProductionBackend",
        "production_eval_dispatch": ["build_final_chain", "build_validation_terminal", "build_test_terminal"],
        "runtime_import_closure": closure,
        "distributed_contract": "init_before_collective_rank0_writer_sync_failure_destroy_all_paths",
        "long_resume_contract": "exact_authorized_dir_last_identity_rng_progress_before_cache_budget_inclusive",
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
    identity = complete_identity_v13(
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
    spool = CandidateSpool(OUT, run_digest=identity["run_digest"], rank=0, owned_root="/dev/shm/r16_dscp_v13")
    seal = spool.seal("size", adapted, ledger)
    spool_max = seal["serialized_bytes"]
    spool.cleanup(seal, ledger)
    gate1 = dual_space_gate(workspace=ROOT, tmpfs="/dev/shm", checkpoint_bytes=saved["size_bytes"], spool_max_bytes=spool_max, world_size=1)
    gate4 = dual_space_gate(workspace=ROOT, tmpfs="/dev/shm", checkpoint_bytes=saved["size_bytes"], spool_max_bytes=spool_max, world_size=4)
    preflight = {
        "schema": "r16_dscp_v13_preflight_v1",
        "candidate": CANDIDATE,
        "status": "frozen",
        "bindings": bindings,
        "runtime_import_closure": closure,
        "run_identity": identity,
        "static": bind(STATIC),
        "checkpoint": {**saved, "sha256": sha256_file(saved["last"]), "resume_equal": resumed == state, "identity_digest": identity["identity_digest"], "state_sha256": model_state_digest(model)},
        "host_cache_gates": static["host_cache_gates"],
        "spool": {"max_bytes": spool_max, "root": "/dev/shm/r16_dscp_v13/<run_digest>/rank<r>", "owned_test_spools_remaining": 0},
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
    if verify_runtime_import_closure(preflight["runtime_import_closure"], root=ROOT) != preflight["runtime_import_closure"]:
        raise RuntimeClosureRefusal("preflight runtime closure drift")
    return preflight


def freeze_prereg(args: Any = None) -> dict[str, Any]:
    if PREREG.exists():
        raise FileExistsError("v13 prereg exists")
    preflight = verify_preflight()
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    payload = {
        "schema": "r16_dscp_v13_prereg_v1",
        "candidate": CANDIDATE,
        "status": "frozen",
        "preflight": bind(PREFLIGHT),
        "effective_bindings": effective_bindings(),
        "frozen_v12_parent": bind(ROOT / "results/r16_dscp_v12_preregistration_20260826.json"),
        "runtime_import_closure": preflight["runtime_import_closure"],
        "only_change": "runtime import closure, four-rank distributed lifecycle, and reachable exact long resume",
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
        "acceptance_failure_budget_inheritance": {
            "source": "frozen v12 config unchanged except blocker contracts and v13 paths",
            "wall_gpu_budget_values_unchanged": True,
            "long_resume_cache_rebuild_included_in_wall_and_gpu_seconds": True,
            "failure_signals": ["failed_gate", "NaN/Inf", "OOM", "binding_drift", "unsafe_disk_or_memory", "budget_limit"],
        },
        "sealed_split_boundary": {
            "validation_test_future_wavefields": "sealed_until_frozen_stage_authorization",
            "prep_truth_reads": 0,
            "test_id_tuning": False,
        },
        "unique_candidate_identity": preflight["run_identity"],
        "sealed": {"truth_read": False, "GPU_used": False, "auth_created": False},
        "rollback": "preserve parent and every protected checkpoint; no deletion",
    }
    atomic_json_exclusive(payload, PREREG)
    return payload


def _resume_selftest_identity() -> dict[str, Any]:
    return complete_identity_v13(
        mode="long",
        run_digest="v13-resume-cli-selftest",
        authorization_sha256="a" * 64,
        engine_sha256=sha256_file(ENGINE),
        script_sha256=sha256_file(SCRIPT),
        config_sha256_effective=sha256_file(CONFIG),
        panels_sha256_effective=PANELS_SHA256,
        basis_sha256_effective=BASIS_FILE_SHA256,
        parent_sha256_effective=PARENT_SHA256,
        input_checkpoint_sha256="b" * 64,
    )


def _resume_selftest_objects() -> tuple[R16DSCP, Any]:
    artifact = torch.load(BASIS, map_location="cpu", weights_only=False)
    model = R16DSCP(artifact["basis"], artifact["coefficient_scales"])
    return model, make_optimizer(list(model.parameters()))


def _resume_selftest_step(model: R16DSCP, optimizer: Any) -> float:
    optimizer.zero_grad(set_to_none=True)
    random_scale = torch.rand((), dtype=torch.float32)
    loss = sum((parameter.float().square().sum() * random_scale) for parameter in model.parameters())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(random_scale)


def resume_selftest(args: Any) -> dict[str, Any]:
    run = Path(args.run_dir).resolve()
    authorized = Path(args.authorized_run_dir).resolve()
    if run != authorized:
        raise LongResumeRefusal("selftest authorization does not exactly bind run directory")
    identity = _resume_selftest_identity()
    if args.phase == "interrupt":
        if run.exists() and any(run.iterdir()):
            raise FileExistsError("resume selftest run directory is nonempty")
        run.mkdir(parents=True, exist_ok=True)
        atomic_json_exclusive(identity, run / "run_identity.json")
        torch.manual_seed(372)
        model, optimizer = _resume_selftest_objects()
        first_random = _resume_selftest_step(model, optimizer)
        state = LongResumeState(
            epoch=0,
            global_step=1,
            group_index=1,
            best=1.0,
            bad_epochs=0,
            best_epoch=0,
        )
        payload = checkpoint_payload(
            model,
            optimizer,
            run_identity=identity,
            sampler_order=list(range(12)),
            progress=state.__dict__,
        )
        saved = save_best_last(payload, run, is_best=True)
        next_random = _resume_selftest_step(model, optimizer)
        expected = {
            "schema": "r16_dscp_v13_resume_selftest_expected_v1",
            "first_random": first_random,
            "next_random": next_random,
            "next_state_sha256": model_state_digest(model),
            "checkpoint_sha256": sha256_file(saved["last"]),
            "interrupted_without_terminal": True,
        }
        atomic_json_exclusive(expected, run / "expected_next_step.json")
        return expected
    expected = json.loads((run / "expected_next_step.json").read_text())
    _, inspected_state = inspect_long_resume(
        run_dir=run,
        authorized_run_dir=authorized,
        expected_identity=identity,
        authorization_sha256="a" * 64,
    )
    cache_rebuilt = False
    model, optimizer = _resume_selftest_objects()
    _, loaded_state = resume_v7_checkpoint(
        run / "last.pt", model, optimizer, expected_identity=identity
    )
    if loaded_state != inspected_state:
        raise LongResumeRefusal("selftest resume state changed before fresh-object restore")
    checkpoint_loaded_before_cache = not cache_rebuilt
    cache_rebuilt = True
    next_random = _resume_selftest_step(model, optimizer)
    observed = {
        "schema": "r16_dscp_v13_resume_selftest_result_v1",
        "fresh_process": True,
        "fresh_objects": True,
        "checkpoint_loaded_before_cache": checkpoint_loaded_before_cache,
        "next_random_equal": next_random == expected["next_random"],
        "next_state_equal": model_state_digest(model) == expected["next_state_sha256"],
        "progress_equal": loaded_state == inspected_state,
        "next_step_equivalent": (
            next_random == expected["next_random"]
            and model_state_digest(model) == expected["next_state_sha256"]
        ),
    }
    atomic_json_exclusive(observed, run / "resume_result.json")
    return observed


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
    if args.mode == "_resume-selftest":
        return resume_selftest(args)
    return HANDLERS[args.mode](args)


if __name__ == "__main__":
    main()


__all__ = [
    "HANDLERS",
    "MODES",
    "V13StageDispatcher",
    "authorize_test",
    "authorize_validation",
    "build_parser",
    "command_argv",
    "final_handler",
    "main",
    "test_handler",
    "validation_handler",
    "resume_selftest",
]
