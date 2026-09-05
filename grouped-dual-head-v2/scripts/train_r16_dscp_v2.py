#!/usr/bin/env python3
"""Frozen harness and reproducibility entrypoint for ``r16_dscp_v2``."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any, Mapping

import h5py
import numpy as np
import torch
from torch import nn
import yaml


ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import (
    R16DSCP, RidgePointwiseBaseline, analytic_model_macs, predictor_parameter_count,
)
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import (
    BASIS_FILE_SHA256, BASIS_TENSOR_SHA256, BindingRefusal, FROZEN_OPTIMIZATION,
    PANELS_SHA256, PARENT_PATH, PARENT_SHA256, StageGateRefusal,
    atomic_json_exclusive, checkpoint_payload, configure_determinism,
    make_optimizer, require_cuda_environment, require_stage_terminal,
    save_best_last, sha256_file, space_gate, write_failure_terminal,
)
from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime
from scripts import probe_r4_family_temporal_pod_capacity as legacy_pod
from scripts import train_r16_dscp as v1_prep


CANDIDATE = "r16_dscp_v2"
SCRIPT = Path(__file__).resolve()
MODULE = ROOT / "saved_time_phase_operator_v4/instance_adaptation/r16_dscp_training_v2.py"
MODEL_MODULE = ROOT / "saved_time_phase_operator_v4/instance_adaptation/r16_dscp.py"
TEST = ROOT / "tests/saved_time_phase_operator_v4/test_r16_dscp_v2.py"
CONFIG = ROOT / "configs/r16_dscp_v2.yaml"
RESULT_DIR = ROOT / "results/r16_dscp_v2"
PREFLIGHT = RESULT_DIR / "design_preflight.json"
STATIC = RESULT_DIR / "static_evidence.json"
TOY_DIR = RESULT_DIR / "toy_checkpoint"
REPLAY_DIR = RESULT_DIR / "replay_basis_verify"
REPLAY_VERIFY = REPLAY_DIR / "verification.json"
PREREG = ROOT / "results/r16_dscp_v2_preregistration_20260826.json"
V1_BASIS = ROOT / "results/r16_dscp_v1/basis_rank16.pt"
V1_PANELS = ROOT / "results/r16_dscp_v1/panels.json"
V1_PREREG = ROOT / "results/r16_dscp_v1_preregistration_20260826.json"
V1_SCALE_SHA256 = "179ba6567af6d0fb00dbf4efccb269b728713efed5f11baec7220b51a14803e3"
MODES_CUDA = (
    "replay-basis-verify", "smoke", "pilot", "scale-probe", "long",
    "final-train-confirm", "validation-once", "test-once",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def binding(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve(); stat = target.stat()
    return {"path": str(target), "sha256": sha256_file(target), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256(); digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C")); return digest.hexdigest()


def software() -> dict[str, Any]:
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": str(torch.version.cuda), "numpy": np.__version__,
            "h5py": h5py.__version__, "platform": platform.platform()}


def load_v1_basis() -> Mapping[str, Any]:
    if sha256_file(V1_BASIS) != BASIS_FILE_SHA256:
        raise BindingRefusal("v1 basis file hash mismatch")
    artifact = torch.load(V1_BASIS, map_location="cpu", weights_only=False)
    if tensor_sha256(artifact["basis"]) != BASIS_TENSOR_SHA256:
        raise BindingRefusal("v1 basis tensor hash mismatch")
    if tensor_sha256(artifact["coefficient_scales"]) != V1_SCALE_SHA256:
        raise BindingRefusal("v1 coefficient scale hash mismatch")
    return artifact


def effective_bindings() -> dict[str, Any]:
    if sha256_file(V1_PANELS) != PANELS_SHA256:
        raise BindingRefusal("v1 panels hash mismatch")
    if sha256_file(PARENT_PATH) != PARENT_SHA256:
        raise BindingRefusal("parent hash mismatch")
    load_v1_basis()
    return {
        "v2_training_module": binding(MODULE), "v2_script": binding(SCRIPT),
        "v2_test": binding(TEST), "v2_config": binding(CONFIG),
        "effective_model_module": binding(MODEL_MODULE), "v1_basis": binding(V1_BASIS),
        "v1_panels": binding(V1_PANELS), "parent": binding(PARENT_PATH),
        "manifest": binding(parent_runtime.MANIFEST_PATH),
        "normalization": binding(parent_runtime.NORMALIZATION_PATH),
        "run_identity": binding(parent_runtime.RUN_IDENTITY_PATH),
    }


def validate_commands(config: Mapping[str, Any]) -> None:
    for name, command in config["commands"].items():
        if not command.startswith("env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="):
            raise BindingRefusal(f"CUDA command lacks explicit deterministic prefix: {name}")


def static_payload() -> dict[str, Any]:
    basis = load_v1_basis()
    model = R16DSCP(basis["basis"], basis["coefficient_scales"])
    ridge = RidgePointwiseBaseline()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf8")); validate_commands(config)
    if predictor_parameter_count(model) != 1202 or predictor_parameter_count(ridge) != 480:
        raise BindingRefusal("parameter contract mismatch")
    return {
        "schema": "r16_dscp_v2_static_v1", "candidate": CANDIDATE,
        "parameters": 1202, "ridge_parameters": 480,
        "model_only_macs": analytic_model_macs(), "cuda_commands_explicit": True,
        "optimization": asdict(FROZEN_OPTIMIZATION),
        "sealed": {"validation_opened": False, "test_id_opened": False},
    }


def make_toy_checkpoint(run_identity: Mapping[str, Any]) -> dict[str, Any]:
    basis = load_v1_basis(); model = R16DSCP(basis["basis"], basis["coefficient_scales"])
    optimizer = make_optimizer(list(model.parameters()))
    optimizer.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 1.0e-4)
    torch.nn.utils.clip_grad_norm_(model.parameters(), FROZEN_OPTIMIZATION.grad_clip_norm)
    optimizer.step()
    payload = checkpoint_payload(model, optimizer, run_identity=run_identity,
                                 sampler_order=list(range(12)),
                                 progress={"mode": "toy_size_probe", "update": 1, "epoch": 0})
    record = save_best_last(payload, TOY_DIR, is_best=True)
    gate = space_gate(record["size_bytes"], ROOT)
    if record["size_bytes"] > 2 * 1024**2 or not gate["passed"] or not record["hardlinked"]:
        raise BindingRefusal("toy checkpoint/space/retention gate failed")
    return {**record, "space_gate": gate, "checkpoint_file_sha256": sha256_file(record["last"])}


def preflight() -> None:
    if RESULT_DIR.exists() or PREREG.exists():
        raise FileExistsError("v2 target exists; refuse overwrite")
    bindings = effective_bindings(); static = static_payload()
    panels = json.loads(V1_PANELS.read_text(encoding="utf8"))
    if panels["census"]["validation_records"] != 0 or panels["census"]["test_id_records"] != 0:
        raise BindingRefusal("frozen development panels overlap sealed splits")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf8"))
    run_identity = {
        "candidate": CANDIDATE, "seed": 372,
        "config_sha256": bindings["v2_config"]["sha256"],
        "code_sha256": bindings["v2_script"]["sha256"],
        "model_sha256": bindings["effective_model_module"]["sha256"],
        "basis_sha256": BASIS_FILE_SHA256, "basis_tensor_sha256": BASIS_TENSOR_SHA256,
        "panels_sha256": PANELS_SHA256, "parent_sha256": PARENT_SHA256,
    }
    run_identity["run_digest"] = hashlib.sha256(
        json.dumps(run_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    RESULT_DIR.mkdir(parents=False, exist_ok=False)
    atomic_json_exclusive(static, STATIC)
    toy = make_toy_checkpoint(run_identity)
    payload = {
        "schema": "r16_dscp_v2_design_preflight_v1", "candidate": CANDIDATE,
        "status": "frozen_pre_replay", "created_utc": utc_now(),
        "hypothesis": "The unchanged v1 R16-DSCP design can be trained and resumed through a fail-closed deterministic harness whose basis replay is bitwise bound before any training authorization.",
        "effective_bindings": bindings,
        "v1_correction_chain_provenance_only": {
            "v1_prereg": binding(V1_PREREG),
            "correction_v2": binding(ROOT / "results/r16_dscp_v1/preflight_correction_v2.json"),
            "correction_v3": binding(ROOT / "results/r16_dscp_v1/preflight_correction_v3.json"),
        },
        "run_identity": run_identity, "toy_checkpoint": toy,
        "panel_census": panels["census"], "config": config,
        "static": binding(STATIC), "software": software(),
        "sealed": {"validation_opened": False, "test_id_opened": False,
                   "smoke_or_training_started": False},
    }
    atomic_json_exclusive(payload, PREFLIGHT)
    print(json.dumps({"status": "frozen_pre_replay", "preflight": binding(PREFLIGHT),
                      "toy_checkpoint": toy}))


def verify_preflight() -> Mapping[str, Any]:
    payload = json.loads(PREFLIGHT.read_text(encoding="utf8"))
    if payload.get("status") != "frozen_pre_replay": raise BindingRefusal("preflight status mismatch")
    current = effective_bindings()
    for key, value in payload["effective_bindings"].items():
        if current[key]["sha256"] != value["sha256"]:
            raise BindingRefusal(f"effective binding drift: {key}")
    return payload


def _fix_signs(basis: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(basis, dtype=torch.float64, device="cpu").clone()
    for mode in range(value.shape[1]):
        pivot = int(value[:, mode].abs().argmax())
        if float(value[pivot, mode]) < 0: value[:, mode].neg_()
    return value


@torch.inference_mode()
def replay_basis_verify(devices: str) -> None:
    if REPLAY_DIR.exists(): raise FileExistsError("replay run exists; refuse overwrite")
    require_cuda_environment(visible_devices=devices)
    if devices != "0": raise BindingRefusal("basis replay is frozen to physical GPU0")
    pre = verify_preflight(); configure_determinism(372)
    REPLAY_DIR.mkdir(parents=True, exist_ok=False)
    identity = {**pre["run_identity"], "mode": "replay-basis-verify", "devices": devices,
                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}
    atomic_json_exclusive(identity, REPLAY_DIR / "run_identity.json")
    atomic_json_exclusive({"parent": binding(PARENT_PATH), "basis_reference": binding(V1_BASIS)},
                          REPLAY_DIR / "checkpoint_binding.json")
    started = time.monotonic(); parent_before = sha256_file(PARENT_PATH)
    try:
        device = torch.device("cuda:0")
        gpu = parent_runtime.gpu_identity(0, device)
        model, normalizer, manifest, _ = parent_runtime.load_model_context(device)
        records = v1_prep.basis_records(manifest)
        bases=[]; scales=[]; torch.cuda.reset_peak_memory_stats(device)
        for family in v1_prep.FAMILIES:
            covariance = torch.zeros((401, 401), device=device, dtype=torch.float32)
            residuals=[]
            for record in records[family]:
                loaded = parent_runtime.load_record_input(record)
                parent = legacy_pod.generate_parent_full401(model, normalizer, loaded, manifest,
                                                            device=device, time_block=16)
                truth, _ = legacy_pod.load_train_truth(record, device=device)
                residual=(truth-parent).contiguous()
                legacy_pod.stream_accumulate_temporal_covariance(covariance, residual)
                residuals.append(residual.cpu()); del parent, truth, residual
                torch.cuda.empty_cache()
            _, vectors = legacy_pod.temporal_pod(covariance)
            basis = _fix_signs(vectors[:, :16]).float().contiguous()
            square=torch.zeros(16,dtype=torch.float64); count=0
            for residual in residuals:
                coefficient=basis.T @ residual.reshape(401,-1)
                square += coefficient.double().square().sum(1); count += coefficient.shape[1]
            scale=(3.0*torch.sqrt(square/float(count)).clamp_min(1e-12)).float()
            bases.append(basis); scales.append(scale)
        basis_tensor=torch.stack(bases).contiguous(); scale_tensor=torch.stack(scales).contiguous()
        observed_basis=tensor_sha256(basis_tensor); observed_scale=tensor_sha256(scale_tensor)
        if observed_basis != BASIS_TENSOR_SHA256 or observed_scale != V1_SCALE_SHA256:
            raise BindingRefusal("in-memory replay basis/scale hash mismatch")
        if sha256_file(PARENT_PATH) != parent_before: raise BindingRefusal("parent changed during replay")
        result={
            "schema":"r16_dscp_v2_basis_replay_v1", "status":"passed",
            "runtime_s":time.monotonic()-started, "basis_tensor_sha256":observed_basis,
            "coefficient_scales_tensor_sha256":observed_scale,
            "v1_basis_file_sha256":sha256_file(V1_BASIS),
            "peak_cuda_allocated_bytes":int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_reserved_bytes":int(torch.cuda.max_memory_reserved(device)),
            "gpu":gpu, "optimizer_used":False, "backward_used":False,
            "arrays_written":False, "validation_opened":False, "test_id_opened":False,
            "parent_sha256_before":parent_before, "parent_sha256_after":sha256_file(PARENT_PATH),
        }
        atomic_json_exclusive(result, REPLAY_VERIFY)
        atomic_json_exclusive({"schema":"r16_dscp_v2_terminal_v1","status":"passed",
                               "mode":"replay-basis-verify","verification_sha256":sha256_file(REPLAY_VERIFY)},
                              REPLAY_DIR / "terminal.json")
        print(json.dumps(result))
    except Exception as exc:
        if not (REPLAY_DIR / "terminal.json").exists():
            write_failure_terminal(REPLAY_DIR, mode="replay-basis-verify", reason=str(exc),
                                   run_identity=identity)
        raise


def guarded_stage(mode: str, devices: str) -> None:
    require_cuda_environment(visible_devices=devices)
    if not PREREG.is_file(): raise StageGateRefusal("frozen v2 preregistration absent")
    prerequisites = {
        "smoke": REPLAY_DIR / "terminal.json",
        "pilot": RESULT_DIR / "smoke/terminal.json",
        "scale-probe": RESULT_DIR / "pilot/terminal.json",
        "long": RESULT_DIR / "scale-probe/terminal.json",
        "final-train-confirm": RESULT_DIR / "long/terminal.json",
        "validation-once": RESULT_DIR / "final-train-confirm/terminal.json",
        "test-once": RESULT_DIR / "validation-once/terminal.json",
    }
    require_stage_terminal(prerequisites[mode])
    authorization = RESULT_DIR / f"authorizations/{mode}.json"
    if not authorization.is_file():
        raise StageGateRefusal(f"explicit frozen authorization absent: {authorization}")
    if mode in {"validation-once", "test-once"}:
        token = RESULT_DIR / f"{mode}.claimed.json"
        atomic_json_exclusive({"mode":mode,"claimed_utc":utc_now(),
                               "authorization_sha256":sha256_file(authorization)}, token)
    raise StageGateRefusal("stage is gated for a later experiment-worker implementation; prep cannot train/evaluate")


def freeze_prereg() -> None:
    if PREREG.exists(): raise FileExistsError("prereg exists; refuse overwrite")
    pre=verify_preflight(); terminal=require_stage_terminal(REPLAY_DIR / "terminal.json")
    verification=json.loads(REPLAY_VERIFY.read_text(encoding="utf8"))
    if verification["basis_tensor_sha256"] != BASIS_TENSOR_SHA256: raise BindingRefusal("replay mismatch")
    config=yaml.safe_load(CONFIG.read_text(encoding="utf8")); validate_commands(config)
    checkpoint_bytes=int(pre["toy_checkpoint"]["size_bytes"]); gate=space_gate(checkpoint_bytes, ROOT)
    if not gate["passed"]: raise BindingRefusal("measured checkpoint space gate failed")
    payload={
        "schema":"r16_dscp_v2_preregistration_v1","candidate":CANDIDATE,"status":"frozen",
        "created_utc":utc_now(),
        "hypothesis":pre["hypothesis"],
        "claim_boundary":"Harness and train-only basis replay only. No predictor training, validation, or test_id result. Parent speedup is only 1.608x; candidate may not claim 10x or faster-than-parent without measured complete E2E evidence.",
        "effective_bindings":effective_bindings(),
        "v1_correction_chain_provenance_only":pre["v1_correction_chain_provenance_only"],
        "preflight":binding(PREFLIGHT),"static":binding(STATIC),
        "replay_verification":binding(REPLAY_VERIFY),"replay_terminal":binding(REPLAY_DIR/"terminal.json"),
        "toy_checkpoint":pre["toy_checkpoint"],"space_gate":gate,
        "config":config,"commands":config["commands"],
        "stage_order":["smoke","pilot","scale-probe","long","final-train-confirm","validation-once","test-once"],
        "once_only":{"validation_token":"results/r16_dscp_v2/validation-once.claimed.json",
                     "test_token":"results/r16_dscp_v2/test-once.claimed.json",
                     "test_requires_validation_pass":True,
                     "validation_requires_locked_checkpoint_hash_thresholds":True,
                     "any_change_requires_new_version":True},
        "rollback":"Any failure writes atomic terminal, preserves minimum best/last/log, forbids validation/test, never touches parent/protected files, and deletes nothing.",
        "sealed":{"validation_opened":False,"test_id_opened":False,"smoke_or_training_started":False},
        "software":software(),"disk_free_bytes":int(shutil.disk_usage(ROOT).free),
    }
    atomic_json_exclusive(payload, PREREG)
    print(json.dumps({"status":"frozen","preregistration":binding(PREREG)}))


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",required=True,choices=("static","preflight",*MODES_CUDA,"freeze-prereg"))
    parser.add_argument("--devices",default="0")
    return parser.parse_args()


def main() -> None:
    args=parse_args()
    if args.mode=="static": print(json.dumps(static_payload(),sort_keys=True))
    elif args.mode=="preflight": preflight()
    elif args.mode=="replay-basis-verify": replay_basis_verify(args.devices)
    elif args.mode=="freeze-prereg": freeze_prereg()
    else: guarded_stage(args.mode,args.devices)


if __name__=="__main__": main()
