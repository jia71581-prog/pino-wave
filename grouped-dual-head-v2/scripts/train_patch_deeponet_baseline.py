#!/usr/bin/env python
"""Train-only Patch-DeepONet entry gate and full-support runner.

The registered configuration is launch-locked. ``audit`` is CPU/static. ``entry``
is a later one-GPU memorization gate. ``full`` requires both an explicit config
authorization and a passed, hash-bound entry report. No validation or test split
is accepted by this command.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grouped_ufno_mionet_v3.config import ALLOWED_MEDIUM_TYPES, V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts
from patch_deeponet_baseline.model import PatchDeepONet, PatchDeepONetConfig
from patch_deeponet_baseline.training import (
    dense_training_pair,
    patch_deeponet_loss,
    pi_deeponet_lwc84_residual_loss,
)
from saved_time_phase_operator_v4.data import ExactStoredTimeBatchDataset, split_pilot_batch
from saved_time_phase_operator_v4.full_support import (
    FullSupportStepSpec,
    audit_epoch_schedule,
    build_full_support_schedule,
    schedule_digest,
    warmup_cosine_factor,
)
from saved_time_phase_operator_v4.streaming_metrics import ExactWavefieldMetricAccumulator
from scripts.train_grouped_v3_pilot import load_normalizer
from scripts.train_saved_time_v4_full_support import validation_panel_indices


IDENTITY_SCHEMA = "patch_deeponet_run_identity_v1"
CHECKPOINT_SCHEMA = "patch_deeponet_checkpoint_v1"
ENTRY_SCHEMA = "patch_deeponet_entry_gate_v1"
PATCH_TRAINING_SCHEMA = "patch_deeponet_training_config_v1"
PI_TRAINING_SCHEMA = "pi_deeponet_training_config_v1"
EVALUATION_ENERGY_FLOOR_FRACTION = 0.01


def _physics_config(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Validate and return the registered discrete PI-DeepONet objective."""

    schema = config.get("schema")
    physics = config.get("physics")
    if schema == PATCH_TRAINING_SCHEMA:
        if physics is not None:
            raise ValueError("ordinary Patch-DeepONet config cannot enable physics loss")
        return None
    if schema != PI_TRAINING_SCHEMA or not isinstance(physics, Mapping):
        raise ValueError("unexpected DeepONet training schema or missing physics config")
    if physics.get("formulation") != "forced_saved_grid_lwc84_v1":
        raise ValueError("PI-DeepONet must use the audited forced saved-grid LWC84 defect")
    positive = {
        "weight": float(physics["weight"]),
        "dt_output_s": float(physics["dt_output_s"]),
        "dx_m": float(physics["dx_m"]),
        "dz_m": float(physics["dz_m"]),
        "denominator_epsilon": float(physics["denominator_epsilon"]),
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in positive.values()):
        raise ValueError("PI-DeepONet physics constants must be finite and positive")
    if positive["weight"] > 0.1:
        raise ValueError("unregistered PI-DeepONet physics weight above 0.1")
    exact = {
        "dt_output_s": 0.0025,
        "dx_m": 10.0,
        "dz_m": 10.0,
        "denominator_epsilon": 1.0e-8,
    }
    if any(positive[name] != value for name, value in exact.items()):
        raise ValueError("PI-DeepONet physics constants changed from dataset provenance")
    if int(physics["maximum_triplets_per_record"]) not in range(1, 7):
        raise ValueError("PI-DeepONet triplet cap must lie in [1,6]")
    if int(physics["warmup_updates"]) < 0:
        raise ValueError("PI-DeepONet physics warmup updates must be nonnegative")
    return physics


def _model_from_config(config: Mapping[str, Any]) -> PatchDeepONet:
    values = config.get("model", {})
    if not isinstance(values, Mapping):
        raise ValueError("Patch-DeepONet model configuration must be a mapping")
    return PatchDeepONet(PatchDeepONetConfig(**dict(values)))


def _require_appearance_time_policy(config: Mapping[str, Any]) -> None:
    policy = str(config["schedule"]["time_policy"])
    if policy != "appearance16":
        raise ValueError(
            "Patch-DeepONet training requires appearance16 so repeated records "
            "cover changing exact saved-time frames"
        )


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_checkpoint(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_resume_checkpoint(
    checkpoint_path: Path,
    *,
    model: PatchDeepONet,
    optimizer: torch.optim.Optimizer,
    expected_config_sha256: str,
    expected_manifest_digest: str,
    device: torch.device,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    identity_path = checkpoint_path.parent / "run_identity.json"
    if not checkpoint_path.is_file() or not identity_path.is_file():
        raise FileNotFoundError("resume checkpoint and parent run identity are required")
    parent_identity = json.loads(identity_path.read_text(encoding="utf8"))
    if parent_identity.get("schema") != IDENTITY_SCHEMA:
        raise ValueError("unexpected parent Patch-DeepONet run identity")
    if parent_identity.get("config_sha256") != expected_config_sha256:
        raise ValueError("resume checkpoint config binding changed")
    if parent_identity.get("manifest_digest") != expected_manifest_digest:
        raise ValueError("resume checkpoint manifest binding changed")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping) or checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected resume checkpoint schema")
    if checkpoint.get("run_digest") != parent_identity.get("run_digest"):
        raise ValueError("resume checkpoint and parent identity disagree")
    if checkpoint.get("manifest_digest") != expected_manifest_digest:
        raise ValueError("resume checkpoint manifest changed")
    if checkpoint.get("selection_split") != "train":
        raise ValueError("resume checkpoint was not selected on train")
    if "model_state" not in checkpoint or "optimizer_state" not in checkpoint:
        raise ValueError("resume checkpoint is incomplete")
    checkpoint = dict(checkpoint)
    if not isinstance(checkpoint.get("selection_metrics"), Mapping):
        if checkpoint.get("checkpoint_kind") != "rolling":
            raise ValueError("resume checkpoint has no train-selection metrics")
        best_metadata_path = checkpoint_path.parent / "best.json"
        if not best_metadata_path.is_file():
            raise FileNotFoundError(
                "rolling resume checkpoint requires sibling best.json metadata"
            )
        best_metadata = json.loads(best_metadata_path.read_text(encoding="utf8"))
        best_metrics = best_metadata.get("metrics")
        if not isinstance(best_metrics, Mapping):
            raise ValueError("rolling resume best metadata has no metrics")
        best_score = float(best_metadata.get("score", float("nan")))
        metric_score = float(best_metrics.get("aggregate_relative_l2", float("nan")))
        if (
            not math.isfinite(best_score)
            or not math.isfinite(metric_score)
            or best_score != metric_score
        ):
            raise ValueError("rolling resume best metadata score is invalid")
        if int(best_metadata.get("epoch", -1)) > int(checkpoint.get("epoch", -1)):
            raise ValueError("rolling resume best metadata is newer than checkpoint")
        checkpoint["selection_metrics"] = dict(best_metrics)
        checkpoint["incumbent_best_epoch"] = int(best_metadata["epoch"])
        checkpoint["incumbent_best_metadata_sha256"] = _sha256_file(
            best_metadata_path
        )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint, parent_identity


def _resume_position(resume: Mapping[str, Any] | None) -> tuple[int, int]:
    """Return the zero-based epoch and completed updates within that epoch."""

    if resume is None:
        return 0, 0
    completed_epoch_count = int(resume["epoch"])
    if resume.get("checkpoint_kind") == "rolling":
        update_in_epoch = int(resume.get("update_in_epoch", -1))
        if completed_epoch_count <= 0 or update_in_epoch <= 0:
            raise ValueError("rolling resume checkpoint has an invalid schedule cursor")
        return completed_epoch_count - 1, update_in_epoch
    if completed_epoch_count < 0:
        raise ValueError("resume checkpoint epoch is negative")
    return completed_epoch_count, 0


def _load_initialization_checkpoint(
    checkpoint_path: Path,
    *,
    model: PatchDeepONet,
    expected_manifest_digest: str,
    expected_sha256: str,
    device: torch.device,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Load model weights only for a new, manifest-bound PI objective."""

    identity_path = checkpoint_path.parent / "run_identity.json"
    if not checkpoint_path.is_file() or not identity_path.is_file():
        raise FileNotFoundError("initialization checkpoint and parent run identity are required")
    actual_sha256 = _sha256_file(checkpoint_path)
    if actual_sha256 != str(expected_sha256):
        raise ValueError("initialization checkpoint digest changed")
    parent_identity = json.loads(identity_path.read_text(encoding="utf8"))
    if parent_identity.get("schema") != IDENTITY_SCHEMA:
        raise ValueError("unexpected initialization run identity")
    if parent_identity.get("manifest_digest") != expected_manifest_digest:
        raise ValueError("initialization checkpoint manifest changed")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping) or checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected initialization checkpoint schema")
    if checkpoint.get("manifest_digest") != expected_manifest_digest:
        raise ValueError("initialization checkpoint manifest changed")
    if checkpoint.get("selection_split") != "train":
        raise ValueError("initialization checkpoint was not selected on train")
    if checkpoint.get("run_digest") != parent_identity.get("run_digest"):
        raise ValueError("initialization checkpoint and parent identity disagree")
    if "model_state" not in checkpoint:
        raise ValueError("initialization checkpoint has no model state")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return checkpoint, parent_identity


def _distributed() -> dict[str, int | bool]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
    return {"world_size": world, "rank": rank, "local_rank": local, "main": rank == 0}


def _barrier(context: Mapping[str, int | bool]) -> None:
    if int(context["world_size"]) > 1:
        dist.barrier()


def _average_gradients(model: torch.nn.Module, context: Mapping[str, int | bool]) -> None:
    world = int(context["world_size"])
    if world <= 1:
        return
    for parameter in model.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world)


def _broadcast(value: float, device: torch.device, context: Mapping[str, int | bool]) -> float:
    if int(context["world_size"]) <= 1:
        return float(value)
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    dist.broadcast(tensor, 0)
    return float(tensor.item())


def _entry_indices(manifest) -> tuple[int, ...]:
    train = tuple(record for record in manifest.records if record.split == "train")
    selected = []
    for family in ALLOWED_MEDIUM_TYPES:
        candidates = [
            (record.sample_id, index)
            for index, record in enumerate(train)
            if record.medium_type == family
        ]
        if not candidates:
            raise ValueError(f"train split has no {family} entry record")
        selected.append(min(candidates)[1])
    return tuple(selected)


def _evaluation_schedule(indices: Sequence[int]) -> tuple[FullSupportStepSpec, ...]:
    selected = tuple(int(value) for value in indices)
    return tuple(
        FullSupportStepSpec(
            step=900_000 + start,
            epoch=0,
            record_indices=selected[start : start + 12],
            appearance_indices=(0,) * len(selected[start : start + 12]),
        )
        for start in range(0, len(selected), 12)
    )


@torch.inference_mode()
def _evaluate_train(
    model: PatchDeepONet,
    *,
    base: V3Config,
    manifest,
    normalizer,
    config: Mapping[str, Any],
    indices: Sequence[int],
    device: torch.device,
    frames: int,
    all_saved: bool = False,
) -> dict[str, Any]:
    execution = config["model_execution"]
    dataset = ExactStoredTimeBatchDataset(
        base.data.source_h5,
        manifest,
        split="train",
        schedule=_evaluation_schedule(indices),
        query_points=1,
        seed=int(config["seed"]),
        time_policy="all_saved" if all_saved else "validation_fixed",
        frames_per_record=len(manifest.time_s) if all_saved else int(frames),
        travel_time_h5=config["travel_time_h5"],
    )
    accumulator = ExactWavefieldMetricAccumulator(
        energy_floor_fraction=EVALUATION_ENERGY_FLOOR_FRACTION,
        require_unique=True,
        stored_time_count=len(manifest.time_s),
    )
    model.eval()
    for batch in dataset:
        for micro in split_pilot_batch(batch, microbatch_records=1):
            prediction, target, indices_t = dense_training_pair(
                model,
                micro,
                normalizer,
                device,
                time_block=int(execution["time_block"]),
                query_chunk=int(execution["query_chunk"]),
                hard_causality_lead_cycles=float(config["loss"]["hard_causality_lead_cycles"]),
            )
            source_t0 = float(micro.source_parameters[0, 3])
            onset = int(np.searchsorted(np.asarray(manifest.time_s), source_t0))
            accumulator.update(
                prediction,
                target,
                families=micro.medium_type,
                group_ids=micro.group_id,
                sample_ids=micro.sample_id,
                time_indices=indices_t,
                source_onset_indices=[onset],
            )
    return accumulator.finalize()


@torch.inference_mode()
def _evaluate_train_physics(
    model: PatchDeepONet,
    *,
    base: V3Config,
    manifest,
    normalizer,
    config: Mapping[str, Any],
    indices: Sequence[int],
    device: torch.device,
) -> dict[str, float | int]:
    """Evaluate the PI loss on a fixed train-only appearance panel."""

    physics = _physics_config(config)
    if physics is None:
        raise ValueError("physics evaluation requires a PI-DeepONet config")
    selected = tuple(int(value) for value in indices)
    spec = FullSupportStepSpec(
        step=950_000,
        epoch=0,
        record_indices=selected,
        appearance_indices=(0,) * len(selected),
    )
    dataset = ExactStoredTimeBatchDataset(
        base.data.source_h5,
        manifest,
        split="train",
        schedule=(spec,),
        query_points=1,
        seed=int(config["seed"]),
        time_policy="appearance16",
        frames_per_record=int(config["schedule"]["frames_per_record"]),
        travel_time_h5=config["travel_time_h5"],
    )
    execution = config["model_execution"]
    weighted_loss = 0.0
    triplets = 0
    model.eval()
    for micro in split_pilot_batch(dataset[0], microbatch_records=1):
        prediction, _, time_indices = dense_training_pair(
            model,
            micro,
            normalizer,
            device,
            time_block=int(execution["time_block"]),
            query_chunk=int(execution["query_chunk"]),
            hard_causality_lead_cycles=float(
                config["loss"]["hard_causality_lead_cycles"]
            ),
        )
        velocity = micro.velocity_mps.index_select(
            0, micro.record_to_medium.long()
        )
        residual = pi_deeponet_lwc84_residual_loss(
            prediction,
            velocity,
            micro.source_parameters,
            micro.source_map,
            micro.requested_time_s,
            time_indices,
            pressure_scale_pa=float(normalizer.metadata.pressure_scale_pa),
            dt_s=float(physics["dt_output_s"]),
            dx_m=float(physics["dx_m"]),
            dz_m=float(physics["dz_m"]),
            maximum_triplets_per_record=int(
                physics["maximum_triplets_per_record"]
            ),
            denominator_epsilon=float(physics["denominator_epsilon"]),
        )
        weighted_loss += float(residual.loss) * residual.triplet_count
        triplets += residual.triplet_count
    if triplets <= 0 or not math.isfinite(weighted_loss):
        raise FloatingPointError("fixed PI-DeepONet physics panel is invalid")
    return {
        "normalized_lwc84_defect_mse": weighted_loss / triplets,
        "triplet_count": triplets,
    }


def audit_config(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf8"))
    physics = _physics_config(config)
    if config["schedule"]["split"] != "train" or config["selection"]["split"] != "train":
        raise ValueError("Patch-DeepONet fitting and selection must be train-only")
    if config["claim_scope"]["source_generalization_variable"] != "position_only":
        raise ValueError("Patch-DeepONet source scope must remain position-only")
    if config["claim_scope"]["frequency_generalization_permitted"] is not False:
        raise ValueError("Patch-DeepONet frequency generalization must remain disabled")
    if float(config["claim_scope"]["fixed_position_evaluation_frequency_hz"]) != 19.0:
        raise ValueError("Patch-DeepONet position frequency changed")
    if bool(config["model_execution"]["automatic_mixed_precision"]):
        raise ValueError("unregistered Patch-DeepONet mixed precision")
    if bool(config.get("launch_authorized", False)):
        _require_appearance_time_policy(config)
    base = V3Config.from_yaml(config["base_config"])
    manifest = build_manifest(base.data.source_h5)
    validate_expected_counts(
        manifest,
        {"train": base.data.expected_train_records, "validation": base.data.expected_validation_records},
    )
    model = _model_from_config(config)
    initialization_audit: dict[str, Any] | None = None
    initialization = config.get("initialization")
    if physics is not None:
        if not isinstance(initialization, Mapping):
            raise ValueError("PI-DeepONet requires a bound initialization checkpoint")
        if initialization.get("policy") != "model_state_only_reset_optimizer":
            raise ValueError("PI-DeepONet initialization must reset optimizer state")
        checkpoint_path = Path(str(initialization["checkpoint"]))
        expected_checkpoint_sha256 = str(initialization["checkpoint_sha256"])
        if len(expected_checkpoint_sha256) != 64 or not checkpoint_path.is_file():
            raise ValueError("PI-DeepONet initialization checkpoint binding is invalid")
        actual_checkpoint_sha256 = _sha256_file(checkpoint_path)
        if actual_checkpoint_sha256 != expected_checkpoint_sha256:
            raise ValueError("PI-DeepONet initialization checkpoint digest changed")
        initialization_audit = {
            "policy": initialization["policy"],
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": actual_checkpoint_sha256,
        }
    elif initialization is not None:
        raise ValueError("ordinary Patch-DeepONet config cannot change initialization")
    schedule_config = config["schedule"]
    equal_weights = np.ones(base.data.expected_train_records, dtype=np.float64)
    schedule = build_full_support_schedule(
        base.data.expected_train_records,
        epochs=int(config["epochs"]),
        macro_records=int(schedule_config["macro_records"]),
        macros_per_update=int(schedule_config["macros_per_update"]),
        seed=int(config["seed"]),
        epoch_offset=int(schedule_config["schedule_epoch_offset"]),
        record_weights=equal_weights,
        record_oversample=float(schedule_config["record_oversample"]),
    )
    audits = [
        audit_epoch_schedule(
            schedule,
            epoch=epoch,
            record_count=base.data.expected_train_records,
            macros_per_update=int(schedule_config["macros_per_update"]),
            allow_oversample=True,
        ).__dict__
        for epoch in range(int(config["epochs"]))
    ]
    if int(schedule_config["effective_records_per_update"]) != int(schedule_config["macro_records"]) * int(schedule_config["macros_per_update"]):
        raise ValueError("Patch-DeepONet effective update size changed")
    return {
        "schema": "patch_deeponet_training_static_audit_v1",
        "status": "pass",
        "launch_authorized": bool(config["launch_authorized"]),
        "config": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "manifest_digest": manifest.digest,
        "model_config": asdict(model.config),
        "parameter_match": model.parameter_match(),
        "schedule_digest": schedule_digest(schedule),
        "epoch_audits": audits,
        "selection_split": "train",
        "fixed_position_frequency_hz": 19.0,
        "frequency_generalization_permitted": False,
        "training_method": (
            "discrete_pi_deeponet" if physics is not None else "patch_deeponet"
        ),
        "physics": None if physics is None else dict(physics),
        "initialization": initialization_audit,
    }


def _require_full_authorization(
    config: Mapping[str, Any],
    entry_report: Path | None,
    *,
    expected_config_sha256: str | None = None,
    expected_manifest_digest: str | None = None,
    allow_failed_comparison: bool = False,
) -> Mapping[str, Any]:
    if not bool(config.get("launch_authorized", False)):
        raise PermissionError("full Patch-DeepONet launch is locked in the registered config")
    if entry_report is None or not entry_report.is_file():
        raise FileNotFoundError("a passed Patch-DeepONet entry report is required")
    report = json.loads(entry_report.read_text(encoding="utf8"))
    if report.get("schema") != ENTRY_SCHEMA:
        raise ValueError("unexpected Patch-DeepONet entry report schema")
    status = report.get("status")
    if status != "pass" and not (bool(allow_failed_comparison) and status == "fail"):
        raise ValueError("Patch-DeepONet entry gate has not passed")
    if report.get("selection_split") != "train":
        raise ValueError("Patch-DeepONet entry report was not selected on train")
    if expected_config_sha256 is not None and report.get("config_sha256") != expected_config_sha256:
        raise ValueError("Patch-DeepONet entry report config binding changed")
    if expected_manifest_digest is not None and report.get("manifest_digest") != expected_manifest_digest:
        raise ValueError("Patch-DeepONet entry report manifest binding changed")
    if not isinstance(report.get("run_digest"), str) or len(report["run_digest"]) != 64:
        raise ValueError("Patch-DeepONet entry report run binding is invalid")
    return report


def _train(
    config_path: Path,
    *,
    mode: str,
    entry_report: Path | None,
    smoke_updates: int,
    allow_failed_entry_for_comparison: bool,
    microbatch_records: int,
    checkpoint_every_updates: int,
    run_label: str | None,
    resume_checkpoint: Path | None,
) -> int:
    config = yaml.safe_load(config_path.read_text(encoding="utf8"))
    audit = audit_config(config_path)
    physics_config = _physics_config(config)
    _require_appearance_time_policy(config)
    if mode != "full" and bool(allow_failed_entry_for_comparison):
        raise ValueError("failed-entry comparison override is valid only for full mode")
    if int(microbatch_records) <= 0:
        raise ValueError("physical microbatch record count must be positive")
    if int(checkpoint_every_updates) < 0:
        raise ValueError("checkpoint interval must be nonnegative")
    if run_label is not None and (
        not run_label or any(not (value.isalnum() or value in "-_") for value in run_label)
    ):
        raise ValueError("run label may contain only letters, digits, hyphens, and underscores")
    if resume_checkpoint is not None and mode != "full":
        raise ValueError("checkpoint resume is valid only for full mode")
    authorization_report: Mapping[str, Any] | None = None
    if mode == "full":
        authorization_report = _require_full_authorization(
            config,
            entry_report,
            expected_config_sha256=str(audit["config_sha256"]),
            expected_manifest_digest=str(audit["manifest_digest"]),
            allow_failed_comparison=bool(allow_failed_entry_for_comparison),
        )
    context = _distributed()
    if not torch.cuda.is_available():
        raise RuntimeError("Patch-DeepONet training requires CUDA")
    if mode == "entry" and int(context["world_size"]) != 1:
        raise ValueError("Patch-DeepONet entry gate is a one-GPU run")
    if mode == "full" and int(context["world_size"]) != int(config["schedule"]["macros_per_update"]):
        raise ValueError("full Patch-DeepONet run requires one macro per registered DDP rank")
    device = torch.device(f"cuda:{int(context['local_rank'])}")
    base = V3Config.from_yaml(config["base_config"])
    manifest = build_manifest(base.data.source_h5)
    normalizer = load_normalizer(base, manifest.digest)
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    model = _model_from_config(config).to(device)
    initialization_checkpoint: Mapping[str, Any] | None = None
    initialization_identity: Mapping[str, Any] | None = None
    if physics_config is not None and resume_checkpoint is None:
        initialization = config["initialization"]
        initialization_checkpoint, initialization_identity = _load_initialization_checkpoint(
            Path(str(initialization["checkpoint"])),
            model=model,
            expected_manifest_digest=str(audit["manifest_digest"]),
            expected_sha256=str(initialization["checkpoint_sha256"]),
            device=device,
        )
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        betas=(float(optimizer_config["beta1"]), float(optimizer_config["beta2"])),
        eps=float(optimizer_config["epsilon"]),
    )
    resume: Mapping[str, Any] | None = None
    parent_identity: Mapping[str, Any] | None = None
    if resume_checkpoint is not None:
        resume, parent_identity = _load_resume_checkpoint(
            resume_checkpoint,
            model=model,
            optimizer=optimizer,
            expected_config_sha256=str(audit["config_sha256"]),
            expected_manifest_digest=str(audit["manifest_digest"]),
            device=device,
        )
    root_name = mode if run_label is None else f"{mode}_{run_label}"
    root = Path(config["artifact_dir"]) / root_name
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Patch-DeepONet artifact directory is not empty: {root}")
    identity_base = {
        "schema": IDENTITY_SCHEMA,
        "config_sha256": audit["config_sha256"],
        "manifest_digest": manifest.digest,
        "selection_split": "train",
        "model_config": asdict(model.config),
        "parameter_count": model.parameter_count(),
        "mode": mode,
        "frequency_generalization_permitted": False,
        "physical_microbatch_records": int(microbatch_records),
        "checkpoint_every_updates": int(checkpoint_every_updates),
        "run_label": run_label,
        "training_method": (
            "discrete_pi_deeponet"
            if physics_config is not None
            else "patch_deeponet"
        ),
        "physics": None if physics_config is None else dict(physics_config),
    }
    if (
        initialization_checkpoint is not None
        and initialization_identity is not None
    ):
        initialization = config["initialization"]
        identity_base.update(
            {
                "initialization_policy": str(initialization["policy"]),
                "initialization_checkpoint_sha256": str(
                    initialization["checkpoint_sha256"]
                ),
                "initialization_parent_run_digest": str(
                    initialization_identity["run_digest"]
                ),
                "initialization_parent_global_step": int(
                    initialization_checkpoint["global_step"]
                ),
            }
        )
    if mode == "full":
        assert entry_report is not None and authorization_report is not None
        identity_base.update(
            {
                "entry_gate_status": str(authorization_report["status"]),
                "entry_report_sha256": hashlib.sha256(entry_report.read_bytes()).hexdigest(),
                "comparison_only": str(authorization_report["status"]) != "pass",
            }
        )
    if resume is not None and parent_identity is not None and resume_checkpoint is not None:
        identity_base.update(
            {
                "parent_checkpoint_sha256": _sha256_file(resume_checkpoint),
                "parent_run_digest": str(parent_identity["run_digest"]),
                "resume_epoch": int(resume["epoch"]),
                "resume_update_in_epoch": int(resume.get("update_in_epoch", 0)),
                "resume_global_step": int(resume["global_step"]),
                "incumbent_best_epoch": int(
                    resume.get("incumbent_best_epoch", resume["epoch"])
                ),
                "incumbent_best_metadata_sha256": resume.get(
                    "incumbent_best_metadata_sha256"
                ),
            }
        )
    identity = {**identity_base, "run_digest": _digest(identity_base)}
    if bool(context["main"]):
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json(identity, root / "run_identity.json")
        _atomic_json(audit, root / "static_audit.json")
    _barrier(context)

    if mode == "entry":
        selected = _entry_indices(manifest)
        maximum = int(config["entry_gate"]["maximum_updates"])
        steps = maximum if int(smoke_updates) <= 0 else min(maximum, int(smoke_updates))
        schedule = tuple(
            FullSupportStepSpec(i, 0, selected, (i, i, i)) for i in range(steps)
        )
        epochs = 1
    else:
        schedule_config = config["schedule"]
        weights = np.ones(base.data.expected_train_records, dtype=np.float64)
        schedule = build_full_support_schedule(
            base.data.expected_train_records,
            epochs=int(config["epochs"]),
            macro_records=int(schedule_config["macro_records"]),
            macros_per_update=int(schedule_config["macros_per_update"]),
            seed=int(config["seed"]),
            epoch_offset=int(schedule_config["schedule_epoch_offset"]),
            record_weights=weights,
            record_oversample=float(schedule_config["record_oversample"]),
        )
        epochs = int(config["epochs"])
    execution, loss_config = config["model_execution"], config["loss"]
    best_score = (
        float(resume["selection_metrics"]["aggregate_relative_l2"])
        if resume is not None
        else float("inf")
    )
    global_step = int(resume["global_step"]) if resume is not None else 0
    start_epoch, resume_update_in_epoch = _resume_position(resume)
    if start_epoch < 0 or start_epoch >= epochs:
        raise ValueError("resume checkpoint epoch is outside the configured schedule")
    if resume is not None:
        updates_by_epoch = []
        for epoch in range(epochs):
            epoch_specs = sum(int(spec.epoch) == epoch for spec in schedule)
            if epoch_specs % int(context["world_size"]) != 0:
                raise ValueError("registered schedule is not divisible across DDP ranks")
            updates_by_epoch.append(epoch_specs // int(context["world_size"]))
        if resume_update_in_epoch > updates_by_epoch[start_epoch]:
            raise ValueError("rolling resume update is outside its registered epoch")
        expected_global_step = (
            sum(updates_by_epoch[:start_epoch]) + resume_update_in_epoch
        )
        if global_step != expected_global_step:
            raise ValueError(
                "resume checkpoint global step disagrees with its schedule cursor"
            )
    entry_physics_initial: dict[str, float | int] | None = None
    if mode == "entry" and physics_config is not None and bool(context["main"]):
        entry_physics_initial = _evaluate_train_physics(
            model,
            base=base,
            manifest=manifest,
            normalizer=normalizer,
            config=config,
            indices=selected,
            device=device,
        )
    started = time.monotonic()
    for epoch in range(start_epoch, epochs):
        epoch_specs = tuple(spec for spec in schedule if int(spec.epoch) == epoch)
        if mode == "full":
            local_specs = tuple(
                spec
                for index, spec in enumerate(epoch_specs)
                if index % int(context["world_size"]) == int(context["rank"])
            )
            macros_per_update = 1
        else:
            local_specs, macros_per_update = epoch_specs, 1
        completed_updates = resume_update_in_epoch if epoch == start_epoch else 0
        completed_specs = completed_updates * macros_per_update
        if completed_specs > len(local_specs):
            raise ValueError("resume cursor exceeds the local epoch schedule")
        local_specs = local_specs[completed_specs:]
        dataset = ExactStoredTimeBatchDataset(
            base.data.source_h5,
            manifest,
            split="train",
            schedule=local_specs,
            query_points=1,
            seed=int(config["seed"]),
            time_policy=str(config["schedule"]["time_policy"]),
            frames_per_record=int(config["schedule"]["frames_per_record"]),
            travel_time_h5=config["travel_time_h5"],
        )
        iterator = iter(dataset)
        update_count = len(local_specs) // macros_per_update
        if int(smoke_updates) > 0:
            update_count = min(update_count, int(smoke_updates))
        factor = warmup_cosine_factor(
            epoch,
            total_epochs=max(3, epochs),
            warmup_epochs=int(optimizer_config["warmup_epochs"]),
            minimum_factor=float(optimizer_config["minimum_factor"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = float(optimizer_config["learning_rate"]) * factor
        model.train()
        for update in range(update_count):
            update_in_epoch = completed_updates + update + 1
            optimizer.zero_grad(set_to_none=True)
            component = {
                name: 0.0
                for name in (
                    "total",
                    "supervised_total",
                    "record_relative_l2",
                    "temporal_difference",
                    "spatial_gradient",
                    "spectrum",
                    "physics_residual",
                )
            }
            records_local = 0
            physics_triplets_local = 0
            physics_weight = 0.0
            if physics_config is not None:
                target_physics_weight = float(physics_config["weight"])
                physics_warmup = int(physics_config["warmup_updates"])
                physics_weight = target_physics_weight * (
                    1.0
                    if physics_warmup == 0
                    else min(1.0, float(global_step + 1) / float(physics_warmup))
                )
            for _ in range(macros_per_update):
                batch = next(iterator)
                macro_record_count = len(batch.sample_id)
                micros = split_pilot_batch(
                    batch, microbatch_records=int(microbatch_records)
                )
                for micro in micros:
                    micro_record_count = len(micro.sample_id)
                    prediction, target, indices_t = dense_training_pair(
                        model,
                        micro,
                        normalizer,
                        device,
                        time_block=int(execution["time_block"]),
                        query_chunk=int(execution["query_chunk"]),
                        hard_causality_lead_cycles=float(loss_config["hard_causality_lead_cycles"]),
                    )
                    loss = patch_deeponet_loss(
                        prediction,
                        target,
                        indices_t,
                        temporal_weight=float(loss_config["temporal_difference"]),
                        gradient_weight=float(loss_config["spatial_gradient"]),
                        spectrum_weight=float(loss_config["spectrum"]),
                    )
                    total_loss = loss.total
                    physics_loss = prediction.new_zeros(())
                    if physics_config is not None:
                        velocity = micro.velocity_mps.index_select(
                            0, micro.record_to_medium.long()
                        )
                        physics = pi_deeponet_lwc84_residual_loss(
                            prediction,
                            velocity,
                            micro.source_parameters,
                            micro.source_map,
                            micro.requested_time_s,
                            indices_t,
                            pressure_scale_pa=float(
                                normalizer.metadata.pressure_scale_pa
                            ),
                            dt_s=float(physics_config["dt_output_s"]),
                            dx_m=float(physics_config["dx_m"]),
                            dz_m=float(physics_config["dz_m"]),
                            maximum_triplets_per_record=int(
                                physics_config["maximum_triplets_per_record"]
                            ),
                            denominator_epsilon=float(
                                physics_config["denominator_epsilon"]
                            ),
                        )
                        physics_loss = physics.loss
                        physics_triplets_local += physics.triplet_count
                        total_loss = total_loss + physics_weight * physics_loss
                    (
                        total_loss
                        * (micro_record_count / macro_record_count)
                        / macros_per_update
                    ).backward()
                    for name, value in (
                        ("total", total_loss),
                        ("supervised_total", loss.total),
                        ("record_relative_l2", loss.record_relative_l2),
                        ("temporal_difference", loss.temporal_difference),
                        ("spatial_gradient", loss.spatial_gradient),
                        ("spectrum", loss.spectrum),
                        ("physics_residual", physics_loss),
                    ):
                        component[name] += float(value.detach()) * micro_record_count
                    records_local += micro_record_count
            _average_gradients(model, context)
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(optimizer_config["gradient_clip"])
                )
            )
            if not math.isfinite(gradient_norm):
                raise FloatingPointError("Patch-DeepONet gradient is non-finite")
            optimizer.step()
            global_step += 1
            if bool(context["main"]):
                row = {
                    "event": "optimizer_update",
                    "epoch": epoch + 1,
                    "update": update_in_epoch,
                    "global_step": global_step,
                    "records_per_rank": records_local,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm_before_clip": gradient_norm,
                    "loss_components": {name: value / records_local for name, value in component.items()},
                    "physics_weight": physics_weight,
                    "physics_triplets_per_rank": physics_triplets_local,
                    "elapsed_seconds": time.monotonic() - started,
                }
                with (root / "updates.jsonl").open("a", encoding="utf8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            if int(checkpoint_every_updates) > 0 and global_step % int(checkpoint_every_updates) == 0:
                _barrier(context)
                if bool(context["main"]):
                    rolling = {
                        "schema": CHECKPOINT_SCHEMA,
                        "checkpoint_kind": "rolling",
                        "manifest_digest": manifest.digest,
                        "selection_split": "train",
                        "run_digest": identity["run_digest"],
                        "epoch": epoch + 1,
                        "update_in_epoch": update_in_epoch,
                        "global_step": global_step,
                        "model_state": {
                            name: value.detach().cpu()
                            for name, value in model.state_dict().items()
                        },
                        "optimizer_state": optimizer.state_dict(),
                    }
                    _atomic_checkpoint(rolling, root / "latest.pt")
                    _atomic_json(
                        {
                            "checkpoint_kind": "rolling",
                            "epoch": epoch + 1,
                            "update_in_epoch": update_in_epoch,
                            "global_step": global_step,
                        },
                        root / "latest.json",
                    )
                _barrier(context)
        _barrier(context)
        if mode == "entry":
            evaluation_indices = selected
            frames = int(config["selection"]["frames_per_record"])
        else:
            evaluation_indices = validation_panel_indices(
                validation_records=base.data.expected_train_records,
                panel_records=int(config["selection"]["panel_records"]),
                epoch=1,
                seed=int(config["seed"]),
            )
            frames = int(config["selection"]["frames_per_record"])
        metrics = (
            _evaluate_train(
                model,
                base=base,
                manifest=manifest,
                normalizer=normalizer,
                config=config,
                indices=evaluation_indices,
                device=device,
                frames=frames,
                all_saved=False,
            )
            if bool(context["main"])
            else {}
        )
        score = _broadcast(
            float(metrics.get("aggregate_relative_l2", float("nan"))), device, context
        )
        if score < best_score:
            best_score = score
            if bool(context["main"]):
                checkpoint = {
                    "schema": CHECKPOINT_SCHEMA,
                    "manifest_digest": manifest.digest,
                    "selection_split": "train",
                    "run_digest": identity["run_digest"],
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "selection_metrics": metrics,
                    "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "optimizer_state": optimizer.state_dict(),
                }
                _atomic_checkpoint(checkpoint, root / "best.pt")
                _atomic_json({"score": score, "epoch": epoch + 1, "metrics": metrics}, root / "best.json")
    if mode == "entry" and int(smoke_updates) <= 0 and bool(context["main"]):
        full = _evaluate_train(
            model,
            base=base,
            manifest=manifest,
            normalizer=normalizer,
            config=config,
            indices=selected,
            device=device,
            frames=len(manifest.time_s),
            all_saved=True,
        )
        gate = config["entry_gate"]
        entry_physics_final = (
            _evaluate_train_physics(
                model,
                base=base,
                manifest=manifest,
                normalizer=normalizer,
                config=config,
                indices=selected,
                device=device,
            )
            if physics_config is not None
            else None
        )
        physics_ratio = None
        physics_passed = True
        if physics_config is not None:
            assert entry_physics_initial is not None and entry_physics_final is not None
            initial_value = float(
                entry_physics_initial["normalized_lwc84_defect_mse"]
            )
            final_value = float(entry_physics_final["normalized_lwc84_defect_mse"])
            physics_ratio = final_value / max(initial_value, 1.0e-8)
            physics_passed = (
                math.isfinite(physics_ratio)
                and physics_ratio
                <= float(gate["maximum_physics_final_to_initial_ratio"])
            )
        passed = (
            float(full["aggregate_relative_l2"]) <= float(gate["maximum_aggregate_relative_l2"])
            and all(float(full["family_relative_l2"][family]) <= float(gate["maximum_each_family_relative_l2"]) for family in ALLOWED_MEDIUM_TYPES)
            and all(math.isfinite(float(value)) for value in full["source_relative_l2"].values())
            and all(float(value) < 1.0 for value in full["source_relative_l2"].values())
            and physics_passed
        )
        _atomic_json(
            {
                "schema": ENTRY_SCHEMA,
                "status": "pass" if passed else "fail",
                "run_digest": identity["run_digest"],
                "config_sha256": audit["config_sha256"],
                "manifest_digest": manifest.digest,
                "selection_split": "train",
                "all_predictions_finite_and_noncollapsed": all(
                    math.isfinite(float(value)) and float(value) < 1.0
                    for value in full["source_relative_l2"].values()
                ),
                "metrics": full,
                "physics_initial": entry_physics_initial,
                "physics_final": entry_physics_final,
                "physics_final_to_initial_ratio": physics_ratio,
            },
            root / "entry_gate.json",
        )
    _barrier(context)
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("audit", "entry", "full"), default="audit")
    parser.add_argument("--entry-report", type=Path)
    parser.add_argument("--smoke-updates", type=int, default=0)
    parser.add_argument("--allow-failed-entry-for-comparison", action="store_true")
    parser.add_argument("--microbatch-records", type=int, default=1)
    parser.add_argument("--checkpoint-every-updates", type=int, default=0)
    parser.add_argument("--run-label")
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "audit":
        print(json.dumps(audit_config(args.config), indent=2, sort_keys=True))
        return 0
    return _train(
        args.config,
        mode=args.mode,
        entry_report=args.entry_report,
        smoke_updates=int(args.smoke_updates),
        allow_failed_entry_for_comparison=bool(args.allow_failed_entry_for_comparison),
        microbatch_records=int(args.microbatch_records),
        checkpoint_every_updates=int(args.checkpoint_every_updates),
        run_label=args.run_label,
        resume_checkpoint=args.resume_checkpoint,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_SCHEMA",
    "ENTRY_SCHEMA",
    "EVALUATION_ENERGY_FLOOR_FRACTION",
    "IDENTITY_SCHEMA",
    "_entry_indices",
    "_model_from_config",
    "_require_appearance_time_policy",
    "_require_full_authorization",
    "audit_config",
]
