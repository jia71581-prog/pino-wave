#!/usr/bin/env python
"""Sealed, resumable exact/interpolated full-data V3 pilot training."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from typing import Mapping, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.config import ALLOWED_MEDIUM_TYPES, V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts
from grouped_ufno_mionet_v3.data.pilot import (
    PilotBatch,
    PilotBatchDataset,
    build_pilot_schedule,
    make_pilot_loader,
)
from grouped_ufno_mionet_v3.losses import V3LossResult, V3LossWeights, compute_v3_losses
from grouped_ufno_mionet_v3.normalization import PhysicalNormalizer
from grouped_ufno_mionet_v3.training.checkpoint import (
    CHECKPOINT_FORMAT,
    load_checkpoint,
)
from grouped_ufno_mionet_v3.training.pilot import PilotIdentity, validate_pilot_prerequisite
from grouped_ufno_mionet_v3.training.trainer import GuardedV3Trainer
from scripts.train_grouped_v3 import build_model


def _atomic_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        with partial.open("x", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _atomic_hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial.{os.getpid()}")
    try:
        partial.unlink(missing_ok=True)
        os.link(source, partial)
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def pilot_run_digest(config_digest: str, identity: PilotIdentity) -> str:
    payload = {"pilot_config_digest": str(config_digest), **identity.to_dict()}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf8")
    return hashlib.sha256(encoded).hexdigest()


def dense_at_pilot_queries(
    dense_prediction: torch.Tensor,
    query_coords: torch.Tensor,
    requested_time_s: torch.Tensor,
    *,
    x_m: torch.Tensor,
    z_m: torch.Tensor,
) -> torch.Tensor:
    if dense_prediction.ndim != 4 or query_coords.ndim != 3 or requested_time_s.ndim != 2:
        raise ValueError("pilot dense/query/time tensors have invalid ranks")
    records, times, nz, nx = dense_prediction.shape
    if query_coords.shape[0] != records or requested_time_s.shape != (records, times):
        raise ValueError("pilot dense/query/time record shapes disagree")
    dx = float(x_m[1] - x_m[0])
    dz = float(z_m[1] - z_m[0])
    x_index = torch.round((query_coords[..., 0] - x_m[0]) / dx).long()
    z_index = torch.round((query_coords[..., 1] - z_m[0]) / dz).long()
    if torch.any(x_index < 0) or torch.any(x_index >= nx) or torch.any(z_index < 0) or torch.any(z_index >= nz):
        raise RuntimeError("pilot query coordinate lies outside the dense grid")
    difference = (query_coords[..., 2, None] - requested_time_s[:, None, :]).abs()
    time_index = difference.argmin(dim=-1)
    if difference.gather(-1, time_index[..., None]).max() > 1.0e-6:
        raise RuntimeError("pilot query time does not match a dense target time")
    flat_index = time_index * (nz * nx) + z_index * nx + x_index
    return dense_prediction.reshape(records, -1).gather(1, flat_index)


def _per_record_relative(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape[0] == 0:
        raise ValueError("pilot validation prediction and target shapes must match")
    numerator = (prediction.float() - target.float()).flatten(start_dim=1).norm(dim=-1)
    denominator = target.float().flatten(start_dim=1).norm(dim=-1).clamp_min(1.0e-8)
    return numerator / denominator


def relative_metrics_by_family(
    prediction_dense: torch.Tensor,
    prediction_query: torch.Tensor,
    target_dense: torch.Tensor,
    target_query: torch.Tensor,
    families: Sequence[str],
    *,
    dense_at_query: torch.Tensor | None = None,
) -> dict[str, object]:
    if len(families) != prediction_dense.shape[0] or set(families) - set(ALLOWED_MEDIUM_TYPES):
        raise ValueError("pilot validation family labels are invalid")
    dense = _per_record_relative(prediction_dense, target_dense)
    query = _per_record_relative(prediction_query, target_query)
    late = _per_record_relative(prediction_dense[:, -1:], target_dense[:, -1:])
    family_dense: dict[str, float] = {}
    family_query: dict[str, float] = {}
    family_late: dict[str, float] = {}
    counts: dict[str, int] = {}
    for family in ALLOWED_MEDIUM_TYPES:
        indices = torch.tensor([value == family for value in families], device=dense.device)
        if not bool(indices.any()):
            raise ValueError(f"pilot validation has no {family} records")
        counts[family] = int(indices.sum())
        family_dense[family] = float(dense[indices].mean())
        family_query[family] = float(query[indices].mean())
        family_late[family] = float(late[indices].mean())
    result = {
        "record_count_by_family": counts,
        "aggregate_dense_relative_l2": float(dense.mean()),
        "aggregate_query_relative_l2": float(query.mean()),
        "aggregate_late_relative_l2": float(late.mean()),
        "family_dense_relative_l2": family_dense,
        "family_query_relative_l2": family_query,
        "family_late_relative_l2": family_late,
    }
    if dense_at_query is not None:
        if dense_at_query.shape != prediction_query.shape:
            raise ValueError("pilot validation dual-head query shapes must match")
        numerator = (dense_at_query.float() - prediction_query.float()).flatten(start_dim=1).norm(dim=-1)
        denominator = target_query.float().flatten(start_dim=1).norm(dim=-1).clamp_min(1.0e-8)
        consistency = numerator / denominator
        family_consistency: dict[str, float] = {}
        for family in ALLOWED_MEDIUM_TYPES:
            indices = torch.tensor(
                [value == family for value in families], device=consistency.device
            )
            family_consistency[family] = float(consistency[indices].mean())
        result["aggregate_head_consistency_relative_l2"] = float(consistency.mean())
        result["family_head_consistency_relative_l2"] = family_consistency
    return result


def combine_validation_metrics(
    exact: Mapping[str, object], interpolated: Mapping[str, object]
) -> dict[str, object]:
    score = sum(
        float(values[name])
        for values in (exact, interpolated)
        for name in ("aggregate_query_relative_l2", "aggregate_dense_relative_l2")
    )
    return {"exact": dict(exact), "interpolated": dict(interpolated), "score": score}


def load_normalizer(config: V3Config, manifest_digest: str) -> PhysicalNormalizer:
    with Path(config.data.normalization_json).open(encoding="utf8") as handle:
        normalizer = PhysicalNormalizer.from_dict(
            json.load(handle), expected_manifest=manifest_digest
        )
    if normalizer.metadata.record_count != config.data.expected_train_records:
        raise ValueError("pilot normalization record count does not match the train census")
    return normalizer


def _to_device(batch: PilotBatch, device: torch.device) -> dict[str, torch.Tensor]:
    names = (
        "velocity_mps",
        "record_to_medium",
        "source_parameters",
        "source_map",
        "requested_time_s",
        "dense_target_physical",
        "query_coords",
        "query_target_physical",
        "query_probability",
        "x_m",
        "z_m",
    )
    return {
        name: getattr(batch, name).to(device, non_blocking=device.type == "cuda")
        for name in names
    }


def pilot_forward_loss(
    model,
    tensors: Mapping[str, torch.Tensor],
    normalizer: PhysicalNormalizer,
    weights: V3LossWeights,
    *,
    phase_energy_fraction: float,
    relative_energy_floor_fraction: float = 0.0,
) -> tuple[V3LossResult, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source = tensors["source_parameters"]
    prepared = model.prepare_sources(
        model.encode_medium(tensors["velocity_mps"], normalizer),
        source,
        tensors["source_map"],
        normalizer,
        record_to_medium=tensors["record_to_medium"],
    )
    prediction_dense = model.dense_normalized(
        prepared,
        tensors["requested_time_s"],
        x_m=tensors["x_m"],
        z_m=tensors["z_m"],
        time_block=1,
    )
    prediction_query = model.query_normalized(
        prepared, tensors["query_coords"], chunk_size=1024
    )
    target_dense = normalizer.encode_pressure(
        tensors["dense_target_physical"], source[:, 4]
    )
    target_query = normalizer.encode_pressure(
        tensors["query_target_physical"], source[:, 4]
    )
    dense_at_query = dense_at_pilot_queries(
        prediction_dense,
        tensors["query_coords"],
        tensors["requested_time_s"],
        x_m=tensors["x_m"],
        z_m=tensors["z_m"],
    )
    result = compute_v3_losses(
        prediction_query=prediction_query,
        target_query=target_query,
        query_probability=tensors["query_probability"],
        prediction_dense=prediction_dense,
        target_dense=target_dense,
        dense_at_query=dense_at_query,
        weights=weights,
        phase_energy_fraction=phase_energy_fraction,
        relative_energy_floor_fraction=relative_energy_floor_fraction,
    )
    return result, prediction_dense, prediction_query, target_dense, target_query


@torch.no_grad()
def evaluate_pilot_batch(
    model,
    batch: PilotBatch,
    normalizer: PhysicalNormalizer,
    *,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    tensors = _to_device(batch, device)
    source = tensors["source_parameters"]
    prepared = model.prepare_sources(
        model.encode_medium(tensors["velocity_mps"], normalizer),
        source,
        tensors["source_map"],
        normalizer,
        record_to_medium=tensors["record_to_medium"],
    )
    prediction_dense = model.dense_normalized(
        prepared,
        tensors["requested_time_s"],
        x_m=tensors["x_m"],
        z_m=tensors["z_m"],
        time_block=1,
    )
    prediction_query = model.query_normalized(
        prepared, tensors["query_coords"], chunk_size=1024
    )
    target_dense = normalizer.encode_pressure(
        tensors["dense_target_physical"], source[:, 4]
    )
    target_query = normalizer.encode_pressure(
        tensors["query_target_physical"], source[:, 4]
    )
    dense_at_query = dense_at_pilot_queries(
        prediction_dense,
        tensors["query_coords"],
        tensors["requested_time_s"],
        x_m=tensors["x_m"],
        z_m=tensors["z_m"],
    )
    return relative_metrics_by_family(
        prediction_dense,
        prediction_query,
        target_dense,
        target_query,
        batch.medium_type,
        dense_at_query=dense_at_query,
    )


def _weights(config: V3Config) -> V3LossWeights:
    return V3LossWeights(
        point=config.loss.point,
        frame=config.loss.frame,
        complex_spectrum=config.loss.complex_spectrum,
        spectral_phase=config.loss.spectral_phase,
        spatial_gradient=config.loss.spatial_gradient,
        time_difference=config.loss.time_difference,
        consistency=config.loss.consistency,
    )


def _prepare_validation_batches(config: V3Config, manifest) -> tuple[PilotBatch, PilotBatch]:
    schedule = build_pilot_schedule(
        manifest, split="validation", steps=1, seed=config.train.seed + 7919
    )
    common = dict(
        source_h5=config.data.source_h5,
        manifest=manifest,
        split="validation",
        schedule=schedule,
        query_points=config.train.query_points_per_step,
        seed=config.train.seed + 7919,
    )
    exact = PilotBatchDataset(continuous_fraction=0.0, **common)[0]
    interpolated = PilotBatchDataset(continuous_fraction=1.0, **common)[0]
    return exact, interpolated


def _validate_artifact_disk(path: Path) -> None:
    existing = path
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if free < 20 * 1024**3:
        raise RuntimeError(f"pilot artifact disk has less than 20 GiB free: {existing}")


def run_pilot(
    config: V3Config,
    *,
    prerequisite_report: Path,
    initial_checkpoint: Path,
    artifact_dir: Path,
    device_name: str,
    resume: Path | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    if config.train.batch_records != 12:
        raise ValueError("the V3 pilot grouped schedule requires batch_records=12")
    if config.train.max_steps != config.train.epochs * config.train.steps_per_epoch:
        raise ValueError("pilot max_steps must equal epochs * steps_per_epoch")
    manifest = build_manifest(config.data.source_h5)
    validate_expected_counts(
        manifest,
        {
            "train": config.data.expected_train_records,
            "validation": config.data.expected_validation_records,
        },
    )
    identity = validate_pilot_prerequisite(
        prerequisite_report,
        initial_checkpoint,
        expected_manifest_digest=manifest.digest,
    )
    config_digest = config.digest()
    run_digest = pilot_run_digest(config_digest, identity)
    schedule = build_pilot_schedule(
        manifest,
        split="train",
        steps=config.train.max_steps,
        seed=config.train.seed,
    )
    summary = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "manifest_digest": manifest.digest,
        "pilot_config_digest": config_digest,
        "run_digest": run_digest,
        "prerequisite": identity.to_dict(),
        "train_records": manifest.counts_after["train"],
        "validation_records": manifest.counts_after["validation"],
        "batch_records": config.train.batch_records,
        "batch_medium_count": 6,
        "continuous_fraction": config.data.continuous_fraction,
        "steps": len(schedule),
        "epochs": config.train.epochs,
    }
    if dry_run:
        summary["parameter_count"] = sum(
            parameter.numel() for parameter in build_model(config).parameters()
        )
        print(json.dumps(summary, sort_keys=True))
        return summary

    _validate_artifact_disk(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    identity_path = artifact_dir / "run_identity.json"
    if identity_path.exists():
        with identity_path.open(encoding="utf8") as handle:
            existing_identity = json.load(handle)
        if existing_identity != summary:
            raise ValueError("existing pilot artifact directory has a different run identity")
        if resume is None and (artifact_dir / "metrics.jsonl").exists():
            raise RuntimeError("pilot artifact directory already contains training metrics; use --resume")
    else:
        _atomic_json(summary, identity_path)

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA pilot requested but unavailable")
    random.seed(config.train.seed)
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.train.seed)
    normalizer = load_normalizer(config, manifest.digest)
    model = build_model(config).to(device)
    parent = torch.load(initial_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(parent["model_state"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    trainer = GuardedV3Trainer(
        model,
        optimizer,
        checkpoint_dir=artifact_dir / "checkpoints",
        manifest_digest=manifest.digest,
        config_digest=run_digest,
        gradient_clip=config.train.gradient_clip,
    )
    start_step = 0
    if resume is not None:
        metadata = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            expected_manifest_digest=manifest.digest,
            expected_config_digest=run_digest,
            restore_rng=True,
            map_location=device,
        )
        start_step = metadata.global_step
        trainer.global_step = start_step
        if start_step % config.train.steps_per_epoch:
            raise ValueError("pilot resume checkpoint is not on an epoch boundary")
    if start_step >= config.train.max_steps:
        raise ValueError("pilot resume checkpoint is already at or beyond max_steps")

    train_dataset = PilotBatchDataset(
        config.data.source_h5,
        manifest,
        split="train",
        schedule=schedule[start_step:],
        continuous_fraction=config.data.continuous_fraction,
        query_points=config.train.query_points_per_step,
        seed=config.train.seed,
    )
    loader = make_pilot_loader(
        train_dataset,
        workers=config.train.workers,
        prefetch_factor=config.train.prefetch_factor,
        pin_memory=device.type == "cuda",
    )
    validation_exact, validation_interpolated = _prepare_validation_batches(config, manifest)
    weights = _weights(config)
    metrics_path = artifact_dir / "metrics.jsonl"
    best_path = artifact_dir / "best.pt"
    best_score = math.inf
    best_validation_path = artifact_dir / "best_validation.json"
    if best_validation_path.exists():
        with best_validation_path.open(encoding="utf8") as handle:
            best_score = float(json.load(handle)["validation"]["score"])
    iterator = iter(loader)
    started = time.monotonic()
    last_report: dict[str, object] = {}
    for _ in range(start_step, config.train.max_steps):
        wait_started = time.monotonic()
        batch = next(iterator)
        data_wait = time.monotonic() - wait_started
        if Counter(batch.medium_type) != {family: 4 for family in ALLOWED_MEDIUM_TYPES}:
            raise RuntimeError("pilot training batch lost its family balance")
        if not bool((~batch.target_exact).any()):
            raise RuntimeError("pilot training batch contains no interpolated targets")
        transfer_started = time.monotonic()
        tensors = _to_device(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        transfer_seconds = time.monotonic() - transfer_started
        captured: dict[str, object] = {}

        def closure() -> torch.Tensor:
            result, _, _, _, _ = pilot_forward_loss(
                model,
                tensors,
                normalizer,
                weights,
                phase_energy_fraction=config.loss.phase_energy_fraction,
                relative_energy_floor_fraction=config.loss.relative_energy_floor_fraction,
            )
            captured["result"] = result
            return result.total

        compute_started = time.monotonic()
        loss = trainer.train_step(closure)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_seconds = time.monotonic() - compute_started
        result = captured["result"]
        assert isinstance(result, V3LossResult)
        global_step = trainer.global_step
        epoch = global_step // config.train.steps_per_epoch
        last_report = {
            "global_step": global_step,
            "epoch": epoch,
            "loss": float(loss),
            "components": {
                name: float(value.detach()) for name, value in result.unweighted.items()
            },
            "family_counts": dict(Counter(batch.medium_type)),
            "exact_target_fraction": float(batch.target_exact.float().mean()),
            "interpolated_target_fraction": float((~batch.target_exact).float().mean()),
            "timing_seconds": {
                "data_wait": data_wait,
                "host_to_device": transfer_seconds,
                "forward_backward_update": compute_seconds,
                "total": data_wait + transfer_seconds + compute_seconds,
            },
            "records_per_second": config.train.batch_records
            / max(data_wait + transfer_seconds + compute_seconds, 1.0e-9),
            "cuda_peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "elapsed_seconds": time.monotonic() - started,
        }
        with metrics_path.open("a", encoding="utf8") as handle:
            handle.write(json.dumps(last_report, sort_keys=True) + "\n")
            handle.flush()
        print(json.dumps({"event": "train_step", **last_report}, sort_keys=True), flush=True)
        if global_step % config.train.steps_per_epoch:
            continue
        validation = combine_validation_metrics(
            evaluate_pilot_batch(
                model, validation_exact, normalizer, device=device
            ),
            evaluate_pilot_batch(
                model, validation_interpolated, normalizer, device=device
            ),
        )
        checkpoint = trainer.save_epoch(
            epoch,
            metrics={
                "loss": float(loss),
                "validation_score": float(validation["score"]),
            },
        )
        _atomic_hardlink(checkpoint, artifact_dir / "latest.pt")
        validation_report = {
            "epoch": epoch,
            "global_step": global_step,
            "checkpoint": str(checkpoint.resolve()),
            "run_digest": run_digest,
            "manifest_digest": manifest.digest,
            "validation": validation,
        }
        _atomic_json(validation_report, artifact_dir / "validation_latest.json")
        if float(validation["score"]) < best_score:
            best_score = float(validation["score"])
            _atomic_hardlink(checkpoint, best_path)
            _atomic_json(validation_report, best_validation_path)
        print(json.dumps({"event": "epoch", **validation_report}, sort_keys=True), flush=True)
        _validate_artifact_disk(artifact_dir)
    terminal = {
        "status": "complete",
        "global_step": trainer.global_step,
        "epochs": trainer.global_step // config.train.steps_per_epoch,
        "best_validation_score": best_score,
        "last_train_report": last_report,
        "run_digest": run_digest,
    }
    _atomic_json(terminal, artifact_dir / "terminal_report.json")
    print(json.dumps({"event": "terminal", **terminal}, sort_keys=True), flush=True)
    return terminal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prerequisite-report", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = V3Config.from_yaml(args.config)
    artifact_dir = Path(args.artifact_dir)
    try:
        run_pilot(
            config,
            prerequisite_report=Path(args.prerequisite_report),
            initial_checkpoint=Path(args.init_checkpoint),
            artifact_dir=artifact_dir,
            device_name=args.device,
            resume=Path(args.resume) if args.resume else None,
            dry_run=args.dry_run,
        )
    except Exception as error:
        if not args.dry_run:
            _atomic_json(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
                artifact_dir / "terminal_report.json",
            )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
