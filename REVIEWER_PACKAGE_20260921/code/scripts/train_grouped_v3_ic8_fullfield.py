#!/usr/bin/env python
"""IC-conditioned full-horizon V3 training (exploratory, non-sealed entry).

Task: predict the full wavefield evolution given the velocity model, the
source description, and the first ``ic_frames`` stored snapshots after source
onset.  Dense supervision samples ``dense_time_steps`` target frames per
record covering the post-IC horizon; inference materialises every stored
time step one snapshot at a time (``iter_dense_normalized(time_block=1)``).

This entry never touches the sealed pilot artifacts, uses only the train and
validation splits, and labels all outputs as exploratory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
from typing import Mapping

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts
from grouped_ufno_mionet_v3.data.pilot import (
    PilotBatch,
    PilotBatchDataset,
    build_pilot_schedule,
    make_pilot_loader,
)
from grouped_ufno_mionet_v3.losses import V3LossResult, V3LossWeights, compute_v3_losses
from grouped_ufno_mionet_v3.normalization import PhysicalNormalizer
from grouped_ufno_mionet_v3.training.audit import audit_required_gradients
from grouped_ufno_mionet_v3.training.trainer import GuardedV3Trainer
from scripts.train_grouped_v3 import build_model
from scripts.train_grouped_v3_pilot import (
    _atomic_hardlink,
    _atomic_json,
    _expected_batch_balance,
    dense_at_pilot_queries,
    load_normalizer,
    relative_metrics_by_family,
)


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
        "ic_snapshots_physical",
    )
    tensors = {}
    for name in names:
        value = getattr(batch, name)
        if value is None:
            raise ValueError(f"ic full-field batch is missing tensor {name}")
        tensors[name] = value.to(device, non_blocking=device.type == "cuda")
    return tensors


def ic_forward_loss(
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
        ic_snapshots_physical=tensors["ic_snapshots_physical"],
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
def evaluate_ic_batch(
    model,
    batch: PilotBatch,
    normalizer: PhysicalNormalizer,
    *,
    device: torch.device,
    micro_batch: int = 1,
) -> dict[str, object]:
    model.eval()
    tensors = _to_device(batch, device)
    dense_parts: list[torch.Tensor] = []
    query_parts: list[torch.Tensor] = []
    dense_target_parts: list[torch.Tensor] = []
    query_target_parts: list[torch.Tensor] = []
    medium_labels: list[str] = []
    for start in range(0, tensors["source_parameters"].shape[0], int(micro_batch)):
        micro = {
            name: value[start : start + int(micro_batch)]
            for name, value in tensors.items()
            if name not in ("x_m", "z_m", "velocity_mps")
        }
        # x_m/z_m are grid axes and velocity_mps is the shared medium group
        # encoding; neither is sliced per record.
        micro["x_m"] = tensors["x_m"]
        micro["z_m"] = tensors["z_m"]
        micro["velocity_mps"] = tensors["velocity_mps"]
        source = micro["source_parameters"]
        prepared = model.prepare_sources(
            model.encode_medium(micro["velocity_mps"], normalizer),
            source,
            micro["source_map"],
            normalizer,
            record_to_medium=micro["record_to_medium"],
            ic_snapshots_physical=micro["ic_snapshots_physical"],
        )
        prediction_dense = model.dense_normalized(
            prepared,
            micro["requested_time_s"],
            x_m=micro["x_m"],
            z_m=micro["z_m"],
            time_block=1,
        )
        prediction_query = model.query_normalized(
            prepared, micro["query_coords"], chunk_size=1024
        )
        target_dense = normalizer.encode_pressure(
            micro["dense_target_physical"], source[:, 4]
        )
        target_query = normalizer.encode_pressure(
            micro["query_target_physical"], source[:, 4]
        )
        dense_parts.append(prediction_dense)
        query_parts.append(prediction_query)
        dense_target_parts.append(target_dense)
        query_target_parts.append(target_query)
        medium_labels.extend(_micro_of(batch.medium_type, start, int(micro_batch)))
    prediction_dense = torch.cat(dense_parts, dim=0)
    prediction_query = torch.cat(query_parts, dim=0)
    target_dense = torch.cat(dense_target_parts, dim=0)
    target_query = torch.cat(query_target_parts, dim=0)
    return relative_metrics_by_family(
        prediction_dense,
        prediction_query,
        target_dense,
        target_query,
        medium_labels,
    )


def _dataset(config: V3Config, manifest, *, split: str, schedule, seed: int) -> PilotBatchDataset:
    return PilotBatchDataset(
        config.data.source_h5,
        manifest,
        split=split,
        schedule=schedule,
        continuous_fraction=config.data.continuous_fraction,
        query_points=config.train.query_points_per_step,
        seed=seed,
        dense_time_steps=config.data.dense_time_steps,
        active_horizon_s=config.data.active_horizon_s,
        ic_frames=config.data.ic_frames,
        time_sampling_policy=config.data.time_sampling_policy,
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


def _micro_of(values, start: int, count: int) -> list[str]:
    return list(values[start : start + count])


def run(
    config: V3Config,
    *,
    artifact_dir: Path,
    device_name: str,
    init_checkpoint: Path | None = None,
    max_steps_override: int | None = None,
) -> dict[str, object]:
    if config.data.ic_frames <= 0:
        raise ValueError("this entry requires data.ic_frames > 0")
    device = torch.device(device_name)
    manifest = build_manifest(config.data.source_h5)
    validate_expected_counts(
        manifest,
        {
            "train": config.data.expected_train_records,
            "validation": config.data.expected_validation_records,
        },
    )
    normalizer = load_normalizer(config, manifest.digest)
    max_steps = int(max_steps_override or config.train.max_steps)
    schedule = build_pilot_schedule(
        manifest,
        split="train",
        steps=max_steps,
        seed=config.train.seed,
        families=config.data.train_families,
    )
    train_dataset = _dataset(config, manifest, split="train", schedule=schedule, seed=config.train.seed)
    loader = make_pilot_loader(
        train_dataset,
        workers=config.train.workers,
        prefetch_factor=config.train.prefetch_factor,
        pin_memory=device.type == "cuda",
    )
    validation_schedule = build_pilot_schedule(
        manifest,
        split="validation",
        steps=1,
        seed=config.train.seed + 7919,
        families=config.data.train_families,
    )
    validation_batch = _dataset(
        config, manifest, split="validation", schedule=validation_schedule, seed=config.train.seed + 7919
    )[0]

    # Seed model initialization so a config's seed controls the network (not just
    # the data schedule); this is what makes multi-seed arms reproducible.
    torch.manual_seed(config.train.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.train.seed)
    model = build_model(config).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    loaded_from = ""
    if init_checkpoint is not None:
        payload = torch.load(init_checkpoint, map_location=device, weights_only=False)
        state = payload.get("model_state", payload)
        missing, unexpected = model.load_state_dict(state, strict=False)
        loaded_from = str(init_checkpoint)
        print(json.dumps({
            "event": "init_checkpoint",
            "path": loaded_from,
            "missing_keys": sorted(missing),
            "unexpected_keys": sorted(unexpected),
        }), flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    # GuardedV3Trainer owns checkpointing/digests; the training loop below
    # accumulates gradients across record micro-batches then clips and steps
    # once, so a full 12-record batch does not have to fit on one GPU at once.
    trainer = GuardedV3Trainer(
        model,
        optimizer,
        checkpoint_dir=Path(config.train.checkpoint_dir),
        manifest_digest=manifest.digest,
        config_digest=config.digest(),
        gradient_clip=config.train.gradient_clip,
    )
    weights = _weights(config)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        {
            "entry": "train_grouped_v3_ic8_fullfield",
            "status": "exploratory_non_sealed",
            "manifest_digest": manifest.digest,
            "config_digest": config.digest(),
            "parameter_count": parameter_count,
            "ic_frames": config.data.ic_frames,
            "dense_time_steps": config.data.dense_time_steps,
            "active_horizon_s": config.data.active_horizon_s,
            "init_checkpoint": loaded_from,
            "max_steps": max_steps,
        },
        artifact_dir / "run_config.json",
    )
    metrics_path = artifact_dir / "metrics.jsonl"
    best_path = artifact_dir / "best.pt"
    best_score = math.inf
    iterator = iter(loader)
    started = time.monotonic()
    last_report: dict[str, object] = {}
    for _ in range(max_steps):
        batch = next(iterator)
        if Counter(batch.medium_type) != _expected_batch_balance(config):
            raise RuntimeError("training batch lost its family balance")
        tensors = _to_device(batch, device)
        losses: list[float] = []
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_started = time.monotonic()
        effective_micro = min(int(config.train.micro_batch_records), tensors["source_parameters"].shape[0])
        for micro_start in range(0, tensors["source_parameters"].shape[0], effective_micro):
            micro = {
                name: value[micro_start : micro_start + effective_micro]
                for name, value in tensors.items()
                if name not in ("x_m", "z_m", "velocity_mps")
            }
            # x_m/z_m are the grid axes and velocity_mps is the shared medium
            # group encoding; neither is sliced per record.
            micro["x_m"] = tensors["x_m"]
            micro["z_m"] = tensors["z_m"]
            micro["velocity_mps"] = tensors["velocity_mps"]
            result, _, _, _, _ = ic_forward_loss(
                model,
                micro,
                normalizer,
                weights,
                phase_energy_fraction=config.loss.phase_energy_fraction,
                relative_energy_floor_fraction=config.loss.relative_energy_floor_fraction,
            )
            loss = result.total
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite micro-batch loss")
            loss.backward()
            losses.append(float(loss.detach()))
            components = {
                name: float(value.detach()) for name, value in result.unweighted.items()
            }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_seconds = time.monotonic() - step_started
        audit_required_gradients(model.required_gradient_groups())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
        if not torch.isfinite(torch.as_tensor(norm)):
            raise RuntimeError("nonfinite V3 gradient norm")
        optimizer.step()
        trainer.global_step += 1
        mean_loss = sum(losses) / len(losses)
        last_report = {
            "global_step": trainer.global_step,
            "loss": mean_loss,
            "microbatches": len(losses),
            "components": components,
            "step_seconds": step_seconds,
            "cuda_peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "elapsed_seconds": time.monotonic() - started,
        }
        with metrics_path.open("a", encoding="utf8") as handle:
            handle.write(json.dumps(last_report, sort_keys=True) + "\n")
        print(json.dumps({"event": "train_step", **last_report}, sort_keys=True), flush=True)
        if trainer.global_step % config.train.steps_per_epoch:
            continue
        epoch = trainer.global_step // config.train.steps_per_epoch
        validation = evaluate_ic_batch(
            model,
            validation_batch,
            normalizer,
            device=device,
            micro_batch=config.train.micro_batch_records,
        )
        score = float(validation["aggregate_dense_relative_l2"]) + float(
            validation["aggregate_query_relative_l2"]
        )
        checkpoint = trainer.save_epoch(
            epoch, metrics={"loss": mean_loss, "validation_score": score}
        )
        _atomic_hardlink(checkpoint, artifact_dir / "latest.pt")
        report = {
            "epoch": epoch,
            "global_step": trainer.global_step,
            "checkpoint": str(checkpoint.resolve()),
            "validation_score": score,
            "validation": validation,
        }
        _atomic_json(report, artifact_dir / "validation_latest.json")
        if score < best_score:
            best_score = score
            _atomic_hardlink(checkpoint, best_path)
            _atomic_json(report, artifact_dir / "best_validation.json")
        print(json.dumps({"event": "epoch", **report}, sort_keys=True), flush=True)
        model.train()
    terminal = {
        "status": "complete",
        "global_step": trainer.global_step,
        "best_validation_score": best_score,
        "last_train_report": last_report,
    }
    _atomic_json(terminal, artifact_dir / "terminal_report.json")
    print(json.dumps({"event": "terminal", **terminal}, sort_keys=True), flush=True)
    return terminal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args(argv)
    config = V3Config.from_yaml(args.config)
    run(
        config,
        artifact_dir=Path(args.artifact_dir),
        device_name=args.device,
        init_checkpoint=Path(args.init_checkpoint) if args.init_checkpoint else None,
        max_steps_override=args.max_steps or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
