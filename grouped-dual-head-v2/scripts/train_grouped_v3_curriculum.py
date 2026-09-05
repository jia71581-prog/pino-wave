#!/usr/bin/env python
"""Train guarded uniform-to-layered-to-Marmousi V3 curriculum stages."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.config import ALLOWED_MEDIUM_TYPES, V3Config
from grouped_ufno_mionet_v3.data.curriculum import (
    CurriculumStepDataset,
    build_curriculum_schedule,
    make_curriculum_loader,
)
from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts
from grouped_ufno_mionet_v3.training.checkpoint import save_checkpoint_atomic
from grouped_ufno_mionet_v3.training.curriculum import (
    accept_stage_checkpoint,
    curriculum_run_digest,
    validate_curriculum_parent,
)
from scripts.train_grouped_v3 import build_model
from scripts.train_grouped_v3_pilot import (
    _atomic_hardlink,
    _atomic_json,
    _prepare_validation_batches,
    _to_device,
    _validate_artifact_disk,
    _weights,
    combine_validation_metrics,
    evaluate_pilot_batch,
    load_normalizer,
    pilot_forward_loss,
)


def accumulate_curriculum_update(
    *,
    optimizer: torch.optim.Optimizer,
    parameters: Sequence[torch.nn.Parameter],
    required_gradient_groups: Mapping[str, Sequence[torch.nn.Parameter]],
    closures: Sequence[Callable[[], torch.Tensor]],
    record_counts: Sequence[int],
    loss_scales: Sequence[float],
    effective_records: int,
    gradient_clip: float,
) -> dict[str, object]:
    if not closures or not (
        len(closures) == len(record_counts) == len(loss_scales)
    ):
        raise ValueError("curriculum accumulation inputs have inconsistent lengths")
    if effective_records <= 0 or gradient_clip <= 0 or any(count <= 0 for count in record_counts):
        raise ValueError("curriculum effective records or gradient clip is invalid")
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    weighted_loss = 0.0
    for closure, record_count, loss_scale in zip(
        closures, record_counts, loss_scales, strict=True
    ):
        loss = closure()
        if loss.ndim or not bool(torch.isfinite(loss)):
            raise FloatingPointError("curriculum microbatch loss is not a finite scalar")
        weight = float(loss_scale) * int(record_count) / int(effective_records)
        (loss * weight).backward()
        losses.append(float(loss.detach()))
        weighted_loss += float(loss.detach()) * weight
    missing = [
        name
        for name, group in required_gradient_groups.items()
        if not any(parameter.grad is not None for parameter in group)
    ]
    if missing:
        raise RuntimeError(f"curriculum missing required gradient groups: {missing}")
    for parameter in parameters:
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError("curriculum gradient contains nonfinite values")
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, float(gradient_clip))
    if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
        raise FloatingPointError("curriculum gradient norm is nonfinite")
    optimizer.step()
    return {
        "microbatch_count": len(closures),
        "actual_record_count": int(sum(record_counts)),
        "effective_record_count": int(effective_records),
        "microbatch_losses": losses,
        "weighted_loss": weighted_loss,
        "gradient_norm": float(gradient_norm),
        "missing_gradient_groups": missing,
    }


def family_validation_scores(validation: Mapping[str, object]) -> dict[str, float]:
    scores = {family: 0.0 for family in ALLOWED_MEDIUM_TYPES}
    for mode in ("exact", "interpolated"):
        metrics = validation.get(mode)
        if not isinstance(metrics, Mapping):
            raise ValueError(f"curriculum validation is missing {mode} metrics")
        for key in ("family_dense_relative_l2", "family_query_relative_l2"):
            values = metrics.get(key)
            if not isinstance(values, Mapping) or set(values) != set(ALLOWED_MEDIUM_TYPES):
                raise ValueError(f"curriculum validation has invalid {mode} {key}")
            for family in ALLOWED_MEDIUM_TYPES:
                scores[family] += float(values[family])
    return scores


def select_stage_output(parent_checkpoint: str, accepted_best: str | None) -> str:
    return str(accepted_best) if accepted_best is not None else str(parent_checkpoint)


def rejection_recovery(
    *,
    stage_parent: str,
    accepted_checkpoint: str | None,
    learning_rate: float,
    factor: float,
    minimum: float,
) -> tuple[str, float]:
    if learning_rate <= 0 or minimum <= 0 or not 0.0 < factor < 1.0:
        raise ValueError("rejection recovery learning-rate settings are invalid")
    anchor = select_stage_output(stage_parent, accepted_checkpoint)
    return anchor, max(float(minimum), float(learning_rate) * float(factor))


def _load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_curriculum(path: Path) -> tuple[dict[str, object], str]:
    encoded = path.read_bytes()
    payload = yaml.safe_load(encoded)
    if not isinstance(payload, dict) or not isinstance(payload.get("stages"), list):
        raise ValueError("curriculum configuration is invalid")
    names = [stage.get("name") for stage in payload["stages"]]
    if names != list(ALLOWED_MEDIUM_TYPES):
        raise ValueError("curriculum stages must be uniform, layered, marmousi")
    return payload, hashlib.sha256(encoded).hexdigest()


def _stage_learned_families(stage: str) -> tuple[str, ...]:
    position = ALLOWED_MEDIUM_TYPES.index(stage)
    return tuple(ALLOWED_MEDIUM_TYPES[:position])


def _stage_reference_scores(
    parent_scores: Mapping[str, float],
    target_family: str,
    accepted_scores: Mapping[str, float] | None,
) -> dict[str, float]:
    result = {family: float(parent_scores[family]) for family in ALLOWED_MEDIUM_TYPES}
    if accepted_scores is not None:
        result[target_family] = float(accepted_scores[target_family])
    return result


def _load_model_state(model, checkpoint: Path, device: torch.device) -> None:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)


def run_curriculum(
    curriculum_path: Path,
    *,
    artifact_dir: Path,
    device_name: str,
    benchmark_steps: int = 0,
) -> dict[str, object]:
    curriculum, curriculum_config_digest = _load_curriculum(curriculum_path)
    base_path = (curriculum_path.parent.parent.parent / str(curriculum["base_config"])).resolve()
    if not base_path.is_file():
        base_path = Path(str(curriculum["base_config"])).resolve()
    base_config = V3Config.from_yaml(base_path)
    manifest = build_manifest(base_config.data.source_h5)
    validate_expected_counts(
        manifest,
        {
            "train": base_config.data.expected_train_records,
            "validation": base_config.data.expected_validation_records,
        },
    )
    parent_dir = Path(str(curriculum["parent_artifact_dir"])).resolve()
    terminal = _load_json(parent_dir / "terminal_report.json")
    best_validation = _load_json(parent_dir / "best_validation.json")
    parent_checkpoint = (parent_dir / "best.pt").resolve()
    parent = validate_curriculum_parent(
        terminal,
        best_validation,
        parent_checkpoint,
        expected_manifest_digest=manifest.digest,
        expected_run_digest=str(terminal["run_digest"]),
    )
    run_digest = curriculum_run_digest(curriculum_config_digest, parent)
    identity = {
        "schema": "grouped_v3_easy_to_hard_curriculum_v1",
        "manifest_digest": manifest.digest,
        "curriculum_config_digest": curriculum_config_digest,
        "run_digest": run_digest,
        "parent": parent.to_dict(),
        "benchmark_steps_per_stage": int(benchmark_steps),
    }
    _validate_artifact_disk(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    identity_path = artifact_dir / "run_identity.json"
    if identity_path.exists():
        if _load_json(identity_path) != identity:
            raise ValueError("existing curriculum artifact identity mismatch")
        if (artifact_dir / "metrics.jsonl").exists():
            raise RuntimeError("curriculum artifact already contains training metrics")
    else:
        _atomic_json(identity, identity_path)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA curriculum requested but unavailable")
    seed = int(curriculum["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    normalizer = load_normalizer(base_config, manifest.digest)
    model = build_model(base_config).to(device)
    _load_model_state(model, parent_checkpoint, device)
    validation_exact, validation_interpolated = _prepare_validation_batches(base_config, manifest)
    weights = _weights(base_config)
    current_checkpoint = parent_checkpoint
    current_validation = best_validation["validation"]
    if not isinstance(current_validation, Mapping):
        raise ValueError("baseline best validation metrics are missing")
    global_step = 0
    global_epoch = 0
    metrics_path = artifact_dir / "metrics.jsonl"
    stage_reports: list[dict[str, object]] = []
    started = time.monotonic()
    optimizer_values = curriculum["optimizer"]
    acceptance_values = curriculum["acceptance"]
    assert isinstance(optimizer_values, Mapping) and isinstance(acceptance_values, Mapping)
    for stage_index, stage_values in enumerate(curriculum["stages"]):
        assert isinstance(stage_values, Mapping)
        stage = str(stage_values["name"])
        configured_steps = int(stage_values["epochs"]) * int(stage_values["steps_per_epoch"])
        stage_steps = int(benchmark_steps) if benchmark_steps else configured_steps
        steps_per_epoch = stage_steps if benchmark_steps else int(stage_values["steps_per_epoch"])
        schedule = build_curriculum_schedule(
            manifest,
            split="train",
            stage=stage,
            optimizer_steps=stage_steps,
            seed=seed + stage_index * 1009,
        )
        dataset = CurriculumStepDataset(
            base_config.data.source_h5,
            manifest,
            split="train",
            schedule=schedule,
            continuous_fraction=float(curriculum["continuous_fraction"]),
            query_points=int(curriculum["query_points_per_record"]),
            seed=seed + stage_index * 1009,
        )
        loader = make_curriculum_loader(
            dataset,
            workers=int(curriculum["workers"]),
            prefetch_factor=int(curriculum["prefetch_factor"]),
            pin_memory=device.type == "cuda",
        )
        stage_dir = artifact_dir / stage
        checkpoint_dir = stage_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stage_parent_checkpoint = current_checkpoint
        stage_parent_validation = dict(current_validation)
        parent_family_scores = family_validation_scores(stage_parent_validation)
        parent_aggregate = float(stage_parent_validation["score"])
        accepted_checkpoint: Path | None = None
        accepted_validation: dict[str, object] | None = None
        accepted_scores: dict[str, float] | None = None
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_values["learning_rate"]),
            weight_decay=float(optimizer_values["weight_decay"]),
        )
        parameters = tuple(model.parameters())
        gradient_groups = model.required_gradient_groups()
        iterator = iter(loader)
        for local_step in range(1, stage_steps + 1):
            wait_started = time.monotonic()
            step_batch = next(iterator)
            data_wait = time.monotonic() - wait_started
            transfer_started = time.monotonic()
            device_batches = [_to_device(batch, device) for batch in step_batch.microbatches]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            transfer_seconds = time.monotonic() - transfer_started
            captured = []
            closures = []
            for tensors in device_batches:
                def closure(tensors=tensors):
                    result, _, _, _, _ = pilot_forward_loss(
                        model,
                        tensors,
                        normalizer,
                        weights,
                        phase_energy_fraction=base_config.loss.phase_energy_fraction,
                        relative_energy_floor_fraction=base_config.loss.relative_energy_floor_fraction,
                    )
                    captured.append(result)
                    return result.total
                closures.append(closure)
            compute_started = time.monotonic()
            update = accumulate_curriculum_update(
                optimizer=optimizer,
                parameters=parameters,
                required_gradient_groups=gradient_groups,
                closures=closures,
                record_counts=tuple(len(batch.sample_id) for batch in step_batch.microbatches),
                loss_scales=step_batch.loss_scale,
                effective_records=int(stage_values["effective_batch_records"]),
                gradient_clip=float(optimizer_values["gradient_clip"]),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            compute_seconds = time.monotonic() - compute_started
            global_step += 1
            family_counts = Counter(
                family for batch in step_batch.microbatches for family in batch.medium_type
            )
            replay_counts = Counter(
                family
                for batch, replay in zip(step_batch.microbatches, step_batch.replay, strict=True)
                if replay
                for family in batch.medium_type
            )
            row = {
                "event": "train_step",
                "stage": stage,
                "stage_step": local_step,
                "global_step": global_step,
                "weighted_loss": update["weighted_loss"],
                "microbatch_losses": update["microbatch_losses"],
                "actual_record_count": update["actual_record_count"],
                "effective_record_count": update["effective_record_count"],
                "family_counts": dict(family_counts),
                "replay_family_counts": dict(replay_counts),
                "timing_seconds": {
                    "data_wait": data_wait,
                    "host_to_device": transfer_seconds,
                    "forward_backward_update": compute_seconds,
                    "total": data_wait + transfer_seconds + compute_seconds,
                },
                "effective_records_per_second": int(stage_values["effective_batch_records"])
                / max(data_wait + transfer_seconds + compute_seconds, 1e-9),
                "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0,
                "elapsed_seconds": time.monotonic() - started,
            }
            with metrics_path.open("a", encoding="utf8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
            print(json.dumps(row, sort_keys=True), flush=True)
            if local_step % steps_per_epoch:
                continue
            global_epoch += 1
            validation = combine_validation_metrics(
                evaluate_pilot_batch(model, validation_exact, normalizer, device=device),
                evaluate_pilot_batch(model, validation_interpolated, normalizer, device=device),
            )
            candidate_scores = family_validation_scores(validation)
            decision = accept_stage_checkpoint(
                parent_family_scores=_stage_reference_scores(
                    parent_family_scores, stage, accepted_scores
                ),
                candidate_family_scores=candidate_scores,
                target_family=stage,
                learned_families=_stage_learned_families(stage),
                aggregate_parent=parent_aggregate,
                aggregate_candidate=float(validation["score"]),
                aggregate_tolerance=float(acceptance_values["aggregate_tolerance"]),
                replay_tolerance=float(acceptance_values["replay_tolerance"]),
            )
            checkpoint = save_checkpoint_atomic(
                checkpoint_dir / f"checkpoint_epoch_{global_epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=global_epoch,
                global_step=global_step,
                manifest_digest=manifest.digest,
                config_digest=run_digest,
                metrics={
                    "loss": float(update["weighted_loss"]),
                    "validation_score": float(validation["score"]),
                },
            )
            _atomic_hardlink(checkpoint, stage_dir / "latest.pt")
            validation_report = {
                "stage": stage,
                "epoch": global_epoch,
                "global_step": global_step,
                "checkpoint": str(checkpoint.resolve()),
                "validation": validation,
                "family_scores": candidate_scores,
                "decision": decision.to_dict(),
            }
            _atomic_json(validation_report, stage_dir / "validation_latest.json")
            if decision.accepted:
                accepted_checkpoint = checkpoint
                accepted_validation = validation_report
                accepted_scores = candidate_scores
                _atomic_hardlink(checkpoint, stage_dir / "best.pt")
                _atomic_json(validation_report, stage_dir / "best_validation.json")
            print(json.dumps({"event": "epoch", **validation_report}, sort_keys=True), flush=True)
            _validate_artifact_disk(artifact_dir)
        selected = Path(
            select_stage_output(
                str(stage_parent_checkpoint),
                None if accepted_checkpoint is None else str(accepted_checkpoint),
            )
        )
        _load_model_state(model, selected, device)
        current_checkpoint = selected
        if accepted_validation is not None:
            current_validation = accepted_validation["validation"]
        stage_report = {
            "stage": stage,
            "parent_checkpoint": str(stage_parent_checkpoint),
            "accepted_checkpoint": None if accepted_checkpoint is None else str(accepted_checkpoint),
            "selected_checkpoint": str(selected),
            "accepted": accepted_checkpoint is not None,
            "parent_validation_score": parent_aggregate,
            "selected_validation_score": float(current_validation["score"]),
        }
        _atomic_json(stage_report, stage_dir / "stage_report.json")
        stage_reports.append(stage_report)
    _atomic_hardlink(current_checkpoint, artifact_dir / "best.pt")
    terminal_report = {
        "status": "complete",
        "global_step": global_step,
        "epochs": global_epoch,
        "selected_checkpoint": str(current_checkpoint.resolve()),
        "selected_validation_score": float(current_validation["score"]),
        "run_digest": run_digest,
        "stages": stage_reports,
    }
    _atomic_json(terminal_report, artifact_dir / "terminal_report.json")
    print(json.dumps({"event": "terminal", **terminal_report}, sort_keys=True), flush=True)
    return terminal_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--benchmark-steps", type=int, default=0)
    args = parser.parse_args(argv)
    run_curriculum(
        Path(args.config).resolve(),
        artifact_dir=Path(args.artifact_dir).resolve(),
        device_name=args.device,
        benchmark_steps=int(args.benchmark_steps),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
