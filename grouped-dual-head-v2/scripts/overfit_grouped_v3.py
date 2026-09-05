#!/usr/bin/env python
"""Train and seal the strict one-record V3 phase-accuracy gate."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts
from grouped_ufno_mionet_v3.losses import V3LossWeights, compute_v3_losses, per_frame_relative_l2
from grouped_ufno_mionet_v3.normalization import PhysicalNormalizer
from grouped_ufno_mionet_v3.training.audit import (
    energy_ratio,
    optimal_amplitude_scale,
    radial_centroid_displacement_m,
)
from grouped_ufno_mionet_v3.training.checkpoint import CHECKPOINT_FORMAT, save_checkpoint_atomic
from grouped_ufno_mionet_v3.training.gates import OneRecordGateMetrics, evaluate_one_record_gate
from grouped_ufno_mionet_v3.training.trainer import GuardedV3Trainer, PlateauDetector
from scripts.train_grouped_v3 import build_model


@dataclass(frozen=True)
class GateRecord:
    velocity_mps: torch.Tensor
    source_parameters: torch.Tensor
    source_map: torch.Tensor
    dense_time_indices: torch.Tensor
    dense_target_physical: torch.Tensor
    query_coords: torch.Tensor
    query_target_physical: torch.Tensor
    sample_probability: torch.Tensor
    time_s: torch.Tensor
    x_m: torch.Tensor
    z_m: torch.Tensor
    sample_id: str
    group_id: str
    medium_type: str


def load_gate_record(config: V3Config, *, require_uniform: bool = True) -> GateRecord:
    sample_id = config.data.gate_sample_id
    if not sample_id:
        raise ValueError("data.gate_sample_id is required")
    with h5py.File(config.data.train_cache, "r", swmr=True) as cache:
        sample_ids = cache["sample_id"].asstr()[:]
        matches = np.flatnonzero(sample_ids == sample_id)
        if len(matches) != 1:
            raise ValueError(f"gate sample ID must match exactly one cache row: {sample_id}")
        row = int(matches[0])
        medium_type = cache["medium_type"].asstr()[row]
        if require_uniform and medium_type != "uniform":
            raise ValueError("one-record gate sample must be uniform and non-anomaly")
        if medium_type not in ("uniform", "layered", "marmousi"):
            raise ValueError("V3 gate sample uses a forbidden medium family")
        values = dict(
            velocity_mps=torch.from_numpy(np.asarray(cache["velocity_mps"][row], np.float32))[None],
            source_parameters=torch.from_numpy(np.asarray(cache["source_parameters"][row], np.float32)),
            source_map=torch.from_numpy(np.asarray(cache["source_map"][row], np.float32))[None],
            dense_time_indices=torch.from_numpy(np.asarray(cache["dense_time_indices"][row], np.int64)),
            dense_target_physical=torch.from_numpy(np.asarray(cache["dense_target"][row], np.float32)),
            query_coords=torch.from_numpy(np.asarray(cache["query_coords"][row], np.float32)),
            query_target_physical=torch.from_numpy(np.asarray(cache["query_target"][row], np.float32)),
            sample_probability=torch.from_numpy(np.asarray(cache["sample_probability"][row], np.float32)),
            time_s=torch.from_numpy(np.asarray(cache["time_s"][:], np.float32)),
            sample_id=sample_id,
            group_id=cache["group_id"].asstr()[row],
            medium_type=medium_type,
        )
    with h5py.File(config.data.source_h5, "r", swmr=True) as source:
        values["x_m"] = torch.from_numpy(np.asarray(source["x_m"][:], np.float32))
        values["z_m"] = torch.from_numpy(np.asarray(source["z_m"][:], np.float32))
    return GateRecord(**values)


def load_normalizer(config: V3Config, manifest_digest: str) -> PhysicalNormalizer:
    with Path(config.data.normalization_json).open(encoding="utf8") as handle:
        payload = json.load(handle)
    normalizer = PhysicalNormalizer.from_dict(payload, expected_manifest=manifest_digest)
    if normalizer.metadata.record_count != config.data.expected_train_records:
        raise ValueError("V3 normalization record count does not match the filtered train split")
    return normalizer


def _phase_positions(step: int) -> torch.Tensor:
    early = (0, 1, 2, 3)
    middle = (4, 5, 6, 7, 8, 9, 10, 11)
    late = (12, 13, 14, 15)
    return torch.tensor(
        [early[step % len(early)], middle[step % len(middle)], late[step % len(late)]],
        dtype=torch.long,
    )


def _adaptive_query_indices(
    record: GateRecord,
    dense_positions: torch.Tensor,
    priority: torch.Tensor,
    *,
    count: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_times = record.time_s[record.dense_time_indices[dense_positions]]
    matches = torch.zeros(len(record.query_coords), dtype=torch.bool)
    for current in selected_times:
        matches |= torch.isclose(record.query_coords[:, 2], current, rtol=0.0, atol=1.0e-7)
    candidates = torch.nonzero(matches, as_tuple=True)[0]
    if len(candidates) == 0:
        raise RuntimeError("selected dense frames have no matching query targets")
    candidate_priority = priority[candidates].clamp_min(0)
    weighted = candidate_priority + torch.finfo(torch.float32).eps
    weighted /= weighted.sum()
    probability = 0.2 / len(candidates) + 0.8 * weighted
    draws = min(int(count), len(candidates))
    local = torch.multinomial(probability, draws, replacement=False, generator=generator)
    return candidates[local], probability[local]


def _dense_at_query(
    dense_prediction: torch.Tensor,
    query_coords: torch.Tensor,
    dense_times: torch.Tensor,
    *,
    x_m: torch.Tensor,
    z_m: torch.Tensor,
) -> torch.Tensor:
    dx = float(x_m[1] - x_m[0])
    dz = float(z_m[1] - z_m[0])
    x_index = torch.round((query_coords[:, 0] - x_m[0]) / dx).long()
    z_index = torch.round((query_coords[:, 1] - z_m[0]) / dz).long()
    differences = (query_coords[:, 2, None] - dense_times[None]).abs()
    time_index = differences.argmin(dim=1)
    if differences.gather(1, time_index[:, None]).max() > 1.0e-6:
        raise RuntimeError("query time does not match the selected dense target frames")
    return dense_prediction[0, time_index, z_index, x_index][None]


@torch.no_grad()
def evaluate_model(
    model,
    record: GateRecord,
    normalizer: PhysicalNormalizer,
    *,
    device: torch.device,
) -> tuple[
    OneRecordGateMetrics,
    dict[str, object],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    model.eval()
    velocity = record.velocity_mps[None].to(device)
    source = record.source_parameters[None].to(device)
    source_map = record.source_map[None].to(device)
    times = record.time_s[record.dense_time_indices].to(device)
    prepared = model.prepare_sources(model.encode_medium(velocity, normalizer), source, source_map, normalizer)
    prediction_dense = model.dense_normalized(
        prepared,
        times,
        x_m=record.x_m.to(device),
        z_m=record.z_m.to(device),
        time_block=1,
    )
    target_dense = normalizer.encode_pressure(
        record.dense_target_physical[None].to(device), source[:, 4]
    )
    prediction_query = model.query_normalized(
        prepared,
        record.query_coords[None].to(device),
        chunk_size=2048,
    )
    target_query = normalizer.encode_pressure(
        record.query_target_physical[None].to(device), source[:, 4]
    )
    _, frame_error = per_frame_relative_l2(prediction_dense, target_dense)
    early = float(frame_error[:, :4].mean())
    middle = float(frame_error[:, 6:10].mean())
    late = float(frame_error[:, -4:].mean())
    query_error = float(
        (prediction_query - target_query).norm() / target_query.norm().clamp_min(1.0e-8)
    )
    displacement, valid = radial_centroid_displacement_m(
        prediction_dense[:, -4:],
        target_dense[:, -4:],
        source_xy_m=source[:, :2],
        x_m=record.x_m.to(device),
        z_m=record.z_m.to(device),
    )
    centroid = float(displacement[valid].mean()) if bool(valid.any()) else float("nan")
    metrics = OneRecordGateMetrics(
        sample_id=record.sample_id,
        medium_type=record.medium_type,
        early_relative_l2=early,
        middle_relative_l2=middle,
        late_relative_l2=late,
        query_relative_l2=query_error,
        radial_centroid_displacement_m=centroid,
        grid_spacing_m=float(record.x_m[1] - record.x_m[0]),
        zero_prediction_relative_l2=1.0,
        missing_gradient_groups=(),
    )
    diagnostics = {
        "frame_relative_l2": frame_error.cpu().tolist()[0],
        "energy_ratio": float(energy_ratio(prediction_dense, target_dense)),
        "optimal_amplitude_scale": float(optimal_amplitude_scale(prediction_dense, target_dense)),
        "valid_centroid_frames": int(valid.sum()),
    }
    query_abs_error = (prediction_query - target_query).abs().squeeze(0).cpu()
    return metrics, diagnostics, prediction_dense.cpu(), target_dense.cpu(), query_abs_error


def _atomic_json(payload: dict[str, object], path: Path) -> None:
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


def _atomic_torch(payload: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        with partial.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_one_record_gate(
    config: V3Config,
    *,
    device_name: str,
    artifact_dir: Path,
    initial_checkpoint: Path | None = None,
    learning_rate: float | None = None,
    max_steps: int | None = None,
) -> dict[str, object]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA gate requested but unavailable")
    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)
    manifest = build_manifest(config.data.source_h5)
    validate_expected_counts(
        manifest,
        {"train": config.data.expected_train_records, "validation": config.data.expected_validation_records},
    )
    normalizer = load_normalizer(config, manifest.digest)
    record = load_gate_record(config)
    model = build_model(config).to(device)
    parent_checkpoint_sha256 = ""
    if initial_checkpoint is not None:
        payload = torch.load(initial_checkpoint, map_location=device, weights_only=False)
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("initial checkpoint is not a V3 checkpoint")
        if payload.get("manifest_digest") != manifest.digest:
            raise ValueError("initial checkpoint manifest does not match the active V3 data")
        model.load_state_dict(payload["model_state"], strict=True)
        parent_checkpoint_sha256 = _file_sha256(initial_checkpoint)
    actual_learning_rate = (
        config.train.learning_rate if learning_rate is None else float(learning_rate)
    )
    actual_max_steps = config.train.max_steps if max_steps is None else int(max_steps)
    if actual_learning_rate <= 0 or actual_max_steps <= 0:
        raise ValueError("fine-tuning learning rate and max steps must be positive")
    run_identity = {
        "base_config_digest": config.digest(),
        "learning_rate": actual_learning_rate,
        "max_steps": actual_max_steps,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
    }
    run_digest = hashlib.sha256(
        json.dumps(run_identity, sort_keys=True, separators=(",", ":")).encode("utf8")
    ).hexdigest()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=actual_learning_rate, weight_decay=config.train.weight_decay
    )
    trainer = GuardedV3Trainer(
        model,
        optimizer,
        checkpoint_dir=artifact_dir / "checkpoints",
        manifest_digest=manifest.digest,
        config_digest=run_digest,
        gradient_clip=config.train.gradient_clip,
    )
    weights = V3LossWeights(
        point=config.loss.point,
        frame=config.loss.frame,
        complex_spectrum=config.loss.complex_spectrum,
        spectral_phase=config.loss.spectral_phase,
        spatial_gradient=config.loss.spatial_gradient,
        time_difference=config.loss.time_difference,
        consistency=config.loss.consistency,
    )
    priority = record.query_target_physical.abs().float()
    generator = torch.Generator().manual_seed(config.train.seed)
    plateau = PlateauDetector(
        patience=config.train.plateau_patience, min_delta=config.train.plateau_min_delta
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / "metrics.jsonl"
    best_score = math.inf
    best_report: dict[str, object] | None = None
    started = time.monotonic()

    for step in range(actual_max_steps):
        dense_positions = _phase_positions(step)
        query_indices, draw_probability = _adaptive_query_indices(
            record,
            dense_positions,
            priority,
            count=config.train.query_points_per_step,
            generator=generator,
        )
        captured: dict[str, object] = {}

        def closure() -> torch.Tensor:
            velocity = record.velocity_mps[None].to(device)
            source = record.source_parameters[None].to(device)
            source_map = record.source_map[None].to(device)
            dense_times = record.time_s[record.dense_time_indices[dense_positions]].to(device)
            prepared = model.prepare_sources(
                model.encode_medium(velocity, normalizer), source, source_map, normalizer
            )
            prediction_dense = model.dense_normalized(
                prepared,
                dense_times,
                x_m=record.x_m.to(device),
                z_m=record.z_m.to(device),
                time_block=1,
            )
            target_dense = normalizer.encode_pressure(
                record.dense_target_physical[dense_positions][None].to(device), source[:, 4]
            )
            coords = record.query_coords[query_indices][None].to(device)
            prediction_query = model.query_normalized(prepared, coords, chunk_size=2048)
            target_query = normalizer.encode_pressure(
                record.query_target_physical[query_indices][None].to(device), source[:, 4]
            )
            dense_at_query = _dense_at_query(
                prediction_dense,
                coords[0],
                dense_times,
                x_m=record.x_m.to(device),
                z_m=record.z_m.to(device),
            )
            result = compute_v3_losses(
                prediction_query=prediction_query,
                target_query=target_query,
                query_probability=draw_probability[None].to(device),
                prediction_dense=prediction_dense,
                target_dense=target_dense,
                dense_at_query=dense_at_query,
                weights=weights,
                phase_energy_fraction=config.loss.phase_energy_fraction,
            )
            captured["components"] = {
                name: float(value.detach()) for name, value in result.unweighted.items()
            }
            return result.total

        loss = trainer.train_step(closure)
        current_step = step + 1
        if current_step % config.train.steps_per_epoch == 0:
            trainer.save_epoch(current_step // config.train.steps_per_epoch, metrics={"loss": float(loss)})
        if current_step % config.train.evaluation_every != 0 and current_step != actual_max_steps:
            continue

        metrics, diagnostics, prediction, target, query_abs_error = evaluate_model(
            model, record, normalizer, device=device
        )
        decision = evaluate_one_record_gate(metrics)
        score = (
            metrics.early_relative_l2
            + metrics.middle_relative_l2
            + metrics.late_relative_l2
            + metrics.query_relative_l2
            + metrics.radial_centroid_displacement_m / metrics.grid_spacing_m
        )
        report = {
            "checkpoint_format": "phase_aligned_complex_fno_mionet_v3",
            "config_digest": run_digest,
            "run_identity": run_identity,
            "manifest_digest": manifest.digest,
            "step": current_step,
            "elapsed_seconds": time.monotonic() - started,
            "loss": float(loss),
            "metrics": metrics.to_dict(),
            "diagnostics": diagnostics,
            "decision": decision.to_dict(),
            "components": captured.get("components", {}),
        }
        with log_path.open("a", encoding="utf8") as handle:
            handle.write(json.dumps(report, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True), flush=True)
        if score < best_score:
            best_score = score
            best_report = report
            save_checkpoint_atomic(
                artifact_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=current_step // config.train.steps_per_epoch,
                global_step=current_step,
                manifest_digest=manifest.digest,
                config_digest=run_digest,
                metrics={"score": score, "query_relative_l2": metrics.query_relative_l2},
            )
            _atomic_json(report, artifact_dir / "best_report.json")
            _atomic_torch(
                {"prediction": prediction, "target": target},
                artifact_dir / "best_fields.pt",
            )
        priority = 0.5 * priority + 0.5 * query_abs_error
        if decision.passed:
            _atomic_json(report, artifact_dir / "passed_report.json")
            return report
        if plateau.update(score):
            report["stopped_reason"] = "adamw_plateau_requires_fresh_full_batch_lbfgs"
            _atomic_json(report, artifact_dir / "plateau_report.json")
            break

    if best_report is None:
        raise RuntimeError("one-record gate produced no evaluation report")
    best_report = dict(best_report)
    best_report["passed"] = False
    _atomic_json(best_report, artifact_dir / "failed_report.json")
    return best_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--artifact-dir", default="artifacts/grouped_ufno_mionet_v3/one_record_gate")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)
    config = V3Config.from_yaml(args.config)
    manifest = build_manifest(config.data.source_h5)
    validate_expected_counts(
        manifest,
        {"train": config.data.expected_train_records, "validation": config.data.expected_validation_records},
    )
    normalizer = load_normalizer(config, manifest.digest)
    record = load_gate_record(config)
    if args.dry_run:
        model = build_model(config)
        print(
            json.dumps(
                {
                    "sample_id": record.sample_id,
                    "medium_type": record.medium_type,
                    "dense_frames": len(record.dense_time_indices),
                    "query_points": len(record.query_coords),
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "manifest_digest": manifest.digest,
                    "normalization_manifest": normalizer.metadata.train_manifest_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    report = run_one_record_gate(
        config,
        device_name=args.device,
        artifact_dir=Path(args.artifact_dir),
        initial_checkpoint=None if args.init_checkpoint is None else Path(args.init_checkpoint),
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
    )
    return 0 if report.get("decision", {}).get("passed", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
