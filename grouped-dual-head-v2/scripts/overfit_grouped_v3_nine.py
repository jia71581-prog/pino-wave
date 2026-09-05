#!/usr/bin/env python
"""Balanced three-family, nine-record V3 accuracy gate."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.config import ALLOWED_MEDIUM_TYPES, V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts
from grouped_ufno_mionet_v3.losses import V3LossWeights, compute_v3_losses
from grouped_ufno_mionet_v3.training.checkpoint import CHECKPOINT_FORMAT, save_checkpoint_atomic
from grouped_ufno_mionet_v3.training.gates import NineRecordGateMetrics, evaluate_nine_record_gate
from grouped_ufno_mionet_v3.training.trainer import GuardedV3Trainer, PlateauDetector
from scripts.overfit_grouped_v3 import (
    GateRecord,
    _adaptive_query_indices,
    _atomic_json,
    _dense_at_query,
    _file_sha256,
    _phase_positions,
    evaluate_model,
    load_gate_record,
    load_normalizer,
)
from scripts.train_grouped_v3 import build_model


def load_balanced_records(config: V3Config) -> tuple[GateRecord, ...]:
    if len(config.data.gate_sample_ids) != 9:
        raise ValueError("nine-record gate requires exactly nine pinned sample IDs")
    records = []
    for sample_id in config.data.gate_sample_ids:
        current = replace(config, data=replace(config.data, gate_sample_id=sample_id))
        records.append(load_gate_record(current, require_uniform=False))
    counts = {family: sum(record.medium_type == family for record in records) for family in ALLOWED_MEDIUM_TYPES}
    if counts != {"uniform": 3, "layered": 3, "marmousi": 3}:
        raise ValueError(f"nine-record family balance is invalid: {counts}")
    if len({record.sample_id for record in records}) != 9:
        raise ValueError("nine-record sample IDs must be unique")
    return tuple(records)


def validate_one_record_prerequisite(path: Path) -> dict[str, object]:
    with path.open(encoding="utf8") as handle:
        report = json.load(handle)
    if not report.get("decision", {}).get("passed", False):
        raise RuntimeError("nine-record gate is blocked by a failed one-record prerequisite")
    return report


@torch.no_grad()
def evaluate_nine(
    model,
    records: tuple[GateRecord, ...],
    normalizer,
    *,
    device: torch.device,
) -> tuple[NineRecordGateMetrics, dict[str, object], dict[str, torch.Tensor]]:
    record_metrics: list[dict[str, object]] = []
    priority_updates: dict[str, torch.Tensor] = {}
    for record in records:
        one, diagnostics, _, _, query_error = evaluate_model(
            model, record, normalizer, device=device
        )
        dense = float(np.mean(diagnostics["frame_relative_l2"]))
        late = float(np.mean(diagnostics["frame_relative_l2"][-4:]))
        record_metrics.append(
            {
                "sample_id": record.sample_id,
                "medium_type": record.medium_type,
                "query_relative_l2": one.query_relative_l2,
                "dense_relative_l2": dense,
                "late_relative_l2": late,
                "centroid_m": one.radial_centroid_displacement_m,
            }
        )
        priority_updates[record.sample_id] = query_error
    family_query: dict[str, float] = {}
    family_dense: dict[str, float] = {}
    family_late: dict[str, float] = {}
    counts: dict[str, int] = {}
    for family in ALLOWED_MEDIUM_TYPES:
        selected = [item for item in record_metrics if item["medium_type"] == family]
        counts[family] = len(selected)
        family_query[family] = float(np.mean([item["query_relative_l2"] for item in selected]))
        family_dense[family] = float(np.mean([item["dense_relative_l2"] for item in selected]))
        family_late[family] = float(np.mean([item["late_relative_l2"] for item in selected]))
    metrics = NineRecordGateMetrics(
        record_count_by_family=counts,
        aggregate_query_relative_l2=float(np.mean([item["query_relative_l2"] for item in record_metrics])),
        aggregate_dense_relative_l2=float(np.mean([item["dense_relative_l2"] for item in record_metrics])),
        family_query_relative_l2=family_query,
        family_dense_relative_l2=family_dense,
        family_late_relative_l2=family_late,
        zero_prediction_relative_l2=1.0,
        missing_gradient_groups=(),
    )
    return metrics, {"records": record_metrics}, priority_updates


def run_nine_record_gate(
    config: V3Config,
    *,
    device_name: str,
    artifact_dir: Path,
    initial_checkpoint: Path,
    prerequisite_report: Path,
) -> dict[str, object]:
    prerequisite = validate_one_record_prerequisite(prerequisite_report)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA nine-record gate requested but unavailable")
    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)
    manifest = build_manifest(config.data.source_h5)
    validate_expected_counts(
        manifest,
        {"train": config.data.expected_train_records, "validation": config.data.expected_validation_records},
    )
    normalizer = load_normalizer(config, manifest.digest)
    records = load_balanced_records(config)
    model = build_model(config).to(device)
    parent = torch.load(initial_checkpoint, map_location=device, weights_only=False)
    if parent.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("nine-record initialization checkpoint is not V3")
    if parent.get("manifest_digest") != manifest.digest:
        raise ValueError("nine-record initialization manifest mismatch")
    model.load_state_dict(parent["model_state"], strict=True)
    run_identity = {
        "base_config_digest": config.digest(),
        "parent_checkpoint_sha256": _file_sha256(initial_checkpoint),
        "prerequisite_report_sha256": _file_sha256(prerequisite_report),
    }
    run_digest = hashlib.sha256(
        json.dumps(run_identity, sort_keys=True, separators=(",", ":")).encode("utf8")
    ).hexdigest()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay
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
    by_family = {
        family: tuple(record for record in records if record.medium_type == family)
        for family in ALLOWED_MEDIUM_TYPES
    }
    priorities = {record.sample_id: record.query_target_physical.abs().float() for record in records}
    generator = torch.Generator().manual_seed(config.train.seed)
    plateau = PlateauDetector(
        patience=config.train.plateau_patience, min_delta=config.train.plateau_min_delta
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / "metrics.jsonl"
    best_score = math.inf
    best_report: dict[str, object] | None = None
    started = time.monotonic()

    for step in range(config.train.max_steps):
        records_step = tuple(by_family[family][step % 3] for family in ALLOWED_MEDIUM_TYPES)
        dense_positions = _phase_positions(step)
        query_indices = []
        query_probability = []
        for record in records_step:
            indices, probability = _adaptive_query_indices(
                record,
                dense_positions,
                priorities[record.sample_id],
                count=config.train.query_points_per_step,
                generator=generator,
            )
            query_indices.append(indices)
            query_probability.append(probability)
        captured: dict[str, object] = {}

        def closure() -> torch.Tensor:
            velocity = torch.stack([record.velocity_mps for record in records_step]).to(device)
            source = torch.stack([record.source_parameters for record in records_step]).to(device)
            source_map = torch.stack([record.source_map for record in records_step]).to(device)
            dense_times = torch.stack(
                [record.time_s[record.dense_time_indices[dense_positions]] for record in records_step]
            ).to(device)
            prepared = model.prepare_sources(
                model.encode_medium(velocity, normalizer),
                source,
                source_map,
                normalizer,
                record_to_medium=torch.arange(3, device=device),
            )
            x_m = records_step[0].x_m.to(device)
            z_m = records_step[0].z_m.to(device)
            prediction_dense = model.dense_normalized(
                prepared, dense_times, x_m=x_m, z_m=z_m, time_block=1
            )
            target_dense = normalizer.encode_pressure(
                torch.stack(
                    [record.dense_target_physical[dense_positions] for record in records_step]
                ).to(device),
                source[:, 4],
            )
            coords = torch.stack(
                [record.query_coords[index] for record, index in zip(records_step, query_indices, strict=True)]
            ).to(device)
            prediction_query = model.query_normalized(prepared, coords, chunk_size=1024)
            target_query = normalizer.encode_pressure(
                torch.stack(
                    [record.query_target_physical[index] for record, index in zip(records_step, query_indices, strict=True)]
                ).to(device),
                source[:, 4],
            )
            dense_at_query = torch.cat(
                [
                    _dense_at_query(
                        prediction_dense[index : index + 1],
                        coords[index],
                        dense_times[index],
                        x_m=x_m,
                        z_m=z_m,
                    )
                    for index in range(3)
                ],
                dim=0,
            )
            result = compute_v3_losses(
                prediction_query=prediction_query,
                target_query=target_query,
                query_probability=torch.stack(query_probability).to(device),
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
        if current_step % config.train.evaluation_every != 0 and current_step != config.train.max_steps:
            continue
        metrics, diagnostics, updates = evaluate_nine(
            model, records, normalizer, device=device
        )
        for sample_id, error in updates.items():
            priorities[sample_id] = 0.5 * priorities[sample_id] + 0.5 * error
        decision = evaluate_nine_record_gate(metrics)
        score = (
            metrics.aggregate_query_relative_l2
            + metrics.aggregate_dense_relative_l2
            + sum(metrics.family_query_relative_l2.values())
            + sum(metrics.family_dense_relative_l2.values())
            + sum(metrics.family_late_relative_l2.values())
        )
        report = {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "manifest_digest": manifest.digest,
            "config_digest": run_digest,
            "run_identity": run_identity,
            "prerequisite_step": prerequisite.get("step"),
            "step": current_step,
            "elapsed_seconds": time.monotonic() - started,
            "loss": float(loss),
            "components": captured.get("components", {}),
            "metrics": metrics.to_dict(),
            "diagnostics": diagnostics,
            "decision": decision.to_dict(),
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
                metrics={"score": score},
            )
            _atomic_json(report, artifact_dir / "best_report.json")
        if decision.passed:
            _atomic_json(report, artifact_dir / "passed_report.json")
            return report
        if plateau.update(score):
            report["stopped_reason"] = "adamw_plateau"
            _atomic_json(report, artifact_dir / "plateau_report.json")
            break
    if best_report is None:
        raise RuntimeError("nine-record gate produced no evaluation")
    failed = dict(best_report)
    failed["passed"] = False
    _atomic_json(failed, artifact_dir / "failed_report.json")
    return failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--artifact-dir", default="artifacts/grouped_ufno_mionet_v3/nine_record_gate")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--one-record-report", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = V3Config.from_yaml(args.config)
    prerequisite = validate_one_record_prerequisite(Path(args.one_record_report))
    records = load_balanced_records(config)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "sample_ids": [record.sample_id for record in records],
                    "family_counts": {
                        family: sum(record.medium_type == family for record in records)
                        for family in ALLOWED_MEDIUM_TYPES
                    },
                    "prerequisite_passed": prerequisite["decision"]["passed"],
                },
                sort_keys=True,
            )
        )
        return 0
    report = run_nine_record_gate(
        config,
        device_name=args.device,
        artifact_dir=Path(args.artifact_dir),
        initial_checkpoint=Path(args.init_checkpoint),
        prerequisite_report=Path(args.one_record_report),
    )
    return 0 if report.get("decision", {}).get("passed", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
