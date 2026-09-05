#!/usr/bin/env python3
"""Run guarded residual-iterator adaptation and sealed future evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet_v3.data.index import ALLOWED_MEDIUM_TYPES, build_manifest
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.instance_adaptation.residual_iterator import (
    RESIDUAL_ITERATOR_SCHEMA_VERSION,
    MultiScaleResidualCorrector,
    refine_with_residual_iterator,
)
from scripts.evaluate_v5_instance_adaptation import evaluate_after_adaptation, write_report
from scripts.run_v5_instance_adaptation import (
    _load_background_provider,
    _load_parent,
    _predict_parent,
    _read_future_truth,
    _write_manifest,
    build_all_validation_manifest,
    build_instance_manifest,
)
from scripts.train_v5_residual_meta import _sha256


def _load_corrector(
    path: str | Path,
    *,
    parent_checkpoint: str | Path,
    manifest_digest: str,
    device: torch.device,
) -> tuple[MultiScaleResidualCorrector, dict[str, object]]:
    checkpoint = Path(path).resolve()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if int(payload.get("schema_version", 0)) != RESIDUAL_ITERATOR_SCHEMA_VERSION:
        raise ValueError("residual iterator checkpoint schema mismatch")
    if payload.get("manifest_digest") != manifest_digest:
        raise ValueError("residual iterator manifest digest mismatch")
    expected_parent = Path(parent_checkpoint).resolve()
    if Path(str(payload.get("parent_checkpoint", ""))).resolve() != expected_parent:
        raise ValueError("residual iterator is bound to a different parent checkpoint")
    expected_sha = str(payload.get("parent_checkpoint_sha256", ""))
    if not expected_sha or _sha256(expected_parent) != expected_sha:
        raise ValueError("residual iterator parent checkpoint hash mismatch")
    corrector = MultiScaleResidualCorrector(
        width=int(payload.get("corrector_width", 24))
    ).to(device)
    corrector.load_state_dict(payload["corrector_state"], strict=True)
    corrector.eval()
    return corrector, {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "parent_checkpoint": str(expected_parent),
        "parent_checkpoint_sha256": expected_sha,
        "best_epoch": payload.get("best_epoch"),
        "best_train_crop_relative_l2": payload.get("best_train_crop_relative_l2"),
    }


def run(
    config_path: str | Path,
    *,
    corrector_checkpoint: str | Path,
    output_dir: str | Path,
    device_name: str = "cuda",
    sample_ids: tuple[str, ...] | None = None,
    per_family: int = 3,
    all_validation: bool = False,
    save_fields: bool = True,
    parent_checkpoint: str | Path | None = None,
    parent_run_identity: str | Path | None = None,
) -> list[dict[str, object]]:
    config = yaml.safe_load(Path(config_path).read_text())
    if parent_checkpoint is not None:
        config["parent_checkpoint"] = str(Path(parent_checkpoint).resolve())
    if parent_run_identity is not None:
        config["parent_run_identity"] = str(Path(parent_run_identity).resolve())
    manifest = build_manifest(config["source_h5"])
    if all_validation:
        rows = build_all_validation_manifest(manifest)
    elif sample_ids:
        by_sample = {
            row.sample_id: row for row in manifest.records if row.split == "validation"
        }
        missing = tuple(value for value in sample_ids if value not in by_sample)
        if missing:
            raise ValueError(f"selected validation samples are missing: {missing[:3]}")
        rows = tuple(by_sample[value] for value in sample_ids)
    else:
        rows = build_instance_manifest(
            manifest,
            seed=int(config.get("seed", 17)),
            per_family=int(per_family),
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_manifest(rows, output / "instance_manifest.json")
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    parent, normalizer = _load_parent(config, manifest, device)
    provider = _load_background_provider(config, (row.sample_id for row in rows))
    corrector, corrector_info = _load_corrector(
        corrector_checkpoint,
        parent_checkpoint=config["parent_checkpoint"],
        manifest_digest=manifest.digest,
        device=device,
    )
    dataset = GuardedOnsetDataset(
        config["source_h5"],
        manifest,
        split="validation",
        sample_ids=tuple(row.sample_id for row in rows),
        travel_time_h5=config.get("travel_time_h5"),
    )
    reports: list[dict[str, object]] = []
    try:
        for record in dataset:
            record.audit.read(record.observed_indices)
            source = record.source_parameters.to(device).unsqueeze(0)
            starter_normalized = _predict_parent(
                parent,
                normalizer,
                record,
                device,
                normalized=True,
                background_provider=provider,
            )
            result = refine_with_residual_iterator(
                corrector,
                starter_normalized,
                record.velocity_mps.to(device).unsqueeze(0),
                record.time_s.to(device),
                record.observed_indices,
                iterations=int(config.get("residual_iterator_iterations", 4)),
                step_candidates=tuple(
                    float(value)
                    for value in config.get(
                        "residual_iterator_step_candidates", (0.0, 0.25, 0.5, 1.0)
                    )
                ),
                minimum_relative_improvement=float(
                    config.get("minimum_physics_improvement", 1.0e-4)
                ),
                minimum_energy_ratio=float(config.get("minimum_energy_ratio", 0.8)),
                maximum_energy_ratio=float(config.get("maximum_energy_ratio", 1.25)),
            )
            starter = normalizer.decode_pressure(starter_normalized, source[:, 4])
            adapted = normalizer.decode_pressure(result.field, source[:, 4])
            artifact = output / record.sample_id
            artifact.mkdir(parents=True, exist_ok=True)
            adaptation_payload = {
                "accepted_steps": result.accepted_steps,
                "residual_history": result.residual_history,
                "step_size_history": result.step_size_history,
                "energy_ratio": result.energy_ratio,
                "stopped_reason": result.stopped_reason,
                "accessed_true_indices": tuple(record.audit.requested_indices),
                "future_truth_used": bool(record.audit.payload()["future_truth_used"]),
                "corrector": corrector_info,
            }
            sealed = {"adaptation": adaptation_payload}
            if save_fields:
                sealed.update(
                    {"parent_field": starter.cpu(), "adapted_field": adapted.cpu()}
                )
            torch.save(sealed, artifact / "adaptation.pt")

            truth = _read_future_truth(config["source_h5"], record.source_index)
            report = evaluate_after_adaptation(
                {"parent_field": starter.cpu(), "adapted_field": adapted.cpu()},
                truth.unsqueeze(0),
                observed_indices=(record.observed_indices,),
                families=(record.medium_type,),
                group_ids=(record.group_id,),
                sample_ids=(record.sample_id,),
                sealed=True,
            )
            report.update(
                {
                    "sample_id": record.sample_id,
                    "medium_type": record.medium_type,
                    "adaptation": adaptation_payload,
                }
            )
            write_report(report, artifact / "evaluation.json")
            reports.append(report)
    finally:
        dataset.close()
        if provider is not None:
            provider.close()
    write_report(
        {
            "records": reports,
            "family_count": {
                family: sum(item["medium_type"] == family for item in reports)
                for family in ALLOWED_MEDIUM_TYPES
            },
            "corrector": corrector_info,
        },
        output / "summary.json",
    )
    return reports


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--corrector-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--parent-run-identity")
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--per-family", type=int, default=3)
    parser.add_argument("--all-validation", action="store_true")
    parser.add_argument("--no-fields", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({
            "config_exists": Path(args.config).is_file(),
            "corrector_checkpoint_exists": Path(args.corrector_checkpoint).is_file(),
            "future_truth_opened": False,
            "allowed_true_snapshot_count": 2,
        }, sort_keys=True))
        return 0
    run(
        args.config,
        corrector_checkpoint=args.corrector_checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
        sample_ids=None if args.sample_id is None else tuple(args.sample_id),
        per_family=args.per_family,
        all_validation=args.all_validation,
        save_fields=not args.no_fields,
        parent_checkpoint=args.parent_checkpoint,
        parent_run_identity=args.parent_run_identity,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
