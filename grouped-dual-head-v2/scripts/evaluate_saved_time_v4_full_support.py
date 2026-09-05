#!/usr/bin/env python
"""Stream all 480 held-out records over all 401 exact stored time indices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.data.records import V3WavefieldDataset
from grouped_ufno_mionet_v3.training.checkpoint import load_checkpoint
from saved_time_phase_operator_v4.evaluation import (
    sha256_file,
    time_axis_sha256,
    validate_evaluation_identity,
)
from saved_time_phase_operator_v4.streaming_metrics import ExactWavefieldMetricAccumulator
from scripts.refine_saved_time_v4_lbfgs import _append_jsonl, _gpu_snapshot
from scripts.train_grouped_v3_pilot import load_normalizer
from scripts.train_saved_time_v4_full_support import (
    _load_context,
    _load_parent_model,
)
from scripts.train_saved_time_v4_probe import _atomic_json


def final_time_blocks(
    *, stored_time_count: int, block_size: int
) -> tuple[tuple[int, int], ...]:
    count = int(stored_time_count)
    block = int(block_size)
    if count <= 0 or block <= 0:
        raise ValueError("stored time count and block size must be positive")
    return tuple((start, min(start + block, count)) for start in range(0, count, block))


def evaluation_checkpoint_identity(
    config,
    *,
    checkpoint_identity: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    identity_path = (
        Path(config["artifact_dir"]) / "run" / "run_identity.json"
        if checkpoint_identity is None
        else Path(checkpoint_identity)
    )
    if not identity_path.is_file():
        raise FileNotFoundError("evaluation checkpoint identity is unavailable")
    identity = json.loads(identity_path.read_text())
    if not isinstance(identity, dict) or not identity.get("run_digest"):
        raise ValueError("evaluation checkpoint identity is invalid")
    return identity_path.resolve(), identity


@torch.inference_mode()
def _evaluate(config, checkpoint: Path, output: Path, *, device: torch.device, block_size: int,
              checkpoint_identity: Path | None = None,
              evaluation_split: str = "validation"):
    base, manifest, parent_identity = _load_context(config)
    split = str(evaluation_split)
    if split not in {"validation", "test_id"}:
        raise ValueError("full-support evaluation split must be validation or test_id")
    expected_records = sum(record.split == split for record in manifest.records)
    if expected_records <= 0:
        raise ValueError(f"full-support evaluation split is empty: {split}")
    run_identity_path, run_identity = evaluation_checkpoint_identity(
        config, checkpoint_identity=checkpoint_identity
    )
    if run_identity["manifest_digest"] != manifest.digest:
        raise ValueError("final evaluation manifest identity mismatch")
    binding = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest_digest": manifest.digest,
        "time_axis_sha256": time_axis_sha256(manifest.time_s),
        "record_census": {split: expected_records},
        "model_config_digest": run_identity["run_digest"],
        "checkpoint_identity": str(run_identity_path),
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "evaluation_identity.json"
    if identity_path.exists():
        validate_evaluation_identity(json.loads(identity_path.read_text()), binding)
    else:
        _atomic_json(binding, identity_path)
    report_path = output / "evaluation_report.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text())
        if existing.get("status") == "complete":
            if str(existing.get("evaluation_split", "validation")) != split:
                raise ValueError("cached full-support evaluation split mismatch")
            return existing

    model = _load_parent_model(config, base, manifest, parent_identity, device)
    load_checkpoint(
        checkpoint,
        model=model,
        expected_manifest_digest=manifest.digest,
        expected_config_digest=run_identity["run_digest"],
        map_location=device,
    )
    model.eval()
    normalizer = load_normalizer(base, manifest.digest)
    dataset = V3WavefieldDataset(base.data.source_h5, manifest, split=split)
    if len(dataset) != expected_records:
        raise RuntimeError("final validation record census changed")
    blocks = final_time_blocks(
        stored_time_count=len(manifest.time_s), block_size=block_size
    )
    accumulator = ExactWavefieldMetricAccumulator(
        energy_floor_fraction=float(config["energy_floor_fraction"]),
        require_unique=True,
        stored_time_count=len(manifest.time_s),
    )
    progress_path = output / "progress.jsonl"
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for record_index in range(len(dataset)):
        record = dataset[record_index]
        velocity = record.velocity_mps.unsqueeze(0).to(device)
        source = record.source_parameters.unsqueeze(0).to(device)
        source_map = record.source_map.unsqueeze(0).to(device)
        prepared = model.prepare_sources(
            model.encode_medium(velocity, normalizer),
            source,
            source_map,
            normalizer,
            record_to_medium=torch.zeros(1, dtype=torch.long, device=device),
        )
        onset_index = int(np.searchsorted(manifest.time_s, float(record.source_parameters[3])))
        for start, stop in blocks:
            requested = record.time_s[start:stop]
            target_physical = dataset.read_wavefield(record_index, requested)
            if not bool(target_physical.exact.all()) or not torch.equal(
                target_physical.left_index, target_physical.right_index
            ):
                raise RuntimeError("final evaluation encountered an interpolated target")
            times = target_physical.requested_time_s.unsqueeze(0).to(device)
            prediction = model.dense_normalized(
                prepared,
                times,
                x_m=record.x_m.to(device),
                z_m=record.z_m.to(device),
                time_block=1,
            )
            target = normalizer.encode_pressure(
                target_physical.values.unsqueeze(0).to(device), source[:, 4]
            )
            accumulator.update(
                prediction,
                target,
                families=(record.medium_type,),
                group_ids=(record.group_id,),
                sample_ids=(record.sample_id,),
                time_indices=torch.arange(start, stop, dtype=torch.long).unsqueeze(0),
                source_onset_indices=(onset_index,),
            )
        progress = {
            "event": "record_complete",
            "record": record_index + 1,
            "record_count": len(dataset),
            "sample_id": record.sample_id,
            "elapsed_seconds": time.monotonic() - started,
            "gpu": _gpu_snapshot(),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        }
        _append_jsonl(progress_path, progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    metrics = accumulator.finalize()
    aggregate_target = float(config["gate"]["target_aggregate_relative_l2"])
    family_target = float(config["gate"]["target_family_relative_l2"])
    report = {
        "status": "complete",
        "evaluation_split": split,
        "binding": binding,
        "stored_times_only": True,
        "interpolated_targets": 0,
        "time_block": int(block_size),
        "metrics": metrics,
        "passes_accuracy_gate": metrics["aggregate_relative_l2"] < aggregate_target
        and all(
            value < family_target for value in metrics["family_relative_l2"].values()
        ),
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }
    if metrics["record_count"] != expected_records:
        raise RuntimeError("final evaluation did not cover all held-out records")
    if metrics["unique_time_index_count"] != len(manifest.time_s):
        raise RuntimeError("final evaluation did not cover all stored time indices")
    _atomic_json(report, report_path)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint-identity", type=Path)
    parser.add_argument("--time-block", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--evaluation-split", choices=("validation", "test_id"), default="validation"
    )
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    checkpoint = args.checkpoint or Path(config["artifact_dir"]) / "run" / "best.pt"
    output = args.output or Path(config["artifact_dir"]) / "run" / "sealed_evaluation"
    report = _evaluate(
        config,
        checkpoint,
        output,
        device=torch.device(args.device),
        block_size=int(args.time_block),
        checkpoint_identity=args.checkpoint_identity,
        evaluation_split=args.evaluation_split,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["evaluation_checkpoint_identity", "final_time_blocks"]
