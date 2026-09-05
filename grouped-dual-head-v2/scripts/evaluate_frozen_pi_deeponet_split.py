#!/usr/bin/env python3
"""Comparison-only evaluation of the frozen, full-data PI-DeepONet."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from grouped_ufno_mionet_v3.config import ALLOWED_MEDIUM_TYPES, V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts
from patch_deeponet_baseline.model import PatchDeepONet, PatchDeepONetConfig
from patch_deeponet_baseline.training import dense_training_pair
from saved_time_phase_operator_v4.data import ExactStoredTimeBatchDataset
from saved_time_phase_operator_v4.full_support import FullSupportStepSpec
from scripts.train_grouped_v3_pilot import load_normalizer


CHECKPOINT_SCHEMA = "patch_deeponet_checkpoint_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _terms(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    prediction64 = prediction.astype(np.float64)
    target64 = target.astype(np.float64)
    return (
        float(np.sum((prediction64 - target64) ** 2)),
        float(np.sum(target64**2)),
    )


def _add(left: tuple[float, float], right: tuple[float, float]) -> tuple[float, float]:
    return left[0] + right[0], left[1] + right[1]


def _relative(values: tuple[float, float]) -> float:
    return float(math.sqrt(values[0] / max(values[1], 1.0e-30)))


def _model_from_training_config(config: dict) -> PatchDeepONet:
    values = config.get("model", {})
    return PatchDeepONet(PatchDeepONetConfig(**dict(values)))


def _progress_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.progress.json")


def run(args: argparse.Namespace) -> dict:
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must be in [0, num_shards)")
    if args.maximum_records is not None and args.split != "train":
        raise PermissionError("record limiting is allowed only for train smoke tests")
    training_config = yaml.safe_load(args.training_config.read_text(encoding="utf-8"))
    if training_config.get("schema") != "pi_deeponet_training_config_v1":
        raise ValueError("comparison requires the registered PI-DeepONet config")
    base_path = ROOT / str(training_config["base_config"])
    travel_time_h5 = (
        Path(training_config["travel_time_h5"])
        if args.travel_time_h5 is None
        else args.travel_time_h5
    ).expanduser().resolve()
    base = V3Config.from_yaml(base_path)
    manifest = build_manifest(base.data.source_h5)
    validate_expected_counts(
        manifest,
        {
            "train": int(base.data.expected_train_records),
            "validation": int(base.data.expected_validation_records),
            "test_id": 480,
        },
    )
    if manifest.digest != "55fbffa9a66b0cb547657d2d5cd8cc140c4f7d970e37f3828144e7778d182e09":
        raise RuntimeError("PI-DeepONet manifest binding changed")
    split_records = tuple(record for record in manifest.records if record.split == args.split)
    selected_all = list(range(len(split_records)))
    if args.maximum_records is not None:
        selected_all = selected_all[: int(args.maximum_records)]
    selected = [
        index
        for position, index in enumerate(selected_all)
        if position % args.num_shards == args.shard_index
    ]
    if not selected:
        raise ValueError("selected PI-DeepONet evaluation shard is empty")
    schedule = tuple(
        FullSupportStepSpec(
            step=1_100_000 + index,
            epoch=0,
            record_indices=(index,),
            appearance_indices=(0,),
        )
        for index in selected
    )
    dataset = ExactStoredTimeBatchDataset(
        base.data.source_h5,
        manifest,
        split=args.split,
        schedule=schedule,
        query_points=1,
        seed=int(training_config["seed"]),
        time_policy="validation_fixed",
        frames_per_record=int(args.frames_per_record),
        travel_time_h5=travel_time_h5,
    )
    normalizer = load_normalizer(base, manifest.digest)
    device = torch.device(args.device)
    model = _model_from_training_config(training_config).to(device)
    checkpoint_sha256 = _sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected PI-DeepONet checkpoint schema")
    if checkpoint.get("manifest_digest") != manifest.digest:
        raise ValueError("PI-DeepONet checkpoint manifest changed")
    if checkpoint.get("selection_split") != "train":
        raise ValueError("PI-DeepONet checkpoint was not selected on train")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    execution = training_config["model_execution"]
    lead_cycles = float(training_config["loss"]["hard_causality_lead_cycles"])

    total = (0.0, 0.0)
    family_terms = {family: (0.0, 0.0) for family in ALLOWED_MEDIUM_TYPES}
    measurements = []
    progress_path = _progress_path(args.output)
    with torch.inference_mode():
        for position, batch in enumerate(dataset, start=1):
            if len(batch.sample_id) != 1:
                raise RuntimeError("PI comparison worker expects one record per batch")
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            prediction, target, time_indices = dense_training_pair(
                model,
                batch,
                normalizer,
                device,
                time_block=int(execution["time_block"]),
                query_chunk=int(execution["query_chunk"]),
                hard_causality_lead_cycles=lead_cycles,
            )
            prediction_cpu = prediction.detach().cpu().numpy()
            target_cpu = target.detach().cpu().numpy()
            torch.cuda.synchronize(device)
            runtime_s = float(time.perf_counter() - started)
            values = _terms(prediction_cpu, target_cpu)
            total = _add(total, values)
            family = str(batch.medium_type[0])
            family_terms[family] = _add(family_terms[family], values)
            measurement = {
                "source_index": int(split_records[selected[position - 1]].source_index),
                "sample_id": str(batch.sample_id[0]),
                "split": args.split,
                "family": family,
                "source_f0_hz": float(batch.source_parameters[0, 2]),
                "relative_l2": _relative(values),
                "error_numerator": values[0],
                "truth_denominator": values[1],
                "runtime_s": runtime_s,
                "time_indices": [int(value) for value in time_indices[0].tolist()],
            }
            measurements.append(measurement)
            _atomic_json(
                {
                    "schema": "frozen_pi_deeponet_split_progress_v1",
                    "status": "running",
                    "split": args.split,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "completed_records": position,
                    "total_records": len(selected),
                    "last_measurement": measurement,
                },
                progress_path,
            )
            print(
                f"[{args.split} shard {args.shard_index}] {position}/{len(selected)} "
                f"{measurement['sample_id']} rel={measurement['relative_l2']:.6g} "
                f"runtime={runtime_s:.4f}s",
                flush=True,
            )

    runtimes = [row["runtime_s"] for row in measurements]
    report = {
        "schema": "frozen_pi_deeponet_split_worker_v1",
        "status": "complete",
        "role": "comparison_only",
        "split": args.split,
        "selection_scope": "complete_registered_split_32_exact_frames"
        if args.maximum_records is None
        else "limited_train_smoke",
        "frames_per_record": int(args.frames_per_record),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "global_record_count": len(selected_all),
        "record_count": len(measurements),
        "aggregate_relative_l2": _relative(total),
        "error_terms": {
            "aggregate": list(total),
            "family": {family: list(values) for family, values in family_terms.items()},
        },
        "runtime_s": {
            "minimum": min(runtimes),
            "mean": float(sum(runtimes) / len(runtimes)),
            "maximum": max(runtimes),
        },
        "measurements": measurements,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_sha256,
            "epoch": int(checkpoint["epoch"]),
            "global_step": int(checkpoint["global_step"]),
            "training_records": 2240,
            "training_epochs": 100,
        },
        "bindings": {
            "training_config_sha256": _sha256(args.training_config),
            "base_config_sha256": _sha256(base_path),
            "manifest_digest": manifest.digest,
            "source_h5_sha256": manifest.source_file_sha256,
            "normalization_sha256": _sha256(Path(base.data.normalization_json)),
            "travel_time_h5": str(travel_time_h5),
            "travel_time_h5_sha256": _sha256(travel_time_h5),
            "evaluation_script_sha256": _sha256(Path(__file__)),
        },
    }
    _atomic_json(report, args.output)
    _atomic_json(
        {
            "schema": "frozen_pi_deeponet_split_progress_v1",
            "status": "complete",
            "split": args.split,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "completed_records": len(measurements),
            "total_records": len(selected),
            "terminal_output": str(args.output.resolve()),
        },
        progress_path,
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--travel-time-h5", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test_id"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--frames-per-record", type=int, default=32)
    parser.add_argument("--maximum-records", type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
