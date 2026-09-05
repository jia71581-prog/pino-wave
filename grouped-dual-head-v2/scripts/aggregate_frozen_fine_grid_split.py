#!/usr/bin/env python3
"""Aggregate frozen fine-grid split workers and apply the same-protocol gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


FAMILIES = ("uniform", "layered", "marmousi")


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _add(left: list[float], right: list[float]) -> list[float]:
    return [float(left[0]) + float(right[0]), float(left[1]) + float(right[1])]


def _relative(terms: list[float]) -> float:
    return float(math.sqrt(float(terms[0]) / max(float(terms[1]), 1.0e-30)))


def _nearest_rank(values: list[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[max(1, int(math.ceil(probability * len(ordered)))) - 1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", nargs="+", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test_id"), required=True)
    parser.add_argument("--expected-records", type=int, default=480)
    parser.add_argument("--speed-reference-s", type=float, default=20.34137312322855)
    parser.add_argument("--maximum-relative-l2", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    workers = [json.loads(path.read_text()) for path in args.workers]
    if any(worker.get("status") != "complete" for worker in workers):
        raise RuntimeError("all workers must be complete")
    if any(worker.get("split") != args.split for worker in workers):
        raise RuntimeError("worker split mismatch")
    shard_pairs = {(worker["shard_index"], worker["num_shards"]) for worker in workers}
    expected_shards = {(index, len(workers)) for index in range(len(workers))}
    if shard_pairs != expected_shards:
        raise RuntimeError(f"incomplete shard set: {sorted(shard_pairs)}")
    binding_keys = (
        "source_h5_sha256",
        "manifest_sha256",
        "marmousi_npy_sha256",
        "fused_solver_sha256",
        "functional_kernel_sha256",
        "gate_helper_sha256",
        "evaluation_script_sha256",
    )
    bindings = {
        key: {worker["bindings"][key] for worker in workers} for key in binding_keys
    }
    if any(len(values) != 1 for values in bindings.values()):
        raise RuntimeError(f"worker binding mismatch: {bindings}")
    measurements = [
        row for worker in workers for row in worker["measurements"]
    ]
    indices = [int(row["source_index"]) for row in measurements]
    if len(indices) != len(set(indices)):
        raise RuntimeError("duplicate records across workers")
    if len(indices) != args.expected_records:
        raise RuntimeError(
            f"expected {args.expected_records} records, got {len(indices)}"
        )
    if any(worker["global_record_count"] != args.expected_records for worker in workers):
        raise RuntimeError("worker global record count mismatch")

    total = [0.0, 0.0]
    family = {name: [0.0, 0.0] for name in FAMILIES}
    temporal = {name: [0.0, 0.0] for name in ("early", "middle", "late")}
    for worker in workers:
        total = _add(total, worker["error_terms"]["aggregate"])
        for name in FAMILIES:
            family[name] = _add(family[name], worker["error_terms"]["family"][name])
        for name in temporal:
            temporal[name] = _add(
                temporal[name], worker["error_terms"]["temporal"][name]
            )
    family_counts = {
        name: sum(row["family"] == name for row in measurements) for name in FAMILIES
    }
    aggregate_relative = _relative(total)
    family_relative = {name: _relative(terms) for name, terms in family.items()}
    temporal_relative = {name: _relative(terms) for name, terms in temporal.items()}
    maximum_instance = max(float(row["relative_l2"]) for row in measurements)
    runtimes = [float(row["runtime_s"]) for row in measurements]
    runtime_mean = sum(runtimes) / len(runtimes)
    runtime_p95 = _nearest_rank(runtimes, 0.95)
    runtime_limit = args.speed_reference_s / 10.0
    passes_accuracy = bool(
        aggregate_relative <= args.maximum_relative_l2
        and max(family_relative.values()) <= args.maximum_relative_l2
        and maximum_instance <= args.maximum_relative_l2
    )
    passes_runtime = bool(runtime_mean <= runtime_limit and runtime_p95 <= runtime_limit)
    passed = bool(passes_accuracy and passes_runtime)
    report = {
        "schema": "frozen_fine_grid_complete_split_gate_v1",
        "status": "target_gate_passed" if passed else "target_gate_failed",
        "split": args.split,
        "selection_scope": f"complete_registered_{args.split}_target_families",
        "record_count": len(measurements),
        "family_counts": family_counts,
        "excluded_family": "anomaly",
        "aggregate_relative_l2": aggregate_relative,
        "family_relative_l2": family_relative,
        "maximum_instance_relative_l2": maximum_instance,
        "temporal_band_relative_l2": temporal_relative,
        "runtime_s": {
            "minimum": min(runtimes),
            "mean": runtime_mean,
            "p95_nearest_rank": runtime_p95,
            "maximum": max(runtimes),
        },
        "speedup_over_traditional_reference": {
            "reference_s": args.speed_reference_s,
            "mean": args.speed_reference_s / runtime_mean,
            "p95": args.speed_reference_s / runtime_p95,
        },
        "promotion_gate": {
            "maximum_relative_l2": args.maximum_relative_l2,
            "maximum_runtime_s": runtime_limit,
            "passes_accuracy": passes_accuracy,
            "passes_runtime": passes_runtime,
            "passed": passed,
        },
        "worst_instances": sorted(
            measurements, key=lambda row: float(row["relative_l2"]), reverse=True
        )[:20],
        "bindings": {
            key: next(iter(values)) for key, values in bindings.items()
        },
        "worker_outputs": [str(path.resolve()) for path in args.workers],
        "worker_sha256": {str(path.resolve()): _sha256(path) for path in args.workers},
        "aggregation_script_sha256": _sha256(Path(__file__)),
    }
    _atomic_json(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
