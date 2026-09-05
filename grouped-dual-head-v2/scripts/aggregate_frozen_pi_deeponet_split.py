#!/usr/bin/env python3
"""Aggregate complete comparison-only PI-DeepONet split workers."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


FAMILIES = ("uniform", "layered", "marmousi")


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


def _add(left: list[float], right: list[float]) -> list[float]:
    return [float(left[0]) + float(right[0]), float(left[1]) + float(right[1])]


def _relative(values: list[float]) -> float:
    return float(math.sqrt(values[0] / max(values[1], 1.0e-30)))


def _nearest_rank(values: list[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[max(1, int(math.ceil(probability * len(ordered)))) - 1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", nargs="+", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test_id"), required=True)
    parser.add_argument("--expected-records", type=int, default=480)
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
        "training_config_sha256",
        "base_config_sha256",
        "manifest_digest",
        "source_h5_sha256",
        "normalization_sha256",
        "travel_time_h5_sha256",
        "evaluation_script_sha256",
    )
    bindings = {
        key: {worker["bindings"][key] for worker in workers} for key in binding_keys
    }
    if any(len(values) != 1 for values in bindings.values()):
        raise RuntimeError(f"worker binding mismatch: {bindings}")
    checkpoint_hashes = {worker["checkpoint"]["sha256"] for worker in workers}
    if checkpoint_hashes != {"e2d6d7a9ec5481268bae2f41cff2400ee78b93a3a9216b2c338db572d405c4c1"}:
        raise RuntimeError("PI-DeepONet checkpoint binding mismatch")
    measurements = [row for worker in workers for row in worker["measurements"]]
    indices = [int(row["source_index"]) for row in measurements]
    if len(indices) != len(set(indices)) or len(indices) != args.expected_records:
        raise RuntimeError("PI-DeepONet complete split coverage failed")
    if any(worker["global_record_count"] != args.expected_records for worker in workers):
        raise RuntimeError("worker global record count mismatch")
    if any(worker["frames_per_record"] != 32 for worker in workers):
        raise RuntimeError("PI-DeepONet comparison must use 32 exact frames")

    total = [0.0, 0.0]
    family_terms = {family: [0.0, 0.0] for family in FAMILIES}
    for worker in workers:
        total = _add(total, worker["error_terms"]["aggregate"])
        for family in FAMILIES:
            family_terms[family] = _add(
                family_terms[family], worker["error_terms"]["family"][family]
            )
    family_rows = {
        family: [row for row in measurements if row["family"] == family]
        for family in FAMILIES
    }
    runtimes = [float(row["runtime_s"]) for row in measurements]
    report = {
        "schema": "frozen_pi_deeponet_complete_split_comparison_v1",
        "status": "complete_comparison_only",
        "split": args.split,
        "selection_scope": "complete_registered_split_32_exact_frames",
        "record_count": len(measurements),
        "frames_per_record": 32,
        "family_counts": {family: len(rows) for family, rows in family_rows.items()},
        "global_energy_relative_l2": _relative(total),
        "record_mean_relative_l2": sum(float(row["relative_l2"]) for row in measurements)
        / len(measurements),
        "family_energy_relative_l2": {
            family: _relative(values) for family, values in family_terms.items()
        },
        "family_record_mean_relative_l2": {
            family: sum(float(row["relative_l2"]) for row in rows) / len(rows)
            for family, rows in family_rows.items()
        },
        "maximum_instance_relative_l2": max(
            float(row["relative_l2"]) for row in measurements
        ),
        "runtime_s_for_32_frames": {
            "minimum": min(runtimes),
            "mean": sum(runtimes) / len(runtimes),
            "p95_nearest_rank": _nearest_rank(runtimes, 0.95),
            "maximum": max(runtimes),
        },
        "comparison_caveat": "Runtime materializes only 32 of 401 frames and therefore favors PI-DeepONet relative to the proposed solver's complete 401-frame runtime.",
        "worst_instances": sorted(
            measurements, key=lambda row: float(row["relative_l2"]), reverse=True
        )[:20],
        "checkpoint": workers[0]["checkpoint"],
        "bindings": {key: next(iter(values)) for key, values in bindings.items()},
        "worker_outputs": [str(path.resolve()) for path in args.workers],
        "worker_sha256": {str(path.resolve()): _sha256(path) for path in args.workers},
        "aggregation_script_sha256": _sha256(Path(__file__)),
    }
    _atomic_json(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
