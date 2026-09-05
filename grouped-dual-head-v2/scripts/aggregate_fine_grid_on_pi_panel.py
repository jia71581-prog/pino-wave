#!/usr/bin/env python3
"""Aggregate frozen fine-grid workers evaluated on the exact PI panel."""
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
    ordered = sorted(values)
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
    pairs = {(worker["shard_index"], worker["num_shards"]) for worker in workers}
    if pairs != {(index, len(workers)) for index in range(len(workers))}:
        raise RuntimeError("incomplete shard set")
    if any(
        worker.get("frames_scored_per_record") != 32
        or worker.get("frames_materialized_per_record") != 401
        or worker.get("global_record_count") != args.expected_records
        for worker in workers
    ):
        raise RuntimeError("matched-panel frame or coverage contract changed")
    binding_keys = (
        "source_h5_sha256",
        "manifest_sha256",
        "marmousi_npy_sha256",
        "fused_solver_sha256",
        "functional_kernel_sha256",
        "comparison_script_sha256",
    )
    bindings = {
        key: {worker["bindings"][key] for worker in workers} for key in binding_keys
    }
    if any(len(values) != 1 for values in bindings.values()):
        raise RuntimeError(f"worker binding mismatch: {bindings}")
    measurements = [row for worker in workers for row in worker["measurements"]]
    if len(measurements) != args.expected_records:
        raise RuntimeError("incomplete matched-panel coverage")
    keys = {(row["source_index"], tuple(row["time_indices"])) for row in measurements}
    if len(keys) != args.expected_records:
        raise RuntimeError("duplicate matched-panel records")
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
    runtimes = [float(row["runtime_s_for_complete_401_frames"]) for row in measurements]
    report = {
        "schema": "fine_grid_on_pi_exact_panel_complete_split_v1",
        "status": "complete",
        "split": args.split,
        "selection_scope": "same_records_and_32_exact_frames_as_frozen_pi_deeponet",
        "record_count": len(measurements),
        "frames_scored_per_record": 32,
        "frames_materialized_per_record": 401,
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
        "runtime_s_for_complete_401_frames": {
            "minimum": min(runtimes),
            "mean": sum(runtimes) / len(runtimes),
            "p95_nearest_rank": _nearest_rank(runtimes, 0.95),
            "maximum": max(runtimes),
        },
        "bindings": {key: next(iter(values)) for key, values in bindings.items()},
        "pi_worker_sha256": {
            int(worker["shard_index"]): worker["bindings"]["pi_worker_sha256"]
            for worker in workers
        },
        "worker_outputs": [str(path.resolve()) for path in args.workers],
        "worker_sha256": {str(path.resolve()): _sha256(path) for path in args.workers},
        "aggregation_script_sha256": _sha256(Path(__file__)),
    }
    _atomic_json(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
