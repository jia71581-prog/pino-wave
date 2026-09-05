#!/usr/bin/env python3
"""Aggregate exact-panel CPADC workers and compare them with frozen PI-DeepONet."""
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


def _relative(terms: list[float]) -> float:
    return float(math.sqrt(float(terms[0]) / max(float(terms[1]), 1.0e-30)))


def _nearest_rank(values: list[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[max(1, int(math.ceil(probability * len(ordered)))) - 1]


def aggregate(
    worker_paths: list[Path],
    pi_complete_path: Path,
    *,
    split: str,
    expected_records: int = 480,
) -> dict[str, object]:
    workers = [json.loads(path.read_text(encoding="utf-8")) for path in worker_paths]
    pi = json.loads(pi_complete_path.read_text(encoding="utf-8"))
    if len(workers) != 4 or any(
        worker.get("schema") != "frozen_cpadc_on_pi_panel_worker_v1"
        or worker.get("status") != "complete"
        for worker in workers
    ):
        raise ValueError("four complete CPADC comparison workers are required")
    if any(worker.get("split") != split for worker in workers) or pi.get("split") != split:
        raise ValueError("comparison split mismatch")
    shards = {(int(worker["shard_index"]), int(worker["num_shards"])) for worker in workers}
    if shards != {(index, 4) for index in range(4)}:
        raise ValueError(f"incomplete CPADC shard set: {sorted(shards)}")
    if pi.get("schema") != "frozen_pi_deeponet_complete_split_comparison_v1":
        raise ValueError("unexpected PI complete-report schema")
    if int(pi.get("record_count", 0)) != expected_records:
        raise ValueError("PI complete-report coverage mismatch")

    measurements = [row for worker in workers for row in worker["measurements"]]
    sample_ids = [str(row["sample_id"]) for row in measurements]
    if len(sample_ids) != expected_records or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("CPADC complete comparison coverage failed")
    if any(int(row["frames_per_record"]) != 32 for row in measurements):
        raise ValueError("CPADC comparison must use exactly 32 PI-selected frames")
    cpadc_hashes = {worker["cpadc_checkpoint"]["sha256"] for worker in workers}
    pi_hashes = {worker["pi_panel"]["checkpoint_sha256"] for worker in workers}
    manifest_digests = {worker["pi_panel"]["manifest_digest"] for worker in workers}
    config_hashes = {worker["bindings"]["config_sha256"] for worker in workers}
    if any(len(values) != 1 for values in (cpadc_hashes, pi_hashes, manifest_digests, config_hashes)):
        raise ValueError("worker identity binding mismatch")

    adapted_total = [0.0, 0.0]
    parent_total = [0.0, 0.0]
    adapted_family = {family: [0.0, 0.0] for family in FAMILIES}
    parent_family = {family: [0.0, 0.0] for family in FAMILIES}
    for worker in workers:
        adapted_total = _add(adapted_total, worker["error_terms"]["aggregate"])
        parent_total = _add(parent_total, worker["error_terms"]["parent_aggregate"])
        for family in FAMILIES:
            adapted_family[family] = _add(
                adapted_family[family], worker["error_terms"]["family"][family]
            )
            parent_family[family] = _add(
                parent_family[family], worker["error_terms"]["parent_family"][family]
            )
    family_rows = {
        family: [row for row in measurements if row["family"] == family]
        for family in FAMILIES
    }
    cpadc_global = _relative(adapted_total)
    parent_global = _relative(parent_total)
    pi_global = float(pi["global_energy_relative_l2"])
    cpadc_family = {family: _relative(values) for family, values in adapted_family.items()}
    parent_family_relative = {
        family: _relative(values) for family, values in parent_family.items()
    }
    pi_family = {
        family: float(pi["family_energy_relative_l2"][family]) for family in FAMILIES
    }
    runtimes = [float(row["total_inference_elapsed_s_for_401_frames"]) for row in measurements]
    adaptation_runtimes = [float(row["adaptation_elapsed_s"]) for row in measurements]
    report = {
        "schema": "frozen_cpadc_vs_pi_deeponet_exact_panel_v1",
        "status": "complete_comparison_only",
        "split": split,
        "selection_scope": "same_480_records_same_32_exact_frames_same_energy_relative_l2",
        "record_count": len(measurements),
        "frames_per_record_for_accuracy": 32,
        "family_counts": {family: len(rows) for family, rows in family_rows.items()},
        "global_energy_relative_l2": {
            "cpadc_r7_adapted": cpadc_global,
            "cpadc_parent": parent_global,
            "pi_deeponet": pi_global,
        },
        "family_energy_relative_l2": {
            family: {
                "cpadc_r7_adapted": cpadc_family[family],
                "cpadc_parent": parent_family_relative[family],
                "pi_deeponet": pi_family[family],
            }
            for family in FAMILIES
        },
        "cpadc_vs_pi": {
            "global_relative_error_reduction": (pi_global - cpadc_global) / pi_global,
            "global_pi_error_over_cpadc_error": pi_global / cpadc_global,
            "family_relative_error_reduction": {
                family: (pi_family[family] - cpadc_family[family]) / pi_family[family]
                for family in FAMILIES
            },
            "family_pi_error_over_cpadc_error": {
                family: pi_family[family] / cpadc_family[family] for family in FAMILIES
            },
        },
        "cpadc_instance_adaptation_effect": {
            "global_relative_error_reduction_from_parent": (
                parent_global - cpadc_global
            )
            / parent_global,
            "family_relative_error_reduction_from_parent": {
                family: (parent_family_relative[family] - cpadc_family[family])
                / parent_family_relative[family]
                for family in FAMILIES
            },
            "accepted_fraction": sum(bool(row["adaptation_accepted"]) for row in measurements)
            / len(measurements),
        },
        "record_mean_relative_l2": {
            "cpadc_r7_adapted": sum(float(row["adapted_relative_l2"]) for row in measurements)
            / len(measurements),
            "cpadc_parent": sum(float(row["parent_relative_l2"]) for row in measurements)
            / len(measurements),
            "pi_deeponet": float(pi["record_mean_relative_l2"]),
        },
        "maximum_instance_relative_l2": {
            "cpadc_r7_adapted": max(float(row["adapted_relative_l2"]) for row in measurements),
            "cpadc_parent": max(float(row["parent_relative_l2"]) for row in measurements),
            "pi_deeponet": float(pi["maximum_instance_relative_l2"]),
        },
        "cpadc_runtime_s_for_full_401_frame_output_including_adaptation": {
            "mean": sum(runtimes) / len(runtimes),
            "p95_nearest_rank": _nearest_rank(runtimes, 0.95),
            "maximum": max(runtimes),
        },
        "cpadc_adaptation_runtime_s": {
            "mean": sum(adaptation_runtimes) / len(adaptation_runtimes),
            "p95_nearest_rank": _nearest_rank(adaptation_runtimes, 0.95),
            "maximum": max(adaptation_runtimes),
        },
        "pi_runtime_s_for_32_frame_output": pi["runtime_s_for_32_frames"],
        "runtime_comparison_caveat": (
            "Do not divide the PI and CPADC runtimes: PI materializes 32 frames, while "
            "the archived CPADC deployment path materializes all 401 frames."
        ),
        "worst_cpadc_instances": sorted(
            measurements, key=lambda row: float(row["adapted_relative_l2"]), reverse=True
        )[:20],
        "bindings": {
            "cpadc_checkpoint_sha256": next(iter(cpadc_hashes)),
            "pi_checkpoint_sha256": next(iter(pi_hashes)),
            "manifest_digest": next(iter(manifest_digests)),
            "cpadc_config_sha256": next(iter(config_hashes)),
            "pi_complete_report": str(pi_complete_path.resolve()),
            "pi_complete_report_sha256": _sha256(pi_complete_path),
            "worker_outputs": [str(path.resolve()) for path in worker_paths],
            "worker_sha256": {str(path.resolve()): _sha256(path) for path in worker_paths},
            "aggregation_script_sha256": _sha256(Path(__file__)),
        },
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=Path, nargs="+", required=True)
    parser.add_argument("--pi-complete", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test_id"), required=True)
    parser.add_argument("--expected-records", type=int, default=480)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = aggregate(
        args.workers,
        args.pi_complete,
        split=args.split,
        expected_records=args.expected_records,
    )
    _atomic_json(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
