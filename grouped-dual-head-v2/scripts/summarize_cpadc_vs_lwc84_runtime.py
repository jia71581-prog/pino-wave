#!/usr/bin/env python3
"""Summarize the archived proposed deployment path against LWC-84.

This script is deliberately read-only with respect to model and solver execution.  It
combines per-record timing fields already sealed in CPADC evaluation summaries with a
previously completed LWC-84 runtime report.  No GPU kernels, training, prediction, or
future-truth reads are performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nearest_rank(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("runtime percentile requires at least one value")
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("runtime percentile must lie in [0, 1]")
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(float(fraction) * len(ordered)) - 1),
    )
    return ordered[index]


def _runtime_stats(values: list[float]) -> dict[str, float]:
    if not values or not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("runtime values must be finite, nonnegative, and nonempty")
    return {
        "mean_s": fmean(values),
        "p50_s": _nearest_rank(values, 0.50),
        "p95_s": _nearest_rank(values, 0.95),
        "maximum_s": max(values),
    }


def _validate_traditional(report: dict[str, object]) -> None:
    protocol = dict(report.get("protocol", {}))
    required = {
        "status": report.get("status") == "complete",
        "schema": report.get("schema") == "lwc84_traditional_runtime_reference_v1",
        "same_gpu_as_deployment": protocol.get("same_gpu_as_deployment") is True,
        "cuda_synchronized_timing": protocol.get("cuda_synchronized_timing") is True,
        "includes_output_materialization": protocol.get("includes_output_materialization")
        is True,
        "excludes_disk_io": protocol.get("excludes_disk_io") is True,
        "saved_frames_401": int(protocol.get("saved_frames", -1)) == 401,
        "saved_grid_201": list(protocol.get("saved_grid", [])) == [201, 201],
        "solver_grid_401": list(protocol.get("solver_grid", [])) == [401, 401],
    }
    failed = [name for name, passed in required.items() if not passed]
    if failed:
        raise ValueError(f"traditional runtime contract failed: {', '.join(failed)}")


def _summarize_cpadc(
    report: dict[str, object],
    *,
    split: str,
    traditional: dict[str, object],
) -> dict[str, object]:
    records = list(report.get("records", []))
    if not records:
        raise ValueError(f"CPADC {split} summary contains no records")

    adaptation: list[float] = []
    total: list[float] = []
    timing_tags: list[bool] = []
    runtime_protocols: list[str] = []
    for record in records:
        payload = dict(record.get("adaptation", {}))
        adaptation_s = float(payload["adaptation_elapsed_s"])
        total_s = float(payload["total_inference_elapsed_s"])
        if adaptation_s > total_s:
            raise ValueError(f"CPADC {split} adaptation exceeds total latency")
        adaptation.append(adaptation_s)
        total.append(total_s)
        timing_tags.append(payload.get("cuda_synchronized_timing") is True)
        runtime_protocols.append(str(payload.get("runtime_protocol", "")))

    adaptation_stats = _runtime_stats(adaptation)
    total_stats = _runtime_stats(total)
    non_adaptation_stats = _runtime_stats(
        [total_s - adaptation_s for total_s, adaptation_s in zip(total, adaptation)]
    )
    traditional_mean = float(traditional["mean_runtime_s"])
    traditional_minimum = float(traditional["minimum_runtime_s"])
    ratio_of_means = traditional_mean / max(total_stats["mean_s"], 1.0e-12)
    conservative_mean = traditional_minimum / max(total_stats["mean_s"], 1.0e-12)
    conservative_p95 = traditional_minimum / max(total_stats["p95_s"], 1.0e-12)
    zero_adaptation_mean_speedup = traditional_mean / max(
        non_adaptation_stats["mean_s"], 1.0e-12
    )
    tenfold_latency_budget = traditional_minimum / 10.0

    archived_timing_metadata_complete = all(timing_tags) and all(
        value == "input_ready_to_401_frame_output_materialized_no_disk_io_v1"
        for value in runtime_protocols
    )
    return {
        "split": split,
        "record_count": len(records),
        "instance_fine_tuning": adaptation_stats,
        "inference_and_output_materialization": non_adaptation_stats,
        "end_to_end_fine_tuning_plus_inference": total_stats,
        "adaptation_fraction_of_mean_total": adaptation_stats["mean_s"]
        / max(total_stats["mean_s"], 1.0e-12),
        "speedup": {
            "ratio_of_mean_runtimes": ratio_of_means,
            "conservative_mean_using_fastest_traditional": conservative_mean,
            "conservative_p95_using_fastest_traditional": conservative_p95,
            "mean_time_reduction_fraction": 1.0
            - total_stats["mean_s"] / max(traditional_mean, 1.0e-12),
            "mean_speedup_if_adaptation_were_zero": zero_adaptation_mean_speedup,
            "tenfold_latency_budget_using_fastest_traditional_s": tenfold_latency_budget,
            "required_mean_total_reduction_fraction_for_tenfold": 1.0
            - tenfold_latency_budget / max(total_stats["mean_s"], 1.0e-12),
        },
        "archived_timing_metadata_complete": archived_timing_metadata_complete,
        "future_truth_sealed": all(
            dict(record.get("adaptation", {})).get("future_truth_used") is False
            for record in records
        ),
    }


def build_report(
    *,
    validation_path: Path,
    test_id_path: Path,
    traditional_path: Path,
) -> dict[str, object]:
    traditional = json.loads(traditional_path.read_text())
    _validate_traditional(traditional)
    validation = json.loads(validation_path.read_text())
    test_id = json.loads(test_id_path.read_text())
    split_reports = [
        _summarize_cpadc(validation, split="validation", traditional=traditional),
        _summarize_cpadc(test_id, split="test_id", traditional=traditional),
    ]
    all_archived_metadata = all(
        bool(item["archived_timing_metadata_complete"]) for item in split_reports
    )
    all_future_truth_sealed = all(bool(item["future_truth_sealed"]) for item in split_reports)
    return {
        "schema": "proposed_operator_instance_finetune_vs_lwc84_runtime_summary_v1",
        "status": "complete_descriptive_comparison",
        "method": {
            "name": "Ours: parent operator + CPADC instance fine-tuning",
            "archived_lineage": "CPADC R7 with its bound frozen parent operator",
            "online_update": (
                "instance fine-tuning of 16 CPU ridge coefficients; "
                "no neural-weight update"
            ),
            "end_to_end_definition": (
                "input preparation + frozen parent inference + basis generation + "
                "instance fine-tuning + inverse normalization + CPU output materialization"
            ),
            "disk_io_included": False,
            "offline_pretraining_included": False,
        },
        "traditional_reference": {
            "name": "LWC-84 with three-sided CFS-CPML",
            "measurement_count": len(traditional["measurements"]),
            "mean_runtime_s": float(traditional["mean_runtime_s"]),
            "p50_runtime_s": float(traditional["p50_runtime_s"]),
            "p95_runtime_s": float(traditional["p95_runtime_s"]),
            "minimum_runtime_s": float(traditional["minimum_runtime_s"]),
            "protocol": traditional["protocol"],
            "device": traditional["device"],
        },
        "splits": split_reports,
        "claim_gate": {
            "future_truth_sealed": all_future_truth_sealed,
            "traditional_runtime_contract_complete": True,
            "cpadc_archived_timing_metadata_complete": all_archived_metadata,
            "same_case_runtime_pairing": False,
            "matched_absolute_accuracy": False,
            "tenfold_speedup_mean": all(
                float(item["speedup"]["conservative_mean_using_fastest_traditional"])
                >= 10.0
                for item in split_reports
            ),
            "tenfold_speedup_p95": all(
                float(item["speedup"]["conservative_p95_using_fastest_traditional"])
                >= 10.0
                for item in split_reports
            ),
            "speed_superiority_claim_allowed": False,
        },
        "interpretation": (
            "The archived comparison supports a descriptive approximately twofold "
            "latency reduction for the complete proposed deployment path, including "
            "instance fine-tuning and inference. It does not support a "
            "same-accuracy or tenfold-speedup claim because the methods are not paired "
            "on identical cases, CPADC absolute accuracy is not matched to LWC-84, and "
            "the archived CPADC records predate explicit synchronized-timing metadata."
        ),
        "sources": {
            "validation": {"path": str(validation_path), "sha256": _sha256(validation_path)},
            "test_id": {"path": str(test_id_path), "sha256": _sha256(test_id_path)},
            "traditional": {"path": str(traditional_path), "sha256": _sha256(traditional_path)},
        },
    }


def _write_csv(report: dict[str, object], path: Path) -> None:
    traditional = dict(report["traditional_reference"])
    rows: list[dict[str, object]] = [
        {
            "method": traditional["name"],
            "split": "three-family runtime reference",
            "records_or_measurements": traditional["measurement_count"],
            "mean_instance_fine_tuning_s": "",
            "mean_solve_or_inference_and_output_s": traditional["mean_runtime_s"],
            "mean_end_to_end_s": traditional["mean_runtime_s"],
            "p95_end_to_end_s": traditional["p95_runtime_s"],
            "mean_speedup_ratio": 1.0,
            "conservative_p95_speedup": 1.0,
        }
    ]
    for item in report["splits"]:
        rows.append(
            {
                "method": report["method"]["name"],
                "split": item["split"],
                "records_or_measurements": item["record_count"],
                "mean_instance_fine_tuning_s": item["instance_fine_tuning"]["mean_s"],
                "mean_solve_or_inference_and_output_s": item[
                    "inference_and_output_materialization"
                ]["mean_s"],
                "mean_end_to_end_s": item[
                    "end_to_end_fine_tuning_plus_inference"
                ]["mean_s"],
                "p95_end_to_end_s": item[
                    "end_to_end_fine_tuning_plus_inference"
                ]["p95_s"],
                "mean_speedup_ratio": item["speedup"]["ratio_of_mean_runtimes"],
                "conservative_p95_speedup": item["speedup"][
                    "conservative_p95_using_fastest_traditional"
                ],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-summary", required=True, type=Path)
    parser.add_argument("--test-id-summary", required=True, type=Path)
    parser.add_argument("--traditional-runtime", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(
        validation_path=args.validation_summary.resolve(),
        test_id_path=args.test_id_summary.resolve(),
        traditional_path=args.traditional_runtime.resolve(),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_csv(report, args.output_csv)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
