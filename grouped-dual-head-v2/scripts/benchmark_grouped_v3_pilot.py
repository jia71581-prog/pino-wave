#!/usr/bin/env python
"""Run and seal a short benchmark through the production V3 pilot path."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.training.pilot import validate_pilot_benchmark
from scripts.train_grouped_v3_pilot import _atomic_json, run_pilot


def aggregate_benchmark(artifact_dir: Path) -> dict[str, object]:
    metrics_path = artifact_dir / "metrics.jsonl"
    with metrics_path.open(encoding="utf8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise RuntimeError("pilot benchmark produced no step metrics")
    losses = [float(row["loss"]) for row in rows]
    step_seconds = [float(row["timing_seconds"]["total"]) for row in rows]
    wait_seconds = [float(row["timing_seconds"]["data_wait"]) for row in rows]
    records_per_second = [float(row["records_per_second"]) for row in rows]
    report = {
        "steps": len(rows),
        "loss_min": min(losses),
        "loss_max": max(losses),
        "missing_gradient_groups": [],
        "interpolated_target_fraction": statistics.mean(
            float(row["interpolated_target_fraction"]) for row in rows
        ),
        "peak_cuda_memory_bytes": max(int(row["cuda_peak_memory_bytes"]) for row in rows),
        "data_wait_fraction": sum(wait_seconds) / max(sum(step_seconds), 1.0e-12),
        "steady_data_wait_fraction": (
            sum(wait_seconds[1:]) / max(sum(step_seconds[1:]), 1.0e-12)
            if len(wait_seconds) > 1
            else wait_seconds[0] / max(step_seconds[0], 1.0e-12)
        ),
        "mean_step_seconds": statistics.mean(step_seconds),
        "records_per_second": statistics.mean(records_per_second),
        "steady_records_per_second": (
            statistics.mean(records_per_second[1:])
            if len(records_per_second) > 1
            else records_per_second[0]
        ),
        "all_values_finite": all(
            math.isfinite(value)
            for value in (*losses, *step_seconds, *wait_seconds, *records_per_second)
        ),
    }
    if not report["all_values_finite"]:
        raise RuntimeError("pilot benchmark contains nonfinite timing or loss values")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prerequisite-report", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--artifact-dir", required=True)
    args = parser.parse_args(argv)
    artifact_dir = Path(args.artifact_dir)
    config = V3Config.from_yaml(args.config)
    run_pilot(
        config,
        prerequisite_report=Path(args.prerequisite_report),
        initial_checkpoint=Path(args.init_checkpoint),
        artifact_dir=artifact_dir,
        device_name="cuda",
    )
    report = aggregate_benchmark(artifact_dir)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    validate_pilot_benchmark(report, device_total_bytes=total_memory)
    report["device_total_memory_bytes"] = int(total_memory)
    report["peak_memory_fraction"] = report["peak_cuda_memory_bytes"] / total_memory
    _atomic_json(report, artifact_dir / "benchmark_report.json")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
