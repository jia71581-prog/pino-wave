#!/usr/bin/env python3
"""Same-case runtime benchmark for the frozen fixed-19-Hz position control.

Run ``model`` and ``lwc`` in separate processes on the same idle GPU, then run
``merge`` on CPU. Disk I/O and one-time object/checkpoint construction are
excluded; per-instance transfers, source construction, preprocessing, complete
401-frame inference, synchronization, and CPU output materialization are
included for both methods.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for value in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from fno_acoustic.data_generation.source import bilinear_point_source
from grouped_ufno_mionet_v3.training.checkpoint import load_checkpoint
from saved_time_phase_operator_v4.evaluation import sha256_file
from scripts.evaluate_marmousi_source_position_control import (
    _load_config_context,
    _load_parent_model,
    _reference_solver,
    _validate_reference_binding,
    load_locked_inputs,
    load_protocol,
    locked_cases,
    reconstruct_locked_fine_velocities,
)
from scripts.train_grouped_v3_pilot import load_normalizer


MODEL_SCHEMA = "saved_time_fixed19hz_position_runtime_v1"
LWC_SCHEMA = "lwc84_fixed19hz_position_runtime_v1"
MERGED_SCHEMA = "saved_time_vs_lwc84_fixed19hz_position_runtime_v1"
BENCHMARK_SLICE_RANKS = (1, 15, 30)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with partial.open("x", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def nearest_rank(values: list[float], fraction: float) -> float:
    if not values or not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("runtime percentile requires values and a fraction in [0,1]")
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(float(fraction) * len(ordered)) - 1))
    return ordered[index]


def summarize_runtimes(values: list[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in values):
        raise ValueError("runtime measurements must be finite and positive")
    return {
        "count": len(values),
        "mean_s": float(statistics.fmean(values)),
        "p50_s": nearest_rank(values, 0.50),
        "p95_s": nearest_rank(values, 0.95),
        "minimum_s": min(values),
        "maximum_s": max(values),
    }


def summarize_speedups(values: list[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in values):
        raise ValueError("speedup measurements must be finite and positive")
    return {
        "count": len(values),
        "mean_x": float(statistics.fmean(values)),
        "p50_x": nearest_rank(values, 0.50),
        "p95_x": nearest_rank(values, 0.95),
        "minimum_x": min(values),
        "maximum_x": max(values),
    }


def _device_identity(device: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "name": props.name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": int(props.total_memory),
        "torch": torch.__version__,
        "python": platform.python_version(),
    }


def _benchmark_cases(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in locked_cases(protocol) if int(row["slice_rank"]) in BENCHMARK_SLICE_RANKS]
    rows.sort(key=lambda row: (int(row["slice_rank"]), str(row["case_id"])))
    if len(rows) != 24 or {float(row["source_parameters"][2]) for row in rows} != {19.0}:
        raise ValueError("runtime panel must contain 24 fixed-19-Hz cases")
    return rows


def _timed_cuda_call(call: Callable[[], tuple[tuple[int, ...], float]], device: torch.device):
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    output_shape, output_max_abs = call()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "runtime_s": elapsed,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "output_shape": list(output_shape),
        "output_max_abs": float(output_max_abs),
    }


@torch.inference_mode()
def benchmark_model(
    *,
    config_path: Path,
    checkpoint_path: Path,
    checkpoint_identity_path: Path,
    protocol_path: Path,
    output_path: Path,
    device: torch.device,
    time_block: int,
    warmup_runs: int,
) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("certification runtime is GPU-only")
    protocol = load_protocol(protocol_path)
    cases = _benchmark_cases(protocol)
    config, base, manifest, parent_identity = _load_config_context(config_path)
    identity = json.loads(checkpoint_identity_path.read_text(encoding="utf8"))
    if identity.get("manifest_digest") != manifest.digest:
        raise ValueError("checkpoint identity and manifest disagree")
    model = _load_parent_model(config, base, manifest, parent_identity, device)
    load_checkpoint(
        checkpoint_path,
        model=model,
        expected_manifest_digest=manifest.digest,
        expected_config_digest=str(identity["run_digest"]),
        map_location=device,
    )
    model.eval()
    normalizer = load_normalizer(base, manifest.digest)
    inputs, time_s, x_m, z_m = load_locked_inputs(Path(base.data.source_h5), manifest, protocol)
    by_rank = {rank: [row for row in cases if int(row["slice_rank"]) == rank] for rank in BENCHMARK_SLICE_RANKS}

    def run_case(row: Mapping[str, Any]) -> tuple[tuple[int, ...], float]:
        velocity_cpu = torch.from_numpy(inputs[int(row["slice_rank"])]["velocity_mps"])[None, None]
        source_values = row["source_parameters"]
        source_map_np = bilinear_point_source(
            float(source_values[0]),
            float(source_values[1]),
            nx=len(x_m),
            nz=len(z_m),
            dx_m=float(x_m[1] - x_m[0]),
            dz_m=float(z_m[1] - z_m[0]),
            centering="node",
        ).source_map
        velocity = velocity_cpu.to(device)
        source = torch.tensor(source_values, dtype=torch.float32, device=device)[None]
        source_map = torch.from_numpy(source_map_np)[None, None].to(device)
        medium = model.encode_medium(velocity, normalizer)
        prepared = model.prepare_sources(
            medium,
            source,
            source_map,
            normalizer,
            record_to_medium=torch.zeros(1, dtype=torch.long, device=device),
        )
        normalized = model.dense_normalized(
            prepared,
            torch.from_numpy(time_s.astype(np.float32)).to(device),
            x_m=torch.from_numpy(x_m.astype(np.float32)).to(device),
            z_m=torch.from_numpy(z_m.astype(np.float32)).to(device),
            time_block=int(time_block),
        )
        output = normalizer.decode_pressure(normalized.float(), source[:, 4]).cpu().numpy()
        if output.shape != (1, 401, 201, 201) or not np.isfinite(output).all():
            raise RuntimeError("model runtime output is invalid")
        return tuple(output.shape), float(np.abs(output).max())

    warmup_case = cases[len(cases) // 2]
    for _ in range(int(warmup_runs)):
        run_case(warmup_case)
    measurements: list[dict[str, Any]] = []
    for rank in BENCHMARK_SLICE_RANKS:
        for row in by_rank[rank]:
            measurement = _timed_cuda_call(lambda row=row: run_case(row), device)
            measurements.append(
                {
                    "record_id": row["record_id"],
                    "slice_rank": int(row["slice_rank"]),
                    "case_id": row["case_id"],
                    "role": row["role"],
                    "source_parameters": row["source_parameters"],
                    **measurement,
                }
            )
    payload = {
        "schema": MODEL_SCHEMA,
        "status": "complete",
        "method": "saved_time_operator",
        "protocol_sha256": sha256_file(protocol_path),
        "fixed_source_frequency_hz": 19.0,
        "frequency_generalization_claim_permitted": False,
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_identity_sha256": sha256_file(checkpoint_identity_path),
        "device": _device_identity(device),
        "timing_boundary": {
            "one_time_model_and_checkpoint_load_excluded": True,
            "disk_io_excluded": True,
            "per_instance_host_to_device_transfer_included": True,
            "per_instance_medium_encoding_included": True,
            "cross_source_medium_encoding_reuse": False,
            "source_construction_included": True,
            "travel_and_conditioning_included": True,
            "all_401_frames_materialized_on_cpu": True,
            "cuda_synchronized": True,
        },
        "warmup_runs": int(warmup_runs),
        "slice_ranks": list(BENCHMARK_SLICE_RANKS),
        "summary": summarize_runtimes([float(row["runtime_s"]) for row in measurements]),
        "measurements": measurements,
    }
    _atomic_json(payload, output_path)
    return payload


def benchmark_lwc(
    *, protocol_path: Path, output_path: Path, device: torch.device, warmup_runs: int
) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("certification runtime is GPU-only")
    protocol = load_protocol(protocol_path)
    cases = _benchmark_cases(protocol)
    reference = protocol["reference_solver"]
    from fno_acoustic.data_generation.config import load_config

    config_path = Path(reference["frozen_config"])
    config = load_config(config_path)
    source_h5 = Path(protocol["velocity_panel"]["source_h5"])
    _validate_reference_binding(protocol, config, source_h5)
    fine = reconstruct_locked_fine_velocities(protocol, config, source_h5)
    solver = _reference_solver(config, device)

    def run_case(row: Mapping[str, Any]) -> tuple[tuple[int, ...], float]:
        source = row["source_parameters"]
        result = solver.simulate(
            fine[int(row["slice_rank"])],
            source_x_m=float(source[0]),
            source_z_m=float(source[1]),
            source_f0_hz=float(source[2]),
            source_t0_s=float(source[3]),
            source_amplitude=float(source[4]),
        )
        output = result.wavefield
        if output.shape != (1, 401, 201, 201) or not np.isfinite(output).all():
            raise RuntimeError("LWC runtime output is invalid")
        return tuple(output.shape), float(np.abs(output).max())

    warmup_case = cases[len(cases) // 2]
    for _ in range(int(warmup_runs)):
        run_case(warmup_case)
    measurements = []
    for row in cases:
        measurement = _timed_cuda_call(lambda row=row: run_case(row), device)
        measurements.append(
            {
                "record_id": row["record_id"],
                "slice_rank": int(row["slice_rank"]),
                "case_id": row["case_id"],
                "role": row["role"],
                "source_parameters": row["source_parameters"],
                **measurement,
            }
        )
    payload = {
        "schema": LWC_SCHEMA,
        "status": "complete",
        "method": "LWC-84_CPML",
        "protocol_sha256": sha256_file(protocol_path),
        "fixed_source_frequency_hz": 19.0,
        "frequency_generalization_claim_permitted": False,
        "solver_config_sha256": sha256_file(config_path),
        "device": _device_identity(device),
        "timing_boundary": {
            "one_time_solver_construction_excluded": True,
            "disk_io_excluded": True,
            "per_instance_host_to_device_transfer_included": True,
            "source_construction_included": True,
            "all_401_frames_materialized_on_cpu": True,
            "cuda_synchronized": True,
        },
        "warmup_runs": int(warmup_runs),
        "slice_ranks": list(BENCHMARK_SLICE_RANKS),
        "summary": summarize_runtimes([float(row["runtime_s"]) for row in measurements]),
        "measurements": measurements,
    }
    _atomic_json(payload, output_path)
    return payload


def merge_reports(
    model_report: Mapping[str, Any],
    lwc_report: Mapping[str, Any],
    score_report: Mapping[str, Any],
) -> dict[str, Any]:
    if model_report.get("schema") != MODEL_SCHEMA or lwc_report.get("schema") != LWC_SCHEMA:
        raise ValueError("unexpected runtime report schema")
    if model_report.get("status") != "complete" or lwc_report.get("status") != "complete":
        raise ValueError("runtime reports are incomplete")
    if model_report.get("protocol_sha256") != lwc_report.get("protocol_sha256"):
        raise ValueError("runtime protocols differ")
    if model_report.get("device") != lwc_report.get("device"):
        raise ValueError("runtime devices or software environments differ")
    model_rows = {row["record_id"]: row for row in model_report["measurements"]}
    lwc_rows = {row["record_id"]: row for row in lwc_report["measurements"]}
    if set(model_rows) != set(lwc_rows) or len(model_rows) != 24:
        raise ValueError("runtime cases are not exactly paired")
    for record_id in model_rows:
        for name in ("slice_rank", "case_id", "role", "source_parameters", "output_shape"):
            if model_rows[record_id].get(name) != lwc_rows[record_id].get(name):
                raise ValueError(f"runtime case mismatch for {record_id}: {name}")
    if (
        score_report.get("status") != "complete"
        or score_report.get("source_generalization_variable") != "position_only"
        or score_report.get("frequency_generalization_claim_permitted") is not False
        or score_report.get("protocol_sha256") != model_report.get("protocol_sha256")
    ):
        raise ValueError("accuracy score is not the matching fixed-position evaluation")
    aggregate = float(score_report["overall"]["record_relative_l2"]["mean"])
    role_values = {
        role: float(values["record_relative_l2"]["mean"])
        for role, values in score_report["by_position_role"].items()
    }
    finite_mean = float(score_report["overall"]["prediction_finite"]["mean"])
    nonzero_mean = float(score_report["overall"]["prediction_nonzero"]["mean"])
    accuracy_gate = (
        aggregate <= 0.05
        and set(role_values) == {"interpolation", "outside_train_position_range"}
        and all(value <= 0.08 for value in role_values.values())
        and finite_mean == 1.0
        and nonzero_mean == 1.0
    )
    model_times = [float(row["runtime_s"]) for row in model_rows.values()]
    lwc_times = [float(row["runtime_s"]) for row in lwc_rows.values()]
    paired_speedups = [
        float(lwc_rows[record_id]["runtime_s"]) / float(model_rows[record_id]["runtime_s"])
        for record_id in sorted(model_rows)
    ]
    mean_speedup = statistics.fmean(lwc_times) / statistics.fmean(model_times)
    conservative_p95_speedup = min(lwc_times) / nearest_rank(model_times, 0.95)
    speed_gate = mean_speedup >= 10.0 and conservative_p95_speedup >= 10.0
    return {
        "schema": MERGED_SCHEMA,
        "status": "complete",
        "protocol_sha256": model_report["protocol_sha256"],
        "fixed_source_frequency_hz": 19.0,
        "frequency_generalization_claim_permitted": False,
        "same_device_and_software": True,
        "paired_case_count": 24,
        "model_summary": summarize_runtimes(model_times),
        "lwc_summary": summarize_runtimes(lwc_times),
        "paired_speedup_summary": summarize_speedups(paired_speedups),
        "mean_runtime_speedup": float(mean_speedup),
        "conservative_p95_latency_speedup": float(conservative_p95_speedup),
        "accuracy_gate": {
            "aggregate_relative_l2": aggregate,
            "maximum_aggregate_relative_l2": 0.05,
            "position_role_relative_l2": role_values,
            "maximum_each_position_role_relative_l2": 0.08,
            "all_predictions_finite": finite_mean == 1.0,
            "all_predictions_nonzero": nonzero_mean == 1.0,
            "passed": bool(accuracy_gate),
        },
        "speed_gate": {
            "minimum_mean_speedup": 10.0,
            "minimum_conservative_p95_latency_speedup": 10.0,
            "passed": bool(speed_gate),
        },
        "matched_accuracy_speed_claim_permitted": bool(accuracy_gate and speed_gate),
        "claim_boundary": (
            "faster at the frozen reported accuracy on this hardware and 401-frame protocol only"
            if accuracy_gate and speed_gate
            else "runtime may be reported descriptively; a matched-accuracy speed claim is forbidden"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    model = sub.add_parser("model")
    model.add_argument("--config", type=Path, required=True)
    model.add_argument("--checkpoint", type=Path, required=True)
    model.add_argument("--checkpoint-identity", type=Path, required=True)
    model.add_argument("--protocol", type=Path, required=True)
    model.add_argument("--output", type=Path, required=True)
    model.add_argument("--device", default="cuda:0")
    model.add_argument("--time-block", type=int, default=16)
    model.add_argument("--warmup-runs", type=int, default=2)
    lwc = sub.add_parser("lwc")
    lwc.add_argument("--protocol", type=Path, required=True)
    lwc.add_argument("--output", type=Path, required=True)
    lwc.add_argument("--device", default="cuda:0")
    lwc.add_argument("--warmup-runs", type=int, default=1)
    merge = sub.add_parser("merge")
    merge.add_argument("--model-report", type=Path, required=True)
    merge.add_argument("--lwc-report", type=Path, required=True)
    merge.add_argument("--score-report", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "model":
        payload = benchmark_model(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            checkpoint_identity_path=args.checkpoint_identity,
            protocol_path=args.protocol,
            output_path=args.output,
            device=torch.device(args.device),
            time_block=args.time_block,
            warmup_runs=args.warmup_runs,
        )
    elif args.command == "lwc":
        payload = benchmark_lwc(
            protocol_path=args.protocol,
            output_path=args.output,
            device=torch.device(args.device),
            warmup_runs=args.warmup_runs,
        )
    else:
        payload = merge_reports(
            json.loads(args.model_report.read_text(encoding="utf8")),
            json.loads(args.lwc_report.read_text(encoding="utf8")),
            json.loads(args.score_report.read_text(encoding="utf8")),
        )
        _atomic_json(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["merge_reports", "nearest_rank", "summarize_runtimes", "summarize_speedups"]
