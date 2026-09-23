#!/usr/bin/env python3
"""Benchmark a single-shot 401x401 LWC-84/CPML solve on the deployment GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT / "src"), str(ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from fno_acoustic.data_generation.config import (
    boundaries_from_config,
    grid_from_config,
    time_from_config,
)
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver
from grouped_ufno_mionet_v3.data.index import ALLOWED_MEDIUM_TYPES, build_manifest
from grouped_ufno_mionet_v3.data.records import V3WavefieldDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _nearest_rank(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("traditional runtime benchmark is empty")
    ordered = sorted(float(value) for value in values)
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(float(fraction) * len(ordered)) - 1),
    )
    return ordered[index]


def _fine_velocity(saved_velocity: torch.Tensor, *, shape: tuple[int, int]) -> np.ndarray:
    value = torch.as_tensor(saved_velocity, dtype=torch.float32)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise ValueError("saved velocity must have one scalar spatial channel")
    value = value[None, None]
    fine = F.interpolate(value, size=shape, mode="bilinear", align_corners=True)
    return fine[0, 0].contiguous().numpy()


def benchmark(
    *,
    source_h5: Path,
    solver_config: Path,
    output: Path,
    device: torch.device,
    repeats_per_family: int,
    warmup_runs: int,
) -> dict[str, object]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("speed certification requires a CUDA deployment device")
    if int(repeats_per_family) <= 0 or int(warmup_runs) < 0:
        raise ValueError("benchmark repeats must be positive and warmups nonnegative")
    config = yaml.safe_load(solver_config.read_text())
    if not isinstance(config, dict):
        raise ValueError("traditional solver config must contain a mapping")
    grid = grid_from_config(config)
    boundaries = boundaries_from_config(config)
    times = time_from_config(config).t_s
    if (grid.nz, grid.nx) != (401, 401) or len(times) != 401:
        raise ValueError("speed reference must use the 401x401 by 401-frame protocol")
    dt_s = float(config["time"]["dt_used_s"])
    if not math.isclose(float(times[-1]), 1.0, abs_tol=1.0e-12):
        raise ValueError("speed reference must cover exactly one propagation second")
    solver = LWC84CPMLSolver(
        grid=grid,
        boundaries=boundaries,
        dt_s=dt_s,
        output_times_s=times,
        c_ref_mps=6750.0,
        device=device,
        dtype=torch.float32,
        kappa_max=float(config["boundaries"]["kappa_max"]),
        minimum_frequency_hz=float(
            config["boundaries"]["minimum_frequency_hz"]
        ),
    )
    manifest = build_manifest(source_h5)
    dataset = V3WavefieldDataset(source_h5, manifest, split="validation")
    selected: dict[str, object] = {}
    for index in range(len(dataset)):
        record = dataset[index]
        if record.medium_type not in selected:
            selected[record.medium_type] = record
        if set(selected) == set(ALLOWED_MEDIUM_TYPES):
            break
    if set(selected) != set(ALLOWED_MEDIUM_TYPES):
        raise ValueError("benchmark selection does not cover every medium family")

    prepared: dict[str, tuple[np.ndarray, dict[str, float | str]]] = {}
    for family, record in selected.items():
        source = record.source_parameters.detach().cpu().numpy()
        prepared[family] = (
            _fine_velocity(record.velocity_mps, shape=(grid.nz, grid.nx)),
            {
                "sample_id": str(record.sample_id),
                "source_x_m": float(source[0]),
                "source_z_m": float(source[1]),
                "source_f0_hz": float(source[2]),
                "source_t0_s": float(source[3]),
                "source_amplitude": float(source[4]),
            },
        )

    def run_one(family: str) -> tuple[float, float]:
        velocity, source = prepared[family]
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = solver.simulate(
            velocity,
            source_x_m=float(source["source_x_m"]),
            source_z_m=float(source["source_z_m"]),
            source_f0_hz=float(source["source_f0_hz"]),
            source_t0_s=float(source["source_t0_s"]),
            source_amplitude=float(source["source_amplitude"]),
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if result.wavefield.shape != (1, 401, 201, 201):
            raise RuntimeError("traditional benchmark returned the wrong output shape")
        compute_elapsed = float(result.metrics[0]["compute_elapsed_s"])
        if not 0.0 < compute_elapsed <= elapsed:
            raise RuntimeError("traditional solver internal timing is inconsistent")
        return compute_elapsed, elapsed

    warmup_families = tuple(ALLOWED_MEDIUM_TYPES)
    for index in range(int(warmup_runs)):
        run_one(warmup_families[index % len(warmup_families)])
    measurements: list[dict[str, object]] = []
    for family in ALLOWED_MEDIUM_TYPES:
        for repeat in range(int(repeats_per_family)):
            elapsed, wrapper_elapsed = run_one(family)
            measurements.append(
                {
                    "family": family,
                    "sample_id": prepared[family][1]["sample_id"],
                    "repeat": repeat,
                    "runtime_s": elapsed,
                    "wrapper_runtime_s": wrapper_elapsed,
                }
            )
    runtimes = [float(row["runtime_s"]) for row in measurements]
    report: dict[str, object] = {
        "status": "complete",
        "schema": "lwc84_traditional_runtime_reference_v1",
        "protocol": {
            "solver": "LWC-84 with three-sided CFS-CPML",
            "solver_grid": [401, 401],
            "saved_grid": [201, 201],
            "saved_frames": 401,
            "propagation_time_s": 1.0,
            "single_instance": True,
            "same_gpu_as_deployment": True,
            "cuda_synchronized_timing": True,
            "includes_output_materialization": True,
            "excludes_disk_io": True,
        },
        "source_h5": str(source_h5.resolve()),
        "source_h5_sha256": _sha256(source_h5),
        "solver_config": str(solver_config.resolve()),
        "solver_config_sha256": _sha256(solver_config),
        "device": {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "warmup_runs": int(warmup_runs),
        "repeats_per_family": int(repeats_per_family),
        "measurements": measurements,
        "minimum_runtime_s": min(runtimes),
        "mean_runtime_s": sum(runtimes) / len(runtimes),
        "p50_runtime_s": _nearest_rank(runtimes, 0.50),
        "p95_runtime_s": _nearest_rank(runtimes, 0.95),
        # The fastest observed traditional solve is the conservative denominator:
        # the learned method must still be ten times faster than this value.
        "conservative_reference_runtime_s": min(runtimes),
    }
    _atomic_json(report, output)
    handle = getattr(dataset, "_h5", None)
    if handle is not None and handle.id.valid:
        handle.close()
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--solver-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats-per-family", type=int, default=2)
    parser.add_argument("--warmup-runs", type=int, default=1)
    args = parser.parse_args(argv)
    report = benchmark(
        source_h5=args.source_h5.resolve(),
        solver_config=args.solver_config.resolve(),
        output=args.output.resolve(),
        device=torch.device(args.device),
        repeats_per_family=int(args.repeats_per_family),
        warmup_runs=int(args.warmup_runs),
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["_nearest_rank", "benchmark"]
