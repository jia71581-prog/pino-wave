#!/usr/bin/env python3
"""Train-only fine-grid LWC84/CUDA-graph accuracy and runtime probe.

The saved dataset contains the anti-aliased 201x201 velocity.  This probe
reconstructs the exact 401x401 Marmousi crop from the hash-bound prepared
source, verifies that its registered restriction equals the stored velocity,
and changes only the internal time step relative to the frozen teacher.
Validation and test_id records are rejected before their wavefields are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT / "src"), str(ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig
from fno_acoustic.data_generation.restriction import restrict_nodal_2x
from fno_acoustic.data_generation.solver_lwc84_fused import FusedLWC84CPMLSolver
from fno_acoustic.data_generation.velocity_models_lwc84 import load_marmousi_crop


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction64 = prediction.astype(np.float64)
    truth64 = truth.astype(np.float64)
    numerator = float(np.sum((prediction64 - truth64) ** 2))
    denominator = float(np.sum(truth64**2))
    return float(math.sqrt(numerator / max(denominator, 1.0e-30)))


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict:
    grid = AcousticGrid(
        nx=401,
        nz=401,
        dx_m=5.0,
        dz_m=5.0,
        lx_m=2000.0,
        lz_m=2000.0,
        centering="node",
    )
    with h5py.File(args.source_h5, "r") as handle:
        index = int(args.source_index)
        split = _decode(handle["split"][index])
        family = _decode(handle["medium_type"][index])
        if split != "train":
            raise PermissionError(f"probe permits train only, got {split!r}")
        if family != "marmousi":
            raise ValueError(f"fine crop probe expects Marmousi, got {family!r}")
        row = {
            "sample_id": _decode(handle["sample_id"][index]),
            "split": split,
            "family": family,
            "source_x_m": float(handle["source_x_m"][index]),
            "source_z_m": float(handle["source_z_m"][index]),
            "source_f0_hz": float(handle["source_f0_hz"][index]),
            "source_t0_s": float(handle["source_t0_s"][index]),
            "source_amplitude": float(handle["source_amplitude"][index]),
            "crop_x0_m": float(handle["crop_x0_m"][index]),
            "crop_z0_m": float(handle["crop_z0_m"][index]),
            "stored_velocity": np.asarray(handle["velocity_mps"][index]),
            "truth": np.asarray(handle["wavefield"][index]),
            "time_s": np.asarray(handle["time_s"][:], dtype=np.float64),
        }

    fine_velocity, crop_metadata = load_marmousi_crop(
        args.marmousi_npy,
        grid=grid,
        source_dx_m=4.0,
        source_dz_m=4.0,
        source_unit="m/s",
        crop_x0_m=row["crop_x0_m"],
        crop_z0_m=row["crop_z0_m"],
        interpolation="scipy_regular_grid_linear",
    )
    restricted_velocity = restrict_nodal_2x(fine_velocity)
    restriction_max_abs = float(
        np.max(np.abs(restricted_velocity - row["stored_velocity"]))
    )
    if restriction_max_abs != 0.0:
        raise RuntimeError(
            "reconstructed fine velocity does not exactly reproduce stored restriction: "
            f"max_abs={restriction_max_abs}"
        )

    device = torch.device(args.device)
    solver = FusedLWC84CPMLSolver(
        grid=grid,
        boundaries=BoundaryConfig(npml=40),
        dt_s=float(args.internal_dt_s),
        output_times_s=row["time_s"],
        c_ref_mps=6750.0,
        device=device,
        dtype=torch.float32,
        kappa_max=3.0,
        minimum_frequency_hz=8.0,
        output_restriction_factor=2,
        cuda_graphs=True,
    )
    warmup_s = solver.warmup(batch=1)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = solver.simulate(
        fine_velocity,
        source_x_m=row["source_x_m"],
        source_z_m=row["source_z_m"],
        source_f0_hz=row["source_f0_hz"],
        source_t0_s=row["source_t0_s"],
        source_amplitude=row["source_amplitude"],
    )
    torch.cuda.synchronize(device)
    runtime_s = float(time.perf_counter() - started)
    prediction = result.wavefield[0]
    truth = row["truth"]
    relative_l2 = _relative_l2(prediction, truth)
    temporal_bands = {}
    for name, start, stop in (
        ("early", 0, 134),
        ("middle", 134, 267),
        ("late", 267, 401),
    ):
        temporal_bands[name] = _relative_l2(
            prediction[start:stop], truth[start:stop]
        )
    pred64 = prediction.astype(np.float64)
    truth64 = truth.astype(np.float64)
    scalar = float(
        np.sum(pred64 * truth64) / max(float(np.sum(pred64**2)), 1.0e-30)
    )
    aligned_relative_l2 = _relative_l2(scalar * pred64, truth64)
    speed_reference_s = float(args.speed_reference_s)
    report = {
        "schema": "lwc84_cuda_graph_fine_grid_trainonly_probe_v1",
        "status": "complete",
        "selection_scope": "train_only",
        "validation_access": False,
        "test_id_access": False,
        "source_index": int(args.source_index),
        "sample_id": row["sample_id"],
        "family": row["family"],
        "source_f0_hz": row["source_f0_hz"],
        "internal_grid": {
            "shape": [401, 401],
            "spacing_m": 5.0,
            "npml": 40,
            "internal_dt_s": float(args.internal_dt_s),
            "output_restriction": "binomial5_lowpass_then_decimate2",
        },
        "fine_velocity_range_mps": [
            float(fine_velocity.min()),
            float(fine_velocity.max()),
        ],
        "fine_velocity_crop_sha256": crop_metadata["crop_sha256"],
        "restriction_max_absolute_difference": restriction_max_abs,
        "relative_l2": relative_l2,
        "temporal_band_relative_l2": temporal_bands,
        "scalar_alignment_diagnostic": {
            "scale": scalar,
            "relative_l2": aligned_relative_l2,
        },
        "runtime_s": runtime_s,
        "solver_reported_runtime_s": result.metrics[0]["compute_elapsed_s"],
        "warmup_s_excluded": warmup_s,
        "lwc_qmax": result.metrics[0]["lwc_qmax"],
        "speedup_over_traditional_reference": speed_reference_s / runtime_s,
        "promotion_gate": {
            "maximum_relative_l2": float(args.maximum_relative_l2),
            "maximum_runtime_s": speed_reference_s / 10.0,
            "passes_accuracy": bool(relative_l2 <= args.maximum_relative_l2),
            "passes_runtime": bool(runtime_s <= speed_reference_s / 10.0),
            "passed": bool(
                relative_l2 <= args.maximum_relative_l2
                and runtime_s <= speed_reference_s / 10.0
            ),
        },
        "bindings": {
            "source_h5": str(args.source_h5.resolve()),
            "source_h5_sha256": _sha256(args.source_h5),
            "marmousi_npy": str(args.marmousi_npy.resolve()),
            "marmousi_npy_sha256": _sha256(args.marmousi_npy),
            "raw_marmousi_sha256_from_frozen_provenance": (
                "f4302792e84bb7ddbdc9d4d0f963b9df32d9f648960084884127e186119b99dd"
            ),
            "fused_solver_sha256": _sha256(
                ROOT / "src/fno_acoustic/data_generation/solver_lwc84_fused.py"
            ),
            "functional_kernel_sha256": _sha256(
                ROOT / "src/fno_acoustic/data_generation/fused_lwc84.py"
            ),
        },
    }
    _atomic_json(report, args.output)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--marmousi-npy", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=2293)
    parser.add_argument("--internal-dt-s", type=float, default=6.25e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speed-reference-s", type=float, default=20.34137312322855)
    parser.add_argument("--maximum-relative-l2", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["promotion_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
