#!/usr/bin/env python3
"""Benchmark finer-grid LWC-84 on selected train-only tail records.

The source velocity is interpolated from the stored 201x201 deployment grid to
the requested solver grid, simulated with the same 0.000125 s internal step,
and interpolated back to 201x201 before comparison with train-only truth.
Validation and test records are rejected by construction.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F


SCRIPT_PATH = Path(__file__).resolve()
BASE_PATH = SCRIPT_PATH.with_name("build_r25_coarse_residual_cache.py")
SPEC = importlib.util.spec_from_file_location("r30_cache_components", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import cache components: {BASE_PATH}")
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)


def interpolate_field(
    value: np.ndarray,
    *,
    size: int,
    device: torch.device,
    batch_frames: int = 32,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3:
        chunks: list[np.ndarray] = []
        for start in range(0, array.shape[0], int(batch_frames)):
            tensor = torch.from_numpy(array[start : start + int(batch_frames)])
            tensor = tensor[:, None].to(device)
            resized = F.interpolate(
                tensor, size=(int(size), int(size)), mode="bilinear", align_corners=True
            )[:, 0]
            chunks.append(resized.cpu().numpy())
        return np.concatenate(chunks, axis=0)
    if array.ndim == 2:
        tensor = torch.from_numpy(array)[None, None].to(device)
        return (
            F.interpolate(
                tensor, size=(int(size), int(size)), mode="bilinear", align_corners=True
            )[0, 0]
            .cpu()
            .numpy()
        )
    raise ValueError(f"unsupported interpolation shape: {array.shape}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-id", nargs="+", required=True)
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grid_size = int(args.grid_size)
    if grid_size <= base.GRID_SIZE or (grid_size - 1) % 2:
        raise ValueError("grid size must exceed 201 and have an even interval count")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("R30 benchmark requires CUDA")
    torch.cuda.set_device(device)

    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != base.SCHEMA_MANIFEST:
        raise RuntimeError("unexpected manifest schema")
    payload = dict(manifest)
    expected = str(payload.pop("selection_sha256"))
    if base.canonical_json_sha256(payload) != expected:
        raise RuntimeError("manifest digest mismatch")
    source_h5 = Path(str(manifest["source_h5"])).resolve()
    all_rows = list(manifest["fit_records"]) + list(manifest["holdout_records"])
    by_id = {str(row["sample_id"]): row for row in all_rows}
    if len(by_id) != len(all_rows):
        raise RuntimeError("duplicate manifest sample IDs")
    missing = [sample_id for sample_id in args.sample_id if sample_id not in by_id]
    if missing:
        raise RuntimeError(f"sample IDs are outside train-only manifest: {missing}")
    selected_rows = [by_id[sample_id] for sample_id in args.sample_id]
    if any(str(row["split"]) != "train" for row in selected_rows):
        raise RuntimeError("non-train record selected")
    records = base.rows_to_records(selected_rows)

    time_s, x_201, z_201 = base._read_axes(source_h5)
    base._validate_axes(time_s, x_201, z_201, internal_dt_s=base.INTERNAL_DT_S)
    dx = float(x_201[-1] - x_201[0]) / float(grid_size - 1)
    dz = float(z_201[-1] - z_201[0]) / float(grid_size - 1)
    x_fine = float(x_201[0]) + np.arange(grid_size, dtype=np.float64) * dx
    z_fine = float(z_201[0]) + np.arange(grid_size, dtype=np.float64) * dz
    npml = int(round(base.NPML * (grid_size - 1) / (base.GRID_SIZE - 1)))
    base._warm_up_cuda(
        dx_m=dx,
        dz_m=dz,
        internal_dt_s=base.INTERNAL_DT_S,
        npml=npml,
        c_ref_mps=base.C_REF_MPS,
        device=device,
    )
    solver = base.LWC84CPMLSolver(
        grid=base.AcousticGrid(
            nx=grid_size,
            nz=grid_size,
            dx_m=dx,
            dz_m=dz,
            lx_m=dx * float(grid_size - 1),
            lz_m=dz * float(grid_size - 1),
            centering="node",
        ),
        boundaries=base.BoundaryConfig(npml=npml),
        dt_s=base.INTERNAL_DT_S,
        output_times_s=time_s,
        c_ref_mps=base.C_REF_MPS,
        device=device,
        dtype=torch.float32,
        output_restriction_factor=1,
    )

    rows: list[dict] = []
    started = time.perf_counter()
    for record in records:
        velocity_201, source = base._read_generation_inputs(source_h5, [record])
        velocity_fine = interpolate_field(
            velocity_201[0], size=grid_size, device=device
        )[None]
        torch.cuda.synchronize(device)
        solve_started = time.perf_counter()
        result = solver.simulate(
            velocity_fine,
            source_x_m=source["source_x_m"],
            source_z_m=source["source_z_m"],
            source_f0_hz=source["source_f0_hz"],
            source_t0_s=source["source_t0_s"],
            source_amplitude=source["source_amplitude"],
        )
        torch.cuda.synchronize(device)
        solve_seconds = time.perf_counter() - solve_started
        prediction_fine = np.asarray(result.wavefield[0], dtype=np.float32)
        prediction_201 = interpolate_field(
            prediction_fine, size=base.GRID_SIZE, device=device
        )
        predicted_before_truth_ns = time.monotonic_ns()
        with h5py.File(source_h5, "r", swmr=True) as handle:
            truth = np.asarray(
                handle["wavefield"][int(record.source_index), :], dtype=np.float32
            )
        if time.monotonic_ns() <= predicted_before_truth_ns:
            raise RuntimeError("predict-before-truth monotonic audit failed")
        difference = prediction_201.astype(np.float64) - truth.astype(np.float64)
        error_square = float(np.square(difference).sum())
        target_square = float(np.square(truth.astype(np.float64)).sum())
        relative = math.sqrt(error_square / max(target_square, 1.0e-30))
        row = {
            "sample_id": record.sample_id,
            "family": record.medium_type,
            "group_id": record.group_id,
            "source_index": int(record.source_index),
            "source_f0_hz": float(source["source_f0_hz"][0]),
            "candidate_rel_l2": relative,
            "error_square": error_square,
            "target_square": target_square,
            "solver_seconds": solve_seconds,
        }
        rows.append(row)
        print(json.dumps({"event": "R30_RECORD", **row}, sort_keys=True), flush=True)
        del result, prediction_fine, prediction_201, truth, velocity_fine
        torch.cuda.empty_cache()

    values = [float(row["candidate_rel_l2"]) for row in rows]
    result_payload = {
        "schema": "r30_fine_lwc84_tail_benchmark_v1",
        "role": "selected_R28_train_holdout_diagnostic_only",
        "grid": [grid_size, grid_size],
        "spacing_m": [dx, dz],
        "internal_dt_s": base.INTERNAL_DT_S,
        "npml": npml,
        "interpolation": "bilinear_align_corners_true_201_to_fine_to_201",
        "aggregate": {
            "count": len(rows),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "all_lte_0p05": bool(max(values) <= 0.05),
        },
        "records": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "validation_opened": False,
        "test_id_opened": False,
        "script_sha256": base.sha256_file(SCRIPT_PATH),
        "manifest_sha256": base.sha256_file(manifest_path),
        "selection_sha256": expected,
    }
    base.atomic_json(result_payload, args.output.expanduser().resolve())
    print(json.dumps(result_payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
