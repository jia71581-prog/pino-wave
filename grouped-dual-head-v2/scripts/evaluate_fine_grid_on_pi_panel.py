#!/usr/bin/env python3
"""Evaluate the frozen fine-grid solver on a PI worker's exact frame panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for value in (str(SCRIPTS), str(ROOT / "src"), str(ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from gate_lwc84_cuda_graph_fine_grid_trainonly import (
    ALLOWED_FAMILIES,
    _add_terms,
    _atomic_json,
    _decode,
    _error_terms,
    _fine_velocity,
    _read_manifest_rows,
    _relative_l2,
    _sha256,
)
from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig
from fno_acoustic.data_generation.restriction import restrict_nodal_2x
from fno_acoustic.data_generation.solver_lwc84_fused import FusedLWC84CPMLSolver


def _progress_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.progress.json")


def run(args: argparse.Namespace) -> dict:
    pi_worker = json.loads(args.pi_worker.read_text(encoding="utf-8"))
    if pi_worker.get("schema") != "frozen_pi_deeponet_split_worker_v1":
        raise ValueError("input is not a frozen PI-DeepONet worker report")
    if pi_worker.get("status") != "complete" or pi_worker.get("frames_per_record") != 32:
        raise ValueError("PI-DeepONet worker is incomplete or did not use 32 frames")
    split = str(pi_worker["split"])
    if split not in {"validation", "test_id"}:
        raise ValueError("matched comparison requires validation or test_id")
    rows = list(pi_worker["measurements"])
    if not rows:
        raise ValueError("PI-DeepONet worker contains no measurements")
    grid = AcousticGrid(
        nx=401,
        nz=401,
        dx_m=5.0,
        dz_m=5.0,
        lx_m=2000.0,
        lz_m=2000.0,
        centering="node",
    )
    sample_ids = {str(row["sample_id"]) for row in rows}
    manifest_rows = _read_manifest_rows(args.manifest, sample_ids)
    with h5py.File(args.source_h5, "r") as handle:
        output_times_s = np.asarray(handle["time_s"][:], dtype=np.float64)
        for row in rows:
            index = int(row["source_index"])
            if _decode(handle["sample_id"][index]) != row["sample_id"]:
                raise RuntimeError("PI report/HDF5 sample mismatch")
            if _decode(handle["split"][index]) != split:
                raise RuntimeError("PI report/HDF5 split mismatch")
            if _decode(handle["medium_type"][index]) != row["family"]:
                raise RuntimeError("PI report/HDF5 family mismatch")
            indices = [int(value) for value in row["time_indices"]]
            if len(indices) != 32 or indices != sorted(set(indices)):
                raise RuntimeError("PI exact-frame selector changed")

    device = torch.device(args.device)
    solver = FusedLWC84CPMLSolver(
        grid=grid,
        boundaries=BoundaryConfig(npml=40),
        dt_s=0.000625,
        output_times_s=output_times_s,
        c_ref_mps=6750.0,
        device=device,
        dtype=torch.float32,
        kappa_max=3.0,
        minimum_frequency_hz=8.0,
        output_restriction_factor=2,
        cuda_graphs=True,
    )
    warmup_s = solver.warmup(batch=1)
    total_terms = (0.0, 0.0)
    family_terms = {family: (0.0, 0.0) for family in ALLOWED_FAMILIES}
    measurements = []
    progress_path = _progress_path(args.output)
    with h5py.File(args.source_h5, "r") as handle:
        for position, row in enumerate(rows, start=1):
            index = int(row["source_index"])
            family = str(row["family"])
            time_indices = np.asarray(row["time_indices"], dtype=np.int64)
            truth = np.asarray(handle["wavefield"][index, time_indices, :, :])
            stored_velocity = np.asarray(handle["velocity_mps"][index])
            manifest_row = manifest_rows[str(row["sample_id"])]
            fine_velocity, _ = _fine_velocity(
                family=family,
                manifest_row=manifest_row,
                grid=grid,
                marmousi_npy=args.marmousi_npy,
            )
            restricted_velocity = restrict_nodal_2x(fine_velocity)
            restriction_max_abs = float(
                np.max(np.abs(restricted_velocity - stored_velocity))
            )
            if restriction_max_abs != 0.0:
                raise RuntimeError(f"fine velocity mismatch for {row['sample_id']}")
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = solver.simulate(
                fine_velocity,
                source_x_m=float(handle["source_x_m"][index]),
                source_z_m=float(handle["source_z_m"][index]),
                source_f0_hz=float(handle["source_f0_hz"][index]),
                source_t0_s=float(handle["source_t0_s"][index]),
                source_amplitude=float(handle["source_amplitude"][index]),
            )
            torch.cuda.synchronize(device)
            runtime_s = float(time.perf_counter() - started)
            prediction = result.wavefield[0, time_indices]
            terms = _error_terms(prediction, truth)
            total_terms = _add_terms(total_terms, terms)
            family_terms[family] = _add_terms(family_terms[family], terms)
            measurement = {
                "source_index": index,
                "sample_id": str(row["sample_id"]),
                "split": split,
                "family": family,
                "source_f0_hz": float(handle["source_f0_hz"][index]),
                "relative_l2": _relative_l2(terms),
                "error_numerator": terms[0],
                "truth_denominator": terms[1],
                "runtime_s_for_complete_401_frames": runtime_s,
                "time_indices": [int(value) for value in time_indices.tolist()],
                "restriction_max_absolute_difference": restriction_max_abs,
            }
            measurements.append(measurement)
            _atomic_json(
                {
                    "schema": "fine_grid_on_pi_panel_progress_v1",
                    "status": "running",
                    "split": split,
                    "shard_index": int(pi_worker["shard_index"]),
                    "num_shards": int(pi_worker["num_shards"]),
                    "completed_records": position,
                    "total_records": len(rows),
                    "last_measurement": measurement,
                },
                progress_path,
            )
            print(
                f"[{split} shard {pi_worker['shard_index']}] {position}/{len(rows)} "
                f"{row['sample_id']} rel={measurement['relative_l2']:.6g} "
                f"runtime401={runtime_s:.4f}s",
                flush=True,
            )

    runtimes = [row["runtime_s_for_complete_401_frames"] for row in measurements]
    report = {
        "schema": "fine_grid_on_pi_exact_panel_worker_v1",
        "status": "complete",
        "split": split,
        "selection_scope": "same_records_and_32_exact_frames_as_frozen_pi_deeponet",
        "frames_scored_per_record": 32,
        "frames_materialized_per_record": 401,
        "shard_index": int(pi_worker["shard_index"]),
        "num_shards": int(pi_worker["num_shards"]),
        "global_record_count": int(pi_worker["global_record_count"]),
        "record_count": len(measurements),
        "aggregate_relative_l2": _relative_l2(total_terms),
        "error_terms": {
            "aggregate": list(total_terms),
            "family": {family: list(values) for family, values in family_terms.items()},
        },
        "runtime_s_for_complete_401_frames": {
            "minimum": min(runtimes),
            "mean": sum(runtimes) / len(runtimes),
            "maximum": max(runtimes),
            "warmup_excluded": warmup_s,
        },
        "measurements": measurements,
        "bindings": {
            "source_h5_sha256": _sha256(args.source_h5),
            "manifest_sha256": _sha256(args.manifest),
            "marmousi_npy_sha256": _sha256(args.marmousi_npy),
            "fused_solver_sha256": _sha256(
                ROOT / "src/fno_acoustic/data_generation/solver_lwc84_fused.py"
            ),
            "functional_kernel_sha256": _sha256(
                ROOT / "src/fno_acoustic/data_generation/fused_lwc84.py"
            ),
            "comparison_script_sha256": _sha256(Path(__file__)),
            "pi_worker_sha256": _sha256(args.pi_worker),
        },
    }
    _atomic_json(report, args.output)
    _atomic_json(
        {
            "schema": "fine_grid_on_pi_panel_progress_v1",
            "status": "complete",
            "split": split,
            "shard_index": int(pi_worker["shard_index"]),
            "num_shards": int(pi_worker["num_shards"]),
            "completed_records": len(rows),
            "total_records": len(rows),
            "terminal_output": str(args.output.resolve()),
        },
        progress_path,
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--marmousi-npy", type=Path, required=True)
    parser.add_argument("--pi-worker", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
