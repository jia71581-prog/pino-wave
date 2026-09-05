#!/usr/bin/env python3
"""Evaluate one deterministic shard of a frozen fine-grid solver split."""
from __future__ import annotations

import argparse
import json
import math
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
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must be in [0, num_shards)")
    if args.maximum_records is not None and args.split != "train":
        raise PermissionError("record limiting is allowed only for train smoke tests")
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
        output_times_s = np.asarray(handle["time_s"][:], dtype=np.float64)
        selected_all = [
            index
            for index in range(handle["split"].shape[0])
            if _decode(handle["split"][index]) == args.split
            and _decode(handle["medium_type"][index]) in ALLOWED_FAMILIES
        ]
        if args.maximum_records is not None:
            selected_all = selected_all[: int(args.maximum_records)]
        selected = [
            index
            for position, index in enumerate(selected_all)
            if position % args.num_shards == args.shard_index
        ]
        if not selected:
            raise ValueError("selected evaluation shard is empty")
        metadata = []
        for index in selected:
            metadata.append(
                {
                    "index": index,
                    "sample_id": _decode(handle["sample_id"][index]),
                    "split": _decode(handle["split"][index]),
                    "family": _decode(handle["medium_type"][index]),
                    "source_x_m": float(handle["source_x_m"][index]),
                    "source_z_m": float(handle["source_z_m"][index]),
                    "source_f0_hz": float(handle["source_f0_hz"][index]),
                    "source_t0_s": float(handle["source_t0_s"][index]),
                    "source_amplitude": float(handle["source_amplitude"][index]),
                }
            )
    manifest_rows = _read_manifest_rows(
        args.manifest, {row["sample_id"] for row in metadata}
    )
    for row in metadata:
        manifest_row = manifest_rows[row["sample_id"]]
        if manifest_row["split"] != args.split:
            raise RuntimeError(f"HDF5/manifest split mismatch for {row['sample_id']}")
        if manifest_row["medium_type"] != row["family"]:
            raise RuntimeError(f"HDF5/manifest family mismatch for {row['sample_id']}")

    device = torch.device(args.device)
    solver = FusedLWC84CPMLSolver(
        grid=grid,
        boundaries=BoundaryConfig(npml=40),
        dt_s=float(args.internal_dt_s),
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
    measurements = []
    total_terms = (0.0, 0.0)
    family_terms = {family: (0.0, 0.0) for family in ALLOWED_FAMILIES}
    temporal_terms = {
        name: (0.0, 0.0) for name in ("early", "middle", "late")
    }
    temporal_ranges = {
        "early": (0, 134),
        "middle": (134, 267),
        "late": (267, 401),
    }
    progress_path = _progress_path(args.output)
    with h5py.File(args.source_h5, "r") as handle:
        for position, row in enumerate(metadata, start=1):
            index = row["index"]
            truth = np.asarray(handle["wavefield"][index])
            stored_velocity = np.asarray(handle["velocity_mps"][index])
            fine_velocity, _ = _fine_velocity(
                family=row["family"],
                manifest_row=manifest_rows[row["sample_id"]],
                grid=grid,
                marmousi_npy=args.marmousi_npy,
            )
            restricted_velocity = restrict_nodal_2x(fine_velocity)
            restriction_max_abs = float(
                np.max(np.abs(restricted_velocity - stored_velocity))
            )
            if restriction_max_abs != 0.0:
                raise RuntimeError(
                    f"fine velocity mismatch for {row['sample_id']}: "
                    f"max_abs={restriction_max_abs}"
                )
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
            terms = _error_terms(prediction, truth)
            total_terms = _add_terms(total_terms, terms)
            family_terms[row["family"]] = _add_terms(
                family_terms[row["family"]], terms
            )
            band_relative = {}
            for name, (start, stop) in temporal_ranges.items():
                band_terms = _error_terms(prediction[start:stop], truth[start:stop])
                temporal_terms[name] = _add_terms(temporal_terms[name], band_terms)
                band_relative[name] = _relative_l2(band_terms)
            measurement = {
                "source_index": index,
                "sample_id": row["sample_id"],
                "split": args.split,
                "family": row["family"],
                "source_f0_hz": row["source_f0_hz"],
                "relative_l2": _relative_l2(terms),
                "error_numerator": terms[0],
                "truth_denominator": terms[1],
                "temporal_band_relative_l2": band_relative,
                "runtime_s": runtime_s,
                "solver_reported_runtime_s": result.metrics[0]["compute_elapsed_s"],
                "lwc_qmax": result.metrics[0]["lwc_qmax"],
                "restriction_max_absolute_difference": restriction_max_abs,
            }
            measurements.append(measurement)
            _atomic_json(
                {
                    "schema": "frozen_fine_grid_split_progress_v1",
                    "status": "running",
                    "split": args.split,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "completed_records": position,
                    "total_records": len(metadata),
                    "last_measurement": measurement,
                },
                progress_path,
            )
            print(
                f"[{args.split} shard {args.shard_index}] {position}/{len(metadata)} "
                f"{row['sample_id']} rel={measurement['relative_l2']:.6g} "
                f"runtime={runtime_s:.4f}s",
                flush=True,
            )

    runtimes = [row["runtime_s"] for row in measurements]
    report = {
        "schema": "frozen_fine_grid_split_worker_v1",
        "status": "complete",
        "split": args.split,
        "selection_scope": f"complete_registered_{args.split}_target_families"
        if args.maximum_records is None
        else "limited_train_smoke",
        "target_families": list(ALLOWED_FAMILIES),
        "excluded_family": "anomaly",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "global_record_count": len(selected_all),
        "record_count": len(measurements),
        "internal_dt_s": float(args.internal_dt_s),
        "error_terms": {
            "aggregate": list(total_terms),
            "family": {family: list(terms) for family, terms in family_terms.items()},
            "temporal": {name: list(terms) for name, terms in temporal_terms.items()},
        },
        "aggregate_relative_l2": _relative_l2(total_terms),
        "family_relative_l2": {
            family: _relative_l2(terms)
            for family, terms in family_terms.items()
            if terms[1] > 0.0
        },
        "runtime_s": {
            "minimum": min(runtimes),
            "mean": float(sum(runtimes) / len(runtimes)),
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
            "gate_helper_sha256": _sha256(
                SCRIPTS / "gate_lwc84_cuda_graph_fine_grid_trainonly.py"
            ),
            "evaluation_script_sha256": _sha256(Path(__file__)),
        },
    }
    _atomic_json(report, args.output)
    _atomic_json(
        {
            "schema": "frozen_fine_grid_split_progress_v1",
            "status": "complete",
            "split": args.split,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "completed_records": len(measurements),
            "total_records": len(metadata),
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
    parser.add_argument("--split", choices=("train", "validation", "test_id"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--maximum-records", type=int)
    parser.add_argument("--internal-dt-s", type=float, default=6.25e-4)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
