#!/usr/bin/env python3
"""Generate and seal one deterministic shard of the 480x401 LWC-84 panel."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver
from saved_time_phase_operator_v4.coarse_lwc84 import (
    EXPECTED_VALIDATION_FAMILY_COUNTS,
    CoarseRecordIndex,
    SealedPrediction,
    complete_field_relative_l2,
    seal_prediction,
    select_all_validation_records,
    shard_records,
)
from scripts.evaluate_coarse_lwc84_201 import (
    _write_json_atomic,
    assert_uniform_axis,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--solver-batch-size", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--internal-dt-s", type=float, default=2.5e-4)
    parser.add_argument("--npml", type=int, default=20)
    parser.add_argument("--c-ref-mps", type=float, default=6750.0)
    parser.add_argument("--metric-block-size", type=int, default=20)
    return parser


def _canonicalize_device(device: str | torch.device) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda" and requested.index is None:
        return torch.device("cuda:0")
    return requested


def _source_identity(source_h5: Path) -> dict[str, object]:
    with h5py.File(source_h5, "r", swmr=True) as handle:
        return {
            "path": str(source_h5),
            "byte_count": source_h5.stat().st_size,
            "manifest_sha256": str(handle.attrs.get("manifest_sha256", "")),
            "config_sha256": str(handle.attrs.get("config_sha256", "")),
        }


def _read_axes(source_h5: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(source_h5, "r", swmr=True) as handle:
        time_s = np.asarray(handle["time_s"][:], dtype=np.float64)
        x_m = np.asarray(handle["x_m"][:], dtype=np.float64)
        z_m = np.asarray(handle["z_m"][:], dtype=np.float64)
    return time_s, x_m, z_m


def _validate_axes(
    time_s: np.ndarray,
    x_m: np.ndarray,
    z_m: np.ndarray,
    *,
    internal_dt_s: float,
) -> tuple[float, float, float]:
    stored_dt_s = assert_uniform_axis(time_s, expected_step=None, name="time_s")
    dx_m = assert_uniform_axis(x_m, expected_step=None, name="x_m")
    dz_m = assert_uniform_axis(z_m, expected_step=None, name="z_m")
    if not np.isclose(time_s[0], 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("stored time axis must start at zero")
    if not np.isclose(x_m[0], 0.0, rtol=0.0, atol=1.0e-12) or not np.isclose(
        z_m[0], 0.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("coarse solver axes must start at zero")
    if len(time_s) not in (3, 401):
        raise ValueError("shard worker requires 401 production times or 3 test times")
    if len(time_s) == 401 and (len(x_m), len(z_m)) != (201, 201):
        raise ValueError("production shard requires a 201 by 201 grid")
    if not np.isclose(dx_m, dz_m, rtol=0.0, atol=1.0e-9):
        raise ValueError("LWC84 shard requires equal x/z spacing")
    ratios = time_s / float(internal_dt_s)
    if not np.allclose(ratios, np.rint(ratios), rtol=0.0, atol=1.0e-9):
        raise ValueError("stored times do not align to internal_dt_s")
    return stored_dt_s, dx_m, dz_m


def _numerical_contract(
    *,
    time_s: np.ndarray,
    x_m: np.ndarray,
    z_m: np.ndarray,
    stored_dt_s: float,
    dx_m: float,
    dz_m: float,
    internal_dt_s: float,
    npml: int,
    c_ref_mps: float,
    device: torch.device,
) -> dict[str, object]:
    return {
        "grid_shape": [len(z_m), len(x_m)],
        "dx_m": dx_m,
        "dz_m": dz_m,
        "stored_time_count": len(time_s),
        "stored_dt_s": stored_dt_s,
        "internal_dt_s": float(internal_dt_s),
        "npml": int(npml),
        "c_ref_mps": float(c_ref_mps),
        "dtype": "float32",
        "device_type": device.type,
        "output_restriction_factor": 1,
        "top_boundary": "free_surface_dirichlet",
        "other_boundaries": "cpml",
    }


def _completed_summary(
    output_dir: Path,
    *,
    records: Sequence[CoarseRecordIndex],
    source_identity: Mapping[str, object],
    numerical_contract: Mapping[str, object],
) -> dict[str, object] | None:
    summary_path = output_dir / "shard_summary.json"
    rows_path = output_dir / "records.json"
    if not summary_path.is_file():
        return None
    if not rows_path.is_file():
        raise RuntimeError("completed shard summary lacks records.json")
    summary = json.loads(summary_path.read_text())
    rows = json.loads(rows_path.read_text())
    expected_ids = [record.sample_id for record in records]
    if summary.get("status") != "complete":
        raise RuntimeError("existing shard summary is not complete")
    if summary.get("source_identity") != dict(source_identity):
        raise RuntimeError("existing shard source identity mismatch")
    if summary.get("numerical_contract") != dict(numerical_contract):
        raise RuntimeError("existing shard numerical contract mismatch")
    if summary.get("sample_ids") != expected_ids or len(rows) != len(records):
        raise RuntimeError("existing shard record set mismatch")
    for row in rows:
        path = Path(str(row["prediction_path"]))
        if not path.is_file() or path.stat().st_size != int(row["prediction_byte_count"]):
            raise RuntimeError(f"sealed prediction is missing or truncated: {path}")
    return summary


def _read_generation_inputs(
    source_h5: Path,
    records: Sequence[CoarseRecordIndex],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    indices = [record.source_index for record in records]
    with h5py.File(source_h5, "r", swmr=True) as handle:
        velocity = np.stack(
            [np.asarray(handle["velocity_mps"][index], dtype=np.float32) for index in indices]
        )
        source = {
            name: np.asarray([handle[name][index] for index in indices], dtype=np.float64)
            for name in (
                "source_x_m",
                "source_z_m",
                "source_f0_hz",
                "source_t0_s",
                "source_amplitude",
            )
        }
    return velocity, source


def _validate_solver_batch(
    result,
    velocity: np.ndarray,
    *,
    records: Sequence[CoarseRecordIndex],
    stored_time_count: int,
) -> None:
    expected_shape = (
        len(records),
        stored_time_count,
        velocity.shape[-2],
        velocity.shape[-1],
    )
    if result.wavefield.shape != expected_shape:
        raise ValueError(f"solver batch shape {result.wavefield.shape} != {expected_shape}")
    if not np.isfinite(result.wavefield).all():
        raise FloatingPointError("solver batch contains non-finite pressure")
    if float(np.max(np.abs(result.wavefield[:, :, 0, :]))) != 0.0:
        raise ValueError("solver batch violates the pressure-free top")
    if not np.allclose(
        result.source_map_saved.sum(axis=(1, 2)), 1.0, rtol=0.0, atol=2.0e-7
    ):
        raise ValueError("solver batch source maps are not conservative")
    if not np.array_equal(result.velocity_saved_mps, velocity):
        raise ValueError("factor-one solver changed saved velocity")
    if len(result.metrics) != len(records):
        raise ValueError("solver QC does not contain one row per record")
    for record, metric in zip(records, result.metrics, strict=True):
        if not bool(metric["finite"]) or float(metric["lwc_qmax"]) >= 1.0:
            raise FloatingPointError(
                f"unstable solver symbol for {record.sample_id}: {metric}"
            )


def _warm_up_cuda(
    *,
    dx_m: float,
    dz_m: float,
    internal_dt_s: float,
    npml: int,
    c_ref_mps: float,
    device: torch.device,
) -> None:
    solver = LWC84CPMLSolver(
        grid=AcousticGrid(
            nx=41,
            nz=41,
            dx_m=dx_m,
            dz_m=dz_m,
            lx_m=40.0 * dx_m,
            lz_m=40.0 * dz_m,
            centering="node",
        ),
        boundaries=BoundaryConfig(npml=min(int(npml), 8)),
        dt_s=float(internal_dt_s),
        output_times_s=np.asarray([0.0, internal_dt_s], dtype=np.float64),
        c_ref_mps=float(c_ref_mps),
        device=device,
        dtype=torch.float32,
        output_restriction_factor=1,
    )
    solver.simulate(
        np.full((41, 41), min(2000.0, c_ref_mps), dtype=np.float32),
        source_x_m=20.0 * dx_m,
        source_z_m=10.0 * dz_m,
        source_f0_hz=20.0,
    )
    torch.cuda.synchronize(device)


def run_shard(
    *,
    source_h5: str | Path,
    output_dir: str | Path,
    shard_index: int,
    shard_count: int = 4,
    solver_batch_size: int = 120,
    device: str = "cuda",
    internal_dt_s: float = 2.5e-4,
    npml: int = 20,
    c_ref_mps: float = 6750.0,
    metric_block_size: int = 20,
    expected_family_counts: Mapping[str, int] = EXPECTED_VALIDATION_FAMILY_COUNTS,
) -> dict[str, object]:
    source_path = Path(source_h5).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if min(
        int(solver_batch_size),
        int(metric_block_size),
        int(npml),
    ) < 1:
        raise ValueError("batch size, metric block size, and npml must be positive")
    if float(internal_dt_s) <= 0.0 or float(c_ref_mps) <= 0.0:
        raise ValueError("internal_dt_s and c_ref_mps must be positive")
    requested_device = _canonicalize_device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")

    all_records = select_all_validation_records(
        source_path, expected_family_counts=expected_family_counts
    )
    records = shard_records(
        all_records, shard_index=shard_index, shard_count=shard_count
    )
    if not records:
        raise ValueError("shard is empty")
    time_s, x_m, z_m = _read_axes(source_path)
    stored_dt_s, dx_m, dz_m = _validate_axes(
        time_s, x_m, z_m, internal_dt_s=float(internal_dt_s)
    )
    identity = _source_identity(source_path)
    numerics = _numerical_contract(
        time_s=time_s,
        x_m=x_m,
        z_m=z_m,
        stored_dt_s=stored_dt_s,
        dx_m=dx_m,
        dz_m=dz_m,
        internal_dt_s=float(internal_dt_s),
        npml=int(npml),
        c_ref_mps=float(c_ref_mps),
        device=requested_device,
    )
    completed = _completed_summary(
        destination,
        records=records,
        source_identity=identity,
        numerical_contract=numerics,
    )
    if completed is not None:
        print(
            json.dumps(
                {"event": "shard_reused", "shard_index": shard_index},
                sort_keys=True,
            ),
            flush=True,
        )
        return completed

    destination.mkdir(parents=True, exist_ok=True)
    prediction_dir = destination / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "coarse_lwc84_201_sealed_shard_manifest_v1",
        "status": "planned",
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "solver_batch_size": int(solver_batch_size),
        "source_identity": identity,
        "numerical_contract": numerics,
        "records": [asdict(record) for record in records],
    }
    _write_json_atomic(destination / "manifest.json", manifest)

    if requested_device.type == "cuda":
        torch.cuda.set_device(requested_device)
        torch.cuda.reset_peak_memory_stats(requested_device)
        _warm_up_cuda(
            dx_m=dx_m,
            dz_m=dz_m,
            internal_dt_s=float(internal_dt_s),
            npml=int(npml),
            c_ref_mps=float(c_ref_mps),
            device=requested_device,
        )
    grid = AcousticGrid(
        nx=len(x_m),
        nz=len(z_m),
        dx_m=dx_m,
        dz_m=dz_m,
        lx_m=float(x_m[-1] - x_m[0]),
        lz_m=float(z_m[-1] - z_m[0]),
        centering="node",
    )
    boundaries = BoundaryConfig(npml=int(npml))
    solver = LWC84CPMLSolver(
        grid=grid,
        boundaries=boundaries,
        dt_s=float(internal_dt_s),
        output_times_s=time_s,
        c_ref_mps=float(c_ref_mps),
        device=requested_device,
        dtype=torch.float32,
        output_restriction_factor=1,
    )
    pending: dict[
        str,
        tuple[CoarseRecordIndex, SealedPrediction, dict[str, object]],
    ] = {}
    wall_start = time.perf_counter()
    solver_seconds = 0.0
    sealing_seconds = 0.0
    for start in range(0, len(records), int(solver_batch_size)):
        batch_records = records[start : start + int(solver_batch_size)]
        velocity, source = _read_generation_inputs(source_path, batch_records)
        if requested_device.type == "cuda":
            torch.cuda.synchronize(requested_device)
        solve_start = time.perf_counter()
        result = solver.simulate(
            velocity,
            source_x_m=source["source_x_m"],
            source_z_m=source["source_z_m"],
            source_f0_hz=source["source_f0_hz"],
            source_t0_s=source["source_t0_s"],
            source_amplitude=source["source_amplitude"],
        )
        if requested_device.type == "cuda":
            torch.cuda.synchronize(requested_device)
        batch_solver_seconds = time.perf_counter() - solve_start
        solver_seconds += batch_solver_seconds
        _validate_solver_batch(
            result,
            velocity,
            records=batch_records,
            stored_time_count=len(time_s),
        )
        seal_start = time.perf_counter()
        for local_index, record in enumerate(batch_records):
            qc = dict(result.metrics[local_index])
            sealed = seal_prediction(
                prediction_dir / f"{record.sample_id}.pt",
                torch.from_numpy(result.wavefield[local_index : local_index + 1]),
                metadata={
                    "record": asdict(record),
                    "source_identity": identity,
                    "numerical_contract": numerics,
                    "solver_qc": qc,
                },
            )
            pending[record.sample_id] = (record, sealed, qc)
        batch_sealing_seconds = time.perf_counter() - seal_start
        sealing_seconds += batch_sealing_seconds
        print(
            json.dumps(
                {
                    "event": "batch_complete",
                    "shard_index": int(shard_index),
                    "batch_start": start,
                    "batch_records": len(batch_records),
                    "solver_seconds": batch_solver_seconds,
                    "sealing_seconds": batch_sealing_seconds,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del result, velocity, source
        gc.collect()

    last_seal_ns = time.monotonic_ns()
    if len(pending) != len(records):
        raise RuntimeError("not every shard prediction was sealed")
    print(
        json.dumps(
            {
                "event": "all_predictions_sealed",
                "shard_index": int(shard_index),
                "record_count": len(records),
                "monotonic_ns": last_seal_ns,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    first_truth_ns = time.monotonic_ns()
    if first_truth_ns <= last_seal_ns:
        raise RuntimeError("monotonic seal/truth ordering failed")
    metric_start = time.perf_counter()
    record_rows: list[dict[str, object]] = []
    for record in records:
        _, sealed, qc = pending[record.sample_id]
        payload = torch.load(sealed.path, map_location="cpu", weights_only=False)
        if not bool(payload.get("sealed")):
            raise RuntimeError(f"prediction is not sealed: {sealed.path}")
        prediction = torch.as_tensor(payload["wavefield"])
        with h5py.File(source_path, "r", swmr=True) as handle:
            truth = torch.from_numpy(
                np.asarray(handle["wavefield"][record.source_index], dtype=np.float32)
            )[None]
        relative = complete_field_relative_l2(
            prediction,
            truth,
            block_size=int(metric_block_size),
        )
        row = {
            "source_index": record.source_index,
            "sample_id": record.sample_id,
            "group_id": record.group_id,
            "sample_sha256": record.sample_sha256,
            "family": record.medium_type,
            "relative_l2": relative,
            "stored_time_count": len(time_s),
            "solver_qc": qc,
            "prediction_path": str(sealed.path),
            "prediction_sha256": sealed.sha256,
            "prediction_byte_count": sealed.byte_count,
            "last_prediction_sealed_monotonic_ns": last_seal_ns,
            "first_truth_opened_monotonic_ns": first_truth_ns,
            "truth_opened_after_all_shard_predictions_sealed": first_truth_ns
            > last_seal_ns,
        }
        record_rows.append(row)
        print(
            json.dumps(
                {
                    "event": "record_metric",
                    "shard_index": int(shard_index),
                    "sample_id": record.sample_id,
                    "family": record.medium_type,
                    "relative_l2": relative,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del payload, prediction, truth
    metric_seconds = time.perf_counter() - metric_start
    _write_json_atomic(destination / "records.json", record_rows)
    family_counts = {
        family: sum(record.medium_type == family for record in records)
        for family in EXPECTED_VALIDATION_FAMILY_COUNTS
    }
    summary = {
        "schema": "coarse_lwc84_201_sealed_shard_summary_v1",
        "status": "complete",
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "solver_batch_size": int(solver_batch_size),
        "record_count": len(records),
        "sample_ids": [record.sample_id for record in records],
        "family_counts": family_counts,
        "source_identity": identity,
        "numerical_contract": numerics,
        "prediction_file_count": len(pending),
        "prediction_byte_count": sum(item[1].byte_count for item in pending.values()),
        "last_prediction_sealed_monotonic_ns": last_seal_ns,
        "first_truth_opened_monotonic_ns": first_truth_ns,
        "truth_opened_after_all_shard_predictions_sealed": first_truth_ns
        > last_seal_ns,
        "runtime": {
            "solver_seconds": solver_seconds,
            "sealing_seconds": sealing_seconds,
            "metric_seconds": metric_seconds,
            "wall_seconds": time.perf_counter() - wall_start,
        },
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(requested_device))
            if requested_device.type == "cuda"
            else 0
        ),
    }
    _write_json_atomic(destination / "shard_summary.json", summary)
    print(
        json.dumps(
            {
                "event": "shard_complete",
                "shard_index": int(shard_index),
                "record_count": len(records),
                "wall_seconds": summary["runtime"]["wall_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return summary


def main() -> None:
    args = build_parser().parse_args()
    run_shard(
        source_h5=args.source_h5,
        output_dir=args.output_dir,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        solver_batch_size=args.solver_batch_size,
        device=args.device,
        internal_dt_s=args.internal_dt_s,
        npml=args.npml,
        c_ref_mps=args.c_ref_mps,
        metric_block_size=args.metric_block_size,
    )


if __name__ == "__main__":
    main()
