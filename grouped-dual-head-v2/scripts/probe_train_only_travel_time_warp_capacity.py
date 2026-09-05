#!/usr/bin/env python
"""Train-only oracle capacity probe for causal travel-time warping."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from fno_acoustic.data_generation.hdf5_lwc84 import _sample_sha256
from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _list(value: str, converter) -> tuple:
    result = tuple(converter(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("registered list must be nonempty")
    return result


def _relative_l2(prediction: torch.Tensor, truth: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(prediction.double() - truth.double()).item()
        / max(torch.linalg.vector_norm(truth.double()).item(), 1.0e-30)
    )


def _travel_warp(
    wavefield: torch.Tensor,
    travel_time_s: torch.Tensor,
    time_s: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Evaluate p(t + scale * T(x,z), x,z) by linear interpolation."""

    nt, nz, nx = wavefield.shape
    dt = float(time_s[1] - time_s[0])
    requested = (
        time_s[:, None, None] + float(scale) * travel_time_s[None]
    ) / dt
    low = torch.floor(requested).long().clamp(0, nt - 1)
    high = (low + 1).clamp(0, nt - 1)
    weight = (requested - low.to(requested.dtype)).clamp(0.0, 1.0)
    flat = wavefield.reshape(nt, -1)
    low_values = torch.gather(flat, 0, low.reshape(nt, -1)).reshape(nt, nz, nx)
    high_values = torch.gather(flat, 0, high.reshape(nt, -1)).reshape(nt, nz, nx)
    warped = low_values + weight * (high_values - low_values)
    outside = (requested < 0.0) | (requested > float(nt - 1))
    return torch.where(outside, torch.zeros_like(warped), warped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--travel-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-indices", required=True)
    parser.add_argument("--expected-sample-sha256", required=True)
    parser.add_argument("--warp-scales", required=True)
    parser.add_argument("--route-threshold-hz", type=float, default=24.0)
    parser.add_argument("--candidate-fraction", type=float, default=0.45)
    parser.add_argument("--internal-dt-s", type=float, default=2.5e-4)
    parser.add_argument("--npml", type=int, default=20)
    parser.add_argument("--threads", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    indices = _list(args.source_indices, int)
    hashes = _list(args.expected_sample_sha256, str)
    scales = _list(args.warp_scales, float)
    if len(indices) != len(hashes) or len(set(indices)) != len(indices):
        raise ValueError("registered indices and hashes must be one-to-one and unique")
    if 0.0 not in scales or len(set(scales)) != len(scales):
        raise ValueError("warp scales must be unique and include zero")
    torch.set_num_threads(int(args.threads))
    torch.set_num_interop_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", str(int(args.threads)))
    dataset = Path(args.dataset).resolve()
    travel_path = Path(args.travel_cache).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    with h5py.File(dataset, "r", swmr=True) as source, h5py.File(
        travel_path, "r", swmr=True
    ) as travel:
        travel_indices = np.asarray(travel["source_index"][:], dtype=np.int64)
        travel_lookup = {int(index): row for row, index in enumerate(travel_indices)}
        time_s = np.asarray(source["time_s"][:], dtype=np.float64)
        x_m = np.asarray(source["x_m"][:], dtype=np.float64)
        z_m = np.asarray(source["z_m"][:], dtype=np.float64)
        grid = AcousticGrid(
            nx=x_m.size,
            nz=z_m.size,
            dx_m=float(np.diff(x_m).mean()),
            dz_m=float(np.diff(z_m).mean()),
            lx_m=float(x_m[-1] - x_m[0]),
            lz_m=float(z_m[-1] - z_m[0]),
            centering="node",
        )
        for index, expected_hash in zip(indices, hashes, strict=True):
            if (
                str(source["split"].asstr()[index]) != "train"
                or str(source["medium_type"].asstr()[index]) != "marmousi"
            ):
                raise RuntimeError("capacity probe is restricted to train/Marmousi")
            velocity = np.asarray(source["velocity_mps"][index], dtype=np.float32)
            truth_np = np.asarray(source["wavefield"][index], dtype=np.float32)
            source_map = np.asarray(source["source_map"][index], dtype=np.float32)
            actual_hash = _sample_sha256(velocity, truth_np, source_map)
            if actual_hash != expected_hash or str(
                source["sample_sha256"].asstr()[index]
            ) != expected_hash:
                raise RuntimeError(f"sample content hash mismatch for index {index}")
            if index not in travel_lookup:
                raise KeyError(f"travel cache lacks source index {index}")
            travel_row = travel_lookup[index]
            if str(travel["sample_id"].asstr()[travel_row]) != str(
                source["sample_id"].asstr()[index]
            ):
                raise RuntimeError("travel cache sample identity mismatch")
            travel_time = torch.from_numpy(
                np.asarray(travel["travel_time_s"][travel_row], dtype=np.float32)
            )
            f0_hz = float(source["source_f0_hz"][index])
            fraction = (
                float(args.candidate_fraction)
                if f0_hz >= float(args.route_threshold_hz)
                else None
            )
            solver = LWC84CPMLSolver(
                grid=grid,
                boundaries=BoundaryConfig(npml=int(args.npml)),
                dt_s=float(args.internal_dt_s),
                output_times_s=time_s,
                c_ref_mps=6750.0,
                device="cpu",
                dtype=torch.float32,
                output_restriction_factor=1,
                drp_max_nyquist_fraction=fraction,
            )
            result = solver.simulate(
                velocity,
                source_x_m=float(source["source_x_m"][index]),
                source_z_m=float(source["source_z_m"][index]),
                source_f0_hz=f0_hz,
                source_t0_s=float(source["source_t0_s"][index]),
                source_amplitude=float(source["source_amplitude"][index]),
            )
            prediction = torch.from_numpy(result.wavefield[0])
            truth = torch.from_numpy(truth_np)
            time_tensor = torch.from_numpy(time_s).to(torch.float32)
            candidates: list[dict[str, float]] = []
            for scale in scales:
                warped = _travel_warp(
                    prediction, travel_time, time_tensor, scale=float(scale)
                )
                denominator = torch.sum(warped.double().square()).clamp_min(1.0e-30)
                amplitude = float(
                    (torch.sum(warped.double() * truth.double()) / denominator).item()
                )
                candidates.append(
                    {
                        "warp_scale": float(scale),
                        "fixed_amplitude_relative_l2": _relative_l2(warped, truth),
                        "oracle_amplitude": amplitude,
                        "oracle_amplitude_relative_l2": _relative_l2(
                            amplitude * warped, truth
                        ),
                    }
                )
            row = {
                "source_index": int(index),
                "sample_id": str(source["sample_id"].asstr()[index]),
                "group_id": str(source["group_id"].asstr()[index]),
                "sample_sha256": actual_hash,
                "split": "train",
                "family": "marmousi",
                "f0_hz": f0_hz,
                "drp_route_used": fraction is not None,
                "baseline_relative_l2": _relative_l2(prediction, truth),
                "candidates": candidates,
                "best_fixed_amplitude": min(
                    candidates, key=lambda item: item["fixed_amplitude_relative_l2"]
                ),
                "best_oracle_amplitude": min(
                    candidates, key=lambda item: item["oracle_amplitude_relative_l2"]
                ),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    baseline = np.asarray([float(row["baseline_relative_l2"]) for row in rows])
    fixed = np.asarray(
        [float(row["best_fixed_amplitude"]["fixed_amplitude_relative_l2"]) for row in rows]
    )
    amplitude = np.asarray(
        [float(row["best_oracle_amplitude"]["oracle_amplitude_relative_l2"]) for row in rows]
    )
    payload = {
        "schema": "train_only_travel_time_warp_capacity_v1",
        "status": "completed",
        "claim": "oracle capacity diagnostic only; not deployable and not holdout evidence",
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "travel_cache": str(travel_path),
        "travel_cache_sha256": _sha256(travel_path),
        "source_indices": list(indices),
        "expected_sample_sha256": list(hashes),
        "warp_scales": list(scales),
        "route_threshold_hz": float(args.route_threshold_hz),
        "candidate_fraction": float(args.candidate_fraction),
        "elapsed_seconds": float(time.perf_counter() - started),
        "rows": rows,
        "summary": {
            "record_count": int(len(rows)),
            "baseline_mean_relative_l2": float(baseline.mean()),
            "best_per_record_fixed_amplitude_mean_relative_l2": float(fixed.mean()),
            "best_per_record_oracle_amplitude_mean_relative_l2": float(amplitude.mean()),
            "fixed_amplitude_below_5_percent_count": int(np.sum(fixed <= 0.05)),
            "oracle_amplitude_below_5_percent_count": int(np.sum(amplitude <= 0.05)),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
