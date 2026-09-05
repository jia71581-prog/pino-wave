#!/usr/bin/env python3
"""Disjoint train-only gate for the frozen fine-grid CUDA-graph LWC84 solver."""
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
from fno_acoustic.data_generation.velocity_models_lwc84 import (
    generate_layered_velocity,
    load_marmousi_crop,
)


ALLOWED_FAMILIES = ("uniform", "layered", "marmousi")


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _error_terms(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    prediction64 = prediction.astype(np.float64)
    truth64 = truth.astype(np.float64)
    return (
        float(np.sum((prediction64 - truth64) ** 2)),
        float(np.sum(truth64**2)),
    )


def _relative_l2(terms: tuple[float, float]) -> float:
    numerator, denominator = terms
    return float(math.sqrt(numerator / max(denominator, 1.0e-30)))


def _add_terms(left: tuple[float, float], right: tuple[float, float]) -> tuple[float, float]:
    return left[0] + right[0], left[1] + right[1]


def _nearest_rank(values: list[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    rank = max(1, int(math.ceil(float(probability) * len(ordered))))
    return ordered[rank - 1]


def _read_manifest_rows(path: Path, sample_ids: set[str]) -> dict[str, dict]:
    selected = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in sample_ids:
                selected[sample_id] = row
    missing = sorted(sample_ids.difference(selected))
    if missing:
        raise KeyError(f"manifest is missing selected samples: {missing}")
    return selected


def _fine_velocity(
    *,
    family: str,
    manifest_row: dict,
    grid: AcousticGrid,
    marmousi_npy: Path,
) -> tuple[np.ndarray, dict]:
    parameters = manifest_row["medium_parameters"]
    if family == "uniform":
        value = float(parameters["velocity_mps"])
        return (
            np.full((grid.nz, grid.nx), value, dtype=np.float32),
            {"velocity_mps": value},
        )
    if family == "layered":
        return generate_layered_velocity(grid, seed=int(parameters["seed"]))
    if family == "marmousi":
        return load_marmousi_crop(
            marmousi_npy,
            grid=grid,
            source_dx_m=4.0,
            source_dz_m=4.0,
            source_unit="m/s",
            crop_x0_m=float(manifest_row["crop_x0_m"]),
            crop_z0_m=float(manifest_row["crop_z0_m"]),
            interpolation="scipy_regular_grid_linear",
        )
    raise ValueError(f"unsupported family {family!r}")


def run(args: argparse.Namespace) -> dict:
    indices = [int(value) for value in args.source_indices]
    if len(indices) != len(set(indices)):
        raise ValueError("source indices must be unique")
    grid = AcousticGrid(
        nx=401,
        nz=401,
        dx_m=5.0,
        dz_m=5.0,
        lx_m=2000.0,
        lz_m=2000.0,
        centering="node",
    )
    metadata = []
    with h5py.File(args.source_h5, "r") as handle:
        output_times_s = np.asarray(handle["time_s"][:], dtype=np.float64)
        for index in indices:
            row = {
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
            if row["split"] != "train":
                raise PermissionError(
                    f"gate permits train only, index {index} is {row['split']!r}"
                )
            if row["family"] not in ALLOWED_FAMILIES:
                raise ValueError(
                    f"index {index} has out-of-scope family {row['family']!r}"
                )
            metadata.append(row)
    family_counts = {
        family: sum(row["family"] == family for row in metadata)
        for family in ALLOWED_FAMILIES
    }
    if min(family_counts.values()) <= 0:
        raise ValueError(f"every target family must be represented: {family_counts}")
    manifest_rows = _read_manifest_rows(
        args.manifest, {row["sample_id"] for row in metadata}
    )
    for row in metadata:
        manifest_row = manifest_rows[row["sample_id"]]
        if manifest_row["split"] != "train" or manifest_row["medium_type"] != row["family"]:
            raise RuntimeError(f"HDF5/manifest mismatch for {row['sample_id']}")

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
    with h5py.File(args.source_h5, "r") as handle:
        for row in metadata:
            index = row["index"]
            truth = np.asarray(handle["wavefield"][index])
            stored_velocity = np.asarray(handle["velocity_mps"][index])
            fine_velocity, velocity_metadata = _fine_velocity(
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
                band_terms = _error_terms(
                    prediction[start:stop], truth[start:stop]
                )
                temporal_terms[name] = _add_terms(
                    temporal_terms[name], band_terms
                )
                band_relative[name] = _relative_l2(band_terms)
            measurements.append(
                {
                    "source_index": index,
                    "sample_id": row["sample_id"],
                    "split": "train",
                    "family": row["family"],
                    "source_f0_hz": row["source_f0_hz"],
                    "relative_l2": _relative_l2(terms),
                    "temporal_band_relative_l2": band_relative,
                    "runtime_s": runtime_s,
                    "solver_reported_runtime_s": result.metrics[0][
                        "compute_elapsed_s"
                    ],
                    "lwc_qmax": result.metrics[0]["lwc_qmax"],
                    "fine_velocity_range_mps": [
                        float(fine_velocity.min()),
                        float(fine_velocity.max()),
                    ],
                    "fine_velocity_metadata": velocity_metadata,
                    "restriction_max_absolute_difference": restriction_max_abs,
                }
            )

    runtimes = [row["runtime_s"] for row in measurements]
    family_relative = {
        family: _relative_l2(family_terms[family]) for family in ALLOWED_FAMILIES
    }
    aggregate_relative = _relative_l2(total_terms)
    maximum_instance_relative = max(row["relative_l2"] for row in measurements)
    runtime_mean = float(sum(runtimes) / len(runtimes))
    runtime_p95 = _nearest_rank(runtimes, 0.95)
    runtime_limit = float(args.speed_reference_s) / 10.0
    passes_accuracy = bool(
        aggregate_relative <= args.maximum_relative_l2
        and max(family_relative.values()) <= args.maximum_relative_l2
        and maximum_instance_relative <= args.maximum_relative_l2
    )
    passes_runtime = bool(
        runtime_mean <= runtime_limit and runtime_p95 <= runtime_limit
    )
    report = {
        "schema": "lwc84_cuda_graph_fine_grid_trainonly_gate_v1",
        "status": "accepted_train_only" if passes_accuracy and passes_runtime else "rejected",
        "selection_scope": "train_only",
        "validation_access": False,
        "test_id_access": False,
        "source_indices": indices,
        "family_counts": family_counts,
        "internal_grid": {
            "shape": [401, 401],
            "spacing_m": 5.0,
            "npml": 40,
            "internal_dt_s": float(args.internal_dt_s),
            "output_restriction": "binomial5_lowpass_then_decimate2",
        },
        "aggregate_relative_l2": aggregate_relative,
        "family_relative_l2": family_relative,
        "maximum_instance_relative_l2": maximum_instance_relative,
        "temporal_band_relative_l2": {
            name: _relative_l2(terms) for name, terms in temporal_terms.items()
        },
        "runtime_s": {
            "minimum": min(runtimes),
            "mean": runtime_mean,
            "p95_nearest_rank": runtime_p95,
            "maximum": max(runtimes),
            "warmup_excluded": warmup_s,
        },
        "speedup_over_traditional_reference": {
            "reference_s": float(args.speed_reference_s),
            "mean": float(args.speed_reference_s) / runtime_mean,
            "p95": float(args.speed_reference_s) / runtime_p95,
        },
        "promotion_gate": {
            "maximum_relative_l2": float(args.maximum_relative_l2),
            "maximum_runtime_s": runtime_limit,
            "passes_accuracy": passes_accuracy,
            "passes_runtime": passes_runtime,
            "passed_train_only": bool(passes_accuracy and passes_runtime),
        },
        "measurements": measurements,
        "bindings": {
            "source_h5": str(args.source_h5.resolve()),
            "source_h5_sha256": _sha256(args.source_h5),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": _sha256(args.manifest),
            "marmousi_npy": str(args.marmousi_npy.resolve()),
            "marmousi_npy_sha256": _sha256(args.marmousi_npy),
            "fused_solver_sha256": _sha256(
                ROOT / "src/fno_acoustic/data_generation/solver_lwc84_fused.py"
            ),
            "functional_kernel_sha256": _sha256(
                ROOT / "src/fno_acoustic/data_generation/fused_lwc84.py"
            ),
            "gate_script_sha256": _sha256(Path(__file__)),
        },
    }
    _atomic_json(report, args.output)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--marmousi-npy", type=Path, required=True)
    parser.add_argument("--source-indices", nargs="+", type=int, required=True)
    parser.add_argument("--internal-dt-s", type=float, default=6.25e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speed-reference-s", type=float, default=20.34137312322855)
    parser.add_argument("--maximum-relative-l2", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["promotion_gate"]["passed_train_only"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
