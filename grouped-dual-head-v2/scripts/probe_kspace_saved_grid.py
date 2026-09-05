#!/usr/bin/env python3
"""Train-only accuracy/runtime gate for the saved-grid k-space propagator.

This probe never reads validation or test_id.  It selects deterministic low/mid/high
source-frequency records from each train family, executes one or more internal time
steps, and reports raw plus scalar-aligned relative L2.  Scalar alignment is diagnostic
only; promotion uses raw error, every-family error, and measured end-to-end runtime.
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
from fno_acoustic.data_generation.solver_kspace import KSpacePSTDSolver


FAMILIES = ("uniform", "layered", "marmousi")


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    numerator = np.sum(
        (prediction.astype(np.float64) - truth.astype(np.float64)) ** 2
    )
    denominator = np.sum(truth.astype(np.float64) ** 2)
    return float(math.sqrt(numerator / max(denominator, 1.0e-30)))


def _scalar_alignment(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    pred64 = prediction.astype(np.float64)
    truth64 = truth.astype(np.float64)
    scale = float(
        np.sum(pred64 * truth64) / max(float(np.sum(pred64 * pred64)), 1.0e-30)
    )
    return scale, _relative_l2(scale * pred64, truth64)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _select_train_indices(handle: h5py.File, records_per_family: int) -> dict[str, list[int]]:
    split = [_decode(value) for value in handle["split"][:]]
    family = [_decode(value) for value in handle["medium_type"][:]]
    frequency = np.asarray(handle["source_f0_hz"][:], dtype=np.float64)
    selected: dict[str, list[int]] = {}
    for name in FAMILIES:
        candidates = [
            index
            for index, (record_split, record_family) in enumerate(zip(split, family))
            if record_split == "train" and record_family == name
        ]
        candidates.sort(key=lambda index: (float(frequency[index]), index))
        if len(candidates) < records_per_family:
            raise ValueError(f"train/{name} has too few records")
        positions = np.linspace(
            0, len(candidates) - 1, num=records_per_family, dtype=np.int64
        )
        selected[name] = [candidates[int(position)] for position in positions]
    return selected


def run_probe(
    *,
    source_h5: Path,
    output: Path,
    internal_dt_s: float,
    records_per_family: int,
    device: torch.device,
    temporal_order: int,
    correction_reference_mps: float | None,
    correction_reference_quantile: float | None,
    match_fine_grid_restriction: bool,
) -> dict:
    if records_per_family <= 0:
        raise ValueError("records_per_family must be positive")
    if correction_reference_quantile is not None:
        if correction_reference_mps is not None:
            raise ValueError(
                "correction_reference_mps and correction_reference_quantile are exclusive"
            )
        if not 0.0 < float(correction_reference_quantile) <= 1.0:
            raise ValueError("correction_reference_quantile must lie in (0, 1]")
    with h5py.File(source_h5, "r") as handle:
        if _decode(handle.attrs.get("schema_version")) != "acoustic_lwc84_401_to_201_v1":
            raise ValueError("source dataset is not the sealed LWC-84 protocol")
        if tuple(handle["wavefield"].shape[1:]) != (401, 201, 201):
            raise ValueError("source dataset does not have 401x201x201 outputs")
        time_s = np.asarray(handle["time_s"][:], dtype=np.float64)
        chosen = _select_train_indices(handle, records_per_family)
        rows = []
        for family in FAMILIES:
            for index in chosen[family]:
                rows.append(
                    {
                        "index": index,
                        "family": family,
                        "sample_id": _decode(handle["sample_id"][index]),
                        "velocity": np.asarray(handle["velocity_mps"][index]),
                        "truth": np.asarray(handle["wavefield"][index]),
                        "source_x_m": float(handle["source_x_m"][index]),
                        "source_z_m": float(handle["source_z_m"][index]),
                        "source_f0_hz": float(handle["source_f0_hz"][index]),
                        "source_t0_s": float(handle["source_t0_s"][index]),
                        "source_amplitude": float(handle["source_amplitude"][index]),
                    }
                )

    grid = AcousticGrid(
        nx=201,
        nz=201,
        dx_m=10.0,
        dz_m=10.0,
        lx_m=2000.0,
        lz_m=2000.0,
        centering="node",
    )
    boundaries = BoundaryConfig(npml=20)
    measurements = []
    family_num = {family: 0.0 for family in FAMILIES}
    family_den = {family: 0.0 for family in FAMILIES}
    total_num = 0.0
    total_den = 0.0
    for row in rows:
        # A record-local reference speed avoids the large, unnecessary phase
        # correction induced by the dataset-wide 6750 m/s upper bound.  The CPML
        # coefficients remain bound to 6750 m/s exactly as in the sealed teacher.
        if correction_reference_mps is not None:
            reference_speed = float(correction_reference_mps)
        elif correction_reference_quantile is None:
            reference_speed = float(np.max(row["velocity"]))
        else:
            reference_speed = float(
                np.quantile(row["velocity"], float(correction_reference_quantile))
            )
        solver = KSpacePSTDSolver(
            grid=grid,
            boundaries=boundaries,
            dt_s=float(internal_dt_s),
            output_times_s=time_s,
            c_ref_mps=reference_speed,
            damping_c_ref_mps=6750.0,
            device=device,
            dtype=torch.float32 if device.type == "cuda" else torch.float64,
            temporal_order=int(temporal_order),
            match_fine_grid_restriction=bool(match_fine_grid_restriction),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = solver.simulate(
            row["velocity"],
            source_x_m=row["source_x_m"],
            source_z_m=row["source_z_m"],
            source_f0_hz=row["source_f0_hz"],
            source_t0_s=row["source_t0_s"],
            source_amplitude=row["source_amplitude"],
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        prediction = result.wavefield[0]
        truth = row["truth"]
        raw = _relative_l2(prediction, truth)
        scale, aligned = _scalar_alignment(prediction, truth)
        pred64 = prediction.astype(np.float64)
        truth64 = truth.astype(np.float64)
        numerator = float(np.sum((pred64 - truth64) ** 2))
        denominator = float(np.sum(truth64**2))
        family_num[row["family"]] += numerator
        family_den[row["family"]] += denominator
        total_num += numerator
        total_den += denominator
        measurements.append(
            {
                "sample_id": row["sample_id"],
                "split": "train",
                "family": row["family"],
                "source_f0_hz": row["source_f0_hz"],
                "relative_l2": raw,
                "correction_reference_speed_mps": reference_speed,
                "velocity_max_mps": float(np.max(row["velocity"])),
                "best_scalar": scale,
                "scalar_aligned_relative_l2_diagnostic": aligned,
                "runtime_s": elapsed,
                "solver_runtime_s": result.metrics[0]["compute_elapsed_s"],
            }
        )
    family_relative = {
        family: float(math.sqrt(family_num[family] / max(family_den[family], 1.0e-30)))
        for family in FAMILIES
    }
    maximum_instance_relative = max(
        float(measurement["relative_l2"]) for measurement in measurements
    )
    runtimes = sorted(float(row["runtime_s"]) for row in measurements)
    report = {
        "status": "complete",
        "schema": "kspace_saved_grid_train_only_probe_v1",
        "source_h5": str(source_h5),
        "source_h5_sha256": _sha256(source_h5),
        "selection_scope": "train_only",
        "records_per_family": records_per_family,
        "internal_dt_s": internal_dt_s,
        "temporal_order": int(temporal_order),
        "correction_reference_mps": correction_reference_mps,
        "correction_reference_quantile": correction_reference_quantile,
        "match_fine_grid_restriction": bool(match_fine_grid_restriction),
        "device": str(device),
        "aggregate_relative_l2": float(
            math.sqrt(total_num / max(total_den, 1.0e-30))
        ),
        "family_relative_l2": family_relative,
        "runtime_s": {
            "minimum": runtimes[0],
            "mean": sum(runtimes) / len(runtimes),
            "maximum": runtimes[-1],
        },
        "promotion_gate": {
            "maximum_relative_l2": 0.05,
            "maximum_instance_relative_l2": maximum_instance_relative,
            "passes_accuracy": bool(
                max(family_relative.values()) <= 0.05
                and math.sqrt(total_num / max(total_den, 1.0e-30)) <= 0.05
                and maximum_instance_relative <= 0.05
            ),
            "scalar_alignment_is_diagnostic_only": True,
        },
        "measurements": measurements,
    }
    _atomic_json(report, output)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--internal-dt-s", type=float, default=6.25e-4)
    parser.add_argument("--records-per-family", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temporal-order", type=int, choices=(2, 4), default=2)
    parser.add_argument(
        "--correction-reference-mps",
        type=float,
        default=None,
        help="Fixed k-space reference; use a near-zero value with temporal-order 4",
    )
    parser.add_argument(
        "--correction-reference-quantile",
        type=float,
        default=None,
        help="Per-record velocity quantile in (0, 1] for the k-space reference; "
        "exclusive with --correction-reference-mps; omit both for per-record maximum",
    )
    parser.add_argument("--match-fine-grid-restriction", action="store_true")
    args = parser.parse_args(argv)
    report = run_probe(
        source_h5=args.source_h5.resolve(),
        output=args.output.resolve(),
        internal_dt_s=float(args.internal_dt_s),
        records_per_family=int(args.records_per_family),
        device=torch.device(args.device),
        temporal_order=int(args.temporal_order),
        correction_reference_mps=args.correction_reference_mps,
        correction_reference_quantile=args.correction_reference_quantile,
        match_fine_grid_restriction=bool(args.match_fine_grid_restriction),
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
