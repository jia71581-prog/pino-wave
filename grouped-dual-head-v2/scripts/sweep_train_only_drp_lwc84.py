#!/usr/bin/env python
"""Leakage-safe DRP bandwidth sweep on deterministic train-only records."""

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

from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def _select_train_marmousi_quantiles(
    handle: h5py.File,
    *,
    count: int,
) -> np.ndarray:
    split = _decode(handle["split"][:])
    family = _decode(handle["medium_type"][:])
    candidates = np.flatnonzero((split == "train") & (family == "marmousi"))
    if candidates.size < int(count):
        raise ValueError(
            f"only {candidates.size} train Marmousi records are available for count={count}"
        )
    frequency = np.asarray(handle["source_f0_hz"][:], dtype=np.float64)
    ordered = candidates[np.argsort(frequency[candidates], kind="stable")]
    positions = np.rint(np.linspace(0, ordered.size - 1, int(count))).astype(np.int64)
    selected = ordered[positions]
    if not np.all(split[selected] == "train"):
        raise RuntimeError("non-train record reached the DRP sweep")
    return selected


def _relative_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    difference = prediction.astype(np.float64) - truth.astype(np.float64)
    denominator = float(np.linalg.norm(truth.astype(np.float64)))
    return float(np.linalg.norm(difference) / max(denominator, 1.0e-30))


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    by_variant: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    summaries: dict[str, dict[str, float | int]] = {}
    for variant, selected in by_variant.items():
        errors = np.asarray([float(row["relative_l2"]) for row in selected])
        seconds = np.asarray([float(row["wall_seconds"]) for row in selected])
        summaries[variant] = {
            "record_count": int(errors.size),
            "mean_relative_l2": float(errors.mean()),
            "median_relative_l2": float(np.median(errors)),
            "maximum_relative_l2": float(errors.max()),
            "mean_wall_seconds": float(seconds.mean()),
        }
    ordered = sorted(
        summaries,
        key=lambda name: (
            summaries[name]["mean_relative_l2"],
            summaries[name]["maximum_relative_l2"],
            name,
        ),
    )
    return {"by_variant": summaries, "ranking": ordered}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-count", type=int, default=7)
    parser.add_argument(
        "--fractions",
        default="0.45,0.55,0.65,0.75,0.85",
        help="Comma-separated DRP Nyquist fractions; standard Taylor is always included.",
    )
    parser.add_argument("--internal-dt-s", type=float, default=2.5e-4)
    parser.add_argument("--npml", type=int, default=20)
    parser.add_argument("--c-ref-mps", type=float, default=6750.0)
    parser.add_argument("--threads", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.sample_count) < 2:
        raise ValueError("sample-count must be at least two")
    fractions = tuple(float(value) for value in str(args.fractions).split(","))
    if len(set(fractions)) != len(fractions):
        raise ValueError("DRP fractions must be unique")
    if any(not 0.0 < value <= 1.0 for value in fractions):
        raise ValueError("DRP fractions must lie in (0, 1]")
    torch.set_num_threads(int(args.threads))
    torch.set_num_interop_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", str(int(args.threads)))

    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    jsonl = output.with_suffix(".jsonl")
    if output.exists() or jsonl.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing sweep artifact: {output} or {jsonl}"
        )
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    with h5py.File(dataset, "r", swmr=True) as handle:
        if str(handle.attrs.get("axis_order", "")) != "NTZX":
            raise ValueError("dataset axis_order must be NTZX")
        selected = _select_train_marmousi_quantiles(
            handle,
            count=int(args.sample_count),
        )
        time_s = np.asarray(handle["time_s"][:], dtype=np.float64)
        x_m = np.asarray(handle["x_m"][:], dtype=np.float64)
        z_m = np.asarray(handle["z_m"][:], dtype=np.float64)
        dx_m = float(np.diff(x_m).mean())
        dz_m = float(np.diff(z_m).mean())
        grid = AcousticGrid(
            nx=x_m.size,
            nz=z_m.size,
            dx_m=dx_m,
            dz_m=dz_m,
            lx_m=float(x_m[-1] - x_m[0]),
            lz_m=float(z_m[-1] - z_m[0]),
            centering="node",
        )
        variants: tuple[tuple[str, float | None], ...] = (
            ("standard_taylor8", None),
            *tuple((f"drp_{value:.6g}", value) for value in fractions),
        )
        for index in selected.tolist():
            split = str(handle["split"].asstr()[index])
            family = str(handle["medium_type"].asstr()[index])
            if split != "train" or family != "marmousi":
                raise RuntimeError("selected record violates the train/Marmousi contract")
            record = {
                "velocity": np.asarray(handle["velocity_mps"][index], dtype=np.float32),
                "truth": np.asarray(handle["wavefield"][index], dtype=np.float32),
                "sample_id": str(handle["sample_id"].asstr()[index]),
                "group_id": str(handle["group_id"].asstr()[index]),
                "f0_hz": float(handle["source_f0_hz"][index]),
                "source_x_m": float(handle["source_x_m"][index]),
                "source_z_m": float(handle["source_z_m"][index]),
                "source_t0_s": float(handle["source_t0_s"][index]),
                "source_amplitude": float(handle["source_amplitude"][index]),
            }
            for variant, fraction in variants:
                solver = LWC84CPMLSolver(
                    grid=grid,
                    boundaries=BoundaryConfig(npml=int(args.npml)),
                    dt_s=float(args.internal_dt_s),
                    output_times_s=time_s,
                    c_ref_mps=float(args.c_ref_mps),
                    device="cpu",
                    dtype=torch.float32,
                    output_restriction_factor=1,
                    drp_max_nyquist_fraction=fraction,
                )
                solve_started = time.perf_counter()
                result = solver.simulate(
                    record["velocity"],
                    source_x_m=record["source_x_m"],
                    source_z_m=record["source_z_m"],
                    source_f0_hz=record["f0_hz"],
                    source_t0_s=record["source_t0_s"],
                    source_amplitude=record["source_amplitude"],
                )
                row: dict[str, object] = {
                    "source_index": int(index),
                    "split": split,
                    "family": family,
                    "sample_id": record["sample_id"],
                    "group_id": record["group_id"],
                    "f0_hz": record["f0_hz"],
                    "variant": variant,
                    "drp_max_nyquist_fraction": fraction,
                    "relative_l2": _relative_l2(result.wavefield[0], record["truth"]),
                    "wall_seconds": float(time.perf_counter() - solve_started),
                    "solver_qc": result.metrics[0],
                }
                rows.append(row)
                with jsonl.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)

    payload = {
        "schema": "train_only_drp_lwc84_sweep_v1",
        "status": "completed",
        "dataset": str(dataset),
        "dataset_byte_count": dataset.stat().st_size,
        "dataset_sha256": _sha256(dataset),
        "selection": "stable_f0_quantiles_of_split_train_and_family_marmousi",
        "selected_source_indices": sorted({int(row["source_index"]) for row in rows}),
        "sample_count": int(args.sample_count),
        "fractions": list(fractions),
        "internal_dt_s": float(args.internal_dt_s),
        "npml": int(args.npml),
        "c_ref_mps": float(args.c_ref_mps),
        "threads": int(args.threads),
        "elapsed_seconds": float(time.perf_counter() - started),
        "rows": rows,
        "summary": _aggregate(rows),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **payload["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
