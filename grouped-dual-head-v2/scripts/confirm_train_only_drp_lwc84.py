#!/usr/bin/env python
"""Disjoint train-only confirmation of one preregistered DRP candidate."""

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
from fno_acoustic.data_generation.hdf5_lwc84 import _sample_sha256
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _indices(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise ValueError("source indices must be a nonempty unique list")
    return result


def _hashes(value: str) -> tuple[str, ...]:
    result = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not result or any(
        len(item) != 64 or any(character not in "0123456789abcdef" for character in item)
        for item in result
    ):
        raise ValueError("expected sample hashes must be lowercase SHA-256 values")
    return result


def _relative_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction64 = prediction.astype(np.float64)
    truth64 = truth.astype(np.float64)
    return float(
        np.linalg.norm(prediction64 - truth64)
        / max(float(np.linalg.norm(truth64)), 1.0e-30)
    )


def _paired_summary(
    rows: list[dict[str, object]],
    *,
    high_frequency_cutoff_hz: float,
    minimum_relative_improvement: float,
    minimum_win_fraction: float,
    maximum_absolute_regression: float,
) -> dict[str, object]:
    by_index: dict[int, dict[str, dict[str, object]]] = {}
    for row in rows:
        by_index.setdefault(int(row["source_index"]), {})[str(row["variant"])] = row
    expected = {"standard_taylor8", "drp_candidate"}
    if any(set(pair) != expected for pair in by_index.values()):
        raise ValueError("each source index must have exactly one standard and candidate row")

    standard = np.asarray(
        [float(by_index[index]["standard_taylor8"]["relative_l2"]) for index in sorted(by_index)]
    )
    candidate = np.asarray(
        [float(by_index[index]["drp_candidate"]["relative_l2"]) for index in sorted(by_index)]
    )
    frequency = np.asarray(
        [float(by_index[index]["standard_taylor8"]["f0_hz"]) for index in sorted(by_index)]
    )
    difference = candidate - standard
    high = frequency >= float(high_frequency_cutoff_hz)
    if not np.any(high):
        raise ValueError("confirmation selection has no registered high-frequency records")
    mean_standard = float(standard.mean())
    mean_candidate = float(candidate.mean())
    high_standard = float(standard[high].mean())
    high_candidate = float(candidate[high].mean())
    relative_improvement = float((mean_standard - mean_candidate) / mean_standard)
    high_relative_improvement = float(
        (high_standard - high_candidate) / high_standard
    )
    win_fraction = float(np.mean(candidate < standard))
    maximum_regression = float(np.max(difference))
    gates = {
        "mean_relative_improvement": bool(
            relative_improvement >= float(minimum_relative_improvement)
        ),
        "paired_win_fraction": bool(win_fraction >= float(minimum_win_fraction)),
        "high_frequency_relative_improvement": bool(
            high_relative_improvement >= float(minimum_relative_improvement)
        ),
        "maximum_absolute_regression": bool(
            maximum_regression <= float(maximum_absolute_regression)
        ),
        "maximum_error_nonworse": bool(candidate.max() <= standard.max()),
    }
    return {
        "record_count": int(standard.size),
        "high_frequency_record_count": int(high.sum()),
        "standard_mean_relative_l2": mean_standard,
        "candidate_mean_relative_l2": mean_candidate,
        "relative_improvement": relative_improvement,
        "standard_maximum_relative_l2": float(standard.max()),
        "candidate_maximum_relative_l2": float(candidate.max()),
        "paired_win_count": int(np.sum(candidate < standard)),
        "paired_win_fraction": win_fraction,
        "high_frequency_standard_mean_relative_l2": high_standard,
        "high_frequency_candidate_mean_relative_l2": high_candidate,
        "high_frequency_relative_improvement": high_relative_improvement,
        "maximum_absolute_regression": maximum_regression,
        "gates": gates,
        "promotion_passed": bool(all(gates.values())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-indices", required=True)
    parser.add_argument("--expected-sample-sha256", required=True)
    parser.add_argument("--pilot-source-indices", required=True)
    parser.add_argument("--candidate-fraction", type=float, required=True)
    parser.add_argument("--internal-dt-s", type=float, default=2.5e-4)
    parser.add_argument("--npml", type=int, default=20)
    parser.add_argument("--c-ref-mps", type=float, default=6750.0)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--high-frequency-cutoff-hz", type=float, default=28.0)
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.02)
    parser.add_argument("--minimum-win-fraction", type=float, default=0.625)
    parser.add_argument("--maximum-absolute-regression", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = _indices(args.source_indices)
    expected_sample_hashes = _hashes(args.expected_sample_sha256)
    pilot = _indices(args.pilot_source_indices)
    if len(expected_sample_hashes) != len(selected):
        raise ValueError("one expected sample hash is required per source index")
    if not 0.0 < float(args.candidate_fraction) <= 1.0:
        raise ValueError("candidate fraction must lie in (0, 1]")
    torch.set_num_threads(int(args.threads))
    torch.set_num_interop_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", str(int(args.threads)))

    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    jsonl = output.with_suffix(".jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or jsonl.exists():
        raise FileExistsError(
            f"refusing to overwrite a confirmation artifact: {output} or {jsonl}"
        )

    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    with h5py.File(dataset, "r", swmr=True) as handle:
        if str(handle.attrs.get("axis_order", "")) != "NTZX":
            raise ValueError("dataset axis_order must be NTZX")
        split = np.asarray(handle["split"].asstr()[:])
        family = np.asarray(handle["medium_type"].asstr()[:])
        group = np.asarray(handle["group_id"].asstr()[:])
        if max((*selected, *pilot)) >= split.size or min((*selected, *pilot)) < 0:
            raise IndexError("a registered source index is outside the dataset")
        if not np.all(split[list(selected)] == "train") or not np.all(
            family[list(selected)] == "marmousi"
        ):
            raise RuntimeError("confirmation records must all be train/Marmousi")
        if len(set(group[list(selected)])) != len(selected):
            raise RuntimeError("confirmation records must use unique medium groups")
        if set(group[list(selected)]) & set(group[list(pilot)]):
            raise RuntimeError("confirmation medium groups overlap the pilot")
        stored_hashes = tuple(
            str(handle["sample_sha256"].asstr()[index]).lower() for index in selected
        )
        if stored_hashes != expected_sample_hashes:
            raise RuntimeError("stored confirmation sample hashes differ from registration")

        time_s = np.asarray(handle["time_s"][:], dtype=np.float64)
        x_m = np.asarray(handle["x_m"][:], dtype=np.float64)
        z_m = np.asarray(handle["z_m"][:], dtype=np.float64)
        grid = AcousticGrid(
            nx=x_m.size,
            nz=z_m.size,
            dx_m=float(np.diff(x_m).mean()),
            dz_m=float(np.diff(z_m).mean()),
            lx_m=float(x_m[-1] - x_m[0]),
            lz_m=float(z_m[-1] - z_m[0]),
            centering="node",
        )
        variants = (
            ("standard_taylor8", None),
            ("drp_candidate", float(args.candidate_fraction)),
        )
        for index in selected:
            record = {
                "velocity": np.asarray(handle["velocity_mps"][index], dtype=np.float32),
                "truth": np.asarray(handle["wavefield"][index], dtype=np.float32),
                "source_map": np.asarray(handle["source_map"][index], dtype=np.float32),
                "sample_id": str(handle["sample_id"].asstr()[index]),
                "group_id": str(handle["group_id"].asstr()[index]),
                "f0_hz": float(handle["source_f0_hz"][index]),
                "source_x_m": float(handle["source_x_m"][index]),
                "source_z_m": float(handle["source_z_m"][index]),
                "source_t0_s": float(handle["source_t0_s"][index]),
                "source_amplitude": float(handle["source_amplitude"][index]),
                "sample_sha256": str(handle["sample_sha256"].asstr()[index]).lower(),
            }
            actual_sample_hash = _sample_sha256(
                record["velocity"], record["truth"], record["source_map"]
            )
            if actual_sample_hash != record["sample_sha256"]:
                raise RuntimeError(
                    f"physical content hash mismatch for source index {index}"
                )
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
                    "split": "train",
                    "family": "marmousi",
                    "sample_id": record["sample_id"],
                    "group_id": record["group_id"],
                    "sample_sha256": record["sample_sha256"],
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

    summary = _paired_summary(
        rows,
        high_frequency_cutoff_hz=float(args.high_frequency_cutoff_hz),
        minimum_relative_improvement=float(args.minimum_relative_improvement),
        minimum_win_fraction=float(args.minimum_win_fraction),
        maximum_absolute_regression=float(args.maximum_absolute_regression),
    )
    payload = {
        "schema": "train_only_drp_lwc84_confirmation_v1",
        "status": "accepted" if summary["promotion_passed"] else "rejected",
        "dataset": str(dataset),
        "dataset_byte_count": dataset.stat().st_size,
        "dataset_sha256": _sha256(dataset),
        "selection": "preregistered_disjoint_train_marmousi_source_indices",
        "selected_source_indices": list(selected),
        "expected_sample_sha256": list(expected_sample_hashes),
        "pilot_source_indices": list(pilot),
        "candidate_fraction": float(args.candidate_fraction),
        "internal_dt_s": float(args.internal_dt_s),
        "npml": int(args.npml),
        "c_ref_mps": float(args.c_ref_mps),
        "threads": int(args.threads),
        "high_frequency_cutoff_hz": float(args.high_frequency_cutoff_hz),
        "minimum_relative_improvement": float(args.minimum_relative_improvement),
        "minimum_win_fraction": float(args.minimum_win_fraction),
        "maximum_absolute_regression": float(args.maximum_absolute_regression),
        "elapsed_seconds": float(time.perf_counter() - started),
        "rows": rows,
        "summary": summary,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
