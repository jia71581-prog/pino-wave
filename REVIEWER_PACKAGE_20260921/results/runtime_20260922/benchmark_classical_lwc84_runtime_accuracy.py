#!/usr/bin/env python3
"""Runtime and accuracy of the classical LWC-84 solver on the paper's figure records.

Two classical configurations, timed with the same protocol as the operator and
the frozen generation-protocol benchmark (CUDA-synchronised, single instance,
output materialised, no disk I/O in the timed region):

  fine    -- the generation protocol: 401x401 grid at dx=5 m, dt=1.25e-4 s
             (8000 steps), restricted to the stored 201x201 grid, built from
             the dataset's frozen_config.yaml.  Its velocity input is the
             stored 10 m velocity bilinearly upsampled to 5 m, as in
             benchmark_lwc84_traditional_runtime.py, so its error against the
             stored truth includes that resampling and is reported, not zero.
  native  -- the same solver on the stored 201x201 grid at dx=10 m,
             dt=2.5e-4 s (4000 steps), npml=20, no restriction: the cheapest
             classical configuration that emits the stored product directly.

Accuracy is the paper's metric: relative L2 in float64 over the future window
(onset + 8 IC frames to the last stored frame) and its three equal time bands,
against the stored truth, on the same three family-median records the paper's
figures and operator timing use.  Record identity and onsets are asserted
against the published panel rows.

Runs from the grouped-dual-head-v2 tree, which is the tree that generated the
dataset and the tree today's generation-protocol runtime benchmark ran from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.nn import functional as F

GDH = Path("/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2")
for value in (str(GDH / "src"), str(GDH)):
    if value not in sys.path:
        sys.path.insert(0, value)

from fno_acoustic.data_generation.config import (  # noqa: E402
    boundaries_from_config,
    grid_from_config,
    time_from_config,
)
from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig  # noqa: E402
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver  # noqa: E402

H5 = Path("/data/jiayh/data/acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2/dataset_v1.h5")
FROZEN_CONFIG = H5.with_name("frozen_config.yaml")
PANEL_ROWS = Path(
    "/root/autodl-tmp/staging/coda_panel_eval_20260916/ckpt_29359/panel_29359_ROWS.jsonl")
RECORDS = ("train_uniform_00413", "train_layered_00299", "train_marmousi_00010")
IC_FRAMES = 8


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative_l2(pred: np.ndarray, truth: np.ndarray, frames: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.float64)[frames]
    t = np.asarray(truth, dtype=np.float64)[frames]
    den = float((t ** 2).sum())
    if den <= 0.0:
        raise ValueError("relative L2 denominator is zero over the frame set")
    return float(np.sqrt(float(((p - t) ** 2).sum()) / den))


def fine_velocity(saved: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = torch.as_tensor(saved, dtype=torch.float32)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise ValueError("saved velocity must be a single scalar channel")
    fine = F.interpolate(value[None, None], size=shape, mode="bilinear", align_corners=True)
    return fine[0, 0].contiguous().numpy()


def load_panel_rows() -> dict[str, dict]:
    rows = {}
    for line in PANEL_ROWS.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["sample_id"]] = row
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("the runtime comparison must run on the deployment GPU")

    config = yaml.safe_load(FROZEN_CONFIG.read_text())
    grid_fine = grid_from_config(config)
    boundaries_fine = boundaries_from_config(config)
    times = time_from_config(config).t_s
    dt_fine = float(config["time"]["dt_used_s"])
    if (grid_fine.nz, grid_fine.nx) != (401, 401) or len(times) != 401:
        raise SystemExit("frozen config does not match the 401x401 by 401-frame protocol")

    panel = load_panel_rows()
    with h5py.File(H5, "r", swmr=True) as handle:
        time_s = np.asarray(handle["time_s"][:], dtype=np.float64)
        x_m = np.asarray(handle["x_m"][:], dtype=np.float64)
        z_m = np.asarray(handle["z_m"][:], dtype=np.float64)
        sample_ids = [v.decode() for v in handle["sample_id"][:]]
        inputs = {}
        for sample_id in RECORDS:
            row = panel[sample_id]
            index = row["source_index"]
            if sample_ids[index] != sample_id:
                raise SystemExit(f"h5 index {index} holds {sample_ids[index]}, not {sample_id}")
            onset = int(np.searchsorted(time_s, float(handle["source_t0_s"][index])))
            if onset != row["onset"]:
                raise SystemExit(f"{sample_id}: onset {onset} against panel {row['onset']}")
            inputs[sample_id] = {
                "source_index": index,
                "medium_type": handle["medium_type"][index].decode(),
                "velocity": np.asarray(handle["velocity_mps"][index], dtype=np.float32),
                "truth": np.asarray(handle["wavefield"][index], dtype=np.float64),
                "source_x_m": float(handle["source_x_m"][index]),
                "source_z_m": float(handle["source_z_m"][index]),
                "source_f0_hz": float(handle["source_f0_hz"][index]),
                "source_t0_s": float(handle["source_t0_s"][index]),
                "source_amplitude": float(handle["source_amplitude"][index]),
                "onset": onset,
            }

    dt_native = 2.5e-4
    frame_step = float(time_s[1] - time_s[0])
    if not math.isclose(frame_step / dt_native, round(frame_step / dt_native), abs_tol=1e-9):
        raise SystemExit("native dt does not divide the stored frame step")

    solver_fine = LWC84CPMLSolver(
        grid=grid_fine, boundaries=boundaries_fine, dt_s=dt_fine, output_times_s=times,
        c_ref_mps=6750.0, device=device, dtype=torch.float32,
        kappa_max=float(config["boundaries"]["kappa_max"]),
        minimum_frequency_hz=float(config["boundaries"]["minimum_frequency_hz"]))
    grid_native = AcousticGrid(
        nx=len(x_m), nz=len(z_m), dx_m=float(x_m[1] - x_m[0]), dz_m=float(z_m[1] - z_m[0]),
        lx_m=float(x_m[-1] - x_m[0]), lz_m=float(z_m[-1] - z_m[0]), centering="node")
    solver_native = LWC84CPMLSolver(
        grid=grid_native, boundaries=BoundaryConfig(npml=20), dt_s=dt_native,
        output_times_s=time_s, c_ref_mps=6750.0, device=device, dtype=torch.float32,
        output_restriction_factor=1)

    def run(solver, velocity, entry):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = solver.simulate(
            velocity,
            source_x_m=entry["source_x_m"], source_z_m=entry["source_z_m"],
            source_f0_hz=entry["source_f0_hz"], source_t0_s=entry["source_t0_s"],
            source_amplitude=entry["source_amplitude"])
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if result.wavefield.shape != (1, 401, 201, 201):
            raise SystemExit(f"solver returned shape {result.wavefield.shape}")
        qc = dict(result.metrics[0])
        if not bool(qc["finite"]) or float(qc["lwc_qmax"]) >= 1.0:
            raise SystemExit(f"unstable solve: {qc}")
        if float(np.abs(result.wavefield[0, :, 0, :]).max()) != 0.0:
            raise SystemExit("free surface row is nonzero")
        return elapsed, result.wavefield[0]

    # Warm-up both solvers once (kernel compilation, allocator) before timing.
    first = inputs[RECORDS[0]]
    run(solver_fine, fine_velocity(first["velocity"], (grid_fine.nz, grid_fine.nx)), first)
    run(solver_native, first["velocity"], first)

    measurements = []
    accuracy = []
    for sample_id in RECORDS:
        entry = inputs[sample_id]
        future = np.arange(entry["onset"] + IC_FRAMES, len(time_s))
        bands = [np.asarray(b) for b in np.array_split(future, 3)]
        for label, solver, velocity in (
                ("fine_401_dt1.25e-4", solver_fine,
                 fine_velocity(entry["velocity"], (grid_fine.nz, grid_fine.nx))),
                ("native_201_dt2.5e-4", solver_native, entry["velocity"])):
            wavefield = None
            for repeat in range(args.repeats):
                elapsed, wavefield = run(solver, velocity, entry)
                measurements.append({
                    "solver": label, "sample_id": sample_id,
                    "family": entry["medium_type"], "repeat": repeat,
                    "runtime_s": elapsed,
                })
            accuracy.append({
                "solver": label, "sample_id": sample_id, "family": entry["medium_type"],
                "future_relative_l2": relative_l2(wavefield, entry["truth"], future),
                "time_bands": [relative_l2(wavefield, entry["truth"], b) for b in bands],
                "onset": entry["onset"],
            })
            print(json.dumps(accuracy[-1]), flush=True)

    def stats(label):
        values = [m["runtime_s"] for m in measurements if m["solver"] == label]
        return {
            "n": len(values), "min_s": min(values),
            "mean_s": sum(values) / len(values), "max_s": max(values),
        }

    report = {
        "status": "complete",
        "schema": "lwc84_classical_runtime_accuracy_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "records": list(RECORDS),
            "record_selection": "the paper's figure records: family median by full-future relL2",
            "metric": "relative L2, float64, future window (onset+8..400) and 3 equal bands",
            "single_instance": True,
            "cuda_synchronized_timing": True,
            "includes_output_materialization": True,
            "excludes_disk_io": True,
            "fine_velocity_input": "stored 10 m velocity bilinearly upsampled to 5 m "
                                   "(as benchmark_lwc84_traditional_runtime.py), so the fine "
                                   "solve's error against stored truth includes resampling",
        },
        "source_h5": str(H5),
        "source_h5_sha256": file_sha256(H5),
        "frozen_config": str(FROZEN_CONFIG),
        "frozen_config_sha256": file_sha256(FROZEN_CONFIG),
        "panel_rows": str(PANEL_ROWS),
        "panel_rows_sha256": file_sha256(PANEL_ROWS),
        "code_tree": str(GDH),
        "script_sha256": file_sha256(Path(__file__)),
        "device": {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "native": {"grid": [201, 201], "dx_m": 10.0, "dt_s": dt_native, "steps": 4000,
                   "npml": 20, "restriction": "none"},
        "fine": {"grid": [401, 401], "dx_m": 5.0, "dt_s": dt_fine, "steps": 8000,
                 "restriction": "binomial lowpass + decimation to 201x201 (solver default)"},
        "repeats_per_record": args.repeats,
        "measurements": measurements,
        "accuracy": accuracy,
        "runtime_summary": {label: stats(label)
                            for label in ("fine_401_dt1.25e-4", "native_201_dt2.5e-4")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["runtime_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
