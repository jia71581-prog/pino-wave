#!/usr/bin/env python3
"""Group 3: coarse-grid LWC-84 dispersion comparison (CPU).

Re-solves two paper records with the SAME LWC-84 + three-sided CFS-CPML solver
that generated the dataset, but on coarse node-centred grids:

  coarse51  -- 51x51 nodes, dx=dz=40 m (nearest admissible to the requested
               50x50: the solver requires odd node-centred dimensions).
               Velocity = stored 201x201 velocity nodally decimated stride 4
               (coarse nodes coincide exactly with fine nodes; no smoothing).
  coarse101 -- 101x101 nodes, dx=dz=20 m, stride-2 nodal decimation
               (supplementary point for the resolution trend).

Everything else matches the dataset's native-grid protocol
(benchmark_classical_lwc84_runtime_accuracy.py, native arm): dt=2.5e-4 s
(4000 steps, frame stride 10), npml=20 nodes, c_ref=6750 m/s, float32,
free-surface top, physical point source handled by the solver's own bilinear
stencil with 1/(dx*dz) delta normalisation (no manual source downsampling).
Runs on CPU; wall time recorded but NOT comparable to the GPU runtime table.

Coarse outputs are bilinearly interpolated back to 201x201
(align_corners=True: coarse nodes map exactly onto fine nodes) and scored
against the stored truth with the paper metric: relative L2, float64, future
window (onset+8 .. 400) and 3 equal time bands (np.array_split).

Writes (only) into results/paper_comparisons_20260923/group3_coarse_dispersion/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F

GDH = Path("/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2")
for value in (str(GDH / "src"), str(GDH)):
    if value not in sys.path:
        sys.path.insert(0, value)

from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig  # noqa: E402
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver  # noqa: E402

H5 = Path("/root/autodl-tmp/data/jiayh/data/acoustic_lwc84_frequency_gap_20260922_v1/"
          "combined_dataset_v1.h5")
PRED_DIR = Path("/root/autodl-tmp/staging/scno_homogeneous_longrun_gap15_v1/"
                "evaluations/update_00005000/attempt_001")
OUT = Path("/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/REVIEWER_PACKAGE_20260921/"
           "results/paper_comparisons_20260923/group3_coarse_dispersion")
RECORDS = {  # sample_id -> h5 index (asserted below)
    "train_marmousi_00076": 2176,
    "train_layered_00299": 719,
}
IC_FRAMES = 8
DT_NATIVE = 2.5e-4
C_REF = 6750.0
NPML = 20
LEVELS = {"coarse51": 4, "coarse101": 2}  # name -> decimation stride from 201


def relative_l2(pred: np.ndarray, truth: np.ndarray, frames: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.float64)[frames]
    t = np.asarray(truth, dtype=np.float64)[frames]
    den = float((t ** 2).sum())
    if den <= 0.0:
        raise ValueError("zero denominator")
    return float(np.sqrt(float(((p - t) ** 2).sum()) / den))


def upsample_to_201(field: np.ndarray) -> np.ndarray:
    """(401, n, n) coarse nodal field -> (401, 201, 201), bilinear, exact at nodes."""
    tens = torch.from_numpy(np.ascontiguousarray(field))[:, None]  # (T,1,n,n)
    fine = F.interpolate(tens, size=(201, 201), mode="bilinear", align_corners=True)
    return fine[:, 0].numpy()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    report = {
        "schema": "paper_group3_coarse_lwc_dispersion_v1",
        "solver": "LWC84CPMLSolver (grouped-dual-head-v2, dataset generation solver)",
        "device": "cpu",
        "dt_s": DT_NATIVE,
        "npml_nodes": NPML,
        "c_ref_mps": C_REF,
        "dtype": "float32",
        "records": {},
        "levels": {
            name: {"grid": [201 // stride + 1] * 2, "dx_m": 10.0 * stride,
                   "velocity": f"stored 201x201 nodally decimated stride {stride}"}
            for name, stride in LEVELS.items()
        },
        "metric": "relative L2, float64, future window (onset+8..400) "
                  "and 3 equal bands (np.array_split)",
        "caveats": [
            "50x50 is not admissible: the solver requires odd node-centred grids; "
            "51x51 (dx=40 m) is the nearest admissible configuration.",
            "coarse medium is nodal decimation (no anti-alias smoothing); the coarse "
            "error therefore includes medium-sampling error on top of numerical "
            "dispersion. Both are consequences of running the solver at that grid.",
            "CPU wall times are recorded for provenance only; not comparable to the "
            "GPU runtime table (group 4).",
        ],
    }

    with h5py.File(H5, "r") as handle:
        time_s = np.asarray(handle["time_s"][:], dtype=np.float64)
        x_m = np.asarray(handle["x_m"][:], dtype=np.float64)
        z_m = np.asarray(handle["z_m"][:], dtype=np.float64)
        sample_ids = [v.decode() for v in handle["sample_id"][:]]
        entries = {}
        for sid, idx in RECORDS.items():
            assert sample_ids[idx] == sid, (sid, idx, sample_ids[idx])
            onset = int(np.searchsorted(time_s, float(handle["source_t0_s"][idx])))
            entries[sid] = {
                "index": idx,
                "onset": onset,
                "velocity201": np.asarray(handle["velocity_mps"][idx], dtype=np.float32),
                "truth": np.asarray(handle["wavefield"][idx], dtype=np.float32),
                "source_x_m": float(handle["source_x_m"][idx]),
                "source_z_m": float(handle["source_z_m"][idx]),
                "source_f0_hz": float(handle["source_f0_hz"][idx]),
                "source_t0_s": float(handle["source_t0_s"][idx]),
                "source_amplitude": float(handle["source_amplitude"][idx]),
                "vmax": float(handle["vmax_mps"][idx]),
            }

    solvers = {}
    for name, stride in LEVELS.items():
        n = 201 // stride + 1
        dx = float(x_m[stride] - x_m[0])
        grid = AcousticGrid(nx=n, nz=n, dx_m=dx, dz_m=dx,
                            lx_m=float(x_m[-1] - x_m[0]), lz_m=float(z_m[-1] - z_m[0]),
                            centering="node")
        solvers[name] = LWC84CPMLSolver(
            grid=grid, boundaries=BoundaryConfig(npml=NPML), dt_s=DT_NATIVE,
            output_times_s=time_s, c_ref_mps=C_REF, device=device,
            dtype=torch.float32, output_restriction_factor=1)

    for sid, entry in entries.items():
        onset = entry["onset"]
        future = np.arange(onset + IC_FRAMES, len(time_s))
        bands = [np.asarray(b) for b in np.array_split(future, 3)]
        truth = entry["truth"]
        rec = {"onset": onset, "h5_index": entry["index"],
               "source_f0_hz": entry["source_f0_hz"],
               "cfl": {}, "solves": {}}

        # neural prediction (future window only) for the comparison numbers
        pred = np.load(PRED_DIR / f"{sid}_prediction.npy")
        assert pred.shape == (len(time_s) - future[0], 201, 201), pred.shape
        pred_full = np.zeros_like(truth)
        pred_full[future[0]:] = pred
        rec["neural_operator"] = {
            "future_relative_l2": relative_l2(pred_full, truth, future),
            "time_bands": [relative_l2(pred_full, truth, b) for b in bands],
            "source": str(PRED_DIR / f"{sid}_prediction.npy"),
        }

        for name, stride in LEVELS.items():
            velocity = entry["velocity201"][::stride, ::stride].copy()
            started = time.perf_counter()
            result = solvers[name].simulate(
                velocity,
                source_x_m=entry["source_x_m"], source_z_m=entry["source_z_m"],
                source_f0_hz=entry["source_f0_hz"], source_t0_s=entry["source_t0_s"],
                source_amplitude=entry["source_amplitude"])
            elapsed = time.perf_counter() - started
            qc = dict(result.metrics[0])
            if not bool(qc["finite"]) or float(qc["lwc_qmax"]) >= 1.0:
                raise SystemExit(f"unstable coarse solve {sid}/{name}: {qc}")
            coarse = np.asarray(result.wavefield[0], dtype=np.float32)
            n = 201 // stride + 1
            assert coarse.shape == (len(time_s), n, n), coarse.shape
            if float(np.abs(coarse[:, 0, :]).max()) != 0.0:
                raise SystemExit(f"free surface row nonzero: {sid}/{name}")
            up = upsample_to_201(coarse).astype(np.float32)
            np.save(OUT / f"{sid}_{name}_native.npy", coarse)
            np.save(OUT / f"{sid}_{name}_on201.npy", up)
            rec["cfl"][name] = float(qc.get("cfl_2d", float("nan")))
            rec["solves"][name] = {
                "grid": [n, n], "dx_m": 10.0 * stride,
                "wall_time_cpu_s": elapsed,
                "lwc_qmax": float(qc["lwc_qmax"]),
                "future_relative_l2": relative_l2(up, truth, future),
                "time_bands": [relative_l2(up, truth, b) for b in bands],
                "native_npy": str(OUT / f"{sid}_{name}_native.npy"),
                "on201_npy": str(OUT / f"{sid}_{name}_on201.npy"),
            }
            print(json.dumps({sid: {name: rec['solves'][name]['future_relative_l2'],
                                    "bands": rec["solves"][name]["time_bands"],
                                    "wall_s": elapsed}}), flush=True)
        report["records"][sid] = rec

    (OUT / "SOLVES.json").write_text(json.dumps(report, indent=1))
    print("written", OUT / "SOLVES.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
