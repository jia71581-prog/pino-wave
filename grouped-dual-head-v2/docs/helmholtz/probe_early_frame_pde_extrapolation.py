#!/usr/bin/env python3
"""GO/NO-GO probe: can early true frames + the high-order LWC PDE residual alone
determine the late-time wavefield?

Deployment setting (user-clarified 2026-08-02):
  * a numerical solver cheaply computes EARLY frames only (0..K_end, ~20% of the
    time steps, energy already near peak) -- these are true, high-fidelity, and use
    the SAME CPML/LWC-84 config as training data generation;
  * NO mid/late snapshots are available (getting them = solving the whole PDE);
  * fine-tuning supervision = the full-spacetime high-order LWC PDE residual ONLY
    (needs just predicted field + velocity + source, all known).

Before building the full A+1 + PDE fine-tuning method, this probe answers the
make-or-break question: with early frames 0..K_end PINNED to truth and the late
frames (K_end+1 .. T-1) as free variables, does minimizing the source-free
high-order LWC residual on the source-free interior converge the late field to
truth? The homogeneous wave equation is 2nd order in time, so two consecutive
frames + the operator determine the whole trajectory -- but on the coarse saved
dt=2.5ms axis with an 8th/4th-order stencil the discrete well-posedness is what we
actually test here. Zero learning: pure optimization over the field tensor.

If it converges -> pure-physics extrapolation from early frames is feasible and the
fine-tuning method has a sound target. If it does not -> the PDE residual + early
frames underdetermine the late field and we need data/structure, not just physics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from saved_time_phase_operator_v4.instance_adaptation.losses import lwc84_residual


def _rel(pred, truth, sl):
    p = pred[:, sl].reshape(-1)
    t = truth[:, sl].reshape(-1)
    return float((p - t).norm() / t.norm().clamp_min(1e-30))


def run(*, source_h5, source_index, k_end, steps, lr, device_name, time_order,
        init):
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    with h5py.File(source_h5, "r", swmr=True) as h5:
        wf = torch.from_numpy(np.asarray(h5["wavefield"][int(source_index)], dtype=np.float64))
        vel = torch.from_numpy(np.asarray(h5["velocity_mps"][int(source_index)], dtype=np.float64))
        time_s = torch.from_numpy(np.asarray(h5["time_s"][:], dtype=np.float64))
    wf = wf.unsqueeze(0).to(device)          # [1,T,z,x]
    vel = vel.unsqueeze(0).to(device)        # [1,z,x]
    T = wf.shape[1]
    dt = float(time_s[1] - time_s[0])

    early = slice(0, k_end + 1)              # pinned to truth (solver output)
    late = slice(k_end + 1, T)               # free variables to solve for

    # Initialize late frames: either the last known early frame held constant
    # (a naive constant-in-time guess) or small noise -- both are far from truth,
    # so convergence to truth is a genuine test of the PDE+early-frame constraint.
    field = wf.clone()
    with torch.no_grad():
        if init == "hold":
            field[:, late] = wf[:, k_end : k_end + 1].expand(-1, T - k_end - 1, -1, -1)
        elif init == "zero":
            field[:, late] = 0.0
        else:
            raise ValueError("init must be 'hold' or 'zero'")
    baseline_late = _rel(field, wf, late)

    # Only late frames are trainable; early frames stay exactly at truth.
    late_var = field[:, late].clone().requires_grad_(True)
    opt = torch.optim.Adam([late_var], lr=lr)

    rows = []
    for step in range(1, int(steps) + 1):
        opt.zero_grad(set_to_none=True)
        full = torch.cat([wf[:, early].detach(), late_var], dim=1)
        residual = lwc84_residual(
            full, vel, dt=dt, dx=10.0, dz=10.0,
            observed_indices=(0, 1), time_order=int(time_order),
        )
        loss = residual.square().mean()
        loss.backward()
        opt.step()
        if step % max(1, steps // 20) == 0 or step == steps:
            with torch.no_grad():
                full = torch.cat([wf[:, early], late_var], dim=1)
                late_rel = _rel(full, wf, late)
            rows.append({"step": step, "pde": float(loss), "late_relL2": late_rel})
            print(json.dumps(rows[-1], sort_keys=True), flush=True)

    result = {
        "source_index": int(source_index), "k_end": int(k_end), "T": int(T),
        "time_order": int(time_order), "init": init,
        "baseline_late_relL2": baseline_late,
        "final_late_relL2": rows[-1]["late_relL2"] if rows else None,
        "best_late_relL2": min(r["late_relL2"] for r in rows) if rows else None,
        "schedule": rows,
        "converged_to_truth": bool(rows and min(r["late_relL2"] for r in rows) < 0.1),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-h5", default="/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5")
    ap.add_argument("--source-index", type=int, required=True, help="raw dataset record index")
    ap.add_argument("--k-end", type=int, default=80, help="last pinned early frame (0..k_end truth)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--time-order", type=int, default=4, choices=(2, 4))
    ap.add_argument("--init", default="hold", choices=("hold", "zero"))
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    result = run(
        source_h5=args.source_h5, source_index=args.source_index, k_end=args.k_end,
        steps=args.steps, lr=args.lr, device_name=args.device,
        time_order=args.time_order, init=args.init,
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
