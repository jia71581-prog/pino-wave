#!/usr/bin/env python
"""Probe: coarse-grid LWC-84 accuracy / dispersion / cost vs saved-grid resolution.

The saved grid is 201x201 (dx=10 m, 2000 m extent).  We re-solve on progressively coarser
grids, upsample the result back to 201x201, and measure (relative L2, high-wavenumber
spectral error, wallclock) against the stored fine-grid truth.  This locates the
dispersion-limited "traditional competitor" regime for the matched-compute comparison.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig  # noqa: E402
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver  # noqa: E402

H5 = "/root/autodl-tmp/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5"
EXTENT_M = 2000.0
DEVICE = "cuda"


def relative_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.sum((pred - truth) ** 2)) / np.sqrt(np.sum(truth**2)))


def high_k_error(pred: np.ndarray, truth: np.ndarray) -> float:
    """Relative L2 restricted to spatial wavenumbers |k| >= 0.5 Nyquist, averaged over time."""
    p = torch.fft.rfft2(torch.as_tensor(pred))
    t = torch.fft.rfft2(torch.as_tensor(truth))
    nz, nx = pred.shape[-2], pred.shape[-1]
    kz = torch.fft.fftfreq(nz)[:, None]
    kx = torch.fft.rfftfreq(nx)[None, :]
    radius = torch.sqrt(kz**2 + kx**2) / 0.5  # 0.5 cyc/sample == Nyquist along the shorter axis
    mask = (radius >= 0.5).to(p.real.dtype)
    num = torch.sqrt(torch.sum((mask * (p - t).abs()) ** 2))
    den = torch.sqrt(torch.sum((mask * t.abs()) ** 2)).clamp_min(1e-30)
    return float(num / den)


def solve_at_resolution(vel201, src, time_s, *, n, internal_dt_s):
    dx = EXTENT_M / (n - 1)
    grid = AcousticGrid(nx=n, nz=n, dx_m=dx, dz_m=dx, centering="node")
    solver = LWC84CPMLSolver(
        grid=grid, boundaries=BoundaryConfig(npml=20), dt_s=internal_dt_s,
        output_times_s=np.asarray(time_s, dtype=np.float64), c_ref_mps=6750.0,
        device=DEVICE, dtype=torch.float32, output_restriction_factor=1,
    )
    vel = F.interpolate(torch.as_tensor(vel201[None, None], dtype=torch.float32),
                        size=(n, n), mode="bilinear", align_corners=True)[0, 0].numpy().astype(np.float64)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = solver.simulate(vel, source_x_m=src["x"], source_z_m=src["z"],
                          source_f0_hz=src["f0"], source_t0_s=src["t0"], source_amplitude=src["amp"])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    wf = np.asarray(res.wavefield)[0].astype(np.float32)  # [nt,n,n]
    if n != 201:
        wf = F.interpolate(torch.as_tensor(wf)[None], size=(201, 201), mode="bilinear",
                           align_corners=True)[0].numpy()
    return wf, elapsed, dx


def main() -> int:
    with h5py.File(H5, "r") as f:
        med = [s.decode() for s in f["medium_type"][:]]
        idx = {}
        for fam in ("uniform", "layered", "marmousi"):
            idx[fam] = next(i for i, m in enumerate(med) if m.startswith(fam) and f["split"][i].decode() == "validation") \
                if any(m.startswith(fam) for m in med) else None
        time_s = np.asarray(f["time_s"][:], dtype=np.float64)
        records = {}
        for fam, i in idx.items():
            records[fam] = {
                "vel": np.asarray(f["velocity_mps"][i], dtype=np.float64),
                "truth": np.asarray(f["wavefield"][i], dtype=np.float32),
                "src": {"x": float(f["source_x_m"][i]), "z": float(f["source_z_m"][i]),
                        "f0": float(f["source_f0_hz"][i]), "t0": float(f["source_t0_s"][i]),
                        "amp": float(f["source_amplitude"][i])},
                "f0": float(f["source_f0_hz"][i]), "vmin": float(f["vmin_mps"][i]),
            }
    grids = [201, 151, 101, 81, 68, 51]
    print(f"{'fam':9s} {'f0':>5s} {'vmin':>6s} | {'n':>4s} {'dx':>5s} {'ppw':>5s} {'relL2':>8s} {'highK':>7s} {'sec':>6s}")
    for fam, rec in records.items():
        wl_min = rec["vmin"] / (rec["f0"] * 2.0)  # ~ shortest wavelength at ~2*f0 upper band
        for n in grids:
            try:
                pred, sec, dx = solve_at_resolution(rec["vel"], rec["src"], time_s, n=n, internal_dt_s=2.5e-4)
                ppw = wl_min / dx
                r = relative_l2(pred, rec["truth"])
                hk = high_k_error(pred, rec["truth"])
                print(f"{fam:9s} {rec['f0']:5.1f} {rec['vmin']:6.0f} | {n:4d} {dx:5.1f} {ppw:5.2f} {r:8.4f} {hk:7.3f} {sec:6.2f}")
            except Exception as e:
                print(f"{fam:9s} n={n}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
