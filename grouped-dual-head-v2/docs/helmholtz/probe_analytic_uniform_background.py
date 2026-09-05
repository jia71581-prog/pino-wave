#!/usr/bin/env python3
"""GO/NO-GO: can an ANALYTIC uniform-medium background replace the numerical smoothed-
velocity P_bg, removing the FD solve (P_bg is A+1's ~21s cost bottleneck)?

User idea: the 2-D acoustic point-source field in a UNIFORM medium of speed c has a
closed-form Green's function, so a uniform background needs NO numerical solver. If a
per-record equivalent uniform speed gives a background comparable to the smoothed-velocity
P_bg, A+1's deploy cost drops from ~21s (FD solve) to ~ms (analytic).

Physics. In 2-D, a point source w(t) in a uniform medium c produces (frequency domain)
    P(x, omega) = w_hat(omega) * (i/4) H0^{(1)}(omega r / c),   r = |x - x_s|,
with H0^{(1)} the Hankel function of the first kind. We synthesize this on the saved grid
and inverse-FFT to time. To reduce the analytic-vs-discrete mismatch that sec 10 found with
an ideal Ricker (residual ~0.5), we drive it with the STORED source wavelet's FFT (w_hat =
FFT of the dataset's source_wavelet), not an idealized Ricker.

Tests (per family, held-out records):
  * relL2 of the analytic uniform background vs the true stored field (all-time and
    per-time-bin) -- how good a background is it?
  * for several equivalent-c choices (source-local velocity, whole-field mean, median);
  * baseline references: naive operator (no background) held-out ~0.639; smoothed-velocity
    P_bg full-field ~0.045-0.082 (but costs ~21s).
If the analytic background beats the naive 0.639 substantially, it is a zero-solver
background worth pairing with the learned scattering residual (a speed-accuracy trade),
even if it does not match the numerical P_bg. If it is ~0.6+ (like sec-10 uniform-c0's
late failure), the uniform background is too weak.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch


def analytic_uniform_background(source_wavelet, time_s, x_m, z_m, sx, sz, c, device):
    """2-D uniform-medium field via the TIME-DOMAIN acoustic Green's function convolved with
    the stored source wavelet -- matching the time-domain wave equation the data solves
    (NOT a frequency-domain Hankel / time-harmonic Helmholtz field, which sec 11 showed the
    finite-window CPML data does not satisfy).

    2-D free-space Green's function: G(r,t) = H(c t - r) / (2 pi c^2 sqrt(t^2 - (r/c)^2)),
    a sharp wavefront at t = r/c with a slowly decaying tail (2-D has an afterglow). The
    field is p(x,t) = (w *_t G_r)(t), a per-pixel time convolution of the source wavelet w
    with the pixel's Green kernel. Done by FFT convolution over the time axis in one batch.
    The sqrt singularity at the wavefront is regularized by a half-cell floor.
    """
    T = len(time_s)
    dt = float(time_s[1] - time_s[0])
    w = torch.as_tensor(source_wavelet, dtype=torch.float64, device=device)
    if w.shape[0] != T:
        w = torch.nn.functional.interpolate(w[None, None], size=T, mode="linear", align_corners=True)[0, 0]
    zz, xx = torch.meshgrid(torch.as_tensor(z_m, dtype=torch.float64, device=device),
                            torch.as_tensor(x_m, dtype=torch.float64, device=device),
                            indexing="ij")
    dx = float(x_m[1] - x_m[0])
    r = torch.sqrt((xx - sx) ** 2 + (zz - sz) ** 2).clamp_min(dx * 0.5).reshape(-1)  # [HW]
    HW = r.shape[0]
    tgrid = torch.as_tensor(time_s, dtype=torch.float64, device=device)              # [T]
    # Green kernel G_r(t) on the time grid: nonzero for c t > r, ~1/sqrt(t^2-(r/c)^2).
    tau = (r / c)[:, None]                                                            # [HW,1]
    tt = tgrid[None, :]                                                               # [1,T]
    denom = torch.sqrt((tt * tt - tau * tau).clamp_min((dt * 0.5) ** 2))
    G = torch.where(tt > tau, 1.0 / (2 * np.pi * c * c * denom), torch.zeros_like(denom))  # [HW,T]
    # time convolution (w * G) via FFT, length T (causal, truncated to saved window)
    n = 2 * T
    Wf = torch.fft.rfft(w, n=n)                                                       # [n//2+1]
    Gf = torch.fft.rfft(G, n=n, dim=1)                                               # [HW, n//2+1]
    conv = torch.fft.irfft(Gf * Wf[None, :], n=n, dim=1)[:, :T] * dt                 # [HW,T]
    return conv.T.reshape(T, *zz.shape).float()                                       # [T,Hz,Wx]


def run(*, source_h5, source_index, sample_id, device_name, c_mode):
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    with h5py.File(source_h5, "r", swmr=True) as f:
        wf = torch.tensor(np.asarray(f["wavefield"][source_index], dtype=np.float32), device=device)
        vel = np.asarray(f["velocity_mps"][source_index], dtype=np.float64)
        ts = np.asarray(f["time_s"][:], dtype=np.float64)
        sw = np.asarray(f["source_wavelet"][source_index], dtype=np.float64)
        sx = float(f["source_x_m"][source_index]); sz = float(f["source_z_m"][source_index])
    T, Hz, Wx = wf.shape
    x_m = np.linspace(0, 2000, Wx); z_m = np.linspace(0, 2000, Hz)
    # equivalent uniform speed
    if c_mode == "source_local":
        iz = int(np.clip(sz / 2000 * (vel.shape[0]-1), 0, vel.shape[0]-1))
        ix = int(np.clip(sx / 2000 * (vel.shape[1]-1), 0, vel.shape[1]-1))
        c = float(vel[iz, ix])
    elif c_mode == "mean":
        c = float(vel.mean())
    elif c_mode == "median":
        c = float(np.median(vel))
    else:
        raise ValueError(c_mode)
    import time as _t
    _t0 = _t.time()
    pbg = analytic_uniform_background(sw, ts, x_m, z_m, sx, sz, c, device)  # [T,Hz,Wx]
    cost_s = _t.time() - _t0
    # scale-match (analytic amplitude convention vs stored): best scalar alpha minimizing ||a*pbg - wf||
    a = float((pbg * wf).sum() / (pbg * pbg).sum().clamp_min(1e-30))
    pbg = pbg * a
    def rel(sl):
        return float((pbg[sl] - wf[sl]).norm() / wf[sl].norm().clamp_min(1e-30))
    res = {
        "sample_id": sample_id, "c_mode": c_mode, "c_used": round(c, 1),
        "analytic_cost_s": round(cost_s, 3), "amp_scale": round(a, 4),
        "relL2_all": round(rel(slice(0, T)), 4),
        "relL2_early": round(rel(slice(0, 81)), 4),
        "relL2_mid": round(rel(slice(81, 201)), 4),
        "relL2_late": round(rel(slice(201, T)), 4),
        "ref_naive_operator": 0.639, "ref_smoothed_Pbg_full": "0.045-0.082 (costs ~21s)",
    }
    print(json.dumps(res, sort_keys=True), flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-h5", default="/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5")
    ap.add_argument("--source-index", type=int, required=True)
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--c-mode", default="source_local", choices=("source_local", "mean", "median"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    run(source_h5=args.source_h5, source_index=args.source_index, sample_id=args.sample_id,
        device_name=args.device, c_mode=args.c_mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
