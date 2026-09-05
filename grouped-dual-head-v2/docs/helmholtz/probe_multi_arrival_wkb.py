#!/usr/bin/env python3
"""Zero-training ORACLE probe: does a MULTI-ARRIVAL WKB carrier represent the late
wavefield better than the single-arrival carrier A+1 uses?

Diagnosis (VERDICT sec 22): adding rank to the Helmholtz synthesis head did NOT reduce
the late residual rank -- capacity is not the bottleneck. Re-diagnosis: A+1's WKB carrier
is SINGLE-ARRIVAL, p(t,x) = sum_j [a_j cos(w_j(t-tau)) + b_j sin(...)], tau = eikonal
FIRST arrival. Late multiples are the same wavefront geometry re-arriving LATER (direct +
reflections), i.e. arrival times tau_0=tau < tau+Delta_1 < tau+Delta_2. A single tau
collapses them; no rank recovers the discarded multi-arrival phase.

Method (TRUE oracle, no optimizer): for fixed frequencies and fixed candidate delays
{Delta_k}, the synthesis is LINEAR in the per-pixel coefficients a_kj, b_kj, so we solve
the exact per-pixel least-squares (batched torch.linalg.lstsq) that best fits the true
late field -- no learning-rate / divergence artifacts (the field is ~1e-8, which broke a
naive Adam fit). K=1 reproduces A+1's single-arrival ceiling; K=2,3 add reflected
arrivals with a small delay grid search. If late relL2 drops sharply with K, multi-arrival
is the correct representation and the architecture fix is multiple carriers. The design
matrix has 2*K*nf columns per pixel; only tau (hence the phase) varies per pixel, so we
build the [HW, Tl, 2Knf] basis and batch-solve.

Reality check for learnability: report the cross-(arrival,freq) amplitude rank -- A+1 uses
rank-8, so if the multi-arrival coefficients stay low-rank the fix is learnable.
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


def _load(source_h5, travel_h5, source_index, sample_id, device, stride=2, background_cache=None):
    with h5py.File(source_h5, "r", swmr=True) as f:
        wf = torch.tensor(np.asarray(f["wavefield"][source_index], dtype=np.float32), device=device)
        ts = torch.tensor(np.asarray(f["time_s"][:], dtype=np.float64), device=device)
    with h5py.File(travel_h5, "r", swmr=True) as f:
        sids = [s.decode() if isinstance(s, bytes) else s for s in f["sample_id"][:]]
        row = sids.index(sample_id)
        tau = torch.tensor(np.asarray(f["travel_time_s"][row], dtype=np.float32), device=device)
    if background_cache:
        # Fit the SCATTERING residual scat = wf - P_bg (the synthesis head's real target)
        # instead of the full field. P_bg cache stores physical fields aligned by sample_id.
        from saved_time_phase_operator_v4.background_field import BackgroundFieldProvider
        bg = BackgroundFieldProvider(background_cache)
        pbg = bg.physical([sample_id], torch.arange(wf.shape[0], device="cpu")[None],
                          device=device, dtype=wf.dtype)[0]   # [T,H,W] physical
        wf = wf - pbg
    wf = wf[:, ::stride, ::stride]
    tau = tau[::stride, ::stride]
    return wf, ts, tau


def _basis(tl, tau_flat, omega, delays):
    """Design matrix [HW, Tl, 2*K*nf] for arrivals at tau+Delta_k, cos & sin per freq.

    tl: [Tl] float32 late times; tau_flat: [HW]; omega: [nf]; delays: [K].
    Scaled to unit-ish columns (cos/sin are already O(1)).
    """
    HW = tau_flat.shape[0]
    Tl = tl.shape[0]
    nf = omega.shape[0]
    K = delays.shape[0]
    cols = []
    for k in range(K):
        # retarded time (t - tau - Delta_k): [Tl, HW]
        tmt = tl[:, None] - tau_flat[None, :] - float(delays[k])
        arg = omega[None, :, None] * tmt[:, None, :]         # [Tl, nf, HW]
        cols.append(torch.cos(arg))
        cols.append(torch.sin(arg))
    # stack -> [Tl, 2K*nf, HW] -> [HW, Tl, 2Knf]
    B = torch.cat(cols, dim=1).permute(2, 0, 1).contiguous()  # [HW, Tl, 2Knf]
    return B


def solve_relL2(wf, ts, tau, *, delays, nf, late_start, device):
    T, H, W = wf.shape
    late = slice(late_start, T)
    p = wf[late].reshape(wf[late].shape[0], H * W).permute(1, 0).contiguous()  # [HW, Tl]
    tl = ts[late].float()
    dt = float((ts[-1] - ts[0]) / (len(ts) - 1))
    df = 1.0 / (len(ts) * dt)
    omega = (2 * np.pi) * torch.arange(nf, device=device, dtype=torch.float32) * df
    tau_flat = tau.reshape(H * W)
    delays_t = torch.tensor(delays, device=device, dtype=torch.float32)
    B = _basis(tl, tau_flat, omega, delays_t)               # [HW, Tl, 2Knf]
    # Ridge-regularized normal equations (stable under rank-deficiency: WKB columns are
    # redundant for pixels where tau > t so plain lstsq/gels fails). c = (BtB+lI)^-1 Bt p.
    Bt = B.transpose(1, 2)                                   # [HW, 2Knf, Tl]
    ncol = B.shape[2]
    BtB = torch.bmm(Bt, B)                                   # [HW, 2Knf, 2Knf]
    lam = 1e-6 * BtB.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1e-12)  # [HW]
    eye = torch.eye(ncol, device=device, dtype=B.dtype)[None]
    BtB = BtB + lam[:, None, None] * eye
    Btp = torch.bmm(Bt, p.unsqueeze(-1))                    # [HW, 2Knf, 1]
    coeff = torch.linalg.solve(BtB, Btp)                    # [HW, 2Knf, 1]
    pred = torch.bmm(B, coeff).squeeze(-1)                  # [HW, Tl]
    rel = float((pred - p).norm() / p.norm().clamp_min(1e-30))
    # cross-(arrival,freq) amplitude rank: coeff -> [HW, 2, K, nf] (cos/sin interleaved as
    # [K blocks of (cos_nf, sin_nf)]); build |a|+|b| per (k,freq), SVD over (K*nf, HW).
    K = delays_t.shape[0]
    c = coeff.squeeze(-1).reshape(H * W, K, 2, nf)          # [HW,K,2,nf]
    amp = torch.sqrt(c[..., 0, :] ** 2 + c[..., 1, :] ** 2)  # [HW,K,nf]
    amp = amp.reshape(H * W, K * nf).permute(1, 0)          # [K*nf, HW]
    S = torch.linalg.svdvals(amp.double())
    cum = S.cumsum(0) / S.sum()
    rank95 = int((cum < 0.95).sum()) + 1
    return rel, rank95


def run(*, source_h5, travel_h5, source_index, sample_id, nf, late_start, device_name, background_cache=None):
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    wf, ts, tau = _load(source_h5, travel_h5, source_index, sample_id, device, background_cache=background_cache)
    # candidate reflection delays (seconds); small grid. K=1 uses [0]; K>=2 searches the
    # best single/second extra delay from the grid (greedy: add the delay that helps most).
    grid = [0.02, 0.04, 0.06, 0.08, 0.12, 0.16]
    rows = []
    # K=1
    r1, rk1 = solve_relL2(wf, ts, tau, delays=[0.0], nf=nf, late_start=late_start, device=device)
    rows.append({"k_arrivals": 1, "delays_s": [0.0], "late_relL2": round(r1, 5), "amp_rank95": rk1})
    print(json.dumps(rows[-1], sort_keys=True), flush=True)
    # K=2: pick best second delay
    best2 = None
    for d in grid:
        r, rk = solve_relL2(wf, ts, tau, delays=[0.0, d], nf=nf, late_start=late_start, device=device)
        if best2 is None or r < best2[0]:
            best2 = (r, d, rk)
    rows.append({"k_arrivals": 2, "delays_s": [0.0, best2[1]], "late_relL2": round(best2[0], 5), "amp_rank95": best2[2]})
    print(json.dumps(rows[-1], sort_keys=True), flush=True)
    # K=3: keep best2 delay, add best third
    best3 = None
    for d in grid:
        if abs(d - best2[1]) < 1e-9:
            continue
        r, rk = solve_relL2(wf, ts, tau, delays=[0.0, best2[1], d], nf=nf, late_start=late_start, device=device)
        if best3 is None or r < best3[0]:
            best3 = (r, d, rk)
    rows.append({"k_arrivals": 3, "delays_s": [0.0, best2[1], best3[1]], "late_relL2": round(best3[0], 5), "amp_rank95": best3[2]})
    print(json.dumps(rows[-1], sort_keys=True), flush=True)
    result = {
        "sample_id": sample_id, "source_index": int(source_index), "nf": nf, "late_start": late_start,
        "results_by_k": rows,
        "single_arrival_late_relL2": rows[0]["late_relL2"],
        "best_multi_late_relL2": min(r["late_relL2"] for r in rows),
        "multi_arrival_gain": round(rows[0]["late_relL2"] - min(r["late_relL2"] for r in rows), 5),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-h5", default="/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5")
    ap.add_argument("--travel-h5", default="/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5")
    ap.add_argument("--source-index", type=int, required=True)
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--nf", type=int, default=48)
    ap.add_argument("--late-start", type=int, default=201)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--background-cache", default=None)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    result = run(source_h5=args.source_h5, travel_h5=args.travel_h5, source_index=args.source_index,
                 sample_id=args.sample_id, nf=args.nf, late_start=args.late_start, device_name=args.device, background_cache=args.background_cache)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
