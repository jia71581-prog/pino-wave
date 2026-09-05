#!/usr/bin/env python3
"""GO/NO-GO for observation-conditioned adaptation (ICON / NOP / fine-tuning): do the
OBSERVABLE early true frames contain information about the late scattering residual?

Context. The information-wall verdicts (VERDICT sec 19-24) showed that deployment-available
signals cannot pin A+1's late-time bias: 2 near-zero onset frames carry ~3e-5 of the energy,
A+1's early frames are already accurate (0.4-3.4%, no new info), and mid/late truth is
unobtainable without solving the full PDE. Recent literature offers observation-CONDITIONED
alternatives to weight fine-tuning -- In-Context Operator Networks (ICON, no weight update,
data-prompt conditioning) and Neural Operator Processes (NOP, partial-observation
conditioning). But these change HOW observations are used, not HOW MUCH information the
observations contain. Before implementing any of them, this probe asks the prerequisite:

    Can the early true frames (frames 0..K_end) linearly predict the LATE scattering
    residual scat = p - P_bg (frames > late_start)?

Method (zero-training, upper bound on any conditioning method's linear reach). Build a
low-dimensional context from the early frames -- their leading spatial principal components
per early frame, flattened -- and ridge-regress (shared over pixels' time series is too
local, so we regress the late field's leading spatial PCA coefficients from the early
context). Report the relL2 of the best linear prediction of the late scat.

If low -> early frames DO carry late information -> ICON/NOP/conditioning has headroom the
weight-fine-tune did not exploit. If ~1 (as high as the render feature wall's 0.98) -> the
late information is genuinely absent from the observable early window, and no
observation-conditioned method can recover it (the information wall is fundamental, not a
method artifact).
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


def run(*, source_h5, travel_h5, source_index, sample_id, background_cache,
        k_end, late_start, n_pca, device_name, stride=2):
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    with h5py.File(source_h5, "r", swmr=True) as f:
        wf = torch.tensor(np.asarray(f["wavefield"][source_index], dtype=np.float32), device=device)
    # scattering residual (the synthesis target), subtract physical background
    from saved_time_phase_operator_v4.background_field import BackgroundFieldProvider
    bg = BackgroundFieldProvider(background_cache)
    pbg = bg.physical([sample_id], torch.arange(wf.shape[0], device="cpu")[None],
                      device=device, dtype=wf.dtype)[0]
    scat = (wf - pbg)[:, ::stride, ::stride]                     # [T,Hs,Ws]
    T, Hs, Ws = scat.shape
    HW = Hs * Ws
    early = scat[:k_end + 1].reshape(k_end + 1, HW).double()      # [Ke, HW]
    late = scat[late_start:].reshape(T - late_start, HW).double() # [Tl, HW]

    # Context = leading spatial PCA coefficients of each early frame, concatenated.
    # (A compact, information-preserving summary of the observable early window.)
    def spatial_pca_coeffs(block, k):
        # block [n, HW]; return [n, k] coefficients onto the top-k spatial modes of block
        U, S, Vh = torch.linalg.svd(block, full_matrices=False)   # block = U S Vh
        return (U[:, :k] * S[:k])                                 # [n, k] temporal loadings
    # We want a per-pixel prediction, so instead regress late field from early field via a
    # shared low-rank temporal operator: late[t,x] ~= sum_s W[t,s] early[s,x]. This is the
    # exact linear map from the early time-series to the late time-series, shared over pixels
    # -- the tightest linear test of "early determines late".
    # Solve min_W || W early - late ||^2 with ridge:  W = late early^T (early early^T + lI)^-1
    E = early                                                     # [Ke, HW]
    L = late                                                      # [Tl, HW]
    EEt = E @ E.T                                                 # [Ke, Ke]
    lam = 1e-3 * torch.diagonal(EEt).mean().clamp_min(1e-12)
    W = L @ E.T @ torch.linalg.inv(EEt + lam * torch.eye(k_end + 1, device=device, dtype=torch.float64))
    pred = W @ E                                                  # [Tl, HW]
    rel_linear = float((pred - L).norm() / L.norm().clamp_min(1e-30))

    # Also report the trivial baselines for context.
    rel_zero = 1.0  # predicting zero
    # Correlation of early vs late energy per pixel (is there any pixelwise coupling?)
    e_energy = early.pow(2).sum(0).sqrt()
    l_energy = late.pow(2).sum(0).sqrt()
    corr = float(torch.corrcoef(torch.stack([e_energy, l_energy]))[0, 1])

    result = {
        "sample_id": sample_id, "k_end": k_end, "late_start": late_start,
        "early_frames": k_end + 1, "late_frames": T - late_start,
        "linear_early_to_late_scat_relL2": round(rel_linear, 4),
        "zero_baseline_relL2": rel_zero,
        "early_late_energy_corr": round(corr, 4),
        "interpretation": "early_carries_late_info_if_relL2_well_below_1",
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-h5", default="/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5")
    ap.add_argument("--travel-h5", default="/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5")
    ap.add_argument("--source-index", type=int, required=True)
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--background-cache", required=True)
    ap.add_argument("--k-end", type=int, default=80)
    ap.add_argument("--late-start", type=int, default=201)
    ap.add_argument("--n-pca", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    run(source_h5=args.source_h5, travel_h5=args.travel_h5, source_index=args.source_index,
        sample_id=args.sample_id, background_cache=args.background_cache, k_end=args.k_end,
        late_start=args.late_start, n_pca=args.n_pca, device_name=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
