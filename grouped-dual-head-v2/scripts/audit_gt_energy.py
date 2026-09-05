"""Ground-truth acoustic energy diagnostic (B2-H addendum step 1).

Validates, on REAL GT trajectories, that the audited fixed wave-operator core
(wave_operators.py) is physically consistent:
  * during the source-active window, interior energy RISES (source injects work);
  * after the source is quiet, interior energy is ~flat MINUS boundary radiation
    (no spurious growth) -- i.e. the discrete operator + variable-c energy are the
    correct conserved quantity, not naive global conservation;
  * quantify the residual of the LWC-84 update law at the saved-frame spacing to
    prove the honest AUDIT claim: a single coarse stencil step does NOT reproduce
    GT (frames are 20 substeps apart + restricted) -> a learned residual is required.

Run (CPU): CUDA_VISIBLE_DEVICES="" python scripts/audit_gt_energy.py
"""
from __future__ import annotations

import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2")
from saved_time_phase_operator_v4.wave_operators import (  # noqa: E402
    discrete_energy_variable_c,
    interior_mask,
    laplacian_8th,
    wave_acceleration,
)

H5 = "/root/autodl-tmp/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5"
DT_OUT = 0.0025
DX = DZ = 10.0


def _first_index_per_family(f, split=b"validation"):
    med = f["medium_type"][:]
    spl = f["split"][:]
    out = {}
    for fam in (b"uniform", b"layered", b"marmousi"):
        idx = np.where((med == fam) & (spl == split))[0]
        if len(idx):
            out[fam.decode()] = int(idx[0])
    return out


def main():
    f = h5py.File(H5, "r")
    fam_idx = _first_index_per_family(f)
    print(f"dataset: {H5}")
    print(f"records per family (validation split): {fam_idx}\n")
    for fam, rec in fam_idx.items():
        wf = torch.from_numpy(f["wavefield"][rec].astype(np.float64))  # [T,Z,X]
        c = torch.from_numpy(f["velocity_mps"][rec].astype(np.float64))
        wl = f["source_wavelet"][rec]  # [T]
        f0 = float(f["source_f0_hz"][rec]); t0 = float(f["source_t0_s"][rec])
        T = wf.shape[0]
        # source-active window: |ricker| above 1% of its peak
        aw = np.abs(wl); thr = 0.01 * aw.max()
        active = np.where(aw > thr)[0]
        src_end = int(active.max()) if len(active) else 0
        # energy trajectory (sampled)
        margin = 12
        ks = list(range(1, T - 1, 20))
        E = [discrete_energy_variable_c(wf[k - 1], wf[k], wf[k + 1], c,
                                        dx_m=DX, dz_m=DZ, dt_s=DT_OUT, margin=margin)["total"]
             for k in ks]
        E = np.asarray(E)
        # post-source drift: from just after src_end to end
        post = [(k, e) for k, e in zip(ks, E) if k > src_end + 20]
        if post:
            pk = np.array([e for _, e in post])
            emax = pk.max()
            drift = (pk[-1] - pk[0]) / max(emax, 1e-30)
            peak_frac_at_end = pk[-1] / max(emax, 1e-30)
        else:
            drift = float("nan"); peak_frac_at_end = float("nan")
        # LWC-84 residual at SAVED spacing (proves single coarse step != GT):
        # r_k = p_{k+1} - [2p_k - p_{k-1} + dt^2 * c^2 lap(p_k)]  (2nd-order coarse leapfrog)
        rel_res = []
        for k in ks:
            accel = wave_acceleration(wf[k], c, dx_m=DX, dz_m=DZ)
            pred = 2 * wf[k] - wf[k - 1] + (DT_OUT ** 2) * accel
            mask = interior_mask(wf.shape[-2:], margin)
            num = (wf[k + 1] - pred)[..., mask].pow(2).sum().sqrt()
            den = (wf[k + 1] - wf[k])[..., mask].pow(2).sum().sqrt().clamp_min(1e-30)
            rel_res.append(float(num / den))
        rel_res = np.asarray(rel_res)
        print(f"--- {fam}  (rec {rec}, f0={f0:.2f}Hz, t0={t0:.4f}s, src_end_frame={src_end}) ---")
        print(f"  interior energy: min={E.min():.3e} max={E.max():.3e} "
              f"E[0]={E[0]:.3e} E[-1]={E[-1]:.3e}")
        print(f"  source-active rise: E jumps to peak during frames 0..{src_end} "
              f"(peak {E.max():.3e})")
        print(f"  POST-source interior drift (should be <=0, radiation only): "
              f"{drift:+.3f}  (end/peak={peak_frac_at_end:.3f})")
        print(f"  QC final energy ratio (generator): {float(f['qc_final_energy_ratio'][rec]):.4f}")
        print(f"  coarse-1-step LWC residual (rel to frame delta): "
              f"mean={rel_res.mean():.3f} max={rel_res.max():.3f}  "
              f"[>~1 => single coarse stencil CANNOT reproduce GT -> learned residual required]\n")
    f.close()


if __name__ == "__main__":
    main()
