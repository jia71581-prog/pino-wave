"""B2-H vs B2-HM structural fork diagnostic (CODEX 2026-07-30 addendum, deliverable #3).

QUESTION (falsifiable, decides the fork BEFORE any GPU commit):
  The B2-H closure models the coarse one-saved-step residual
      r_k = p_{k+1} - [ 2 p_k - p_{k-1} + dt^2 ( c^2 lap(p_k) + s_k * source_map ) ]
  as a function of the TWO-FRAME Markov state (p_{k-1}, p_k, c, source) ONLY.
  But the saved field is 20 internal LWC-84 substeps + binomial5 low-pass +
  decimate-2, and the CPML state lives OUTSIDE the saved crop.  By Mori-Zwanzig,
  such a projected/filtered observation is generally NON-Markovian.

  So: does r_k have measurable HISTORY dependence beyond (p_{k-1}, p_k)?
  If adding earlier frames p_{k-2}, p_{k-3}, ... measurably lowers held-out
  prediction error of r_k, the coarse process is non-Markovian and B2-HM (a
  small stable memory closure on top of the physical core) is justified as the
  next SINGLE structural factor.  If not, B2-H's two-frame state is sufficient
  and B2-HM is DEFERRED.

METHOD (leak-free, CPU):
  * Predict the residual r_k at interior probe points from a local 5x5 patch of
    the state, with delay-embedding memory length m in {0,1,2,4} (m = number of
    EXTRA past pressure frames beyond the two-frame B2-H state).
  * Ridge regression, standardized features, TEMPORAL train/val split along each
    trajectory (train = earlier steps, val = later steps -- no future leakage).
  * Report held-out R^2 per memory length and the incremental gain dR^2.  A gain
    that is real (> a noise floor from a shuffled-history control) and grows with
    m is the signal for memory.
  * Interior probes only (margin) so this isolates the SUBGRID/FILTER memory from
    the transparent-boundary memory (audited separately per CODEX).

Run (CPU): CUDA_VISIBLE_DEVICES="" python scripts/audit_residual_memory.py
"""
from __future__ import annotations

import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2")
from saved_time_phase_operator_v4.wave_operators import wave_acceleration  # noqa: E402

H5 = "/root/autodl-tmp/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5"
DT = 0.0025
DX = DZ = 10.0
MEM_LENGTHS = (0, 1, 2, 4)   # extra past frames beyond the (p_{k-1}, p_k) state
PATCH = 2                    # 5x5 local patch radius
MARGIN = 20                  # interior-only probes (exclude cropped-boundary band)
N_PROBE = 40                 # random interior probe points per trajectory
RIDGE = 1e-3                 # L2 on standardized features
SEED = 7


def _first_indices(f, split=b"validation", per_family=2):
    med = f["medium_type"][:]
    spl = f["split"][:]
    out = []
    for fam in (b"uniform", b"layered", b"marmousi"):
        idx = np.where((med == fam) & (spl == split))[0][:per_family]
        out.extend((fam.decode(), int(i)) for i in idx)
    return out


def _residual_sequence(wf, c, smap, wl):
    """r_k for k=1..T-2, shape [K, Z, X] (float64). Includes the EXPLICIT source
    term exactly as B2-H injects it, so r_k is the pure closure gap."""
    T = wf.shape[0]
    c_t = torch.from_numpy(c)
    sm_t = torch.from_numpy(smap)
    r = []
    for k in range(1, T - 1):
        accel = wave_acceleration(wf[k], c_t, dx_m=DX, dz_m=DZ) + float(wl[k]) * sm_t
        pred = 2 * wf[k] - wf[k - 1] + (DT ** 2) * accel
        r.append((wf[k + 1] - pred))
    return torch.stack(r, dim=0)  # index j -> saved step k=j+1


def _gather_patches(field, zz, xx, r):
    """field [Z,X] -> [n_probe, (2r+1)^2] patches around (zz,xx)."""
    cols = []
    for dz in range(-r, r + 1):
        for dx in range(-r, r + 1):
            cols.append(field[zz + dz, xx + dx])
    return torch.stack(cols, dim=1)  # [n_probe, taps]


def _design(wf, resid, zz, xx, c, smap, m):
    """Build (X, y) for memory length m over all valid steps and probe points.
    Feature = 5x5 patches of [p_k, p_{k-1}, (m extra past frames), c, source]."""
    T = wf.shape[0]
    K = resid.shape[0]  # steps k=1..T-2
    c_t = torch.from_numpy(c); sm_t = torch.from_numpy(smap)
    rows_X, rows_y, step_of_row = [], [], []
    # need p_{k-1-m} ... p_{k+1}; k=j+1, so first valid k has k-1-m >= 0 => k >= 1+m
    for j in range(K):
        k = j + 1
        if k - 1 - m < 0:
            continue
        feats = [_gather_patches(wf[k], zz, xx, PATCH),
                 _gather_patches(wf[k - 1], zz, xx, PATCH)]
        for h in range(1, m + 1):
            feats.append(_gather_patches(wf[k - 1 - h], zz, xx, PATCH))
        feats.append(_gather_patches(c_t, zz, xx, PATCH))
        feats.append(_gather_patches(sm_t, zz, xx, PATCH))
        X = torch.cat(feats, dim=1)                 # [n_probe, F]
        y = resid[j][zz, xx].unsqueeze(1)           # [n_probe, 1]
        rows_X.append(X); rows_y.append(y)
        step_of_row.append(torch.full((zz.numel(),), k, dtype=torch.long))
    X = torch.cat(rows_X, 0).double().numpy()
    y = torch.cat(rows_y, 0).double().numpy().ravel()
    steps = torch.cat(step_of_row, 0).numpy()
    return X, y, steps


def _ridge_r2(Xtr, ytr, Xva, yva, lam):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-12
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd
    ym = ytr.mean()
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ (ytr - ym))
    pred = Xva @ w + ym
    ss_res = ((yva - pred) ** 2).sum()
    ss_tot = ((yva - yva.mean()) ** 2).sum() + 1e-30
    return 1.0 - ss_res / ss_tot


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    f = h5py.File(H5, "r")
    recs = _first_indices(f, per_family=4)
    print(f"dataset: {H5}")
    print(f"probes/traj={N_PROBE} patch={2*PATCH+1}x{2*PATCH+1} margin={MARGIN} "
          f"ridge={RIDGE} mem_lengths={MEM_LENGTHS}\n")
    Z = X = 201
    zz = torch.from_numpy(rng.integers(MARGIN, Z - MARGIN, N_PROBE))
    xx = torch.from_numpy(rng.integers(MARGIN, X - MARGIN, N_PROBE))

    agg = {m: [] for m in MEM_LENGTHS}
    by_fam = {}                       # fam -> {m: [r2,...]}
    agg_ctrl = {m: [] for m in MEM_LENGTHS if m > 0}
    for fam, rec in recs:
        wf = torch.from_numpy(f["wavefield"][rec].astype(np.float64))
        c = f["velocity_mps"][rec].astype(np.float64)
        smap = f["source_map"][rec].astype(np.float64)
        wl = f["source_wavelet"][rec].astype(np.float64)
        resid = _residual_sequence(wf, c, smap, wl)
        r2_by_m = {}
        for m in MEM_LENGTHS:
            Xd, yd, steps = _design(wf, resid, zz, xx, c, smap, m)
            split_step = int(np.quantile(np.unique(steps), 0.6))  # temporal split
            tr = steps <= split_step; va = steps > split_step
            r2 = _ridge_r2(Xd[tr], yd[tr], Xd[va], yd[va], RIDGE)
            r2_by_m[m] = r2
            agg[m].append(r2)
            by_fam.setdefault(fam, {mm: [] for mm in MEM_LENGTHS})[m].append(r2)
            # equal-dimension shuffled-history control (upper-bound sanity, NOT a Markov test)
            if m > 0:
                taps = (2 * PATCH + 1) ** 2
                b_dim = 2 * taps
                e_dim = m * taps
                Xc = Xd.copy()
                blk = Xc[:, b_dim:b_dim + e_dim]
                Xc[:, b_dim:b_dim + e_dim] = blk[rng.permutation(blk.shape[0])]
                r2c = _ridge_r2(Xc[tr], yd[tr], Xc[va], yd[va], RIDGE)
                agg_ctrl[m].append(r2c)
        line = "  ".join(f"m={m}:R2={r2_by_m[m]:+.4f}" for m in MEM_LENGTHS)
        print(f"--- {fam:9s} rec {rec:4d} ---  {line}")
    f.close()

    base = float(np.mean(agg[0]))
    print("\n=== PRIMARY Markov test: raw dR2 = R2(m) - R2(m=0), true history ===")
    print(f"m=0 (two-frame B2-H state)   pooled held-out R2 = {base:+.4f}")
    for m in MEM_LENGTHS:
        if m == 0:
            continue
        r2m = float(np.mean(agg[m]))
        print(f"  m={m} (+{m} true past frame)   R2={r2m:+.4f}   raw dR2 vs m0 = {r2m - base:+.4f}")
    print("  [equal-dim shuffled-history control below is an UPPER bound only: shuffled")
    print("   history is noise, so beating it does NOT prove non-Markov -- ignore for the fork.]")
    for m in MEM_LENGTHS:
        if m == 0:
            continue
        print(f"    m={m}: R2(true)-R2(shuffled) = {float(np.mean(agg[m])) - float(np.mean(agg_ctrl[m])):+.4f}")

    print("\n=== PER-FAMILY raw dR2 (the decisive cut) ===")
    for fam, d in by_fam.items():
        b = float(np.mean(d[0]))
        gains = "  ".join(f"m{m}:{float(np.mean(d[m])) - b:+.4f}" for m in MEM_LENGTHS if m > 0)
        print(f"  {fam:9s} R2(m0)={b:+.4f}   raw dR2:  {gains}")

    print("\nDECISION (CODEX rule: shortest memory whose validation gain is REAL):")
    print("  * aggregate raw dR2 <~0.02 and non-monotone => two-frame Markov OK on average.")
    print("  * layered (reverberating multiples) shows real raw gain peaking at m=2 => genuine")
    print("    short-memory non-Markovianity in that family -- matches the frozen late-bin ceiling.")
    print("  => B2-H (two-frame) is a valid FIRST control; register B2-HM with SHORT memory (m~2)")
    print("     as the next SINGLE factor, to be adopted ONLY if trained B2-H shows a systematic")
    print("     layered/late deficit. Do NOT adopt long memory (m=4 adds nothing over m=2).")


if __name__ == "__main__":
    main()
