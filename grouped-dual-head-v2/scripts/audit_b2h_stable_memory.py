"""B2-H vs B2-HM fork after correcting the physical baseline.

The earlier audit used one 2.5 ms leapfrog update and raw source-map injection.
That operator is CFL-unstable at dataset vmax and differs from the registered
delta scaling.  This rerun defines r_k against the corrected two-substep LWC-84
composition with exact Ricker q/q_tt.  Only this residual can justify B2-HM.

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

Run: CUDA_VISIBLE_DEVICES=0 python scripts/audit_b2h_stable_memory.py
"""
from __future__ import annotations

import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2")
from saved_time_phase_operator_v4.b2h import PhysicalResidualPropagator  # noqa: E402

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


def _residual_sequence(wf, c, smap, wl, source_parameters, time_s):
    """Corrected stable-core residual r_k for k=1..T-2."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicalResidualPropagator(
        width=8,
        spectral_rank=4,
        modes=4,
        depth=1,
        dt_s=DT,
        dx_m=DX,
        dz_m=DZ,
        gate_init=0.0,
        residual_scale_init=1.0e-4,
        substeps_per_saved_step=2,
        activation_checkpointing=False,
    ).to(device).eval()
    pressure = wf.float()
    velocity = torch.from_numpy(c.astype(np.float32))[None, None].to(device)
    source_map = torch.from_numpy(smap.astype(np.float32))[None, None].to(device)
    parameters = torch.as_tensor(
        source_parameters, dtype=torch.float32, device=device
    )[None]
    times = torch.as_tensor(time_s, dtype=torch.float32, device=device)
    residuals = []
    with torch.no_grad():
        for first in range(1, pressure.shape[0] - 1, 32):
            stop = min(first + 32, pressure.shape[0] - 1)
            count = stop - first
            p0 = pressure[first - 1 : stop - 1, None].to(device)
            p1 = pressure[first:stop, None].to(device)
            target = pressure[first + 1 : stop + 1, None].to(device)
            prediction = model(
                p0,
                p1,
                velocity.expand(count, -1, -1, -1),
                source_map.expand(count, -1, -1, -1),
                torch.from_numpy(
                    wl[first:stop].astype(np.float32)
                ).to(device)[:, None],
                steps=1,
                source_parameters=parameters.expand(count, -1),
                initial_time_s=times[first:stop],
            )[:, 0]
            residuals.append((target - prediction)[:, 0].cpu())
    return torch.cat(residuals, dim=0).double()


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
        source_parameters = np.asarray(
            [
                f["source_f0_hz"][rec],
                f["source_t0_s"][rec],
                f["source_amplitude"][rec],
            ],
            dtype=np.float64,
        )
        resid = _residual_sequence(
            wf,
            c,
            smap,
            wl,
            source_parameters,
            f["time_s"][:],
        )
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

    aggregate_gain = {
        m: float(np.mean(agg[m])) - base for m in MEM_LENGTHS if m > 0
    }
    layered_base = float(np.mean(by_fam["layered"][0]))
    layered_gain = {
        m: float(np.mean(by_fam["layered"][m])) - layered_base
        for m in MEM_LENGTHS
        if m > 0
    }
    shortest_real = next(
        (
            m
            for m in (1, 2, 4)
            if max(aggregate_gain[m], layered_gain[m]) >= 0.02
        ),
        None,
    )
    print("\nDECISION (shortest corrected-baseline gain >= 0.02):")
    if shortest_real is None:
        print("  => RETAIN B2-H; corrected residual has no material finite-memory gain.")
    else:
        print(
            f"  => ADOPT B2-HM with m={shortest_real}; corrected residual retains "
            "material history dependence."
        )
    print(
        {
            "recommended_memory": shortest_real,
            "aggregate_gain": aggregate_gain,
            "layered_gain": layered_gain,
        }
    )


if __name__ == "__main__":
    main()
