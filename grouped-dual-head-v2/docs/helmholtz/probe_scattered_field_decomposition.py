"""Zero-training probe for DIRECTION A: Lippmann-Schwinger incident/scattered split.

Context (from project memory). The strongest query-invariant lead is the
temporal-frequency Helmholtz synthesis field: converged G2 = 0.206 (below Option-B
LPF 0.29), whole-time-axis stable, sub-grid phase alignment. Oracle proves <5% is
representable (64 complex Helmholtz fields |k|<=40). The remaining wall is the
(c, x_s) -> low-rank complex amplitude field G_k(x) MAPPING, and the dominant
residual is EARLY-TIME (0.249) = the near-source high-wavenumber region.

Physical reason for the early-time wall: a point source's frequency-domain near
field is a Hankel function H0(k|x-x_s|) with a log singularity at r->0 carrying all
the high-wavenumber content. A smooth low-rank basis (rank<=6, |k|<=40) cannot fit
an analytic singularity -> "target simple but mapping unlearnable" for early-time.

DIRECTION A. Do not make the network learn the singular near field. Split
analytically (Lippmann-Schwinger):

    P_total(x,w) = P_inc(x,w) + P_scat(x,w)

  P_inc  = field of the SAME source in a UNIFORM reference medium c0. Carries the
           near-source singularity + all incident high-wavenumber content. Computed
           by the SAME LWC84 solver on a constant-c0 fine grid -> numerically
           consistent with the stored field (self-check below is bit-exact on a
           uniform record). ZERO learning.
  P_scat  = P_total - P_inc = the field produced by velocity heterogeneity. Away
           from the source it should be SMOOTHER (lower wavenumber) and LOWER RANK
           than P_total -> the part actually worth learning.

GO / NO-GO (decisive, zero training). For P_total vs P_scat, per family, measure:
  (1) spatial wavenumber content: fraction of spatial spectral energy within |k|<=m.
      If P_scat concentrates at lower |k| than P_total, the |k| budget shrinks.
  (2) cross-frequency rank of the complex amplitude cube (rank @95% energy). Memory:
      P_total gave rank<=6. Lower for P_scat => cheaper mixing head.
  (3) THE KEY METRIC. Reconstruct with a fixed (K time-freqs, |k|<=m space) budget,
      per time bin (early/mid/late):
        - baseline: truncate P_total, inverse transform.
        - split:    truncate ONLY P_scat, then ADD the exact (untruncated) P_inc.
      If split's EARLY-bin error collapses vs baseline, Direction A removes the
      0.249 early-time wall by handing the singularity to the analytic incident term.

c0 (reference velocity): primary = source-local velocity (the medium the incident
wave is born in, so the near-source singularity is captured exactly); also report
mean-velocity as a sanity variant.

Runtime: GPU solver ~seconds/record; PER_FAMILY records per family. Minutes.
"""
from __future__ import annotations

import sys
import numpy as np
import h5py
import torch

sys.path.insert(0, "src")
from fno_acoustic.data_generation.config import (  # noqa: E402
    load_config,
    resolve_config,
    grid_from_config,
    boundaries_from_config,
    time_from_config,
)
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver  # noqa: E402

DATA_DIR = "/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1"
H5 = f"{DATA_DIR}/dataset_v1.h5"
FROZEN = f"{DATA_DIR}/frozen_config.yaml"
FAMILIES = ("uniform", "layered", "anomaly", "marmousi")
PER_FAMILY = 2
K_TIME = 64                 # time-frequency slices kept (oracle used 64)
M_SPACE_LIST = (24, 32, 40) # isotropic spatial |k| disk radius (oracle used <=40)


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def spatial_lowk_fraction(P, m):
    """Fraction of spatial spectral energy within isotropic disk |k|<=m,
    energy-weighted across kept time-frequency slices."""
    F, nz, nx = P.shape
    S = np.fft.fft2(P, axes=(1, 2))
    kz = np.fft.fftfreq(nz) * nz
    kx = np.fft.fftfreq(nx) * nx
    KZ, KX = np.meshgrid(kz, kx, indexing="ij")
    disk = (KZ ** 2 + KX ** 2) <= (m * m)
    e_tot = (np.abs(S) ** 2).sum()
    e_in = (np.abs(S[:, disk]) ** 2).sum()
    return float(e_in / (e_tot + 1e-30))


def crossfreq_rank(P, keep_freq_idx, energy_frac=0.95):
    """Rank of the [Kfreq, npix] complex amplitude matrix at energy_frac (via SVD)."""
    sub = P[keep_freq_idx].reshape(len(keep_freq_idx), -1)
    s = np.linalg.svd(sub, compute_uv=False)
    e = s ** 2
    c = np.cumsum(e) / (e.sum() + 1e-30)
    return int(np.searchsorted(c, energy_frac) + 1)


def truncate_reconstruct(P, keep_freq_idx, m_space, nt):
    """Keep only keep_freq_idx time-frequency slices; within each, keep isotropic
    spatial |k|<=m_space; inverse-transform back to nt frames."""
    F, nz, nx = P.shape
    kz = np.fft.fftfreq(nz) * nz
    kx = np.fft.fftfreq(nx) * nx
    KZ, KX = np.meshgrid(kz, kx, indexing="ij")
    disk = (KZ ** 2 + KX ** 2) <= (m_space * m_space)
    Pk = np.zeros_like(P)
    for j in keep_freq_idx:
        S = np.fft.fft2(P[j])
        S[~disk] = 0.0
        Pk[j] = np.fft.ifft2(S)
    return np.fft.irfft(Pk, n=nt, axis=0)


def bin_errors(rec, truth, nt):
    out = {}
    for name, lo, hi in [("early", 0, nt // 3), ("mid", nt // 3, 2 * nt // 3),
                         ("late", 2 * nt // 3, nt), ("all", 0, nt)]:
        out[name] = rel_l2(rec[lo:hi], truth[lo:hi])
    return out


def main():
    cfg = resolve_config(load_config(FROZEN))
    grid = grid_from_config(cfg)
    bnd = boundaries_from_config(cfg)
    tg = time_from_config(cfg)
    dt = float(cfg["time"]["dt_used_s"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    solver = LWC84CPMLSolver(
        grid=grid, boundaries=bnd, dt_s=dt, output_times_s=tg.t_s,
        c_ref_mps=6750.0, device=device, dtype=torch.float32,
        kappa_max=float(cfg["boundaries"]["kappa_max"]),
        minimum_frequency_hz=float(cfg["boundaries"]["minimum_frequency_hz"]),
    )

    f = h5py.File(H5, "r")
    time_s = f["time_s"][:]
    nt = len(time_s)
    x_m = f["x_m"][:]
    z_m = f["z_m"][:]
    mt = np.array([s.decode() for s in f["medium_type"][:]])

    def make_incident(ridx, c0):
        velfine = np.full((1, grid.nz, grid.nx), float(c0), dtype=np.float64)
        res = solver.simulate(
            velfine,
            source_x_m=float(f["source_x_m"][ridx]),
            source_z_m=float(f["source_z_m"][ridx]),
            source_f0_hz=float(f["source_f0_hz"][ridx]),
            source_t0_s=float(f["source_t0_s"][ridx]),
            source_amplitude=float(f["source_amplitude"][ridx]),
        )
        return np.asarray(res.wavefield)[0]

    print(f"nt={nt} dt={dt*1e3:.3f}ms device={device} K_time={K_TIME} m_space={M_SPACE_LIST}")
    print("=" * 100)

    for fam in FAMILIES:
        idxs = np.where(np.char.startswith(mt, fam))[0][:PER_FAMILY]
        for ridx in idxs:
            ridx = int(ridx)
            wf = f["wavefield"][ridx].astype(np.float64)
            vel = f["velocity_mps"][ridx].astype(np.float64)
            xs = float(f["source_x_m"][ridx]); zs = float(f["source_z_m"][ridx])
            iz = int(np.argmin(np.abs(z_m - zs))); ix = int(np.argmin(np.abs(x_m - xs)))
            c0_local = float(vel[iz, ix])
            c0_mean = float(vel.mean())

            wf_inc = make_incident(ridx, c0_local)
            # self-check for uniform: incident should equal total
            inc_selfcheck = rel_l2(wf_inc, wf)

            P_total = np.fft.rfft(wf, axis=0)
            P_inc = np.fft.rfft(wf_inc, axis=0)
            P_scat = P_total - P_inc

            # relative scattered energy
            scat_frac = float(np.linalg.norm(P_scat) / (np.linalg.norm(P_total) + 1e-30))

            # top-K time-frequency slices by TOTAL field energy (same bins for fair compare)
            fe = (np.abs(P_total) ** 2).sum(axis=(1, 2))
            keep = np.sort(np.argsort(fe)[::-1][:K_TIME])

            # (1) spatial low-k fraction at m=40
            lowk_tot = spatial_lowk_fraction(P_total[keep], 40)
            lowk_scat = spatial_lowk_fraction(P_scat[keep], 40)
            # (2) cross-frequency rank @95%
            rank_tot = crossfreq_rank(P_total, keep)
            rank_scat = crossfreq_rank(P_scat, keep)

            print(f"[{fam:8s} #{ridx:4d}] c0_local={c0_local:6.1f} c0_mean={c0_mean:6.1f} "
                  f"inc_selfcheck={inc_selfcheck:.4f} scat/total_energy={scat_frac:.3f}")
            print(f"    spatial |k|<=40 energy frac:  total={lowk_tot:.4f}  scat={lowk_scat:.4f}")
            print(f"    cross-freq rank@95%:          total={rank_tot:4d}    scat={rank_scat:4d}")

            for m in M_SPACE_LIST:
                rec_tot = truncate_reconstruct(P_total, keep, m, nt)
                rec_scat = truncate_reconstruct(P_scat, keep, m, nt) + wf_inc
                bt = bin_errors(rec_tot, wf, nt)
                bs = bin_errors(rec_scat, wf, nt)
                print(f"    m={m:3d}  BASELINE(total)  early={bt['early']:.3f} mid={bt['mid']:.3f} "
                      f"late={bt['late']:.3f} all={bt['all']:.3f}")
                print(f"          SPLIT(scat+inc)  early={bs['early']:.3f} mid={bs['mid']:.3f} "
                      f"late={bs['late']:.3f} all={bs['all']:.3f}"
                      f"   {'<< early WIN' if bs['early'] < bt['early']*0.7 else ''}")
    print("=" * 100)
    print("READ: (a) inc_selfcheck ~0 on uniform validates the numerically-consistent incident field.")
    print("      (b) If scat has lower |k| content + lower rank than total, the learning target shrinks.")
    print("      (c) DECISIVE: if SPLIT early-bin error << BASELINE early-bin, Direction A removes the")
    print("          0.249 early-time wall by handing the near-source singularity to the analytic P_inc.")


if __name__ == "__main__":
    sys.exit(main() or 0)
