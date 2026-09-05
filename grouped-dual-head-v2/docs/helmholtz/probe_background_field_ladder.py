"""Deep zero-training probe: BACKGROUND-FIELD decomposition ladder (Direction A+).

Direction A verdict: subtracting the UNIFORM-c0 incident field P_inc removes the
early-time near-source high-wavenumber wall (early error 2.3-3.7x lower) BUT degrades
late time -- because the uniform incident field propagates straight out of the domain
while the true field is reflected/trapped inside by the medium, so late scat energy
> total (|scat|/|total| late = 1.29 on marmousi).

Direction B verdict: a FIXED single-frequency Helmholtz iteration core is falsified --
the stored field's single rfft bin is NOT a time-harmonic Helmholtz field (finite time
window + CPML + transient source), so a fixed spectral resolvent iterates to the wrong
fixed point.

Key surviving idea (this probe): the TIME-DOMAIN solver field is bit-exact (self-check
0.0000) and involves NO time-harmonic assumption. So replace the uniform-c0 background
with a solver run on a SMOOTHED velocity. A smoothed background CONTAINS the reflectors
(so it tracks the late field, fixing A's late blow-up) yet is SMOOTH-medium (cheaper to
solve, more learnable residual). The learned network then only supplies the FINE-SCALE
scattering correction  scat = total - P_bg.

Ladder of backgrounds P_bg, each a real LWC84 solve, per record:
    sigma=inf  -> uniform c0            (= Direction A incident; direct wave only)
    sigma=16,8,4,2 cells -> gaussian-smoothed true velocity (progressively more medium)
    sigma=0    -> true velocity          (= exact; upper reference, scat==0)

For each background, per time bin (early/mid/late), zero training, measure the LEARNED
TARGET COMPLEXITY of the residual scat = wf - wf_bg:
    (1) |scat|/|total|      : how much is left to learn (lower = better) -- MUST be low
                              in LATE too, unlike uniform incident.
    (2) spatial low-k frac  : is the residual smooth (low |k|)? (higher = more learnable)
    (3) K-freq recon error  : truncate scat to (K time-freq, |k|<=40) + add exact P_bg,
                              inverse-FFT, rel-L2 per bin. This is the ORACLE FLOOR for a
                              network that learns a smooth low-rank scat on THIS background.
Compare every background's oracle floor to the pure-total baseline (0.206 converged / the
truncation floor ~0.05-0.09). The winning background minimizes the oracle floor across ALL
bins -- especially it must beat uniform-incident's late degradation.

HONESTY: uniform-c0 background needs only (c0, source) = trivial medium (cheap, generalizes
to any new velocity). Smoothed-velocity background needs a real (if cheaper, smooth-medium)
solve per new velocity -> a HYBRID solver, not a pure surrogate. Both are legitimate; the
probe reports the accuracy each buys so the cost/accuracy trade is explicit.
"""
from __future__ import annotations

import sys
import numpy as np
import h5py
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, "src")
from fno_acoustic.data_generation.config import (  # noqa: E402
    load_config, resolve_config, grid_from_config, boundaries_from_config, time_from_config,
)
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver  # noqa: E402

DATA_DIR = "/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1"
H5 = f"{DATA_DIR}/dataset_v1.h5"
FROZEN = f"{DATA_DIR}/frozen_config.yaml"
FAMILIES = ("uniform", "layered", "anomaly", "marmousi")
PER_FAMILY = 1
K_TIME = 64
M_SPACE = 40
# sigma in SAVED-grid (201, dx=10m) cells; None = uniform c0; 0 = true velocity
SIGMAS = (None, 16.0, 8.0, 4.0, 2.0, 0.0)


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def spatial_lowk_fraction(P, m):
    F, nz, nx = P.shape
    S = np.fft.fft2(P, axes=(1, 2))
    kz = np.fft.fftfreq(nz) * nz
    kx = np.fft.fftfreq(nx) * nx
    KZ, KX = np.meshgrid(kz, kx, indexing="ij")
    disk = (KZ ** 2 + KX ** 2) <= (m * m)
    e_tot = (np.abs(S) ** 2).sum()
    return float((np.abs(S[:, disk]) ** 2).sum() / (e_tot + 1e-30))


def truncate_reconstruct(P, keep, m, nt):
    F, nz, nx = P.shape
    kz = np.fft.fftfreq(nz) * nz
    kx = np.fft.fftfreq(nx) * nx
    KZ, KX = np.meshgrid(kz, kx, indexing="ij")
    disk = (KZ ** 2 + KX ** 2) <= (m * m)
    Pk = np.zeros_like(P)
    for j in keep:
        S = np.fft.fft2(P[j]); S[~disk] = 0.0; Pk[j] = np.fft.ifft2(S)
    return np.fft.irfft(Pk, n=nt, axis=0)


def main():
    cfg = resolve_config(load_config(FROZEN))
    grid = grid_from_config(cfg); bnd = boundaries_from_config(cfg); tg = time_from_config(cfg)
    dt = float(cfg["time"]["dt_used_s"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    solver = LWC84CPMLSolver(
        grid=grid, boundaries=bnd, dt_s=dt, output_times_s=tg.t_s, c_ref_mps=6750.0,
        device=device, dtype=torch.float32, kappa_max=float(cfg["boundaries"]["kappa_max"]),
        minimum_frequency_hz=float(cfg["boundaries"]["minimum_frequency_hz"]),
    )
    f = h5py.File(H5, "r")
    time_s = f["time_s"][:]; nt = len(time_s)
    x_m = f["x_m"][:]; z_m = f["z_m"][:]
    mt = np.array([s.decode() for s in f["medium_type"][:]])
    nz_fine, nx_fine = grid.nz, grid.nx

    def upsample_to_fine(vel201):
        t = torch.as_tensor(vel201[None, None], dtype=torch.float32)
        up = torch.nn.functional.interpolate(t, size=(nz_fine, nx_fine), mode="bilinear",
                                             align_corners=True)
        return up[0, 0].numpy().astype(np.float64)

    def background(ridx, sigma):
        vel = f["velocity_mps"][ridx].astype(np.float64)
        zs = float(f["source_z_m"][ridx]); xs = float(f["source_x_m"][ridx])
        iz = int(np.argmin(np.abs(z_m - zs))); ix = int(np.argmin(np.abs(x_m - xs)))
        if sigma is None:
            velfine = np.full((nz_fine, nx_fine), float(vel[iz, ix]), dtype=np.float64)
        elif sigma == 0.0:
            velfine = upsample_to_fine(vel)
        else:
            velfine = upsample_to_fine(gaussian_filter(vel, sigma=sigma, mode="nearest"))
        res = solver.simulate(
            velfine[None], source_x_m=xs, source_z_m=zs,
            source_f0_hz=float(f["source_f0_hz"][ridx]), source_t0_s=float(f["source_t0_s"][ridx]),
            source_amplitude=float(f["source_amplitude"][ridx]),
        )
        return np.asarray(res.wavefield)[0]

    def bins(rec, truth):
        o = {}
        for name, lo, hi in [("early", 0, nt // 3), ("mid", nt // 3, 2 * nt // 3),
                             ("late", 2 * nt // 3, nt)]:
            o[name] = rel_l2(rec[lo:hi], truth[lo:hi])
        return o

    print(f"nt={nt} dt={dt*1e3:.3f}ms fine={nz_fine}x{nx_fine} device={device} K={K_TIME} m={M_SPACE}")
    print("sigma in saved-grid cells (dx=10m); None=uniform-c0, 0=true-velocity")
    print("=" * 104)

    for fam in FAMILIES:
        idxs = np.where(np.char.startswith(mt, fam))[0][:PER_FAMILY]
        for ridx in idxs:
            ridx = int(ridx)
            wf = f["wavefield"][ridx].astype(np.float64)
            P_total = np.fft.rfft(wf, axis=0)
            fe = (np.abs(P_total) ** 2).sum(axis=(1, 2))
            keep = np.sort(np.argsort(fe)[::-1][:K_TIME])
            # baseline: pure total truncation (the current 0.206-direction oracle floor)
            base = bins(truncate_reconstruct(P_total, keep, M_SPACE, nt), wf)
            print(f"[{fam:8s} #{ridx:4d}]  PURE-TOTAL oracle floor: "
                  f"early={base['early']:.3f} mid={base['mid']:.3f} late={base['late']:.3f}")
            for sigma in SIGMAS:
                wf_bg = background(ridx, sigma)
                scat = wf - wf_bg
                P_scat = np.fft.rfft(scat, axis=0)
                # residual energy per bin
                def scat_frac(lo, hi):
                    return float(np.linalg.norm(scat[lo:hi]) / (np.linalg.norm(wf[lo:hi]) + 1e-30))
                sf = (scat_frac(0, nt // 3), scat_frac(nt // 3, 2 * nt // 3), scat_frac(2 * nt // 3, nt))
                lowk = spatial_lowk_fraction(P_scat[keep], M_SPACE)
                # oracle: truncate scat + exact background
                rec = truncate_reconstruct(P_scat, keep, M_SPACE, nt) + wf_bg
                b = bins(rec, wf)
                tag = "uniform-c0" if sigma is None else ("TRUE-vel" if sigma == 0.0 else f"smooth s={sigma:.0f}")
                win = "  <<" if (b['early'] < base['early'] and b['late'] <= base['late'] * 1.05) else ""
                print(f"    {tag:12s}  scatE early={sf[0]:.2f} mid={sf[1]:.2f} late={sf[2]:.2f} | "
                      f"lowk={lowk:.3f} | ORACLE early={b['early']:.3f} mid={b['mid']:.3f} late={b['late']:.3f}{win}")
    print("=" * 104)
    print("READ: winning background = ORACLE early << baseline AND late <= baseline (fixes A's late).")
    print("      uniform-c0 should show A's pattern (early win, late loss). A smoothed background that")
    print("      keeps late healthy => 'solver-on-smoothed-velocity + learned residual' is the architecture.")


if __name__ == "__main__":
    sys.exit(main() or 0)
