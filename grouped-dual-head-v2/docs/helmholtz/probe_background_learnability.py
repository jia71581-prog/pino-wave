"""Deep zero-training probe: is the SMOOTHED-MEDIUM background field learnable?

Direction A+ (probe_background_field_ladder) proved: solver on a Gaussian-smoothed
velocity (sigma~2-4 cells) gives a background P_bg whose residual scat = wf - P_bg
reconstructs to <1% (all families, all time bins). That fixes A's late blow-up without
B's time-harmonic premise. The open architecture choice:
  A+1 hybrid solver : run the (cheap) smoothed solve at deploy -> not a pure surrogate.
  A+2 pure surrogate: PREDICT P_bg with a small net -> zero-shot, if P_bg is learnable.

A+2 is strictly better (no solve at deploy, oracle <1%) IF the smoothed background field
is genuinely low-dimensional. "Learnable" ultimately needs training, but a NECESSARY
condition is zero-training measurable: is P_bg intrinsically LOWER-dimensional /
LOWER-wavenumber / SMOOTHER-in-time than the full field wf? If P_bg is just as complex
as wf, predicting it is as hard as the original problem and A+2 has no edge.

Measured here, per family, comparing P_bg(sigma=4) vs full wf, ZERO training:
  (1) SPATIAL wavenumber: fraction of spatial spectral energy within |k|<=m, per frame.
      Smoother medium -> fewer scattering wiggles -> P_bg more concentrated at low |k|.
  (2) TEMPORAL rank: SVD of the [nt, npix] space-time matrix; rank@99% energy. The
      background has no fine multiples -> should be far lower temporal rank than wf.
  (3) CROSS-FRAME smoothness: energy of d/dt of the field (finite diff) relative to the
      field -- a smooth-medium background evolves more smoothly frame-to-frame.
  (4) PREDICTABILITY-FROM-CONDITIONING proxy: correlation of P_bg with the eikonal
      travel-time structure. The smoothed-medium field is close to geometric-optics
      (single arrival + smooth amplitude), so its phase should track a single traveltime
      T(x) computed on the smoothed velocity -> high phase-linearity R^2 (the amplitude-
      learnability probe's own criterion). If R^2(P_bg) >> R^2(wf), the background is
      WKB-simple = a small net conditioned on (smoothed-c, source) can produce it.

Decision: if P_bg is uniformly lower-dim on ALL four metrics, A+2 (learn the background)
is well-founded and is the pure-zero-shot optimum. If P_bg matches wf complexity on some
metric, that metric names the residual difficulty and we fall back to A+1 hybrid.
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
FAMILIES = ("layered", "anomaly", "marmousi")   # uniform trivial (bg==full)
PER_FAMILY = 2
SIGMA_BG = 4.0
M_SPACE = 40


def lowk_frac_per_frame(field, m):
    """field [nt,nz,nx] real -> mean over frames of fraction spatial energy within |k|<=m."""
    nt, nz, nx = field.shape
    S = np.fft.fft2(field, axes=(1, 2))
    kz = np.fft.fftfreq(nz) * nz; kx = np.fft.fftfreq(nx) * nx
    KZ, KX = np.meshgrid(kz, kx, indexing="ij")
    disk = (KZ ** 2 + KX ** 2) <= (m * m)
    e = np.abs(S) ** 2
    per = e[:, disk].sum(axis=1) / (e.reshape(nt, -1).sum(axis=1) + 1e-30)
    return float(per.mean())


def temporal_rank(field, frac=0.99):
    nt = field.shape[0]
    M = field.reshape(nt, -1)
    s = np.linalg.svd(M, compute_uv=False)
    e = s ** 2; c = np.cumsum(e) / (e.sum() + 1e-30)
    return int(np.searchsorted(c, frac) + 1)


def temporal_smoothness(field):
    """||d/dt field|| / ||field||, lower = smoother in time."""
    d = np.diff(field, axis=0)
    return float(np.linalg.norm(d) / (np.linalg.norm(field[1:]) + 1e-30))


def phase_linearity_r2(field, dt, source_map, band_lo=4, band_hi=40, npix=400):
    """rfft in time; on high-energy pixels fit unwrapped phase vs omega; mean R^2.
    High R^2 = single-arrival / WKB-simple (traveltime phase)."""
    P = np.fft.rfft(field, axis=0)
    freqs = np.fft.rfftfreq(field.shape[0], d=dt)
    band = np.where((freqs >= band_lo) & (freqs <= band_hi))[0]
    if len(band) < 3:
        return float("nan")
    w = 2 * np.pi * freqs[band]
    Pb = P[band]
    pe = (np.abs(Pb) ** 2).sum(0)
    thr = np.percentile(pe, 90)
    zz, xx = np.where(pe >= thr)
    sel = slice(0, min(npix, len(zz)))
    A = np.vstack([w, np.ones_like(w)]).T
    r2s = []
    for z, x in zip(zz[sel], xx[sel]):
        ph = np.unwrap(np.angle(Pb[:, z, x]))
        coef, *_ = np.linalg.lstsq(A, ph, rcond=None)
        ss_res = ((ph - A @ coef) ** 2).sum(); ss_tot = ((ph - ph.mean()) ** 2).sum() + 1e-30
        r2s.append(1 - ss_res / ss_tot)
    return float(np.mean(r2s)) if r2s else float("nan")


def main():
    cfg = resolve_config(load_config(FROZEN))
    grid = grid_from_config(cfg); bnd = boundaries_from_config(cfg); tg = time_from_config(cfg)
    dt_used = float(cfg["time"]["dt_used_s"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    solver = LWC84CPMLSolver(
        grid=grid, boundaries=bnd, dt_s=dt_used, output_times_s=tg.t_s, c_ref_mps=6750.0,
        device=device, dtype=torch.float32, kappa_max=float(cfg["boundaries"]["kappa_max"]),
        minimum_frequency_hz=float(cfg["boundaries"]["minimum_frequency_hz"]),
    )
    f = h5py.File(H5, "r")
    time_s = f["time_s"][:]; nt = len(time_s); dt = float(np.mean(np.diff(time_s)))
    mt = np.array([s.decode() for s in f["medium_type"][:]])
    nz_fine, nx_fine = grid.nz, grid.nx

    def up(v):
        t = torch.as_tensor(v[None, None], dtype=torch.float32)
        return torch.nn.functional.interpolate(t, size=(nz_fine, nx_fine), mode="bilinear",
                                               align_corners=True)[0, 0].numpy().astype(np.float64)

    def bg(ridx):
        vel = f["velocity_mps"][ridx].astype(np.float64)
        velfine = up(gaussian_filter(vel, sigma=SIGMA_BG, mode="nearest"))
        res = solver.simulate(velfine[None], source_x_m=float(f["source_x_m"][ridx]),
                              source_z_m=float(f["source_z_m"][ridx]),
                              source_f0_hz=float(f["source_f0_hz"][ridx]),
                              source_t0_s=float(f["source_t0_s"][ridx]),
                              source_amplitude=float(f["source_amplitude"][ridx]))
        return np.asarray(res.wavefield)[0]

    print(f"nt={nt} dt={dt*1e3:.3f}ms sigma_bg={SIGMA_BG} m={M_SPACE} device={device}")
    print("compare SMOOTHED-BACKGROUND P_bg vs FULL field wf (lower-dim bg => A+2 learnable)")
    print("=" * 100)
    hdr = f"{'family':10s} {'metric':22s} {'FULL wf':>10s} {'BG P_bg':>10s}  verdict"
    for fam in FAMILIES:
        idxs = np.where(np.char.startswith(mt, fam))[0][:PER_FAMILY]
        for ridx in idxs:
            ridx = int(ridx)
            wf = f["wavefield"][ridx].astype(np.float64)
            sm = f["source_map"][ridx].astype(np.float64)
            P_bg = bg(ridx)
            print(f"[{fam} #{ridx}]")
            # (1) spatial low-k
            a, b = lowk_frac_per_frame(wf, M_SPACE), lowk_frac_per_frame(P_bg, M_SPACE)
            print(f"    spatial |k|<=40 frac      full={a:.4f}  bg={b:.4f}  {'bg smoother' if b>=a else 'bg NOT smoother'}")
            # (2) temporal rank
            a, b = temporal_rank(wf), temporal_rank(P_bg)
            print(f"    temporal rank@99%         full={a:5d}   bg={b:5d}   {'bg lower-dim' if b<a else 'bg NOT lower'}")
            # (3) temporal smoothness
            a, b = temporal_smoothness(wf), temporal_smoothness(P_bg)
            print(f"    d/dt energy (lower=smooth) full={a:.4f}  bg={b:.4f}  {'bg smoother' if b<=a else 'bg NOT smoother'}")
            # (4) phase linearity (WKB-simple)
            a = phase_linearity_r2(wf, dt, sm); b = phase_linearity_r2(P_bg, dt, sm)
            print(f"    phase-linearity R^2 (WKB)  full={a:.4f}  bg={b:.4f}  {'bg more WKB' if b>=a else 'bg NOT more WKB'}")
    print("=" * 100)
    print("READ: if bg is lower-dim on ALL metrics (lower temporal rank, higher low-k frac, smoother")
    print("      in time, higher phase-R^2) => the smoothed background is WKB-simple/low-rank => a small")
    print("      net conditioned on (smoothed-c, source) can predict it => A+2 pure zero-shot is founded.")
    print("      Any metric where bg ~ full names a residual difficulty => fall back to A+1 hybrid solver.")


if __name__ == "__main__":
    sys.exit(main() or 0)
