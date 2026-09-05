"""Direction B core: preconditioned convergent Born series (CBS) iteration kernel,
validated as a zero-training oracle before wiring the learnable architecture.

Frequency-domain acoustic Helmholtz with source (time-FT of the wave equation):
    (Lap + k^2(x)) P(x,w) = -S(x,w),   k^2(x) = w^2 / c(x)^2,  S = Shat(w)*source_map

Split k^2 = k0^2 + V,  k0^2 = w^2/c0^2 (reference), V(x) = w^2(1/c(x)^2 - 1/c0^2).
Bare Born series diverges when the spectral radius of G0*V exceeds 1 (strong
contrast, e.g. marmousi). Osnabrugge-Leedumrongwatthanakun-Vellekoop 2016
("A convergent Born series ... arbitrarily large media", Optics Express) fix this
with an ABSORBING reference + preconditioner:
    add absorption:   k0^2 -> w^2/c0^2 + i*eps,  eps >= max|V|  (guarantees ||M||<1)
    Green (spectral):  g(xi) = 1/(|xi|^2 - k0^2)   [ (Lap+k0^2)G = -delta ]
    preconditioner:    gamma(x) = (i/eps) V(x)
    iterate:           u_{n+1} = u_n - gamma * ( u_n - G[ V u_n + S ] ),   u_0 = G[S]
Fixed point u* = G[V u* + S] is the Lippmann-Schwinger solution = exact Helmholtz P.

This module exposes cbs_solve() reused by the learnable architecture. The probe below
fixes V from the TRUE velocity (oracle) and asks, per family, zero training:
  Q1 CONVERGENCE: does the CBS residual decrease monotonically (not diverge) for all
     families incl. marmousi high-contrast? Report residual vs iteration.
  Q2 RECONSTRUCTION: inverse-FFT the converged per-frequency fields to 401 frames;
     rel-L2 per time bin (early/mid/late). Compare to Direction-A naive split.
  Q3 LATE FIX: Direction A's naive (P_total - P_inc) degraded late (scat energy >
     total). Does CBS -- which represents the FULL field as a fixed point, not a
     subtraction -- keep late healthy?

Continuous spectral G0 on the saved 201 grid carries a dx=10m dispersion gap vs the
post-processed stored field (the same gap analytic Hankel showed in Direction A).
That residual gap is exactly what the learnable per-step correction head will absorb;
here we QUANTIFY it as the oracle floor for B.
"""
from __future__ import annotations

import sys
import numpy as np
import h5py

DATA_DIR = "/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1"
H5 = f"{DATA_DIR}/dataset_v1.h5"
FAMILIES = ("uniform", "layered", "anomaly", "marmousi")
PER_FAMILY = 2
K_TIME = 64          # top time-frequency slices by energy
CBS_ITERS = 60


def cbs_solve(k2_x, k02_real, S, dx, iters, return_trace=False):
    """Solve (Lap + k2_x) u = -S by preconditioned convergent Born series.

    k2_x   : [nz,nx] real, w^2/c(x)^2
    k02_real: scalar, w^2/c0^2 (reference, real part)
    S      : [nz,nx] complex source (Shat(w)*source_map, already scaled)
    dx     : grid spacing (m), isotropic
    """
    nz, nx = k2_x.shape
    xi_z = 2 * np.pi * np.fft.fftfreq(nz, d=dx)
    xi_x = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    XZ, XX = np.meshgrid(xi_z, xi_x, indexing="ij")
    xi2 = XZ ** 2 + XX ** 2

    V0 = k2_x - k02_real                     # real part of scattering potential
    eps = float(np.abs(V0).max()) * 1.05 + 1e-6
    k02 = k02_real + 1j * eps                 # absorbing reference
    V = k2_x - k02                            # complex potential, |V| <= eps by construction
    gamma = (1j / eps) * V
    g = 1.0 / (xi2 - k02)                      # spectral Green, no pole (k02 complex)

    def G(field):
        return np.fft.ifft2(g * np.fft.fft2(field))

    u = G(S)                                   # u_0 = incident field (absorbing ref)
    trace = []
    for _ in range(iters):
        u = u - gamma * (u - G(V * u + S))
        if return_trace:
            # Helmholtz residual on the ORIGINAL (non-absorbing) operator:
            # r = (Lap + k2_x) u + S ; measure ||r||/||S||
            lap_u = np.fft.ifft2(-xi2 * np.fft.fft2(u))
            r = lap_u + k2_x * u + S
            trace.append(float(np.linalg.norm(r) / (np.linalg.norm(S) + 1e-30)))
    return (u, trace) if return_trace else u


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main():
    f = h5py.File(H5, "r")
    time_s = f["time_s"][:]
    nt = len(time_s)
    dt = float(np.mean(np.diff(time_s)))
    x_m = f["x_m"][:]
    dx = float(np.mean(np.diff(x_m)))
    freqs = np.fft.rfftfreq(nt, d=dt)
    mt = np.array([s.decode() for s in f["medium_type"][:]])

    # quick argv filter for fast symbol-debugging on a subset
    only = sys.argv[1:] if len(sys.argv) > 1 else FAMILIES

    print(f"nt={nt} dt={dt*1e3:.3f}ms dx={dx}m K_time={K_TIME} cbs_iters={CBS_ITERS}")
    print("=" * 96)

    for fam in FAMILIES:
        if fam not in only:
            continue
        idxs = np.where(np.char.startswith(mt, fam))[0][:PER_FAMILY]
        for ridx in idxs:
            ridx = int(ridx)
            wf = f["wavefield"][ridx].astype(np.float64)
            vel = f["velocity_mps"][ridx].astype(np.float64)
            wl = f["source_wavelet"][ridx].astype(np.float64)
            sm = f["source_map"][ridx].astype(np.float64)
            xs = float(f["source_x_m"][ridx]); zs = float(f["source_z_m"][ridx])
            z_m = f["z_m"][:]
            iz = int(np.argmin(np.abs(z_m - zs))); ix = int(np.argmin(np.abs(x_m - xs)))
            c0 = float(vel[iz, ix])

            P_total = np.fft.rfft(wf, axis=0)          # [F,nz,nx]
            W = np.fft.rfft(wl)                          # [F]
            fe = (np.abs(P_total) ** 2).sum(axis=(1, 2))
            keep = np.sort(np.argsort(fe)[::-1][:K_TIME])

            P_cbs = np.zeros_like(P_total)
            trace_last = None
            contrast = float((vel.max() / vel.min()))
            for jj, j in enumerate(keep):
                w = 2 * np.pi * freqs[j]
                k2_x = (w ** 2) / (vel ** 2)
                k02_real = (w ** 2) / (c0 ** 2)
                S = W[j] * sm                            # source (global amplitude fit below)
                tr = (jj == len(keep) // 2)              # trace one mid-band frequency
                out = cbs_solve(k2_x, k02_real, S.astype(complex), dx, CBS_ITERS,
                                return_trace=tr)
                if tr:
                    P_cbs[j], trace_last = out
                else:
                    P_cbs[j] = out

            # global complex amplitude calibration (absorbs source normalization)
            num = np.vdot(P_cbs[keep], P_total[keep])
            den = np.vdot(P_cbs[keep], P_cbs[keep]) + 1e-30
            alpha = num / den
            rec = np.fft.irfft(alpha * P_cbs, n=nt, axis=0)

            def binerr(lo, hi):
                return rel_l2(rec[lo:hi], wf[lo:hi])
            e_all = binerr(0, nt); e_e = binerr(0, nt // 3)
            e_m = binerr(nt // 3, 2 * nt // 3); e_l = binerr(2 * nt // 3, nt)

            print(f"[{fam:8s} #{ridx:4d}] c0={c0:6.1f} contrast={contrast:4.2f} |alpha|={abs(alpha):.3e}")
            if trace_last is not None:
                t = trace_last
                print(f"    CBS residual @iter[0,5,20,40,{CBS_ITERS-1}] = "
                      f"{t[0]:.3f} {t[5]:.3f} {t[20]:.3f} {t[40]:.3f} {t[-1]:.3f}  "
                      f"{'DIVERGES' if t[-1] > t[0] else 'converges'}")
            print(f"    recon rel-L2  early={e_e:.3f} mid={e_m:.3f} late={e_l:.3f} all={e_all:.3f}")
    print("=" * 96)
    print("READ: Q1 residual must DECREASE (CBS converges) for ALL families incl marmousi.")
    print("      Q2 recon = oracle floor for learnable B (dx10 continuous-vs-postproc gap).")
    print("      Q3 late must stay healthy (unlike Direction A naive split late blow-up).")


if __name__ == "__main__":
    sys.exit(main() or 0)
