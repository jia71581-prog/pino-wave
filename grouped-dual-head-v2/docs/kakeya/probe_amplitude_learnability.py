"""Zero-training probe #2: is the Helmholtz amplitude field A_k(x) LEARNABLE?

Following probe_source_freq_helmholtz.py (Q1/Q2/Q3 all positive: source factors
out, K~48-64 freqs give <5% incl. marmousi, single-eikonal phase R^2>0.96), the
ONLY thing the network must actually learn is the complex Green's amplitude at
each kept frequency:
    G_hat(x, omega_k) = P_hat(x, omega_k) / w_hat(omega_k)
    A_k(x) := G_hat(x, omega_k) * exp(-i omega_k T(x))     # de-phased by eikonal T

If the eikonal phase model holds, A_k(x) is a SMOOTH, slowly varying complex
amplitude (geometric spreading + reflections), which a modest network can produce.
This probe measures, WITHOUT training, whether A_k is actually low-complexity:

  M1 SPATIAL SMOOTHNESS. Fraction of A_k spectral energy in low spatial
     wavenumbers (|k|<kc). High => smooth => easy to render with a low-mode/
     conv generator (unlike the sharp time-domain wavefront that caused Gibbs).

  M2 CROSS-FREQUENCY LOW-RANK. Stack {A_k(x)} over the K kept freqs into a
     [K, Npix] matrix; SVD; report how many singular components hold 95/99% energy.
     Low rank => the K amplitude fields share a small common basis => the operator
     can output a few basis fields + per-freq mixing, cheap in parameters.

  M3 DE-PHASING GAIN. Compare spatial smoothness of A_k (eikonal-dephased) vs the
     raw G_hat (with phase). If de-phasing sharply increases low-wavenumber energy,
     it confirms the eikonal phase carries the oscillation and the residual amplitude
     is genuinely smooth (the whole point).

T(x) here is estimated per-record from the data itself (phase slope of G_hat in
omega, the same linear fit probe #1 used), NOT from the project's RayTravelTime —
so this stays a standalone numpy probe. Phase-slope T is exactly the traveltime
the eikonal solver would give where the single-arrival model holds.
"""
from __future__ import annotations

import sys
import numpy as np
import h5py


H5 = "/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5"
FAMILIES = ("uniform", "layered", "anomaly", "marmousi")
PER_FAMILY = 2
K = 64  # kept frequencies (probe #1: K~48-64 gives <5% all families)


def _decode(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def low_wavenumber_fraction(field_2d, kc_frac=0.1):
    """Fraction of |FFT2|^2 energy inside the disk |k| < kc_frac * k_nyquist."""
    F = np.fft.fft2(field_2d)
    n = field_2d.shape[0]
    kz = np.fft.fftfreq(n)[:, None]
    kx = np.fft.fftfreq(field_2d.shape[1])[None, :]
    kk = np.sqrt(kz**2 + kx**2)
    e = np.abs(F) ** 2
    return float(e[kk < kc_frac * 0.5].sum() / (e.sum() + 1e-30))


def svd_rank(matrix, thresh):
    s = np.linalg.svd(matrix, compute_uv=False)
    e = np.cumsum(s**2) / (np.sum(s**2) + 1e-30)
    return int(np.searchsorted(e, thresh) + 1)


def main():
    f = h5py.File(H5, "r")
    time_s = f["time_s"][:]
    nt = len(time_s)
    dt = float(np.mean(np.diff(time_s)))
    freqs = np.fft.rfftfreq(nt, d=dt)
    w_ang = 2 * np.pi * freqs
    mt = np.array([_decode(s) for s in f["medium_type"][:]])

    print(f"nt={nt} dt={dt*1e3:.2f}ms  K={K} kept freqs")
    print("=" * 82)
    print(f"{'family':9s} {'rec':>5s} | M1 lowk% raw->dephased | M2 rank@95%/99% (of K={K}) | M3")
    print("-" * 82)

    for fam_name in FAMILIES:
        idx = np.where(np.char.startswith(mt, fam_name))[0][:PER_FAMILY]
        for ridx in idx:
            wf = f["wavefield"][ridx]
            wl = f["source_wavelet"][ridx]
            P = np.fft.rfft(wf, axis=0)          # [F,201,201]
            W = np.fft.rfft(wl)                  # [F]
            fe = (np.abs(P) ** 2).sum(axis=(1, 2))
            keep = np.argsort(fe)[::-1][:K]
            keep = np.sort(keep)
            G = P[keep] / (W[keep][:, None, None] + 1e-30)   # [K,201,201] complex

            # estimate T(x) from phase slope in omega (single-arrival eikonal proxy)
            H, Wd = G.shape[1:]
            wa = w_ang[keep]
            phase = np.unwrap(np.angle(G), axis=0)           # [K,H,W]
            # per-pixel linear fit slope = -T (P ~ e^{i w T}); use lstsq vectorized
            A_ = np.vstack([wa, np.ones_like(wa)]).T          # [K,2]
            pinv = np.linalg.pinv(A_)                          # [2,K]
            coef = np.einsum("ck,kij->cij", pinv, phase)       # [2,H,W]
            Tslope = coef[0]                                   # [H,W]
            # de-phase: A_k = G_k * exp(-i w_k T)
            Adep = G * np.exp(-1j * wa[:, None, None] * Tslope[None])

            # M1: mean low-wavenumber fraction across kept freqs, raw vs dephased
            m1_raw = np.mean([low_wavenumber_fraction(np.abs(G[i])) for i in range(0, K, 8)])
            # for dephased, smoothness is in the COMPLEX field (real+imag both smooth)
            m1_dep = np.mean([
                0.5 * (low_wavenumber_fraction(Adep[i].real)
                       + low_wavenumber_fraction(Adep[i].imag))
                for i in range(0, K, 8)
            ])

            # M2: cross-frequency low-rank of the dephased amplitude stack
            mat = Adep.reshape(K, -1)
            mat = np.concatenate([mat.real, mat.imag], axis=1)  # real-valued [K,2*Npix]
            r95 = svd_rank(mat, 0.95)
            r99 = svd_rank(mat, 0.99)

            gain = m1_dep / (m1_raw + 1e-30)
            print(f"{fam_name:9s} {ridx:5d} |   {m1_raw*100:5.1f}% -> {m1_dep*100:5.1f}%    "
                  f"|      {r95:3d} / {r99:3d}            | x{gain:.2f}")
    print("=" * 82)
    print("READ:")
    print(" M1 dephased lowk% high (>~70%) => amplitude is smooth => low-mode generator ok")
    print(" M2 rank95 << K  => the K freq-fields share a small basis => few params suffice")
    print(" M3 gain >1      => eikonal de-phasing genuinely smooths (phase carried the")
    print("                    oscillation); the residual the net learns is easy")


if __name__ == "__main__":
    sys.exit(main() or 0)
