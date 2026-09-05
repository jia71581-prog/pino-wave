"""Zero-training probe for the SOURCE-AWARE temporal-frequency Helmholtz direction.

We solve the time-domain acoustic wave equation WITH a source. Fourier in time:
    p(x,t) = sum_omega  P_hat(x,omega) e^{i omega t}
and, because the equation is LINEAR in the source, the source separates exactly:
    P_hat(x,omega) = w_hat(omega) * G_hat(x; x_s, omega, c)
where w_hat(omega) = FFT_t of the stored source_wavelet (KNOWN, per record), and
G_hat is the Helmholtz Green's function depending only on velocity + source POSITION
(NOT on f0/t0 beyond what w_hat already carries).

This probe tests, on real data, WITHOUT training, three claims that decide whether
the direction is worth building:

  Q1 SOURCE FACTORIZATION. Is G_hat := P_hat / w_hat effectively independent of the
     wavelet? We cannot vary the wavelet on a fixed medium+position here, so we test
     the weaker necessary condition: dividing out w_hat must not blow up (w_hat has
     no zeros in the band that carries the field's energy) and must leave a field
     whose per-frequency energy tracks |w_hat|^2 (i.e. the field IS the wavelet
     times a medium response). Reported as band-limited-ness + division conditioning.

  Q2 K-FREQUENCY RECONSTRUCTION (the oracle from memory, re-verified HERE with the
     source wavelet in hand). Keep only K complex frequency slices of P_hat and
     inverse-FFT back to 401 frames. Report rel-L2 vs the true wavefield, per family,
     as a function of K. This is the "<5% with 64 freqs" claim, re-checked on this
     exact data path.

  Q3 EIKONAL PHASE MODEL. High-frequency asymptotics predict G_hat ~ A(x) e^{i omega T(x)}
     with T the first-arrival traveltime. We do NOT have T from the project's
     RayTravelTime here (no torch model load), so we estimate the phase slope in omega
     at each pixel (unwrapped phase of P_hat/w_hat vs omega) and check it is smooth /
     low-rank — a proxy for "the phase is a single smooth traveltime surface" vs
     "multipath scramble". Reported as phase-linearity R^2 per family (high = single
     arrival, eikonal model OK; low = multipath, needs sum of arrivals).

Runs on a few records per family, CPU-only numpy, no training, minutes.
"""
from __future__ import annotations

import sys
import numpy as np
import h5py


H5 = "/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5"
FAMILIES = ("uniform", "layered", "anomaly", "marmousi")
PER_FAMILY = 2
K_LIST = (16, 32, 48, 64, 96)


def _decode(x):
    return x.decode() if isinstance(x, (bytes, bytes)) else str(x)


def pick_records(f):
    fam = np.array([_decode(s).split("_")[0] if b"_" in s else _decode(s)
                    for s in f["medium_type"][:]])
    # medium_type may be like 'uniform' / 'layered' / ... ; match by prefix
    mt = np.array([_decode(s) for s in f["medium_type"][:]])
    chosen = {}
    for fam_name in FAMILIES:
        idx = np.where(np.char.startswith(mt, fam_name))[0]
        if len(idx):
            chosen[fam_name] = idx[:PER_FAMILY].tolist()
    return chosen


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main():
    f = h5py.File(H5, "r")
    time_s = f["time_s"][:]                    # [401]
    nt = len(time_s)
    dt = float(np.mean(np.diff(time_s)))
    chosen = pick_records(f)
    if not chosen:
        print("no records matched family names; medium_type sample:",
              [_decode(s) for s in f["medium_type"][:5]])
        return 1
    print(f"nt={nt} dt={dt*1e3:.3f} ms  Nyquist={1/(2*dt):.1f} Hz")
    print(f"records per family: {{ {', '.join(f'{k}:{v}' for k,v in chosen.items())} }}")
    print("=" * 78)

    freqs = np.fft.rfftfreq(nt, d=dt)          # [nt//2+1] Hz

    for fam_name, idxs in chosen.items():
        for ridx in idxs:
            wf = f["wavefield"][ridx]          # [401,201,201]
            wl = f["source_wavelet"][ridx]     # [401]
            f0 = float(f["source_f0_hz"][ridx])
            t0 = float(f["source_t0_s"][ridx])
            # time -> frequency (rfft along time axis)
            P = np.fft.rfft(wf, axis=0)        # [F,201,201] complex
            W = np.fft.rfft(wl)                # [F] complex
            band_energy = np.abs(W) ** 2
            band_energy /= band_energy.sum() + 1e-30
            # energy-carrying band: freqs holding 99% of wavelet energy
            order = np.argsort(band_energy)[::-1]
            cum = np.cumsum(band_energy[order])
            n99 = int(np.searchsorted(cum, 0.99)) + 1
            band = np.sort(order[:n99])
            fmax_band = freqs[band].max()

            # --- Q1: division conditioning (|W| not tiny where field has energy) ---
            field_energy = (np.abs(P) ** 2).sum(axis=(1, 2))
            field_energy /= field_energy.sum() + 1e-30
            # in the field's 99%-energy band, how small does |W| get (relative)?
            forder = np.argsort(field_energy)[::-1]
            fcum = np.cumsum(field_energy[forder])
            fn99 = int(np.searchsorted(fcum, 0.99)) + 1
            fband = np.sort(forder[:fn99])
            Wmag = np.abs(W)
            cond = float(Wmag[fband].max() / (Wmag[fband].min() + 1e-30))

            # --- Q2: K-frequency inverse-FFT reconstruction ---
            # keep the K frequencies with largest field energy, zero the rest, irfft
            q2 = {}
            for K in K_LIST:
                keep = forder[:K]
                Pk = np.zeros_like(P)
                Pk[keep] = P[keep]
                rec = np.fft.irfft(Pk, n=nt, axis=0)
                q2[K] = rel_l2(rec, wf)

            # --- Q3: eikonal phase linearity in omega (per pixel) ---
            # G = P / W on the field band; unwrap phase vs 2*pi*freq; fit slope;
            # R^2 of the linear fit averaged over high-energy pixels = single-arrival-ness
            Gband = P[fband] / (W[fband][:, None, None] + 1e-30)
            w_ang = 2 * np.pi * freqs[fband]
            # choose pixels with strong field energy (skip near-source & quiet zones)
            pix_energy = (np.abs(Gband) ** 2).sum(0)
            thr = np.percentile(pix_energy, 90)
            zz, xx = np.where(pix_energy >= thr)
            sel = slice(0, min(400, len(zz)))
            r2s = []
            for z, x in zip(zz[sel], xx[sel]):
                ph = np.unwrap(np.angle(Gband[:, z, x]))
                A = np.vstack([w_ang, np.ones_like(w_ang)]).T
                coef, res, *_ = np.linalg.lstsq(A, ph, rcond=None)
                ss_res = float(((ph - A @ coef) ** 2).sum())
                ss_tot = float(((ph - ph.mean()) ** 2).sum()) + 1e-30
                r2s.append(1 - ss_res / ss_tot)
            phase_r2 = float(np.mean(r2s)) if r2s else float("nan")

            q2str = "  ".join(f"K{K}:{q2[K]*100:5.2f}%" for K in K_LIST)
            print(f"[{fam_name:8s} #{ridx:4d}] f0={f0:4.1f}Hz t0={t0*1e3:5.1f}ms "
                  f"band<={fmax_band:5.1f}Hz Wcond={cond:6.1f}")
            print(f"           Q2 recon: {q2str}")
            print(f"           Q3 eikonal phase-linearity R^2 = {phase_r2:.3f} "
                  f"({'single-arrival OK' if phase_r2>0.9 else 'MULTIPATH — needs sum of arrivals'})")
    print("=" * 78)
    print("READ: Q2 <5% at some K re-confirms the freq-domain oracle WITH the source.")
    print("      Q3 R^2>0.9 => single eikonal phase suffices; low R^2 (esp marmousi)")
    print("      => multipath, the known hard case — factorization reframes but does")
    print("      not auto-solve it.")


if __name__ == "__main__":
    sys.exit(main() or 0)
