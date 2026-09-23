#!/usr/bin/env python3
"""Group 3 figures: coarse-grid LWC-84 numerical dispersion vs neural operator error.

Per record (train_marmousi_00076, train_layered_00299), one figure with:
  (a) per-frame relative L2 vs physical time (neural, coarse51, coarse101);
  (b) temporal amplitude spectrum of the receiver trace at (z=40 m, x=800 m)
      over the future window (rFFT, Hann window), truth vs the three;
  (c) radially averaged spatial wavenumber spectrum at the middle future frame.

Numbers (NUMBERS.json): future relative L2 + bands (from SOLVES.json),
spectral centroid of the receiver trace (dispersion shifts it down),
and the wavenumber-spectrum ratios in three k-bands at the middle frame.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

H5 = Path("/root/autodl-tmp/data/jiayh/data/acoustic_lwc84_frequency_gap_20260922_v1/"
          "combined_dataset_v1.h5")
PRED_DIR = Path("/root/autodl-tmp/staging/scno_homogeneous_longrun_gap15_v1/"
                "evaluations/update_00005000/attempt_001")
BASE = Path("/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/REVIEWER_PACKAGE_20260921/"
            "results/paper_comparisons_20260923")
OUT = BASE / "group3_coarse_dispersion"
RECORDS = {"train_marmousi_00076": 2176, "train_layered_00299": 719}
IC_FRAMES = 8
RZ, RX = 4, 80  # receiver z=40 m, x=800 m
DX = 10.0


def per_frame_rel(pred, truth, frames):
    num = np.linalg.norm((pred - truth)[frames].reshape(len(frames), -1), axis=1)
    den = np.linalg.norm(truth[frames].reshape(len(frames), -1), axis=1)
    return num / np.clip(den, 1e-30, None)


def trace_spectrum(trace, dt):
    w = np.hanning(len(trace))
    spec = np.abs(np.fft.rfft(trace * w))
    freq = np.fft.rfftfreq(len(trace), dt)
    return freq, spec


def centroid(freq, spec):
    return float((freq * spec).sum() / spec.sum())


def radial_spectrum(frame, dx):
    """Radially averaged magnitude spectrum of one [z,x] frame."""
    f2 = np.fft.fftshift(np.abs(np.fft.fft2(frame)))
    nz, nx = frame.shape
    kz = np.fft.fftshift(np.fft.fftfreq(nz, dx))
    kx = np.fft.fftshift(np.fft.fftfreq(nx, dx))
    kzz, kxx = np.meshgrid(kz, kx, indexing="ij")
    kr = np.sqrt(kzz ** 2 + kxx ** 2)
    k_edges = np.linspace(0, kr.max(), 41)
    centers = 0.5 * (k_edges[1:] + k_edges[:-1])
    prof = np.array([f2[(kr >= k_edges[i]) & (kr < k_edges[i + 1])].mean()
                     for i in range(len(centers))])
    return centers, prof


def main() -> int:
    solves = json.loads((OUT / "SOLVES.json").read_text())
    numbers = {"schema": "paper_group3_dispersion_figures_v1", "records": {}}
    with h5py.File(H5, "r") as handle:
        time_s = np.asarray(handle["time_s"][:], dtype=np.float64)
        dt = float(time_s[1] - time_s[0])
        sample_ids = [v.decode() for v in handle["sample_id"][:]]
        for sid, idx in RECORDS.items():
            assert sample_ids[idx] == sid
            onset = solves["records"][sid]["onset"]
            start = onset + IC_FRAMES
            fut = np.arange(start, len(time_s))
            truth = np.asarray(handle["wavefield"][idx], dtype=np.float64)
            pred_f = np.load(PRED_DIR / f"{sid}_prediction.npy").astype(np.float64)
            pred = np.zeros_like(truth)
            pred[start:] = pred_f
            c51 = np.load(OUT / f"{sid}_coarse51_on201.npy").astype(np.float64)
            c101 = np.load(OUT / f"{sid}_coarse101_on201.npy").astype(np.float64)

            fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), constrained_layout=True)
            series = [("Neural operator", pred, "r", "--"),
                      ("LWC-84 51x51 (dx=40 m)", c51, "tab:blue", "-."),
                      ("LWC-84 101x101 (dx=20 m)", c101, "tab:green", ":")]

            ax = axes[0]
            for name, field, color, ls in series:
                ax.plot(time_s[fut], per_frame_rel(field, truth, fut),
                        color=color, ls=ls, lw=1.4, label=name)
            ax.set_yscale("log")
            ax.set_xlabel("t (s)")
            ax.set_ylabel("per-frame relative L2")
            ax.set_title("(a) Error growth over time")
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=8)

            ax = axes[1]
            rec_spec = {}
            tt = truth[fut, RZ, RX]
            freq, spec_t = trace_spectrum(tt, dt)
            keep = freq <= 60.0
            ax.plot(freq[keep], spec_t[keep] / spec_t.max(), "k-", lw=1.4,
                    label="Truth")
            rec_spec["truth_centroid_hz"] = centroid(freq, spec_t)
            for name, field, color, ls in series:
                tr = field[fut, RZ, RX]
                _, spec = trace_spectrum(tr, dt)
                ax.plot(freq[keep], spec[keep] / spec_t.max(), color=color, ls=ls,
                        lw=1.2, label=name)
                key = name.split()[0].lower() if "Neural" in name else (
                    "coarse51" if "51x51" in name else "coarse101")
                rec_spec[f"{key}_centroid_hz"] = centroid(freq, spec)
            ax.set_xlabel("frequency (Hz)")
            ax.set_ylabel("|P(f)| (norm. to truth peak)")
            ax.set_title(f"(b) Receiver spectrum, z=40 m x=800 m")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

            ax = axes[2]
            mid = start + (len(time_s) - 1 - start) // 2
            k, prof_t = radial_spectrum(truth[mid], DX)
            ax.plot(k * 1000, prof_t, "k-", lw=1.4, label="Truth")
            kband = {}
            for name, field, color, ls in series:
                _, prof = radial_spectrum(field[mid], DX)
                ax.plot(k * 1000, prof, color=color, ls=ls, lw=1.2, label=name)
                key = ("neural" if "Neural" in name else
                       "coarse51" if "51x51" in name else "coarse101")
                thirds = np.array_split(np.arange(len(k)), 3)
                kband[key] = [float(prof[t].sum() / prof_t[t].sum()) for t in thirds]
            ax.set_yscale("log")
            ax.set_xlabel("|k| (cycles/km)")
            ax.set_ylabel("radially averaged |FFT2|")
            ax.set_title(f"(c) Wavenumber spectrum at t = {time_s[mid]:.3f} s")
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=8)

            fig.suptitle(f"{sid}: coarse-grid numerical dispersion vs neural operator error",
                         fontsize=13)
            fig.savefig(OUT / f"dispersion_{sid}.png", dpi=180, bbox_inches="tight")
            fig.savefig(OUT / f"dispersion_{sid}.pdf", bbox_inches="tight")
            plt.close(fig)

            numbers["records"][sid] = {
                "onset": onset,
                "future_relative_l2": {
                    "neural": solves["records"][sid]["neural_operator"]["future_relative_l2"],
                    "coarse51": solves["records"][sid]["solves"]["coarse51"]["future_relative_l2"],
                    "coarse101": solves["records"][sid]["solves"]["coarse101"]["future_relative_l2"],
                },
                "time_bands": {
                    "neural": solves["records"][sid]["neural_operator"]["time_bands"],
                    "coarse51": solves["records"][sid]["solves"]["coarse51"]["time_bands"],
                    "coarse101": solves["records"][sid]["solves"]["coarse101"]["time_bands"],
                },
                "receiver_spectrum_centroids_hz": rec_spec,
                "wavenumber_band_energy_ratio_vs_truth_at_mid_frame": kband,
                "mid_frame_time_s": float(time_s[mid]),
            }
            print("figure written for", sid)
    (OUT / "NUMBERS.json").write_text(json.dumps(numbers, indent=1))
    print("written", OUT / "NUMBERS.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
