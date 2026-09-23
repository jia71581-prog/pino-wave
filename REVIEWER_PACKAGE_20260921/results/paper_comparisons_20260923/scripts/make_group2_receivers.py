#!/usr/bin/env python3
"""Group 2: near-surface receiver waveform comparison.

For train_marmousi_00076 and train_layered_00299: receivers at z = 40 m
(grid row 4; the free surface pins p(z=0)=0) and x = 400/800/1200/1600 m.

Figure A (receivers_<sid>.png): truth vs neural operator overlay + residual,
full stored time axis (prediction exists from the future-window start).
Figure B (receivers_dispersion_<sid>.png): truth vs neural vs coarse-grid
LWC-84 (51x51, dx=40 m, from group 3) overlay per receiver, showing the
coarse solver's numerical dispersion (phase lag, waveform broadening)
against the neural operator's error character.

Quantities in NUMBERS.json per receiver, future window (onset+8..400):
relative L2 (neural, coarse51, coarse101), best cross-correlation lag in ms
(positive = trace arrives late vs truth), and amplitude ratio ||trace||/||truth||.
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
COARSE = BASE / "group3_coarse_dispersion"
OUT = BASE / "group2_receivers"
RECORDS = {"train_marmousi_00076": 2176, "train_layered_00299": 719}
IC_FRAMES = 8
Z_INDEX = 4            # z = 40 m
X_INDICES = [40, 80, 120, 160]   # x = 400, 800, 1200, 1600 m
MAX_LAG = 40           # frames (= 100 ms) searched for cross-correlation lag


def xcorr_lag_ms(trace: np.ndarray, truth: np.ndarray, dt_ms: float) -> float:
    """Lag (ms) maximising correlation; positive = trace late vs truth."""
    n = len(truth)
    best, best_lag = -np.inf, 0
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        if lag >= 0:
            a, b = trace[lag:], truth[:n - lag]
        else:
            a, b = trace[:n + lag], truth[-lag:]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            continue
        c = float(np.dot(a, b) / denom)
        if c > best:
            best, best_lag = c, lag
    return best_lag * dt_ms, best


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    numbers = {"schema": "paper_group2_receivers_v1",
               "receiver_z_m": 40.0,
               "receiver_x_m": [float(i * 10) for i in X_INDICES],
               "window": "future (onset+8..400)",
               "lag_definition": "argmax normalized cross-correlation, "
                                 "search +-100 ms; positive = arrival late vs truth",
               "records": {}}
    with h5py.File(H5, "r") as handle:
        time_s = np.asarray(handle["time_s"][:], dtype=np.float64)
        sample_ids = [v.decode() for v in handle["sample_id"][:]]
        dt_ms = float(time_s[1] - time_s[0]) * 1000.0
        for sid, idx in RECORDS.items():
            assert sample_ids[idx] == sid
            onset = int(np.searchsorted(time_s, float(handle["source_t0_s"][idx])))
            start = onset + IC_FRAMES
            truth = np.asarray(handle["wavefield"][idx], dtype=np.float64)
            pred = np.load(PRED_DIR / f"{sid}_prediction.npy").astype(np.float64)
            c51 = np.load(COARSE / f"{sid}_coarse51_on201.npy").astype(np.float64)
            c101 = np.load(COARSE / f"{sid}_coarse101_on201.npy").astype(np.float64)
            fut = slice(start, len(time_s))

            rec_out = {"onset": onset, "future_start_time_s": float(time_s[start]),
                       "receivers": []}

            # Figure A: truth vs neural + residual
            figA, axA = plt.subplots(len(X_INDICES), 2, figsize=(13, 2.6 * len(X_INDICES)),
                                     sharex=True, constrained_layout=True)
            # Figure B: dispersion overlay with coarse LWC
            figB, axB = plt.subplots(len(X_INDICES), 1, figsize=(13, 2.8 * len(X_INDICES)),
                                     sharex=True, constrained_layout=True)
            for r, xi in enumerate(X_INDICES):
                tt = truth[:, Z_INDEX, xi]
                pp = pred[:, Z_INDEX, xi]
                w51 = c51[:, Z_INDEX, xi]
                w101 = c101[:, Z_INDEX, xi]
                t_axis = time_s
                tp_axis = time_s[fut]

                den = float(np.linalg.norm(tt[fut]))
                stats = {"x_m": float(xi * 10)}
                for name, tr in (("neural", pp), ("coarse51", w51[fut]),
                                 ("coarse101", w101[fut])):
                    rel = float(np.linalg.norm(tr - tt[fut]) / den)
                    lag, corr = xcorr_lag_ms(tr, tt[fut], dt_ms)
                    stats[name] = {"relative_l2": rel, "lag_ms": lag,
                                   "peak_xcorr": corr,
                                   "amplitude_ratio": float(np.linalg.norm(tr) / den)}
                rec_out["receivers"].append(stats)

                ax = axA[r, 0]
                ax.plot(t_axis, tt, "k-", lw=1.1, label="Truth (LWC-84)")
                ax.plot(tp_axis, pp, "r--", lw=1.0, label="Neural operator")
                ax.axvline(time_s[start], color="0.6", lw=0.8, ls=":")
                ax.set_ylabel(f"x = {xi*10:.0f} m\np (Pa)", fontsize=10)
                ax.grid(alpha=0.3)
                if r == 0:
                    ax.legend(fontsize=9, loc="upper right")
                    ax.set_title("Truth vs neural operator, receiver z = 40 m", fontsize=11)
                ax = axA[r, 1]
                ax.plot(tp_axis, pp - tt[fut], "r-", lw=0.9)
                ax.grid(alpha=0.3)
                if r == 0:
                    ax.set_title("Residual (neural - truth), future window", fontsize=11)

                ax = axB[r]
                ax.plot(t_axis, tt, "k-", lw=1.2, label="Truth (fine-grid LWC-84)")
                ax.plot(tp_axis, pp, "r--", lw=1.0, label="Neural operator")
                ax.plot(t_axis, w51, color="tab:blue", ls="-.", lw=1.0,
                        label="LWC-84 51x51 (dx=40 m)")
                ax.axvline(time_s[start], color="0.6", lw=0.8, ls=":")
                ax.set_ylabel(f"x = {xi*10:.0f} m\np (Pa)", fontsize=10)
                ax.grid(alpha=0.3)
                if r == 0:
                    ax.legend(fontsize=9, loc="upper right", ncols=3)
                    ax.set_title(f"{sid}: receiver waveforms with coarse-grid LWC-84 "
                                 "(numerical dispersion) overlay, z = 40 m", fontsize=11)
            axA[-1, 0].set_xlabel("t (s)")
            axA[-1, 1].set_xlabel("t (s)")
            axB[-1].set_xlabel("t (s)")
            figA.suptitle(f"{sid}: near-surface receivers, truth vs neural operator",
                          fontsize=12)
            for fig, name in ((figA, f"receivers_{sid}"),
                              (figB, f"receivers_dispersion_{sid}")):
                fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
                fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
                plt.close(fig)
            numbers["records"][sid] = rec_out
            print("figures written for", sid)
    (OUT / "NUMBERS.json").write_text(json.dumps(numbers, indent=1))
    print("written", OUT / "NUMBERS.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
