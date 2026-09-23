#!/usr/bin/env python3
"""Group 1: wavefield snapshot comparison figures (truth / prediction / error).

One 3x3 panel per record (rows = early/middle/late physical times inside the
future window, columns = truth, neural prediction, difference), for
train_marmousi_00076 and train_layered_00299 (family-median dev records of the
gap15 evaluation).

Colour discipline (project rules):
  - spatial arrays are [z, x], depth first; imshow WITHOUT transpose,
    extent = [0, 2000, 2000, 0] so z runs downward;
  - seismic diverging colormap, vmin=-vmax, truth and prediction share one
    scale per row (per time), error has its own symmetric scale per row.

Writes PNG + PDF into group1_snapshots/.
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
OUT = Path("/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/REVIEWER_PACKAGE_20260921/"
           "results/paper_comparisons_20260923/group1_snapshots")
RECORDS = {"train_marmousi_00076": 2176, "train_layered_00299": 719}
IC_FRAMES = 8
EXTENT = [0.0, 2000.0, 2000.0, 0.0]  # x left->right, z downward (depth first)


def pick_frames(onset: int, n_frames: int) -> list[int]:
    """Early / middle / late absolute frame indices inside the future window."""
    start = onset + IC_FRAMES
    last = n_frames - 1
    early = start + int(round(0.08 * (last - start)))
    middle = start + (last - start) // 2
    return [early, middle, last]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    numbers = {"schema": "paper_group1_snapshots_v1", "records": {}}
    with h5py.File(H5, "r") as handle:
        time_s = np.asarray(handle["time_s"][:], dtype=np.float64)
        sample_ids = [v.decode() for v in handle["sample_id"][:]]
        for sid, idx in RECORDS.items():
            assert sample_ids[idx] == sid
            onset = int(np.searchsorted(time_s, float(handle["source_t0_s"][idx])))
            truth = np.asarray(handle["wavefield"][idx], dtype=np.float64)  # (401,201,201) [t,z,x]
            pred = np.load(PRED_DIR / f"{sid}_prediction.npy").astype(np.float64)
            start = onset + IC_FRAMES
            assert pred.shape == (len(time_s) - start, 201, 201)
            frames = pick_frames(onset, len(time_s))

            fig, axes = plt.subplots(3, 3, figsize=(12.6, 11.4), constrained_layout=True)
            rec_numbers = {"onset": onset, "future_start_frame": start, "frames": {}}
            for row, frame in enumerate(frames):
                t_true = truth[frame]
                t_pred = pred[frame - start]
                diff = t_pred - t_true
                vm = float(np.abs(np.stack([t_true, t_pred])).max())
                ve = float(np.abs(diff).max())
                rel = float(np.sqrt((diff ** 2).sum() / (t_true ** 2).sum()))
                label = f"t = {time_s[frame]:.3f} s"
                panels = [
                    (t_true, vm, f"Truth (LWC-84), {label}"),
                    (t_pred, vm, f"Neural operator, {label}"),
                    (diff, ve, f"Difference (frame relL2 = {rel:.3f})"),
                ]
                for col, (data, v, title) in enumerate(panels):
                    ax = axes[row, col]
                    im = ax.imshow(data, cmap="seismic", vmin=-v, vmax=v,
                                   extent=EXTENT, aspect="equal")
                    ax.set_title(title, fontsize=11)
                    if row == 2:
                        ax.set_xlabel("x (m)", fontsize=11)
                    if col == 0:
                        ax.set_ylabel("z (m)", fontsize=11)
                    cb = fig.colorbar(im, ax=ax, shrink=0.82)
                    cb.ax.tick_params(labelsize=8)
                    cb.formatter.set_powerlimits((-2, 2))
                    cb.update_ticks()
                rec_numbers["frames"][int(frame)] = {
                    "time_s": float(time_s[frame]),
                    "frame_relative_l2": rel,
                    "shared_scale_pa": vm,
                    "error_scale_pa": ve,
                }
            fig.suptitle(f"{sid}: truth vs neural operator, early/middle/late "
                         f"(future window starts t = {time_s[start]:.3f} s)", fontsize=13)
            for ext in ("png", "pdf"):
                fig.savefig(OUT / f"snapshots_{sid}.{ext}", dpi=180 if ext == "png" else None,
                            bbox_inches="tight")
            plt.close(fig)
            numbers["records"][sid] = rec_numbers
            print("figure written for", sid)
    (OUT / "NUMBERS.json").write_text(json.dumps(numbers, indent=1))
    print("written", OUT / "NUMBERS.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
