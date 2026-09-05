#!/usr/bin/env python3
"""True / pretrained-prediction / error wavefield triptychs at three moments.

Uses the ACTUAL frozen pretrained parent (parent_field in the pilot fields.pt,
~0.47 held-out) — NOT the w192 overfit-ceiling model.  Ground truth is loaded
from the dataset H5 by source_index.  Symmetric diverging colormap per the
wavefield-viz convention; error panels annotate per-frame relative L2.
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2")
H5 = "/root/autodl-tmp/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5"
DT = 0.0025
MOMENTS = [80, 240, 400]  # t = 0.20s, 0.60s, 1.00s (early / mid / late)
SRC = {"uniform": 2869, "layered": 2938, "marmousi": 3359}
OUT = ROOT / "results/wavefield_viz"


def _rel(p, g):
    return float(np.linalg.norm(p - g) / max(np.linalg.norm(g), 1e-30))


def _sym(ax, field, title, vmax):
    im = ax.imshow(field, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[0, 2000, 2000, 0], aspect="equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def make(fam):
    f = sorted(glob.glob(str(ROOT / f"results/instance_adaptation/pilot/validation_{fam}_*/fields.pt")))[0]
    d = torch.load(f, map_location="cpu")
    pred = d["parent_field"][0].numpy()          # (401,201,201) pretrained prediction
    with h5py.File(H5, "r", swmr=True) as h:
        truth = np.asarray(h["wavefield"][SRC[fam]], dtype=np.float32)

    fig, axes = plt.subplots(len(MOMENTS), 3, figsize=(13, 4.2 * len(MOMENTS)))
    fig.suptitle(f"Pretrained-model wavefield — {fam} (frozen parent, held-out ≈0.47)",
                 fontsize=13, y=0.995)
    for r, t in enumerate(MOMENTS):
        g, p = truth[t], pred[t]
        err = p - g
        rel = _rel(p, g)
        vmax = max(np.abs(g).max(), 1e-30)
        evmax = max(np.abs(err).max(), 1e-30)
        _sym(axes[r, 0], g, f"True  t={t*DT:.2f}s", vmax)
        _sym(axes[r, 1], p, "Pred (pretrained)", vmax)
        _sym(axes[r, 2], err, f"Error  relL2={rel*100:.1f}%", evmax)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = OUT / f"pretrained_snapshots_{fam}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    rels = {f"{t*DT:.2f}s": round(_rel(pred[t], truth[t]), 4) for t in MOMENTS}
    print(f"{fam}: wrote {out.name}  per-frame relL2 {rels}")
    return rels


if __name__ == "__main__":
    summary = {fam: make(fam) for fam in ("uniform", "layered", "marmousi")}
    (OUT / "pretrained_snapshots_relL2.json").write_text(json.dumps(summary, indent=2))
    print("done")
