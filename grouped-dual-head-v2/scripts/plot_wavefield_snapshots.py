#!/usr/bin/env python3
"""Plot wavefield snapshots + per-timestep error curve from a coarse-field diagnostic npz.

Follows the wavefield-viz skill colour rules: diverging symmetric colormap for
pressure fields and errors (vmin=-vmax, 0 centred), physical extent, z down.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--dx", type=float, default=10.0)
    p.add_argument("--dz", type=float, default=10.0)
    p.add_argument("--tag", default="")
    args = p.parse_args(argv)

    d = np.load(args.npz)
    true, coarse, pred = d["true"], d["coarse"], d["pred"]   # (F,X,Z)
    times = d["times"]; dump_idx = d["dump_idx"]
    rl2_coarse, rl2_pred = d["rl2_coarse"], d["rl2_pred"]
    fam = str(d["family"]) if "family" in d.files else args.tag
    F, X, Z = true.shape
    ext = [0, X * args.dx, Z * args.dz, 0]
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: True | Pred | Error, one row per snapshot ----
    fig, ax = plt.subplots(F, 3, figsize=(11, 3.1 * F), constrained_layout=True)
    if F == 1:
        ax = ax[None, :]
    for r in range(F):
        t = true[r]; pr = pred[r]; er = pr - t
        vm = np.abs(np.concatenate([t, pr])).max() or 1e-9
        ve = np.abs(er).max() or 1e-9
        rl2 = np.linalg.norm(er) / max(np.linalg.norm(t), 1e-9)
        ts = times[dump_idx[r]]
        panels = [(t, vm, f"True  t={ts:.2f}s"),
                  (pr, vm, f"Pred (corrected)"),
                  (er, ve, f"Error  relL2={rl2:.1%}")]
        for c, (dat, v, ttl) in enumerate(panels):
            im = ax[r, c].imshow(dat.T, cmap="seismic", vmin=-v, vmax=v,
                                 extent=ext, aspect="auto")
            ax[r, c].set_title(ttl, fontsize=11)
            ax[r, c].set_xlabel("x (m)"); ax[r, c].set_ylabel("z (m)")
            fig.colorbar(im, ax=ax[r, c], shrink=0.85)
    fig.suptitle(f"Wavefield snapshots — {fam} (w192, corrected prediction)", fontsize=13)
    f1 = outdir / f"snapshots_true_pred_error_{fam}.png"
    fig.savefig(f1, dpi=160, bbox_inches="tight"); plt.close(fig)

    # ---- Figure 2: True | Coarse (MIONet) | Corrected — mechanism view ----
    fig, ax = plt.subplots(F, 3, figsize=(11, 3.1 * F), constrained_layout=True)
    if F == 1:
        ax = ax[None, :]
    for r in range(F):
        t = true[r]; co = coarse[r]; pr = pred[r]
        vm = np.abs(np.concatenate([t, co, pr])).max() or 1e-9
        ts = times[dump_idx[r]]
        rc = np.linalg.norm(co - t) / max(np.linalg.norm(t), 1e-9)
        rp = np.linalg.norm(pr - t) / max(np.linalg.norm(t), 1e-9)
        panels = [(t, f"True  t={ts:.2f}s"),
                  (co, f"Coarse MIONet  relL2={rc:.1%}"),
                  (pr, f"Corrected  relL2={rp:.1%}")]
        for c, (dat, ttl) in enumerate(panels):
            im = ax[r, c].imshow(dat.T, cmap="seismic", vmin=-vm, vmax=vm,
                                 extent=ext, aspect="auto")
            ax[r, c].set_title(ttl, fontsize=11)
            ax[r, c].set_xlabel("x (m)"); ax[r, c].set_ylabel("z (m)")
            fig.colorbar(im, ax=ax[r, c], shrink=0.85)
    fig.suptitle(f"True vs coarse-MIONet vs corrected — {fam} (ringing bottleneck)", fontsize=13)
    f2 = outdir / f"snapshots_true_coarse_corrected_{fam}.png"
    fig.savefig(f2, dpi=160, bbox_inches="tight"); plt.close(fig)

    # ---- Figure 3: per-timestep relative-L2 error curve ----
    # Pre-onset frames have ~0 true-field norm, so frame-normalized relL2 explodes
    # (divide-by-tiny). Mask those out (relL2 >= 1 == essentially no signal yet).
    valid = (rl2_pred < 1.0) & (rl2_coarse < 5.0)
    fig, axc = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    axc.plot(times[valid], rl2_coarse[valid], label="Coarse MIONet", color="#c0392b", lw=1.6)
    axc.plot(times[valid], rl2_pred[valid], label="Corrected (coarse+decoder)", color="#2471a3", lw=1.6)
    for ts in times[dump_idx]:
        axc.axvline(ts, color="0.85", lw=0.6, zorder=0)
    axc.set_xlabel("time (s)"); axc.set_ylabel("relative L2")
    cm, pm = rl2_coarse[valid].mean(), rl2_pred[valid].mean()
    axc.set_title(f"Per-timestep error — {fam} (post-onset window)\n"
                  f"coarse mean={cm:.3f}  corrected mean={pm:.3f}")
    axc.grid(alpha=0.3); axc.legend()
    axc.set_ylim(0, max(rl2_coarse[valid].max(), rl2_pred[valid].max()) * 1.1)
    f3 = outdir / f"error_curve_{fam}.png"
    fig.savefig(f3, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"post-onset ({valid.sum()}/{len(valid)} frames): "
          f"coarse mean={cm:.4f}  corrected mean={pm:.4f}")

    print(f"wrote:\n  {f1}\n  {f2}\n  {f3}")
    print(f"coarse mean relL2={rl2_coarse.mean():.4f}  corrected mean relL2={rl2_pred.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
