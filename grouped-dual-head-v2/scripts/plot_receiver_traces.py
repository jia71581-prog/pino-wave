#!/usr/bin/env python3
"""Receiver-waveform (trace) plots from a full-frames npz.

Picks several receiver points at increasing offset from the source and overlays the
true vs predicted normalized pressure time series u(t). Traces expose phase / travel-time
/ amplitude errors that 2D snapshots hide. Also draws a receiver-gather (offset vs time)
image for true and predicted.
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
    args = p.parse_args(argv)

    d = np.load(args.npz)
    true, pred = d["true"], d["pred"]        # (T,X,Z)
    times = d["times"]                       # (T,)
    fam = str(d["family"])
    sx, sz = float(d["source_xz"][0]), float(d["source_xz"][1])
    T, X, Z = true.shape
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # source grid index
    six = int(round(sx / args.dx)); siz = int(round(sz / args.dz))
    six = min(max(six, 0), X - 1); siz = min(max(siz, 0), Z - 1)

    # ---- pick receivers at a depth where the true field actually carries energy ----
    # choose receiver depth: a mid band, not the (possibly near-boundary) source depth
    # so shallow near-surface sources still see propagating energy below them.
    per_depth_energy = np.linalg.norm(true.reshape(T, X, Z), axis=(0, 1))  # (Z,)
    rz = int(np.argmax(per_depth_energy))
    # spread receivers toward whichever side has more room from the source
    room_right = X - 1 - six
    room_left = six
    sign = 1 if room_right >= room_left else -1
    span = max(room_right, room_left)
    offsets = [int(span * f) for f in (0.15, 0.4, 0.65, 0.9)]
    recs = []
    for o in offsets:
        rx = min(max(six + sign * o, 0), X - 1)
        recs.append((rx, rz))

    # ---- Figure A: overlaid traces ----
    n = len(recs)
    fig, ax = plt.subplots(n, 1, figsize=(9, 2.2 * n), sharex=True, constrained_layout=True)
    if n == 1:
        ax = [ax]
    for k, (rx, rzk) in enumerate(recs):
        ut = true[:, rx, rzk]; up = pred[:, rx, rzk]
        off_m = abs(rx - six) * args.dx
        rl2 = np.linalg.norm(up - ut) / max(np.linalg.norm(ut), 1e-9)
        ax[k].plot(times, ut, color="k", lw=1.4, label="True")
        ax[k].plot(times, up, color="#c0392b", lw=1.2, ls="--", label="Pred")
        ax[k].set_ylabel("u (norm.)")
        ax[k].set_title(f"Receiver @ x-offset {off_m:.0f} m  (z={rzk*args.dz:.0f} m)   trace relL2={rl2:.1%}",
                        fontsize=10)
        ax[k].grid(alpha=0.3)
        if k == 0:
            ax[k].legend(loc="upper right", fontsize=9)
    ax[-1].set_xlabel("time (s)")
    fig.suptitle(f"Receiver waveforms — {fam} (w192)", fontsize=13)
    fA = outdir / f"traces_{fam}.png"
    fig.savefig(fA, dpi=160, bbox_inches="tight"); plt.close(fig)

    # ---- Figure B: receiver gather (all x at fixed depth) true | pred | error ----
    gt = true[:, :, rz].T        # (X, T) -> receivers vs time
    gp = pred[:, :, rz].T
    ge = gp - gt
    x_axis = np.arange(X) * args.dx
    ext = [times[0], times[-1], x_axis[-1], x_axis[0]]
    vm = np.abs(np.concatenate([gt, gp])).max() or 1e-9
    ve = np.abs(ge).max() or 1e-9
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for a, dat, v, ttl in [(ax[0], gt, vm, "True gather"),
                           (ax[1], gp, vm, "Pred gather"),
                           (ax[2], ge, ve, "Error")]:
        im = a.imshow(dat, cmap="seismic", vmin=-v, vmax=v, extent=ext, aspect="auto")
        a.axhline(sx, color="lime", lw=0.8, ls=":")     # source x location
        a.set_title(ttl); a.set_xlabel("time (s)"); a.set_ylabel("receiver x (m)")
        fig.colorbar(im, ax=a, shrink=0.85)
    fig.suptitle(f"Receiver gather at z={rz*args.dz:.0f} m — {fam} (w192)", fontsize=13)
    fB = outdir / f"gather_{fam}.png"
    fig.savefig(fB, dpi=160, bbox_inches="tight"); plt.close(fig)

    print(f"wrote:\n  {fA}\n  {fB}\n  source=({sx:.0f},{sz:.0f})m grid=({six},{siz})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
