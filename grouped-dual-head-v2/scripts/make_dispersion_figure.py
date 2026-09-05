#!/usr/bin/env python3
"""Build publication figures for the coarse-grid numerical-dispersion study."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_all(out_root):
    cases = {}
    for sdir in sorted(glob.glob(str(Path(out_root) / "shard*"))):
        sj = Path(sdir) / "dispersion_summary.json"
        if not sj.exists():
            continue
        payload = json.load(open(sj))
        for rec in payload["per_case"]:
            rid = rec["record_id"]
            npz_path = Path(sdir) / "fields" / f"{rid}.npz"
            if not npz_path.exists():
                continue
            z = np.load(npz_path)
            cases[rid] = {"metrics": rec, "z": z, "shard": sdir}
    return cases


def pick_representative(cases, role="interp_center"):
    """slice whose DFO-vs-PS late error is the median among interp_center."""
    cands = {r: c for r, c in cases.items() if r.endswith(role)}
    vals = sorted(
        (c["metrics"]["moments"]["dfo"]["late"]["B"], r) for r, c in cands.items()
    )  # ordering proxy only; choose middle by index instead below
    keys = sorted(cands)
    mid = keys[len(keys) // 2]
    return mid, cands[mid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--figures-dir", required=True)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    fdir = Path(args.figures_dir)
    fdir.mkdir(parents=True, exist_ok=True)

    # unit calibration: reproduce gamma from any shard summary
    gamma = None
    for sj in sorted(glob.glob(str(Path(args.out_root) / "shard*" / "dispersion_summary.json"))):
        gamma = json.load(open(sj)).get("archive_to_physical_gamma")
        break
    assert gamma, "gamma missing"
    inv = 1.0 / gamma

    cases = load_all(args.out_root)
    print(f"[figs] loaded {len(cases)} cases")

    # aggregate summary across shards
    merged = {"fd2": {}, "dfo": {}, "lwc84": {}}
    for name in merged:
        rows = []
        for sj in sorted(glob.glob(str(Path(args.out_root) / "shard*" / "dispersion_summary.json"))):
            d = json.load(open(sj))["summary_vs_pseudo_spectral"].get(name)
            if d:
                for k, v in d.items():
                    rows.append((k, v["median"], v["iqr"]))
        agg = {}
        for k in set(k for k, _, _ in rows):
            meds = [m for kk, m, _ in rows if kk == k]
            iqrs = [i for kk, _, i in rows if kk == k]
            agg[k] = (float(np.median(meds)), float(np.median(iqrs)))
        merged[name] = agg
    print(json.dumps({n: {k: [round(a, 4) for a in v] for k, v in d.items()}
                      for n, d in merged.items()}, indent=1))
    json.dump({"summary": merged, "archive_to_physical_gamma": gamma},
              open(fdir / "dispersion_summary_merged.json", "w"), indent=2)

    rid, case = pick_representative(cases)
    print("[figs] representative:", rid)
    z = case["z"]
    D = eval_dir
    ref_full = np.load(D / "references" / f"{rid}.npz")["target_tzx"].astype(np.float64) * inv
    pred_full = np.load(D / "predictions" / f"{rid}.npz")["prediction_tzx"].astype(np.float64) * inv

    methods = {
        "FD2 (classical)": z["snap_t060_fd2"].astype(np.float64),
        "LWC-84": z["snap_t060_lwc84"].astype(np.float64) * inv,
        "DFO (ours)": pred_full[240],
        "Pseudo-spectral anchor": z["snap_t060_ps"].astype(np.float64),
    }
    methods_late = {
        "FD2 (classical)": z["snap_t100_fd2"].astype(np.float64),
        "LWC-84": z["snap_t100_lwc84"].astype(np.float64) * inv,
        "DFO (ours)": pred_full[400],
        "Pseudo-spectral anchor": z["snap_t100_ps"].astype(np.float64),
    }

    def panel(fig, axes_row, imgs, titles, vmax):
        for ax, im, ti in zip(axes_row, imgs, titles):
            ax.imshow(im.T[::-1], origin="lower", cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax,
                      extent=[0, 2.0, 0, 2.0], aspect="auto", interpolation="bilinear")
            ax.set_title(ti, fontsize=8.5)
            ax.set_xlabel("x (km)", fontsize=7.5)
            ax.set_ylabel("z (km)", fontsize=7.5)
            ax.tick_params(labelsize=7)

    order = ["FD2 (classical)", "LWC-84", "DFO (ours)", "Pseudo-spectral anchor"]
    vmax_early = max(float(np.percentile(np.abs(methods[k]), 99.5)) for k in order)
    vmax_late = max(float(np.percentile(np.abs(methods_late[k]), 99.5)) for k in order)

    fig, axes = plt.subplots(2, 4, figsize=(12.5, 5.9), constrained_layout=True)
    panel(fig, axes[0], [methods[k] for k in order], [f"{k}\n$t=0.6$ s" for k in order], vmax_early)
    panel(fig, axes[1], [methods_late[k] for k in order],
          [f"{k}\n$t=1.0$ s" for k in order], vmax_late)
    fig.savefig(fdir / "dispersion_snapshots.png", dpi=300)
    fig.savefig(fdir / "dispersion_snapshots.pdf")
    plt.close(fig)

    # --- receiver-gather overlay (chosen case) --------------------------------
    # gathers were stored as [nrec, nt_late] (post 0.55 s), receivers every
    # RECV_X_STEP columns from column 40.
    rx = z["receiver_x_m"] / 1000.0
    tt = np.arange(z["ps"].shape[1]) * 2.5e-3 + 0.55
    fig2, axes2 = plt.subplots(1, 4, figsize=(12.5, 3.4), sharey=True,
                               constrained_layout=True)
    nrec = z["ps"].shape[0]
    pick_rows = [0, nrec // 3, 2 * nrec // 3, nrec - 1]
    for ax, name in zip(axes2, order):
        key = {"FD2 (classical)": "fd2", "LWC-84": "lwc84",
               "DFO (ours)": "dfo", "Pseudo-spectral anchor": "ps"}[name]
        g = z[key].astype(np.float64)                       # [nrec, nt]
        norm = max(float(np.percentile(np.abs(g), 99)), 1e-14)
        for j, ridx in enumerate(pick_rows):
            ax.plot(tt, g[ridx] / norm + j * 1.6, lw=0.7, color="#1f77b4")
        ax.set_title(f"{name}", fontsize=8.5)
        ax.set_xlabel("t (s)", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.set_xlim(tt[0], tt[-1])
    axes2[0].set_ylabel("trace index", fontsize=7.5)
    fig2.savefig(fdir / "dispersion_traces.png", dpi=300)
    fig2.savefig(fdir / "dispersion_traces.pdf")
    plt.close(fig2)
    print("saved:", sorted(p.name for p in fdir.iterdir()))


if __name__ == "__main__":
    main()
