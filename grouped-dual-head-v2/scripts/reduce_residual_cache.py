"""Reduce cached residual .npz -> shared-temporal-basis + per-record rank-r floors.

Reads whatever rec*.npz are present in --cache-dir (each has R=target-warp, G=target,
shape (Tlate, HW)), groups by family via --order-map, and reports the separable-rank floor
per family for however many records have completed. Safe to run while the oracle is still
filling the cache -> gives incremental family floors without waiting for all 12 records.
"""
import argparse, json
from pathlib import Path
import numpy as np

# record_index -> family (matches residual_rank_oracle picking first 4/family on validation split)
ORDER = {90: "layered", 91: "layered", 92: "layered", 93: "layered",
         330: "marmousi", 331: "marmousi", 332: "marmousi", 333: "marmousi",
         0: "uniform", 1: "uniform", 2: "uniform", 3: "uniform"}


def shared_floor(Rs, Gs, r):
    M = np.concatenate(Rs, axis=1)          # (Tlate, Nrec*HW)
    G = np.concatenate(Gs, axis=1)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    r = min(r, len(s))
    Mr = (U[:, :r] * s[:r]) @ Vt[:r]
    return float(np.linalg.norm(M - Mr) / (np.linalg.norm(G) + 1e-12))


def perrec_floor(R, G, r):
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    r = min(r, len(s))
    Rr = (U[:, :r] * s[:r]) @ Vt[:r]
    return float(np.linalg.norm(R - Rr) / (np.linalg.norm(G) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--ranks", default="8,16,32,48,64")
    args = ap.parse_args()
    ranks = [int(x) for x in args.ranks.split(",")]
    fams = {}
    warp = {}
    for f in sorted(args.cache_dir.glob("rec*.npz")):
        idx = int(f.stem.replace("rec", ""))
        fam = ORDER.get(idx, "?")
        z = np.load(f)
        R, G = z["R"], z["G"]
        fams.setdefault(fam, ([], []))
        fams[fam][0].append(R); fams[fam][1].append(G)
        warp.setdefault(fam, []).append(round(float(np.linalg.norm(R) / np.linalg.norm(G)), 4))
    out = {"records_per_family": {f: len(v[0]) for f, v in fams.items()},
           "warp_rel_l2": warp, "shared_temporal_basis_floor": {}, "perrecord_floor_mean": {}}
    for fam, (Rs, Gs) in sorted(fams.items()):
        out["shared_temporal_basis_floor"][fam] = {str(r): round(shared_floor(Rs, Gs, r), 4) for r in ranks}
        out["perrecord_floor_mean"][fam] = {
            str(r): round(float(np.mean([perrec_floor(R, G, r) for R, G in zip(Rs, Gs)])), 4) for r in ranks}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
