"""R57 free probe: do K early wavefield frames (+ velocity, source) determine the future?

Deployment contract: inputs = first-K truth snapshots + velocity + source + PDE.
No parent solver, no future truth. Before any GPU run we test, on already-open
fit records only, whether the early->future map is stable enough to learn:

For record pairs within a family, compare
  d_early  = rel L2 distance over the first K stored frames
  d_cond   = rel L2 distance of (velocity field, source params)
  d_future = rel L2 distance over the remaining stored frames
A learnable operator needs pairs close in (early, conditioning) to be close in
future (one-sided Lipschitz). We report, per family and K, the future distance
of the closest pairs and the Spearman rank correlation.
"""
import json
import random

import h5py
import numpy as np

CACHE = "/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2/results/r38_full_coverage_cache_20260828/fit_shard_0.h5"
KS = (4, 8, 16)
PAIRS_PER_FAMILY = 400
rng = random.Random(570830)


def rel_l2(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    denom = max(np.linalg.norm(a), np.linalg.norm(b), 1e-12)
    return float(np.linalg.norm(a - b) / denom)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx**2).sum() * (ry**2).sum()))


with h5py.File(CACHE, "r") as f:
    families = np.asarray(f["family"].asstr()[:])
    n = len(families)
    out = {"schema": "r57_early_frame_identifiability_probe_v1", "cache": CACHE,
           "pairs_per_family": PAIRS_PER_FAMILY, "results": {}}
    for fam in sorted(set(families)):
        idx = np.flatnonzero(families == fam)
        pairs = set()
        while len(pairs) < min(PAIRS_PER_FAMILY, len(idx) * (len(idx) - 1) // 2):
            i, j = rng.sample(list(idx), 2)
            pairs.add((min(i, j), max(i, j)))
        pairs = sorted(pairs)
        d_early = {k: [] for k in KS}
        d_future, d_vel = [], []
        for i, j in pairs:
            ti = f["truth_norm"][i]  # (64, 201, 201) fp16
            tj = f["truth_norm"][j]
            for k in KS:
                d_early[k].append(rel_l2(ti[:k], tj[:k]))
            d_future.append(rel_l2(ti[16:], tj[16:]))
            d_vel.append(rel_l2(f["static_features"][i][0], f["static_features"][j][0]))
        d_future = np.asarray(d_future)
        fam_out = {"n_records": int(len(idx)), "n_pairs": len(pairs),
                   "future_dist_median": float(np.median(d_future))}
        for k in KS:
            de = np.asarray(d_early[k])
            order = np.argsort(de)
            closest = order[: max(len(order) // 10, 5)]  # closest decile in early space
            fam_out[f"K{k}"] = {
                "spearman_early_vs_future": spearman(de, d_future),
                "early_dist_median": float(np.median(de)),
                "closest_decile_early_median": float(np.median(de[closest])),
                "closest_decile_future_median": float(np.median(d_future[closest])),
                "closest_decile_future_max": float(np.max(d_future[closest])),
            }
        dv = np.asarray(d_vel)
        fam_out["spearman_velocity_vs_future"] = spearman(dv, d_future)
        out["results"][fam] = fam_out
        print(fam, json.dumps(fam_out, indent=1))

with open("/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2/results/r57_early_frame_identifiability_probe_v1_20260830.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("written")
