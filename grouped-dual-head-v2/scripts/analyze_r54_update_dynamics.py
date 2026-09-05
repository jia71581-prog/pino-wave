"""Free R56-pre diagnostic: loss-component and gradient-norm dynamics of R54 v2.

Reads updates.jsonl (train-side only, no sealed data). Questions:
1. Which loss component dominates late training (mean vs hinge vs gradient)?
2. Is the gradient norm noisy/clipped (EMA rationale) or smooth?
3. What lr did the cosine schedule give at the holdout-best epoch (ep15)?
"""
import json

p = "/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2/results/r54_scratch_fullpool_40ep_v2_20260829/updates.jsonl"
rows = [json.loads(l) for l in open(p)]
print(f"{len(rows)} update events; keys: {sorted(rows[0])}")
by_ep = {}
for r in rows:
    by_ep.setdefault(r["epoch"], []).append(r)
print("ep  n  loss      rel_sq    parent_rs hinge     grad_comp corr_en   gnorm(med/max)  lr")
for ep in sorted(by_ep):
    rs = by_ep[ep]
    def med(k):
        v = sorted(float(r[k]) for r in rs)
        return v[len(v)//2]
    gs = sorted(float(r["gradient_norm"]) for r in rs)
    clipped = sum(1 for g in gs if g >= 0.999)
    print(f"{ep:3d} {len(rs):3d} {med('loss'):.6f} {med('relative_square'):.6f} {med('parent_relative_square'):.6f} "
          f"{med('hinge'):.6f} {med('gradient'):.6f} {med('correction_energy'):.5f} "
          f"{gs[len(gs)//2]:.3f}/{gs[-1]:.3f} clip{clipped}/{len(gs)} {rs[0]['learning_rate']:.2e}")
