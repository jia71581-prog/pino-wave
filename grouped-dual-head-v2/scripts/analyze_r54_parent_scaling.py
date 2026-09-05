"""R56-pre probe: parent->candidate error relation on the R54 best-epoch holdout records.

Question: if the marmousi parent (LWC-84 coarse solve) were improved, would the
capped-correction candidate clear max<=0.05? Free analysis, no GPU, no new data opened.
"""
import json

p = "/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2/results/r54_scratch_fullpool_40ep_v2_20260829/best.json"
d = json.load(open(p))
recs = d["metrics"]["records"]
fams = {}
for r in recs:
    fams.setdefault(r["family"], []).append(r)

for fam, rs in sorted(fams.items()):
    rs.sort(key=lambda r: -r["candidate_rel_l2"])
    print(f"== {fam} (n={len(rs)}) worst 6 by candidate ==")
    for r in rs[:6]:
        pe, ce = r["parent_rel_l2"], r["candidate_rel_l2"]
        ri = r["relative_improvement"]
        print(f"  {r['sample_id']}: parent {pe:.4f} -> cand {ce:.4f}  (ri {ri:.3f})")
    ks = [r["candidate_rel_l2"] / r["parent_rel_l2"] for r in rs]
    ks_sorted = sorted(ks)
    n = len(ks)
    print(f"  k=cand/parent: min {min(ks):.3f} med {ks_sorted[n//2]:.3f} max {max(ks):.3f}")
    kw = max(ks)
    print(f"  worst-case-k implied parent_max needed for cand<=0.05: {0.05/kw:.4f}")
