#!/usr/bin/env python
"""Compare V72 full-retrain (per-frame loss) pilot against the frozen-anchor
v63 baseline. Reads existing JSON only -- no model, no GPU. Emits an ablation
table (stdout + CSV) and an overlay bar figure (baseline vs latest V72 epoch)."""
import json, os, sys, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN="/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/saved_time_v72_full_retrain_per_frame_r1/pilot/metrics.jsonl"
V63="/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/saved_time_v63_full_dataset_allband_r1/pilot/terminal.json"
OUT="results/v72_baseline_diag"; os.makedirs(OUT, exist_ok=True)

def flat(m):
    tb=m.get("time_bin_relative_l2",{}); sb=m.get("spectrum_relative_l2",{}); fam=m.get("family_relative_l2",{})
    return dict(agg=m.get("aggregate_relative_l2"), U=fam.get("uniform"), L=fam.get("layered"), M=fam.get("marmousi"),
               pre=tb.get("pre_onset"), early=tb.get("early"), mid=tb.get("middle"), late=tb.get("late"),
               low=sb.get("low"), midf=sb.get("middle"), high=sb.get("high"), phase=m.get("phase_correlation"))

base=flat(json.load(open(V63))["best"]["metrics"]) if os.path.exists(V63) else {}
rows=[]
if os.path.exists(RUN):
    for l in open(RUN):
        r=json.loads(l)
        if r.get("event")=="epoch":
            f=flat(r.get("metrics",{})); f["epoch"]=r.get("epoch")
            f["trainable"]=r.get("trainable_stage",{}).get("trainable_parameters")
            f["peakGiB"]=round(r.get("peak_cuda_bytes",0)/1024**3,2)
            rows.append(f)

cols=["epoch","agg","U","L","M","early","mid","late","low","midf","high","phase","peakGiB","trainable"]
def fmt(v): return f"{v:.3f}" if isinstance(v,float) else str(v)
print("\n=== V72 vs baseline (target agg<0.10, family<0.12) ===")
print("baseline(v63): agg={agg:.3f} U={U:.3f} L={L:.3f} M={M:.3f} late={late:.3f} high={high:.3f} phase={phase:.3f}".format(**base))
hdr=" | ".join(f"{c:>8}" for c in cols); print(hdr); print("-"*len(hdr))
for f in rows: print(" | ".join(f"{fmt(f.get(c,'')):>8}" for c in cols))
with open(f"{OUT}/ablation_table.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=cols); w.writeheader()
    for f in rows: w.writerow({c:f.get(c,"") for c in cols})

# overlay: baseline vs latest V72 epoch
if rows and base:
    cur=rows[-1]
    groups=[("time",["early","mid","late"]),("spectrum",["low","midf","high"]),("family",["U","L","M"])]
    fig,ax=plt.subplots(1,3,figsize=(15,4.3),constrained_layout=True)
    import numpy as np
    for a,(title,keys) in zip(ax,groups):
        x=np.arange(len(keys)); w=0.38
        a.bar(x-w/2,[base[k] for k in keys],w,label="v63 frozen",color="#bbb")
        a.bar(x+w/2,[cur[k] for k in keys],w,label=f"V72 ep{cur['epoch']}",color="#c44")
        a.set_xticks(x); a.set_xticklabels(keys); a.set_ylim(0,1.05); a.grid(alpha=.3,axis="y")
        a.axhline(0.1,ls="--",c="g"); a.set_title(f"{title} relative L2")
    ax[0].legend()
    fig.suptitle(f"V72 (per-frame loss, full retrain) ep{cur['epoch']} vs v63 frozen anchor — agg {cur['agg']:.3f} vs {base['agg']:.3f}")
    fig.savefig(f"{OUT}/v72_vs_baseline_overlay.png",dpi=150,bbox_inches="tight"); plt.close(fig)
    print(f"\noverlay -> {OUT}/v72_vs_baseline_overlay.png  (latest ep{cur['epoch']})")
print(f"table   -> {OUT}/ablation_table.csv  ({len(rows)} epochs)")
