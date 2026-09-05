"""CPU param/activation-memory probe for _ContinuousTemporalLatentBasis vs rank.

Fills the mandate's per-candidate "参数量/激活显存估计" requirement for the A3
rank-escalation queue (rank32->64->96->128).  Measures, WITHOUT a GPU:
  * exact trainable parameter count per rank,
  * peak intra-forward activation bytes (via forward hooks summing every module
    output tensor's numel*element_size) at the realistic training microbatch
    (records=1, count=training_frames_per_record=24) and at the 201x201 full grid
    (upper bound; the coarse field is <= the query grid).
The DELTA between ranks is what determines whether rank64/96/128 fit under the
observed A3-rank32 tracked peak (22.07 GiB, cap 23.5).  Diagnostic only; touches
no training artifact and no GPU.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from saved_time_phase_operator_v4.local_field import _ContinuousTemporalLatentBasis

WIDTH = 128
COUNT = 24            # training_frames_per_record
RECORDS = 1          # microbatch_records
HARMONICS = 4

def probe(rank, H, W):
    op = _ContinuousTemporalLatentBasis(WIDTH, rank=rank, harmonics=HARMONICS, gate_init=0.03)
    nparam = sum(p.numel() for p in op.parameters() if p.requires_grad)
    act = {"bytes": 0}
    def hook(mod, inp, out):
        t = out if isinstance(out, torch.Tensor) else None
        if t is not None:
            act["bytes"] += t.numel() * t.element_size()
    hs = [m.register_forward_hook(hook) for m in op.modules() if len(list(m.children()))==0]
    cond = torch.randn(RECORDS, WIDTH, H, W)
    time_s = torch.rand(RECORDS, COUNT) * 0.4 + 0.05
    src = torch.zeros(RECORDS, 5); src[:,2]=15.0; src[:,3]=0.05
    out = op(cond, time_s, src, domain_t_s=1.0)
    for h in hs: h.remove()
    # dominant tensors, computed analytically for cross-check:
    anchors = RECORDS*rank*H*W*4          # (R,rank,H,W)
    residual = RECORDS*COUNT*H*W*4        # (R,T,H,W) -- rank-INDEPENDENT
    return {"rank": rank, "params": nparam,
            "hook_activation_MiB": round(act["bytes"]/1024**2, 2),
            "anchors_MiB": round(anchors/1024**2,2),
            "residual_MiB": round(residual/1024**2,2),
            "out_shape": list(out.shape)}

if __name__ == "__main__":
    res = {"width": WIDTH, "records": RECORDS, "count": COUNT, "harmonics": HARMONICS,
           "note": "A3-rank32 gate0 observed tracked peak = 22.07 GiB (cap 23.5). "
                   "Deltas below are the FULL temporal_latent forward activation; the "
                   "rank-dependent part is only the anchors tensor.", "grids": {}}
    for H in (128, 201):
        res["grids"][f"{H}x{H}"] = [probe(r, H, H) for r in (32, 64, 96, 128)]
    Path("/root/autodl-tmp/cc_research_supervisor/temporal_latent_rank_memory.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
