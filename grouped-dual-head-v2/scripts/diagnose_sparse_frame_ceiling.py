#!/usr/bin/env python3
"""Feasibility diagnostic v2: with SPARSE multi-time snapshots (early+mid+late),
how low can a high-capacity per-instance fine-tune push the held-out relL2?

This measures the *information-content ceiling*, not a specific adapter's ceiling.
The per-instance correction is a free full-field tensor (max capacity) added to
the frozen parent, optimized by:
  * observed match on K sparse real frames spread across time (whole-field
    energy normalization, NOT the pathological per-near-zero-frame ratio);
  * LWC-84 PDE self-supervision (self-normalized -> scale invariant) on all times.

Held-out relL2 is reported on ENERGETIC frames only (truth per-frame energy >
5% of peak), excluding the observed frames — this is the real target for 1%.

Contract: only the K observed frames drive training; the rest of the truth is
opened solely to score.  With K spread across early/mid/late, the observed set
carries enough wave information to have a shot at 1%.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet_v3.data.index import build_manifest
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.instance_adaptation.losses import lwc84_residual
from scripts.run_v5_instance_adaptation import _load_parent, _predict_parent


def _relL2(pred, truth, mask):
    p = pred[:, mask].reshape(-1); t = truth[:, mask].reshape(-1)
    return float((p - t).norm() / t.norm().clamp_min(1e-12))


def run(config_path, sample_id, device_name, n_obs, steps_schedule, lr, pde_weight, seed):
    config = yaml.safe_load(Path(config_path).read_text())
    manifest = build_manifest(config["source_h5"])
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    model, normalizer = _load_parent(config, manifest, device)

    ds = GuardedOnsetDataset(config["source_h5"], manifest, split="validation",
                             sample_ids=(sample_id,), travel_time_h5=config.get("travel_time_h5"))
    record = ds[0]
    amp = record.source_parameters.to(device).unsqueeze(0)[:, 4]
    time_s = record.time_s.to(device)
    velocity = record.velocity_mps.to(device).unsqueeze(0)
    T = len(time_s)

    parent_n = _predict_parent(model, normalizer, record, device, normalized=True)
    with h5py.File(config["source_h5"], "r", swmr=True) as h5:
        truth_phys = torch.from_numpy(np.asarray(h5["wavefield"][record.source_index], dtype=np.float32)).unsqueeze(0).to(device)
    truth_n = normalizer.encode_pressure(truth_phys, amp)

    # energetic frames (truth energy > 5% of peak)
    fe = truth_n.flatten(2).norm(dim=2)[0]
    peak = fe.max()
    energetic = fe > 0.05 * peak
    en_idx = torch.nonzero(energetic).flatten()

    # SPARSE observed frames: n_obs frames spread evenly across the energetic span
    lo, hi = int(en_idx.min()), int(en_idx.max())
    obs_idx = torch.linspace(lo, hi, n_obs).round().long().unique().tolist()
    obs_set = set(obs_idx)
    # held-out = energetic frames NOT observed
    heldout = energetic.clone()
    for i in obs_idx:
        heldout[i] = False

    obs_frames = truth_n[:, obs_idx].detach()  # only these true frames drive training

    # MAX-CAPACITY per-instance correction: free full-field tensor, zero init
    correction = torch.zeros_like(parent_n, requires_grad=True)
    opt = torch.optim.Adam([correction], lr=lr)
    dt = float(time_s[1] - time_s[0])

    baseline_heldout = _relL2(parent_n, truth_n, heldout)
    baseline_global = _relL2(parent_n, truth_n, energetic)
    whole_ref = truth_n.flatten().norm().clamp_min(1e-12)

    history = []
    done = 0
    torch.manual_seed(seed)
    for target in steps_schedule:
        while done < target:
            opt.zero_grad(set_to_none=True)
            raw = parent_n + correction
            # observed match: whole-field energy normalization over the K frames
            obs_loss = (raw[:, obs_idx] - obs_frames).flatten().norm() / obs_frames.flatten().norm().clamp_min(1e-12)
            res = lwc84_residual(raw, velocity, dt=dt, dx=10.0, dz=10.0, observed_indices=(obs_idx[0], obs_idx[0]+1))
            pde_loss = res.square().mean()
            loss = obs_loss + pde_weight * pde_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            opt.step()
            done += 1
        with torch.no_grad():
            raw = parent_n + correction
            hl = _relL2(raw, truth_n, heldout)
            obsm = _relL2(raw, truth_n, torch.tensor([i in obs_set for i in range(T)], device=device))
        history.append({"step": done, "heldout_relL2": hl, "observed_relL2": obsm,
                        "obs_loss": float(obs_loss), "pde_loss": float(pde_loss)})
        print(f"  step {done:4d}: held-out {hl:.4f}  observed-frames {obsm:.4f}  (obs_l {float(obs_loss):.3f} pde_l {float(pde_loss):.3f})")

    ds.close()
    best = min(h["heldout_relL2"] for h in history)
    print(json.dumps({
        "sample_id": sample_id, "medium": record.medium_type,
        "n_obs": len(obs_idx), "obs_idx": obs_idx,
        "parent_baseline_heldout": baseline_heldout,
        "parent_baseline_energetic_global": baseline_global,
        "best_heldout_relL2": best, "history": history,
    }, indent=2))
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-obs", type=int, default=12)
    ap.add_argument("--steps", default="50,150,400,800")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--pde-weight", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args(argv)
    sched = [int(x) for x in args.steps.split(",")]
    print(f"=== sparse-frame diagnostic: {args.sample_id}  n_obs={args.n_obs} lr={args.lr} pde_w={args.pde_weight} ===")
    run(args.config, args.sample_id, args.device, args.n_obs, sched, args.lr, args.pde_weight, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
