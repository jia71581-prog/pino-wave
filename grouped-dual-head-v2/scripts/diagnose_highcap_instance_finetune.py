#!/usr/bin/env python3
"""Feasibility diagnostic: how low can a 2-frame, high-capacity per-instance
fine-tune push the held-out (unobserved-time) relative L2?

Contract respected: the adapter reads ONLY the two early onset frames.  Later
truth is never used for training; it is opened solely to *score* held-out
frames after each optimization checkpoint.

What is exercised here (beyond the deployed LoRA):
  * the whole residual head is unfrozen (13k params), not just latent_delta+gate;
  * the LWC-84 physics residual (self-normalized -> scale invariant, so valid in
    normalized O(1) space) supervises UNOBSERVED times, pushing the solution
    forward from the 2 observed frames;
  * observed-frame match anchors the two known frames.

Everything runs in the normalizer's O(1) space (decoded pressure ~1e-9 is
ill-conditioned).  Reports held-out relL2 vs the frozen parent baseline at
several step counts so we can see the accuracy ceiling of this route.
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
from saved_time_phase_operator_v4.instance_adaptation.adapters import OnsetAdaptedV5
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.instance_adaptation.losses import (
    lwc84_residual,
    build_fixed_physics_points,
    build_rad_physics_points,
    sample_fixed_physics_residual,
)
from scripts.run_v5_instance_adaptation import (
    _load_parent,
    _load_conditioner,
    _predict_parent,
)


def _heldout_rel_l2(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> float:
    """Energy-aggregated relL2 over masked (unobserved) frames."""
    p = pred[:, mask].reshape(-1)
    t = truth[:, mask].reshape(-1)
    return float((p - t).norm() / t.norm().clamp_min(1e-12))


def run(config_path, sample_id, device_name, steps_schedule, lr, pde_weight, seed,
        *, sampler="full", count=512, k=1.0, c=1.0, time_tilt=1.5, resample_every=4):
    config = yaml.safe_load(Path(config_path).read_text())
    manifest = build_manifest(config["source_h5"])
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    model, normalizer = _load_parent(config, manifest, device)

    # locate the requested validation record
    travel_h5 = config.get("travel_time_h5")
    dataset = GuardedOnsetDataset(
        config["source_h5"], manifest, split="validation",
        sample_ids=(sample_id,), travel_time_h5=travel_h5,
    )
    record = dataset[0]
    src_amp = record.source_parameters.to(device).unsqueeze(0)[:, 4]
    time_s = record.time_s.to(device)
    indices = record.observed_indices

    # parent field in normalized O(1) space
    parent_n = _predict_parent(model, normalizer, record, device, normalized=True)

    # held-out mask: strictly after the 2nd observed frame
    mask = torch.zeros(parent_n.shape[1], dtype=torch.bool, device=device)
    mask[indices[1] + 1:] = True

    # full truth (opened ONLY to score, never to train), in normalized space
    with h5py.File(config["source_h5"], "r", swmr=True) as h5:
        truth_phys = torch.from_numpy(
            np.asarray(h5["wavefield"][record.source_index], dtype=np.float32)
        ).unsqueeze(0).to(device)
    truth_n = normalizer.encode_pressure(truth_phys, src_amp)

    # observed frames in normalized space
    observed_n = normalizer.encode_pressure(
        record.observed_wavefield.to(device).unsqueeze(0), src_amp
    )
    velocity = record.velocity_mps.to(device).unsqueeze(0)
    source = record.source_parameters.to(device).unsqueeze(0)

    adapter = OnsetAdaptedV5(
        model, latent_dim=int(config.get("latent_dim", 32)),
        lora_rank=int(config.get("lora_rank", 4)),
    ).to(device)
    cond_ckpt = config.get("conditioner_checkpoint")
    if cond_ckpt and Path(cond_ckpt).exists():
        _load_conditioner(adapter, cond_ckpt, device, expected_manifest_digest=manifest.digest)

    # HIGH-CAPACITY: unfreeze the whole residual head (not deployment mode)
    adapter._deployment_mode = False
    for p in adapter.conditioner.parameters():
        p.requires_grad_(False)          # keep meta conditioner frozen
    for p in adapter.residual.parameters():
        p.requires_grad_(True)           # unfreeze full residual head
    adapter.latent_delta.requires_grad_(True)
    adapter.residual_gate.requires_grad_(True)
    with torch.no_grad():
        adapter.residual_gate.fill_(1.0)
    params = [p for p in adapter.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)

    dt = float(time_s[1] - time_s[0])
    opt = torch.optim.Adam(params, lr=lr)

    baseline = _heldout_rel_l2(parent_n, truth_n, mask)
    obs_pos = list(indices)
    # Collocation sampler on OUR adapter's PDE term: full (every cell), uniform
    # (512 fixed random points, historical), or rad (512 residual-adaptive +
    # causal-tilt points, resampled every `resample_every` Adam steps).
    rad_generator = torch.Generator(device=device).manual_seed(seed) if sampler == "rad" else None
    colloc = {"points": None}
    history = []
    done = 0
    torch.manual_seed(seed)
    for target in steps_schedule:
        while done < target:
            opt.zero_grad(set_to_none=True)
            raw = adapter.raw_wavefield(parent_n, velocity, source, observed_n, time_s)
            # observed-frame anchor: whole-field energy normalization over the K
            # observed frames (NOT the pathological per-near-zero-frame ratio,
            # which explodes on early onset frames — see REPORT bug #2).
            obs_loss = ((raw[:, obs_pos] - observed_n).flatten().norm()
                        / observed_n.flatten().norm().clamp_min(1e-8))
            # PDE self-supervision on unobserved times (self-normalized residual)
            res = lwc84_residual(raw, velocity, dt=dt, dx=10.0, dz=10.0, observed_indices=indices)
            if sampler == "full":
                pde_loss = res.square().mean()
            elif sampler == "uniform":
                if colloc["points"] is None:
                    colloc["points"] = build_fixed_physics_points(
                        len(time_s), indices, count=count, seed=seed
                    ).to(device)
                pde_loss = sample_fixed_physics_residual(res, colloc["points"], indices).square().mean()
            elif sampler == "rad":
                if colloc["points"] is None or done % max(1, resample_every) == 0:
                    colloc["points"] = build_rad_physics_points(
                        res.detach(), indices, count=count, k=k, c=c,
                        time_tilt=time_tilt, generator=rad_generator,
                    )
                pde_loss = sample_fixed_physics_residual(res, colloc["points"], indices).square().mean()
            else:
                raise ValueError(f"unknown sampler {sampler}")
            loss = obs_loss + pde_weight * pde_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            done += 1
        with torch.no_grad():
            raw = adapter.raw_wavefield(parent_n, velocity, source, observed_n, time_s)
            hl = _heldout_rel_l2(raw, truth_n, mask)
        history.append({"step": done, "heldout_relL2": hl,
                        "obs_loss": float(obs_loss), "pde_loss": float(pde_loss)})
        print(f"  step {done:4d}: held-out relL2 = {hl:.4f}  (obs {float(obs_loss):.3f}  pde {float(pde_loss):.3f})")

    dataset.close()
    best = min(h["heldout_relL2"] for h in history)
    print(json.dumps({
        "sample_id": sample_id, "medium": record.medium_type, "sampler": sampler,
        "trainable_params": n_train, "parent_baseline_heldout_relL2": baseline,
        "best_heldout_relL2": best, "history": history,
    }, indent=2))
    return best, baseline


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", default="20,50,100,200,400")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pde-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--sampler", choices=("full", "uniform", "rad"), default="full",
                    help="PDE collocation sampler on our adapter's fine-tune loss")
    ap.add_argument("--count", type=int, default=512)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--time-tilt", type=float, default=1.5)
    ap.add_argument("--resample-every", type=int, default=4)
    args = ap.parse_args(argv)
    sched = [int(x) for x in args.steps.split(",")]
    print(f"=== diagnostic: {args.sample_id}  lr={args.lr} pde_w={args.pde_weight} sampler={args.sampler} ===")
    run(args.config, args.sample_id, args.device, sched, args.lr, args.pde_weight, args.seed,
        sampler=args.sampler, count=args.count, k=args.k, c=args.c,
        time_tilt=args.time_tilt, resample_every=args.resample_every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
