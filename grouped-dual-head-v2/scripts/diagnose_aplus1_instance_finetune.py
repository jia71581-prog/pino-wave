#!/usr/bin/env python3
"""Instance fine-tuning on the A+1 hybrid-solver parent: does the old information
wall still hold when the parent is a generalizable operator?

Background
----------
Prior instance-adaptation work (memory: fno-acoustic-instance-adaptation) found a
strict 2-early-frame fine-tune could NOT beat the frozen parent -- the information
wall -- but that parent was the old coarse-MIONet with a structural bottleneck. The
parent here is A+1 (Helmholtz-synthesis rank-8 + smoothed-velocity background P_bg),
a generalizable operator whose held-out layered/marmousi are already 0.082/0.045. The
question is whether early snapshots can now sharpen this already-good parent.

Contract respected (identical to diagnose_highcap_instance_finetune):
  * adapter reads ONLY the two early onset frames (GuardedOnsetDataset enforces);
  * later truth is opened solely to SCORE held-out frames, never to train;
  * everything in the normalizer's O(1) space (decoded pressure ~1e-9 is ill-conditioned);
  * the LWC-84 physics residual is the HIGH-ORDER (time_order=4) form matching the
    data-generating solver, supervising unobserved times.

Parent field = A+1 full field = model.dense_normalized (normalized scattering residual)
+ encode_pressure(P_bg).  P_bg comes from the same sigma=2 smoothed-velocity cache used
for G3 (it covers the held-out validation triplet), so this mirrors deployment where a
new record's P_bg is a cheap physical solve.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest
from saved_time_phase_operator_v4.probe import ProbeVariant
from saved_time_phase_operator_v4.instance_adaptation.adapters import OnsetAdaptedV5
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.instance_adaptation.losses import (
    lwc84_residual,
    build_fixed_physics_points,
    build_rad_physics_points,
    sample_fixed_physics_residual,
)
from saved_time_phase_operator_v4.background_field import BackgroundFieldProvider
from scripts.train_saved_time_v4_probe import _model
from scripts.train_grouped_v3_pilot import load_normalizer
from scripts.run_v5_instance_adaptation import _predict_parent

BASE_CONFIG = "configs/grouped_v3/continuous_pilot.yaml"


def _load_aplus1_parent(checkpoint, manifest, device, *, width, dense_depth,
                        dense_spectral_rank, dense_modes, helmholtz_frequencies,
                        helmholtz_rank, normalization_json):
    base = V3Config.from_yaml(BASE_CONFIG)
    if int(width) != int(base.model.width):
        base = dataclasses.replace(base, model=dataclasses.replace(base.model, width=int(width)))
    if normalization_json:
        base = dataclasses.replace(base, data=dataclasses.replace(
            base.data, normalization_json=str(normalization_json)))
    variant = ProbeVariant(
        depth=int(dense_depth), use_local_phase=True,
        spectral_rank=int(dense_spectral_rank), modes=int(dense_modes),
        temporal_basis_rank=0, family_expert_rank=0,
        local_field=True, local_field_residual=False,
        local_field_helmholtz_synthesis=True,
        local_field_helmholtz_synthesis_frequencies=int(helmholtz_frequencies),
        local_field_helmholtz_synthesis_wkb_phase=True,
        local_field_helmholtz_synthesis_rank=int(helmholtz_rank),
    )
    model = _model(base, manifest, variant).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("manifest_digest") not in (None, manifest.digest):
        raise ValueError("A+1 checkpoint manifest digest mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    normalizer = load_normalizer(base, manifest.digest)
    return model, normalizer


def _heldout_rel_l2(pred, truth, mask):
    p = pred[:, mask].reshape(-1)
    t = truth[:, mask].reshape(-1)
    return float((p - t).norm() / t.norm().clamp_min(1e-12))


def run(*, checkpoint, background_cache, source_h5, travel_time_h5, normalization_json,
        sample_id, device_name, steps_schedule, lr, pde_weight, seed, sampler,
        count, k, c, time_tilt, resample_every, width, dense_depth, dense_spectral_rank,
        dense_modes, helmholtz_frequencies, helmholtz_rank):
    torch.manual_seed(int(seed))
    manifest = build_manifest(source_h5)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")

    model, normalizer = _load_aplus1_parent(
        checkpoint, manifest, device, width=width, dense_depth=dense_depth,
        dense_spectral_rank=dense_spectral_rank, dense_modes=dense_modes,
        helmholtz_frequencies=helmholtz_frequencies, helmholtz_rank=helmholtz_rank,
        normalization_json=normalization_json,
    )
    background = BackgroundFieldProvider(str(background_cache))

    dataset = GuardedOnsetDataset(
        source_h5, manifest, split="validation",
        sample_ids=(sample_id,), travel_time_h5=travel_time_h5,
    )
    record = dataset[0]
    if not background.covers([record.sample_id]):
        raise ValueError(f"background cache does not cover {record.sample_id}")
    src_amp = record.source_parameters.to(device).unsqueeze(0)[:, 4]
    time_s = record.time_s.to(device)
    indices = record.observed_indices

    # A+1 parent field in normalized O(1) space, WITH background P_bg added back.
    parent_n = _predict_parent(
        model, normalizer, record, device, normalized=True, background_provider=background,
    )

    mask = torch.zeros(parent_n.shape[1], dtype=torch.bool, device=device)
    mask[indices[1] + 1:] = True

    with h5py.File(source_h5, "r", swmr=True) as h5:
        truth_phys = torch.from_numpy(
            np.asarray(h5["wavefield"][record.source_index], dtype=np.float32)
        ).unsqueeze(0).to(device)
    truth_n = normalizer.encode_pressure(truth_phys, src_amp)

    observed_n = normalizer.encode_pressure(
        record.observed_wavefield.to(device).unsqueeze(0), src_amp
    )
    velocity = record.velocity_mps.to(device).unsqueeze(0)
    source = record.source_parameters.to(device).unsqueeze(0)

    adapter = OnsetAdaptedV5(model, latent_dim=32, lora_rank=4).to(device)
    adapter._deployment_mode = False
    for p in adapter.conditioner.parameters():
        p.requires_grad_(False)
    for p in adapter.residual.parameters():
        p.requires_grad_(True)
    adapter.latent_delta.requires_grad_(True)
    adapter.residual_gate.requires_grad_(True)
    with torch.no_grad():
        adapter.residual_gate.fill_(1.0)
    params = [p for p in adapter.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)

    dt = float(time_s[1] - time_s[0])
    baseline = _heldout_rel_l2(parent_n, truth_n, mask)
    opt = torch.optim.Adam(params, lr=lr)
    rad_generator = torch.Generator(device=device).manual_seed(int(seed))

    schedule = sorted(int(s) for s in steps_schedule.split(","))
    rows = []
    step = 0
    points = None
    for target in schedule:
        while step < target:
            opt.zero_grad(set_to_none=True)
            raw = adapter.raw_wavefield(parent_n, velocity, source, observed_n, time_s)
            # observed_wavefield holds ONLY the two onset frames; compare the raw
            # prediction AT those frame indices against it (whole-field energy norm).
            obs = raw[:, list(indices)]
            obs_loss = ((obs - observed_n).flatten().norm()
                        / observed_n.flatten().norm().clamp_min(1e-8))
            residual_field = lwc84_residual(
                raw, velocity, dt=dt, dx=10.0, dz=10.0, observed_indices=indices,
            )  # time_order=4 (high-order LWC) by default
            if sampler == "full":
                pde = residual_field.square().mean()
            elif sampler == "uniform":
                if points is None:
                    points = build_fixed_physics_points(
                        len(time_s), indices, count=count, seed=seed
                    ).to(device)
                pde = sample_fixed_physics_residual(residual_field, points, indices).square().mean()
            else:  # rad
                if points is None or step % max(1, resample_every) == 0:
                    points = build_rad_physics_points(
                        residual_field.detach(), indices, count=count, k=k, c=c,
                        time_tilt=time_tilt, generator=rad_generator,
                    )
                pde = sample_fixed_physics_residual(residual_field, points, indices).square().mean()
            loss = obs_loss + pde_weight * pde + 1e-5 * (raw - parent_n).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
        with torch.no_grad():
            raw = adapter.raw_wavefield(parent_n, velocity, source, observed_n, time_s)
            held = _heldout_rel_l2(raw, truth_n, mask)
        rows.append({"step": step, "heldout_relL2": held, "obs_loss": float(obs_loss),
                     "pde": float(pde)})
        print(json.dumps(rows[-1], sort_keys=True), flush=True)

    result = {
        "sample_id": sample_id, "trainable_params": n_train,
        "parent_baseline_heldout_relL2": baseline,
        "future_truth_used": False, "observed_indices": list(indices),
        "time_order": 4, "sampler": sampler, "pde_weight": pde_weight,
        "schedule": rows,
        "best_heldout_relL2": min(r["heldout_relL2"] for r in rows),
        "beats_parent": min(r["heldout_relL2"] for r in rows) < baseline,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="A+1 G3 best checkpoint .pt")
    ap.add_argument("--background-cache", required=True, help="sigma=2 P_bg cache covering the held-out record")
    ap.add_argument("--source-h5", default="/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5")
    ap.add_argument("--travel-time-h5", default="/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5")
    ap.add_argument("--normalization-json", default="/root/autodl-tmp/home/jiayh/Data/data/processed/grouped_v3_normalization.before_tgrs_ablation_identity_20260726T1050.json")
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", default="20,50,100,200,400")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pde-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--sampler", choices=("full", "uniform", "rad"), default="rad")
    ap.add_argument("--count", type=int, default=512)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--time-tilt", type=float, default=1.5)
    ap.add_argument("--resample-every", type=int, default=4)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--dense-depth", type=int, default=8)
    ap.add_argument("--dense-spectral-rank", type=int, default=112)
    ap.add_argument("--dense-modes", type=int, default=32)
    ap.add_argument("--helmholtz-frequencies", type=int, default=64)
    ap.add_argument("--helmholtz-rank", type=int, default=8)
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    result = run(
        checkpoint=args.checkpoint, background_cache=args.background_cache,
        source_h5=args.source_h5, travel_time_h5=args.travel_time_h5,
        normalization_json=args.normalization_json, sample_id=args.sample_id,
        device_name=args.device, steps_schedule=args.steps, lr=args.lr,
        pde_weight=args.pde_weight, seed=args.seed, sampler=args.sampler,
        count=args.count, k=args.k, c=args.c, time_tilt=args.time_tilt,
        resample_every=args.resample_every, width=args.width, dense_depth=args.dense_depth,
        dense_spectral_rank=args.dense_spectral_rank, dense_modes=args.dense_modes,
        helmholtz_frequencies=args.helmholtz_frequencies, helmholtz_rank=args.helmholtz_rank,
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
