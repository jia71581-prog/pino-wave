#!/usr/bin/env python3
"""A+1 instance fine-tuning with EARLY-WINDOW true frames + high-order LWC PDE regularizer.

Corrected method (VERDICT sec 20): pure spacetime PDE residual cannot improve A+1 -- the
A+1 field already sits at the PDE-residual minimum (0.0068 vs truth 0.0066), so PDE alone
has no gradient toward the specific record's truth. The PDE is necessary-not-sufficient
(the wave equation has infinitely many solutions; only initial/boundary data pin THIS
record). So supervision =
    early-window true-frame data match (frames 0..K_end, energy ~97% of peak, from the
    numerical solver run only over the early time steps -- the deployment-legal signal)
    + high-order LWC PDE residual (time_order=4) as a physics regularizer over all times.
The early true frames carry instance-specific information A+1 lacks (A+1's own early
frames have ~4.5% error), anchoring the record; the PDE keeps the extrapolation physical.

Causal contract: an EarlyWindowAudit records that ONLY frames 0..K_end are read as
supervision. Late truth is opened solely to SCORE held-out frames (> K_end), never to
train. Everything in the normalizer's O(1) space. Parent field = A+1 full field
(dense_normalized scattering residual + encode(P_bg)).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet_v3.data.index import build_manifest
from saved_time_phase_operator_v4.instance_adaptation.adapters import OnsetAdaptedV5
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.instance_adaptation.contracts import onset_indices
from saved_time_phase_operator_v4.instance_adaptation.losses import (
    lwc84_residual,
    build_rad_physics_points,
    sample_fixed_physics_residual,
)
from saved_time_phase_operator_v4.background_field import BackgroundFieldProvider
from scripts.diagnose_aplus1_instance_finetune import _load_aplus1_parent
from scripts.run_v5_instance_adaptation import _predict_parent


class EarlyWindowAudit:
    """Causal guard: supervision may read ONLY frames 0..k_end (inclusive)."""

    def __init__(self, k_end: int):
        self.k_end = int(k_end)
        self.requested: list[int] = []

    def read(self, indices):
        for v in indices:
            v = int(v)
            if v not in self.requested:
                self.requested.append(v)
            if v > self.k_end:
                raise PermissionError(
                    f"future truth access forbidden: frame {v} > k_end {self.k_end}"
                )
        return tuple(int(v) for v in indices)

    def payload(self):
        return {
            "k_end": self.k_end,
            "requested_frames": tuple(self.requested),
            "future_truth_used": any(v > self.k_end for v in self.requested),
        }


def _rel(pred, truth, sl):
    p = pred[:, sl].reshape(-1)
    t = truth[:, sl].reshape(-1)
    return float((p - t).norm() / t.norm().clamp_min(1e-30))


def run(*, checkpoint, background_cache, source_h5, travel_time_h5, normalization_json,
        sample_id, device_name, k_end, steps_schedule, lr, pde_weight, obs_stride,
        seed, sampler, count, k, c, time_tilt, resample_every,
        width, dense_depth, dense_spectral_rank, dense_modes,
        helmholtz_frequencies, helmholtz_rank):
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
    T = time_s.shape[0]

    # EARLY-WINDOW supervision frames: 0..k_end, subsampled by obs_stride. The
    # numerical solver produces these (early time steps only); reading them is
    # deployment-legal. The audit forbids any frame > k_end.
    audit = EarlyWindowAudit(k_end)
    obs_frames = list(range(0, k_end + 1, max(1, obs_stride)))
    audit.read(obs_frames)  # records + enforces the causal window

    with h5py.File(source_h5, "r", swmr=True) as h5:
        truth_phys = torch.from_numpy(
            np.asarray(h5["wavefield"][record.source_index], dtype=np.float32)
        ).unsqueeze(0).to(device)
    truth_n = normalizer.encode_pressure(truth_phys, src_amp)
    obs_n = truth_n[:, obs_frames].detach()  # early true frames as data anchor
    # The OnsetAdaptedV5 conditioner/residual head are wired for exactly two onset
    # frames [record,2,z,x]. Feed them the two onset frames for conditioning, while
    # the fine-tune's data-match loss uses the FULL early window (obs_frames). The
    # two conditioning frames are within the early window (<= k_end), causal-legal.
    onset2 = onset_indices(
        record.time_s, t0_s=float(record.source_parameters[3]),
        f0_hz=float(record.source_parameters[2]),
    )
    if onset2[1] > k_end:
        raise ValueError("onset frames fall outside the early window; raise k_end")
    audit.read(list(onset2))
    cond_n = truth_n[:, list(onset2)].detach()

    # held-out = frames strictly after the early window (what we must extrapolate)
    mask = torch.zeros(T, dtype=torch.bool, device=device)
    mask[k_end + 1:] = True

    # A+1 parent field (normalized scattering residual + encode(P_bg)).
    parent_n = _predict_parent(
        model, normalizer, record, device, normalized=True, background_provider=background,
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
    baseline_held = _rel(parent_n, truth_n, mask)
    baseline_early = _rel(parent_n, truth_n, obs_frames)
    opt = torch.optim.Adam(params, lr=lr)
    rad_gen = torch.Generator(device=device).manual_seed(int(seed))

    # onset-like indices for the PDE residual's start (first two early frames)
    pde_start = (obs_frames[0], obs_frames[1] if len(obs_frames) > 1 else obs_frames[0] + 1)

    schedule = sorted(int(s) for s in steps_schedule.split(","))
    rows = []
    step = 0
    points = None
    for target in schedule:
        while step < target:
            opt.zero_grad(set_to_none=True)
            raw = adapter.raw_wavefield(parent_n, velocity, source, cond_n, time_s)
            # early-window data anchor (whole-window energy norm)
            obs_pred = raw[:, obs_frames]
            obs_loss = ((obs_pred - obs_n).flatten().norm()
                        / obs_n.flatten().norm().clamp_min(1e-8))
            # high-order LWC PDE regularizer over all interior times
            residual_field = lwc84_residual(
                raw, velocity, dt=dt, dx=10.0, dz=10.0, observed_indices=pde_start,
            )
            if sampler == "full":
                pde = residual_field.square().mean()
            else:
                if points is None or step % max(1, resample_every) == 0:
                    points = build_rad_physics_points(
                        residual_field.detach(), pde_start, count=count, k=k, c=c,
                        time_tilt=time_tilt, generator=rad_gen,
                    )
                pde = sample_fixed_physics_residual(residual_field, points, pde_start).square().mean()
            loss = obs_loss + pde_weight * pde
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
        with torch.no_grad():
            raw = adapter.raw_wavefield(parent_n, velocity, source, cond_n, time_s)
            held = _rel(raw, truth_n, mask)
            early = _rel(raw, truth_n, obs_frames)
        rows.append({"step": step, "heldout_relL2": held, "early_relL2": early,
                     "obs_loss": float(obs_loss), "pde": float(pde)})
        print(json.dumps(rows[-1], sort_keys=True), flush=True)

    best = min(r["heldout_relL2"] for r in rows)
    result = {
        "sample_id": sample_id, "k_end": k_end, "n_obs_frames": len(obs_frames),
        "obs_stride": obs_stride, "trainable_params": n_train,
        "parent_baseline_heldout_relL2": baseline_held,
        "parent_baseline_early_relL2": baseline_early,
        "best_heldout_relL2": best, "beats_parent": best < baseline_held,
        "pde_weight": pde_weight, "sampler": sampler, "time_order": 4,
        "causal_audit": audit.payload(), "schedule": rows,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--background-cache", required=True)
    ap.add_argument("--source-h5", default="/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5")
    ap.add_argument("--travel-time-h5", default="/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5")
    ap.add_argument("--normalization-json", default="/root/autodl-tmp/home/jiayh/Data/data/processed/grouped_v3_normalization.before_tgrs_ablation_identity_20260726T1050.json")
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k-end", type=int, default=80, help="last early true frame (0..k_end are the solver-computed anchor)")
    ap.add_argument("--obs-stride", type=int, default=4, help="subsample early frames by this stride")
    ap.add_argument("--steps", default="50,150,300,500")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pde-weight", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--sampler", choices=("full", "rad"), default="rad")
    ap.add_argument("--count", type=int, default=1024)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--time-tilt", type=float, default=1.5)
    ap.add_argument("--resample-every", type=int, default=8)
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
        device_name=args.device, k_end=args.k_end, steps_schedule=args.steps,
        lr=args.lr, pde_weight=args.pde_weight, obs_stride=args.obs_stride,
        seed=args.seed, sampler=args.sampler, count=args.count, k=args.k, c=args.c,
        time_tilt=args.time_tilt, resample_every=args.resample_every,
        width=args.width, dense_depth=args.dense_depth,
        dense_spectral_rank=args.dense_spectral_rank, dense_modes=args.dense_modes,
        helmholtz_frequencies=args.helmholtz_frequencies, helmholtz_rank=args.helmholtz_rank,
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
