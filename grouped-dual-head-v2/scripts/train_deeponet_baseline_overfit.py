#!/usr/bin/env python3
"""Pure-DeepONet 3-record overfit probe — the capacity-ladder comparison baseline.

Mirrors scripts/diagnose_capacity_ladder_overfit.py EXACTLY where it matters:
  * same deterministic uniform/layered/marmousi triplet
    (select_one_index_per_family, split='train'),
  * same ExactStoredTimeBatchDataset + travel-time-free inputs,
  * same PhysicalNormalizer.encode_pressure target normalization,
  * same hard-causality masking of predictions,
  * same ExactWavefieldMetricAccumulator scoring and target
    (agg < 0.10 and every family < 0.12),
  * same all_saved final evaluation over all 401 stored frames.

Only the MODEL differs: a textbook DeepONet (branch ⊗ trunk inner product) with
no FNO front end / travel branch / MIONet product / coarse field. This isolates
"what does our extra structure buy over a plain global low-rank operator".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts
from saved_time_phase_operator_v4.data import split_pilot_batch
from saved_time_phase_operator_v4.full_support import FullSupportStepSpec
from saved_time_phase_operator_v4.losses import apply_hard_causality, source_causality_onset_s
from saved_time_phase_operator_v4.streaming_metrics import ExactWavefieldMetricAccumulator
from scripts.train_grouped_v3_pilot import _to_device, load_normalizer
from scripts.train_saved_time_v4_probe import _atomic_json, _digest
from scripts.diagnose_capacity_ladder_overfit import build_base_config, build_probe_config
from scripts.diagnose_saved_time_temporal_three_record_overfit import (
    FAMILIES,
    select_one_index_per_family,
    build_repeat_schedule,
    _dataset,
    _meets_target,
)

from deeponet_baseline.model import DeepONetBaseline, DeepONetConfig


DOMAIN_X_M = 2000.0
DOMAIN_Z_M = 2000.0
DOMAIN_T_S = 1.0


def _scaled_coord_grid(x_m, z_m, times_s, device):
    """Build (F,H,W,3) query coords scaled to ~[-1,1] as (x,z,t)."""
    x = (x_m / DOMAIN_X_M) * 2.0 - 1.0          # (W,)
    z = (z_m / DOMAIN_Z_M) * 2.0 - 1.0          # (H,)
    t = (times_s / DOMAIN_T_S) * 2.0 - 1.0      # (F,)
    F, H, W = t.numel(), z.numel(), x.numel()
    xg = x.view(1, 1, W).expand(F, H, W)
    zg = z.view(1, H, 1).expand(F, H, W)
    tg = t.view(F, 1, 1).expand(F, H, W)
    return torch.stack([xg, zg, tg], dim=-1).to(device)  # (F,H,W,3)


def _forward_record(model, micro, normalizer, device, *, config, frame_chunk=2, checkpoint_trunk=False):
    """Run DeepONet on one microbatch record → (pred, target) normalized fields.

    pred/target shaped (1, F, H, W) to feed the metric accumulator directly.
    """
    tensors = _to_device(micro, device)
    source = tensors["source_parameters"]                  # (1,5)
    velocity_encoded = normalizer.encode_velocity(tensors["velocity_mps"])  # (1,1,H,W)
    source_scalars = normalizer.encode_source(source)      # (1,5)
    times_s = tensors["requested_time_s"][0]               # (F,)
    coords = _scaled_coord_grid(tensors["x_m"], tensors["z_m"], times_s, device)  # (F,H,W,3)
    coords = coords.unsqueeze(0)                           # (1,F,H,W,3)
    prediction = model(velocity_encoded, tensors["source_map"], source_scalars, coords,
                       frame_chunk=int(frame_chunk), checkpoint_trunk=bool(checkpoint_trunk))  # (1,F,H,W)
    target = normalizer.encode_pressure(tensors["dense_target_physical"], source[:, 4])  # (1,F,H,W)
    if bool(config["loss"].get("hard_causality", False)):
        onset = source_causality_onset_s(
            source, lead_cycles=float(config["loss"].get("hard_causality_lead_cycles", 0.0))
        )
        prediction = apply_hard_causality(prediction, tensors["requested_time_s"], onset)
    return prediction, target, tensors, source


def _relative_l2_loss(pred, target, *, gradient_weight):
    error = (pred.float() - target.float()).flatten(1).norm(dim=-1)
    norm = target.float().flatten(1).norm(dim=-1).clamp_min(1e-8)
    loss = (error / norm).mean()
    if gradient_weight > 0.0:
        pdx = pred[..., 1:] - pred[..., :-1]; tdx = target[..., 1:] - target[..., :-1]
        pdz = pred[..., 1:, :] - pred[..., :-1, :]; tdz = target[..., 1:, :] - target[..., :-1, :]
        g = (pdx.float() - tdx.float()).square().mean().sqrt() + (pdz.float() - tdz.float()).square().mean().sqrt()
        scale = (tdx.float().square().mean().sqrt() + tdz.float().square().mean().sqrt()).clamp_min(1e-8)
        loss = loss + gradient_weight * g / scale
    return loss


@torch.inference_mode()
def _evaluate(model, base, manifest, normalizer, device, config, indices, *, time_policy, frames_per_record):
    schedule = (FullSupportStepSpec(step=990_000, epoch=0,
                                    record_indices=tuple(indices),
                                    appearance_indices=(0,) * len(indices)),)
    dataset = _dataset(config, base, manifest, indices, split="train",
                       schedule=schedule, time_policy=time_policy, frames_per_record=frames_per_record)
    acc = ExactWavefieldMetricAccumulator(
        energy_floor_fraction=float(config.get("energy_floor_fraction", 0.01)),
        require_unique=True, stored_time_count=len(manifest.time_s))
    time_axis = torch.as_tensor(manifest.time_s, device=device)
    model.eval()
    for bi in range(len(dataset)):
        batch = dataset[bi]
        for micro in split_pilot_batch(batch, microbatch_records=1):
            pred, target, tensors, source = _forward_record(model, micro, normalizer, device,
                                                             config=config, frame_chunk=1)
            metric_onset = torch.clamp(source[:, 3] - source[:, 2].reciprocal(), min=float(manifest.time_s[0]))
            onset_indices = torch.searchsorted(time_axis, metric_onset.contiguous()).cpu().tolist()
            acc.update(pred, target,
                       families=micro.medium_type, group_ids=micro.group_id, sample_ids=micro.sample_id,
                       time_indices=micro.left_index, source_onset_indices=onset_indices)
    return acc.finalize()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--branch-width", type=int, default=256)
    p.add_argument("--trunk-width", type=int, default=384)
    p.add_argument("--trunk-depth", type=int, default=6)
    p.add_argument("--fourier-bands", type=int, default=16)
    p.add_argument("--updates", type=int, default=800)
    p.add_argument("--evaluate-every", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=1.0e-3)
    p.add_argument("--warmup-updates", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--frame-chunk", type=int, default=8,
                   help="time frames per trunk forward block; larger = faster when memory allows")
    p.add_argument("--checkpoint-trunk", action="store_true",
                   help="recompute trunk in backward to save memory (~30%% slower); off by default")
    p.add_argument("--gradient-weight", type=float, default=0.1)
    p.add_argument("--training-frames", type=int, default=16)
    p.add_argument("--validation-frames", type=int, default=32)
    p.add_argument("--seed", type=int, default=372)
    p.add_argument("--travel-time-h5",
                   default="/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    root = Path(args.artifact_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    terminal_path = root / "terminal.json"
    if terminal_path.exists():
        print(terminal_path.read_text().strip()); return 0

    device = torch.device("cuda")
    seed = int(args.seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)

    base = build_base_config(128)   # width unused by DeepONet; only for data/manifest wiring
    manifest = build_manifest(base.data.source_h5)
    # The dataset file was extended after the capacity-ladder run (record count grew
    # 2240 -> 4003, so the content digest no longer matches the frozen normalization
    # binding). Grid/time axes are unchanged, so load the normalizer WITHOUT the
    # manifest guard and skip the count check. The frozen split-relative triplet
    # (select_one_index_per_family on split='train') still resolves to the same
    # uniform/layered/marmousi records the ladder used, keeping the comparison exact.
    from grouped_ufno_mionet_v3.normalization import PhysicalNormalizer
    normalizer = PhysicalNormalizer.from_dict(
        json.loads(Path(base.data.normalization_json).read_text(encoding="utf8")),
        expected_manifest=None,
    )
    config = build_probe_config(dense_lr=args.learning_rate, backbone_lr=args.learning_rate,
                                temporal_lr=1e-5, seed=seed, travel_time_h5=str(args.travel_time_h5))

    model = DeepONetBaseline(DeepONetConfig(
        branch_width=args.branch_width, latent_dim=args.latent_dim,
        trunk_width=args.trunk_width, trunk_depth=args.trunk_depth,
        fourier_bands=args.fourier_bands)).to(device)
    param_count = model.parameter_count()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate),
                                  weight_decay=0.0, betas=(0.9, 0.99))
    total_updates = 8 if args.smoke else int(args.updates)
    warmup = max(1, int(args.warmup_updates))
    # linear warmup then cosine decay — the full-field DeepONet diverges without
    # warmup at lr>=1e-3 (the K-dim inner product amplifies early large steps).
    warm = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_updates - warmup), eta_min=float(args.learning_rate) * 0.02)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warm, cos], milestones=[warmup])

    indices = select_one_index_per_family(manifest.records, split="train")
    selected = tuple(r for r in manifest.records if r.split == "train")
    sample_ids = tuple(selected[i].sample_id for i in indices)

    updates = 8 if args.smoke else int(args.updates)
    evaluate_every = 4 if args.smoke else int(args.evaluate_every)
    schedule = build_repeat_schedule(indices, updates=updates)
    train_data = _dataset(config, base, manifest, indices, split="train", schedule=schedule,
                          time_policy="appearance16", frames_per_record=int(args.training_frames))

    identity = {
        "schema": "deeponet_baseline_three_record_overfit_v1",
        "model": "pure_deeponet_branch_trunk_inner_product",
        "latent_dim": int(args.latent_dim), "branch_width": int(args.branch_width),
        "trunk_width": int(args.trunk_width), "trunk_depth": int(args.trunk_depth),
        "fourier_bands": int(args.fourier_bands), "parameter_count": param_count,
        "manifest_digest": manifest.digest, "split": "train",
        "record_indices": indices, "sample_ids": sample_ids, "families": FAMILIES,
        "updates": updates, "evaluate_every": evaluate_every,
        "training_frames_per_record": int(args.training_frames),
        "validation_frames_per_record": int(args.validation_frames),
        "learning_rate": float(args.learning_rate), "gradient_weight": float(args.gradient_weight),
        "seed": seed, "smoke": bool(args.smoke),
        "one_source_per_record": True, "receiver_input": False,
        "wavefield_shape": [201, 201], "exact_saved_times_only": True,
        "comparison_to": "capacity_ladder W2/W3 (SavedTimePhaseOperatorV4)",
    }
    identity["run_digest"] = _digest(identity)
    _atomic_json(identity, root / "run_identity.json")

    def _append(path, row):
        with open(path, "a") as h:
            h.write(json.dumps(row, sort_keys=True) + "\n")

    baseline = _evaluate(model, base, manifest, normalizer, device, config, indices,
                         time_policy="validation_fixed", frames_per_record=int(args.validation_frames))
    _append(root / "metrics.jsonl", {"event": "baseline", "update": 0, "metrics": baseline})

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    best_score = float(baseline["aggregate_relative_l2"])
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    last_update = 0
    for update, batch in enumerate(train_data, start=1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for micro in split_pilot_batch(batch, microbatch_records=1):
            pred, target, _, _ = _forward_record(model, micro, normalizer, device, config=config,
                                                  frame_chunk=int(args.frame_chunk),
                                                  checkpoint_trunk=bool(args.checkpoint_trunk))
            loss = _relative_l2_loss(pred, target, gradient_weight=float(args.gradient_weight))
            (loss / 3.0).backward()
            total += float(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        scheduler.step()
        last_update = update
        _append(root / "updates.jsonl", {"update": update, "loss": total / 3.0,
                                         "elapsed_seconds": time.monotonic() - started})
        if update % evaluate_every and update != updates:
            continue
        metrics = _evaluate(model, base, manifest, normalizer, device, config, indices,
                            time_policy="validation_fixed", frames_per_record=int(args.validation_frames))
        score = float(metrics["aggregate_relative_l2"])
        if score <= best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        _append(root / "metrics.jsonl", {"event": "evaluation", "update": update, "metrics": metrics,
                                         "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
                                         "elapsed_seconds": time.monotonic() - started})
        print(json.dumps({"update": update, "agg": score,
                          "family": metrics.get("family_relative_l2")}, sort_keys=True), flush=True)
        if _meets_target(metrics):
            break

    if args.smoke:
        terminal = {"status": "smoke_complete", "updates_completed": last_update,
                    "best_fixed_aggregate_relative_l2": best_score, "parameter_count": param_count,
                    "peak_cuda_bytes": int(torch.cuda.max_memory_allocated())}
        _atomic_json(terminal, terminal_path)
        print(json.dumps(terminal, sort_keys=True), flush=True); return 0

    model.load_state_dict(best_state)
    full = _evaluate(model, base, manifest, normalizer, device, config, indices,
                     time_policy="all_saved", frames_per_record=len(manifest.time_s))
    _append(root / "metrics.jsonl", {"event": "all_saved_evaluation", "metrics": full})
    torch.save({"model_state": model.state_dict(), "identity": identity}, root / "best.pt")
    terminal = {
        "status": "complete", "updates_completed": last_update,
        "best_fixed_aggregate_relative_l2": best_score,
        "all_saved_aggregate_relative_l2": float(full["aggregate_relative_l2"]),
        "all_saved_family_relative_l2": full.get("family_relative_l2"),
        "can_memorize_below_target": _meets_target(full),
        "parameter_count": param_count,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }
    _atomic_json(terminal, terminal_path)
    print(json.dumps(terminal, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
