#!/usr/bin/env python3
"""Smoke: B2 anchored rollout with REAL-snapshot IC, 2-frame vs 8-frame arms.

Pipeline being de-risked (first real-data driver for CausalSemigroupPropagator):
  medium+source -> warp_r1 render (solve-free) = base_seq anchor
  truth wavefield first-N frames                = initial_state (deployment input)
  CausalSemigroupPropagator.forward_anchored    = corrected sequence
Loss/metric only on frames >= N_VISIBLE (the IC window is a known input, never
scored).  Both arms see the SAME 8 visible frames; they differ ONLY in how many
enter the IC encoder (2 = last two visible, 8 = all visible), so any arm gap is
attributable to IC capacity alone.

Smoke scope: tiny subset (per-family records from validation), few hundred
optimizer steps, single GPU.  Metrics are pipeline-health signals only, NOT a
judged comparison; the judged run needs the frozen 90-record dev manifest.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from saved_time_phase_operator_v4.propagator import CausalSemigroupPropagator
from saved_time_phase_operator_v4.propagator_harness import aggregate_relative_l2

WARP_CFG = ROOT / "configs/saved_time_v4/generated/local_field_w128_warp_r1.yaml"
WARP_RUN = Path(
    "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/"
    "pretraining/local_field_w128_hicap/warp_r1/run"
)
SOURCE_H5 = (
    "/data/jiayh/data/acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2/dataset_v1.h5"
)
PARENT_IDENTITY = (
    ROOT / "configs/saved_time_v4/generated/"
    "w2_parent_identity_legacy_norm_marmousi1_4m_v2.json"
)
FAMILIES = ("uniform", "layered", "marmousi")
N_VISIBLE = 8          # deployment contract: first 8 true snapshots are given
K_FRAMES = 64          # rollout window length (frames, contiguous saved times)
ONSET_RMS_FRACTION = 0.05  # window starts at the first frame with this energy


def _atomic_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def _static_cond(velocity, x_m, z_m, source_x, source_z):
    """7-channel deployment-legal conditioning (r25 static_features recipe)."""
    value = np.asarray(velocity, dtype=np.float32)
    logv = np.log(np.maximum(value, 1.0))
    grad_z, grad_x = np.gradient(logv)
    grad_x = np.clip(20.0 * grad_x, -4.0, 4.0)
    grad_z = np.clip(20.0 * grad_z, -4.0, 4.0)
    xx, zz = np.meshgrid(x_m.astype(np.float32), z_m.astype(np.float32))
    distance = np.sqrt((xx - float(source_x)) ** 2 + (zz - float(source_z)) ** 2)
    source_map = np.exp(-0.5 * (distance / 20.0) ** 2)
    source_map /= max(float(source_map.max()), 1.0e-12)
    travel = distance / np.maximum(value, 1.0)
    x_norm = 2.0 * xx / max(float(x_m[-1]), 1.0) - 1.0
    z_norm = 2.0 * zz / max(float(z_m[-1]), 1.0) - 1.0
    return np.stack(
        [
            np.clip((value - 4500.0) / 2500.0, -2.0, 2.0),
            grad_x,
            grad_z,
            source_map.astype(np.float32),
            np.clip(travel, 0.0, 1.5).astype(np.float32),
            x_norm.astype(np.float32),
            z_norm.astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def _select_records(per_family: int, seed: int):
    with h5py.File(SOURCE_H5, "r", swmr=True) as f:
        split = f["split"][:].astype(str)
        family = f["medium_type"][:].astype(str)
        sample_ids = f["sample_id"][:].astype(str)
    rng = np.random.default_rng(seed)
    rows = []
    for name in FAMILIES:
        candidates = np.flatnonzero((split == "validation") & (family == name))
        candidates = candidates[np.argsort(sample_ids[candidates])]  # deterministic
        chosen = rng.choice(candidates, size=per_family, replace=False)
        rows.extend(
            {"index": int(i), "family": name, "sample_id": str(sample_ids[i])}
            for i in sorted(int(v) for v in chosen)
        )
    return rows


def _onset_start(wavefield: np.ndarray) -> int:
    """First frame whose RMS reaches ONSET_RMS_FRACTION of the max frame RMS."""
    rms = np.sqrt((wavefield.astype(np.float64) ** 2).mean(axis=(1, 2)))
    threshold = ONSET_RMS_FRACTION * float(rms.max())
    hits = np.flatnonzero(rms >= threshold)
    return int(hits[0]) if hits.size else 0


def _render_base_seq(rows, times_by_row, device):
    """Render warp_r1 base_seq per record (solve-free deployment path)."""
    import importlib.util

    saved_argv = sys.argv
    sys.argv = ["smoke_b2"]
    spec = importlib.util.spec_from_file_location(
        "train_v4", str(ROOT / "scripts/train_saved_time_v4_full_support.py")
    )
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    sys.argv = saved_argv
    from scripts.train_grouped_v3_pilot import _to_device, load_normalizer
    from saved_time_phase_operator_v4.data import ExactStoredTimeBatchDataset
    from grouped_ufno_mionet_v3.data.pilot import PilotStepSpec

    config = yaml.safe_load(WARP_CFG.read_text())
    # Load the TRAINED warp_r1 weights (the config's own parent is opt16) and
    # point the identity at the surviving marmousi1_4m_v2 copy (same manifest
    # digest a20c9a65 as the warp_r1 run; the original _v1 sibling was removed
    # in the marmousi cleanup).
    config["parent_checkpoint"] = str(WARP_RUN / "best.pt")
    config["parent_checkpoint_identity"] = str(WARP_RUN / "run_identity.json")
    config["parent_identity"] = str(PARENT_IDENTITY)
    # warp_r1 predates the marmousi cleanup; explicit weight transfer across the
    # dataset repair is the sanctioned path (parent_manifest_transfer_metadata).
    transfer = dict(config.get("checkpoint_transfer", {}))
    transfer["allow_parent_manifest_mismatch"] = True
    config["checkpoint_transfer"] = transfer
    base, manifest, parent_identity = train._load_context(config)
    model = train._load_parent_model(config, base, manifest, parent_identity, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    normalizer = load_normalizer(base, manifest.digest)

    # map source h5 row index -> position within the validation split VIEW
    # (V3WavefieldDataset filters manifest.records; indices are view-local)
    dataset = ExactStoredTimeBatchDataset(
        base.data.source_h5, manifest, split="validation",
        schedule=[PilotStepSpec(step=0, record_indices=(0,))],  # placeholder
        query_points=1, seed=0,
    )
    index_of = {
        rec.source_index: pos for pos, rec in enumerate(dataset.records.records)
    }
    base_frames, target_frames = [], []
    with torch.no_grad():
        for row, times in zip(rows, times_by_row):
            position = index_of[row["index"]]
            spec_one = PilotStepSpec(step=0, record_indices=(position,))
            batch = dataset.materialize_spec(spec_one)
            tensors = _to_device(batch, device)
            source = tensors["source_parameters"]
            prepared = model.prepare_sources(
                model.encode_medium(tensors["velocity_mps"], normalizer),
                source, tensors["source_map"], normalizer,
                record_to_medium=tensors["record_to_medium"],
            )
            # render in frame chunks; 64 frames at once OOMs the 24G card
            chunks = []
            for lo in range(0, len(times), 8):
                requested = torch.as_tensor(
                    times[lo:lo + 8], dtype=torch.float32, device=device
                )[None]
                chunks.append(model.dense_normalized(
                    prepared, requested,
                    x_m=tensors["x_m"], z_m=tensors["z_m"], time_block=1,
                ).float().cpu())
            prediction = torch.cat(chunks, dim=1)
            target = dataset.records.read_wavefield(position, times).values
            target = normalizer.encode_pressure(
                target[None].to(device), source[:, 4]
            )
            base_frames.append(prediction[0])
            target_frames.append(target[0].float().cpu())
    del model
    torch.cuda.empty_cache()
    return torch.stack(base_frames), torch.stack(target_frames)


def _run_arm(name, ic_frames, base_seq, cond, target, *, steps, lr, seed, device, log,
             micro_records=3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CausalSemigroupPropagator(
        state_channels=ic_frames, cond_channels=cond.shape[1], width=32,
        spectral_rank=16, modes=12, depth=3, gate_init=1.0,
        activation_checkpointing=True,
    ).to(device)
    # Gate-collapse fix (observed: scalar gate driven to ~0 within 25 steps,
    # matching the documented B2-H failure mode).  Replace the collapsible
    # scalar with the repo's proven zero-init output-conv warm-start
    # (CoarseResidualUNet pattern): decoder's last conv starts at zero so the
    # rollout is still an exact no-op at init, but "opening" is learned
    # per-channel/per-pixel; the global gate is pinned at 1 and frozen.
    torch.nn.init.zeros_(model.decoder[-1].weight)
    torch.nn.init.zeros_(model.decoder[-1].bias)
    model.gate.requires_grad_(False)
    # IC = the LAST ic_frames of the visible window (both arms see the same 8).
    # Tensors stay on CPU; microbatches move to GPU per step (24G card).
    initial_state_all = target[:, N_VISIBLE - ic_frames:N_VISIBLE, 0]
    eval_slice = slice(N_VISIBLE, None)
    n_records = base_seq.shape[0]
    micros = [
        slice(lo, min(lo + micro_records, n_records))
        for lo in range(0, n_records, micro_records)
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)

    def _rollout_scores():
        """Eval-window aggregate over all records, microbatched, no grad."""
        num_sq, den_sq = [], []
        with torch.no_grad():
            for sl in micros:
                pred = model.forward_anchored(
                    base_seq[sl].to(device), cond[sl].to(device),
                    initial_state=initial_state_all[sl].to(device),
                )[:, eval_slice]
                t = target[sl, eval_slice].to(device)
                p = pred.reshape(pred.shape[0], -1).double()
                tt = t.reshape(t.shape[0], -1).double()
                num_sq.append((p - tt).pow(2).sum(1).sqrt().cpu())
                den_sq.append(tt.pow(2).sum(1).clamp_min(1e-16).sqrt().cpu())
        return float((torch.cat(num_sq) / torch.cat(den_sq)).mean())

    with torch.no_grad():
        b = base_seq[:, eval_slice].reshape(n_records, -1).double()
        t = target[:, eval_slice].reshape(n_records, -1).double()
        base_score = float(
            ((b - t).pow(2).sum(1).sqrt() / t.pow(2).sum(1).clamp_min(1e-16).sqrt()).mean()
        )
    history = []
    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for sl in micros:
            pred = model.forward_anchored(
                base_seq[sl].to(device), cond[sl].to(device),
                initial_state=initial_state_all[sl].to(device),
            )
            p = pred[:, eval_slice].reshape(pred.shape[0], -1)
            t = target[sl, eval_slice].to(device).reshape(sl.stop - sl.start, -1)
            loss = ((p - t).pow(2).sum(1).clamp_min(0).sqrt()
                    / t.pow(2).sum(1).clamp_min(1e-16).sqrt()).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name}: non-finite loss at step {step}")
            (loss * (sl.stop - sl.start) / n_records).backward()
            loss_total += float(loss.detach()) * (sl.stop - sl.start) / n_records
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 25 == 0 or step == steps - 1:
            model.eval()
            score = _rollout_scores()
            row = {"arm": name, "step": step, "loss": loss_total,
                   "eval_window_agg": score, "base_agg": base_score,
                   "gate": float(model.gate), "step_scale": float(model.step_scale)}
            history.append(row)
            log(row)
    del model
    torch.cuda.empty_cache()
    return {"final": history[-1], "history": history,
            "parameters": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-family", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=372)
    parser.add_argument("--arms", default="ic2,ic8",
                        help="comma list from {ic2,ic8}; single arm enables per-GPU parallelism")
    parser.add_argument("--record-seed", type=int, default=372,
                        help="record-selection seed, kept FIXED across parallel arms/seeds "
                             "so every process trains on identical records")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/smoke_b2_anchored_ic2_vs_ic8.json")
    args = parser.parse_args()
    device = torch.device("cuda")
    started = time.time()

    rows = _select_records(args.per_family, args.record_seed)
    with h5py.File(SOURCE_H5, "r", swmr=True) as f:
        time_axis = np.asarray(f["time_s"][:], dtype=np.float64)
        x_m = np.asarray(f["x_m"][:]); z_m = np.asarray(f["z_m"][:])
        conds, times_by_row = [], []
        for row in rows:
            i = row["index"]
            wavefield = np.asarray(f["wavefield"][i], dtype=np.float32)
            start = _onset_start(wavefield)
            start = min(start, len(time_axis) - K_FRAMES)
            row["window_start"] = start
            times_by_row.append(time_axis[start:start + K_FRAMES])
            conds.append(_static_cond(
                np.asarray(f["velocity_mps"][i]), x_m, z_m,
                float(f["source_x_m"][i]), float(f["source_z_m"][i]),
            ))
    cond = torch.from_numpy(np.stack(conds))

    print(json.dumps({"event": "render_base_seq", "records": len(rows)}), flush=True)
    base_seq, target = _render_base_seq(rows, times_by_row, device)
    base_seq = base_seq[:, :, None]   # [B,K,1,Z,X]
    target = target[:, :, None]

    def log(row):
        print(json.dumps(row, sort_keys=True), flush=True)

    results = {}
    arm_specs = {"ic2": 2, "ic8": 8}
    requested = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in requested if a not in arm_specs]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {sorted(arm_specs)}")
    for name in requested:
        ic = arm_specs[name]
        print(json.dumps({"event": "arm_start", "arm": name, "seed": args.seed}), flush=True)
        results[name] = _run_arm(
            name, ic, base_seq, cond, target,
            steps=args.steps, lr=args.lr, seed=args.seed, device=device, log=log,
        )

    payload = {
        "schema": "smoke_b2_anchored_ic2_vs_ic8_v1",
        "scope": "pipeline_smoke_only_not_judged",
        "n_visible": N_VISIBLE, "k_frames": K_FRAMES,
        "onset_rms_fraction": ONSET_RMS_FRACTION,
        "records": rows, "steps": args.steps, "lr": args.lr, "seed": args.seed,
        "record_seed": args.record_seed, "arms": requested,
        "source_h5": SOURCE_H5, "warp_checkpoint": str(WARP_RUN / "best.pt"),
        "results": results, "elapsed_s": time.time() - started,
        "validation_opened": True, "test_id_opened": False,
    }
    _atomic_json(payload, args.output)
    summary = {"event": "done", "output": str(args.output),
               "base": next(iter(results.values()))["final"]["base_agg"]}
    for name in requested:
        summary[name] = results[name]["final"]["eval_window_agg"]
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
