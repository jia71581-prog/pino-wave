#!/usr/bin/env python
"""
Instance Adaptation with PDE Constraints.

Loads a pretrained Factorized FNO, fine-tunes on 2 observed frames + PDE residual,
generates wavefield snapshots and receiver waveform figures.

Usage:
  python scripts/run_instance_adaptation_pde.py \
    --config configs/factorized_fno_128x128x160_continue.yaml \
    --checkpoint artifacts/factorized_fno/128x128x160_continue/checkpoints/best.pt \
    --sample 5 --steps 50 --lr 1e-5 \
    --output-dir artifacts/instance_adaptation/sample_005
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fno_acoustic.checkpoint import load_checkpoint
from fno_acoustic.config import load_config
from fno_acoustic.data import PinoHDF5Dataset, collate_pino
from fno_acoustic.losses_physics import pde_residual_loss
from fno_acoustic.losses_receiver import receiver_waveform_loss
from fno_acoustic.metrics import basic_metrics, per_time_relative_l2
from fno_acoustic.normalization import decode_standard
from fno_acoustic.train import build_training_model
from fno_acoustic.utils import choose_device, ensure_dir, write_json
from fno_acoustic.visualization import plot_sample, denormalize_wavefield


def _rel_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    d = (pred - target).reshape(1, -1).float()
    r = target.reshape(1, -1).float()
    return float((d.norm() / r.norm().clamp_min(eps)).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--pde-weight", type=float, default=1e-3)
    parser.add_argument("--observed-frames", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = ensure_dir(args.output_dir)
    config = load_config(args.config)

    # ── Load pretrained model ──────────────────────────────────
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    stats = ckpt["normalization_stats"]
    config["model"] = ckpt["model_config"]
    model, _ = build_training_model(config)
    model = model.to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    eps = float(config["normalization"].get("eps", 1e-6))

    # ── Load sample ────────────────────────────────────────────
    dataset = PinoHDF5Dataset(config, [args.sample], normalization_stats=stats, return_normalized=True)
    sample = dataset[0]
    x = sample["input"].unsqueeze(0).to(device)
    y = sample["target"].unsqueeze(0).to(device)
    velocity = sample["velocity"]
    if velocity.ndim == 4:
        velocity = velocity[..., 0] if velocity.shape[-1] == 1 else velocity.squeeze(1)
    elif velocity.ndim == 3 and velocity.shape[0] == 1:
        velocity = velocity[0]
    velocity = velocity.unsqueeze(0).to(device)  # [1, H, W]
    source_map = sample["source_map"]
    if source_map.ndim == 4:
        source_map = source_map[..., 0] if source_map.shape[-1] == 1 else source_map.squeeze(1)
    elif source_map.ndim == 3 and source_map.shape[0] == 1:
        source_map = source_map[0]
    source_map = source_map.unsqueeze(0).to(device)  # [1, H, W]
    time_coords = sample["time"].to(device)
    metadata = sample.get("metadata", {}) or {}
    wavelet = sample.get("wavelet_saved_t", torch.zeros(x.shape[-1])).to(device)
    dataset.close()

    dx = float(metadata.get("dx", 1.0) or 1.0)
    dz = float(metadata.get("dz", dx) or dx)
    dt_saved = float(torch.median(torch.diff(time_coords)).item())
    source_term = source_map.unsqueeze(-1) * wavelet.view(1, 1, 1, -1)
    t_steps = int(y.shape[-1])
    obs_frames = [int(f) for f in args.observed_frames]

    # ── Pretrained prediction ──────────────────────────────────
    model.eval()
    with torch.no_grad():
        pred_before = model(x)
        phys_before = decode_standard(pred_before, stats["wavefield"], eps=eps)
        phys_target = decode_standard(y, stats["wavefield"], eps=eps)

    before_rel = _rel_l2(phys_before, phys_target)
    before_rec = float(receiver_waveform_loss(
        phys_before, phys_target,
        z_indices=list(range(10, min(390, phys_target.shape[2]-10), 40)),
        x_stride=20,
    ).item())
    before_pde = float(pde_residual_loss(
        phys_before, velocity, source_term, dx=dx, dz=dz, dt_saved=dt_saved,
        source_mask=source_map,
    ).item())

    # ── Fine-tuning with PDE constraint ────────────────────────
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    future_frames = [t for t in range(t_steps) if t not in obs_frames]
    ft_log = []

    for step in range(args.steps):
        opt.zero_grad()
        pred = model(x)
        pred_phys = decode_standard(pred, stats["wavefield"], eps=eps)

        obs_loss = torch.nn.functional.mse_loss(
            pred.index_select(-1, torch.tensor(obs_frames, device=device)),
            y.index_select(-1, torch.tensor(obs_frames, device=device)),
        )
        pde_loss = pde_residual_loss(
            pred_phys, velocity, source_term, dx=dx, dz=dz, dt_saved=dt_saved,
            source_mask=source_map,
        )
        loss = obs_loss + args.pde_weight * pde_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 10 == 0 or step == args.steps - 1:
            with torch.no_grad():
                pf = decode_standard(model(x), stats["wavefield"], eps=eps)
                ft_log.append({
                    "step": step,
                    "obs_mse": float(obs_loss.item()),
                    "pde_loss": float(pde_loss.item()),
                    "future_l2": _rel_l2(pf[..., future_frames], phys_target[..., future_frames]),
                })

    # ── Metrics after ─────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        pred_after = model(x)
        phys_after = decode_standard(pred_after, stats["wavefield"], eps=eps)

    after_rel = _rel_l2(phys_after, phys_target)
    after_rec = float(receiver_waveform_loss(
        phys_after, phys_target,
        z_indices=list(range(10, min(390, phys_target.shape[2]-10), 40)),
        x_stride=20,
    ).item())
    after_pde = float(pde_residual_loss(
        phys_after, velocity, source_term, dx=dx, dz=dz, dt_saved=dt_saved,
        source_mask=source_map,
    ).item())

    # ── Per-time L2 ───────────────────────────────────────────
    pt_before = per_time_relative_l2(phys_before, phys_target)[0].cpu().numpy()
    pt_after = per_time_relative_l2(phys_after, phys_target)[0].cpu().numpy()
    time_vals = time_coords.cpu().numpy()

    # ── Generate figures ──────────────────────────────────────
    # Overview + snapshots using existing plot_sample
    sample_for_plot = {k: (v[0] if isinstance(v, torch.Tensor) else v) for k, v in sample.items()}
    sample_for_plot["velocity"] = sample["velocity"]
    sample_for_plot["target"] = sample["target"]
    sample_for_plot["source_map"] = sample["source_map"]
    sample_for_plot["time"] = time_coords
    sample_for_plot["sample_index"] = torch.tensor([args.sample])
    sample_for_plot["input"] = sample["input"]

    before_dir = output_dir / "before"
    after_dir = output_dir / "after"
    plot_sample(sample_for_plot, pred_before.squeeze(0), before_dir, normalization_stats=stats)
    plot_sample(sample_for_plot, pred_after.squeeze(0), after_dir, normalization_stats=stats)

    # Per-time error comparison
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_vals, pt_before, "b-", label="Before", alpha=0.7)
    ax.plot(time_vals, pt_after, "r-", label="After (PDE-adapted)", alpha=0.7)
    for f in obs_frames:
        ax.axvline(time_vals[f], color="gray", ls="--", alpha=0.3)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Relative L2")
    ax.set_title(f"Per-Time Error — Sample {args.sample}")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(output_dir / "per_time_error.png", dpi=150)
    plt.close(fig)

    # Receiver waveform overlay
    rec_x = phys_target.shape[1] // 2
    rec_z = min(390, phys_target.shape[2] - 10)
    t = time_vals
    tgt = phys_target[0, rec_x, rec_z, :].cpu().numpy()
    bef = phys_before[0, rec_x, rec_z, :].cpu().numpy()
    aft = phys_after[0, rec_x, rec_z, :].cpu().numpy()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(t, tgt, "k-", label="Target", lw=1.5)
    axes[0].plot(t, bef, "b--", label="Before", alpha=0.7)
    axes[0].set_title("Pretrained prediction")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, tgt, "k-", label="Target", lw=1.5)
    axes[1].plot(t, aft, "r--", label="After PDE adaptation", alpha=0.7)
    axes[1].set_title("After PDE-constrained adaptation")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(output_dir / "receiver_waveform.png", dpi=150)
    plt.close(fig)

    # ── Save results ──────────────────────────────────────────
    result = {
        "sample": args.sample,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "observed_frames": obs_frames,
        "future_frames": future_frames,
        "pde_weight": args.pde_weight,
        "steps": args.steps,
        "lr": args.lr,
        "before": {"relative_l2": before_rel, "receiver_l2": before_rec, "pde_residual": before_pde},
        "after": {"relative_l2": after_rel, "receiver_l2": after_rec, "pde_residual": after_pde},
        "improvement": {
            "relative_l2": before_rel - after_rel,
            "receiver_l2": before_rec - after_rec,
            "pde_residual": before_pde - after_pde,
        },
        "accepted": float(after_pde) <= float(before_pde) * 1.05,
        "fine_tuning_log": ft_log,
        "artifacts": {
            "before_snapshots": str(before_dir),
            "after_snapshots": str(after_dir),
            "per_time_error": str(output_dir / "per_time_error.png"),
            "receiver_waveform": str(output_dir / "receiver_waveform.png"),
        },
    }
    write_json(output_dir / "adaptation_result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "fine_tuning_log"}, indent=2))


if __name__ == "__main__":
    main()
