#!/usr/bin/env python
"""
Hybrid Numerical-Neural Instance Adaptation.

Combines pretrained Factorized FNO with traditional finite-difference stencils
for fast, accurate test-time adaptation. Balances speed vs accuracy by:
  - Using O(n) FD stencils (not O(n²) autograd) for PDE constraints
  - Training only lightweight adapter parameters (not full model)
  - Limiting to 10-30 optimization steps

Approach:
  1. Pretrained FNO predicts p_nn
  2. Compute PDE residual with 2nd/4th-order FD stencils (fast, O(n))
  3. Minimize: observed_MSE + λ_pde * PDE_residual² + λ_smooth * smoothness
  4. Update only fc1/fc2 head (or small adapter) for speed

Usage:
  python scripts/run_hybrid_numerical_adaptation.py \
    --config configs/factorized_fno_128x128x160_continue.yaml \
    --checkpoint artifacts/factorized_fno/128x128x160_continue/checkpoints/best.pt \
    --sample 5 --steps 20 --lr 1e-4 --pde-weight 1.0 \
    --output-dir artifacts/hybrid_adaptation/sample_005
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
from fno_acoustic.data import PinoHDF5Dataset
from fno_acoustic.finite_difference import second_time_derivative, laplacian_2d_inner
from fno_acoustic.losses_receiver import receiver_waveform_loss
from fno_acoustic.metrics import per_time_relative_l2
from fno_acoustic.normalization import decode_standard
from fno_acoustic.train import build_training_model
from fno_acoustic.utils import ensure_dir, write_json
from fno_acoustic.visualization import plot_sample


def fast_pde_residual(p: torch.Tensor, v: torch.Tensor, src: torch.Tensor,
                      dx: float, dz: float, dt: float) -> torch.Tensor:
    """Compute PDE residual using fast FD stencils (2nd order, O(n)).
    All inputs are [B, H, W, T] in physical units."""
    # Time derivative: [B, H, W, T-2]
    p_tt = second_time_derivative(p, dt)
    # Spatial Laplacian on inner domain: [B, H-2, W-2, T]
    lap = laplacian_2d_inner(p, dx=dx, dz=dz)
    # Align: take inner spatial AND inner temporal
    p_tt_inner = p_tt[:, 1:-1, 1:-1, :]              # [B, H-2, W-2, T-2]
    lap_inner = lap[:, :, :, 1:-1]                    # [B, H-2, W-2, T-2]
    v2 = v[:, 1:-1, 1:-1].unsqueeze(-1).pow(2)       # [B, H-2, W-2, 1]
    src_inner = src[:, 1:-1, 1:-1, 1:-1]             # [B, H-2, W-2, T-2]
    return p_tt_inner - v2 * lap_inner - src_inner


def rel_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    d = (pred - target).reshape(1, -1).float()
    r = target.reshape(1, -1).float()
    return float((d.norm() / r.norm().clamp_min(eps)).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pde-weight", type=float, default=1.0)
    parser.add_argument("--smooth-weight", type=float, default=0.01)
    parser.add_argument("--fd-order", type=int, default=2, choices=[2, 4])
    parser.add_argument("--train-mode", type=str, default="head",
                        choices=["head", "last_layer", "adapter"])
    parser.add_argument("--observed-frames", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = ensure_dir(args.output_dir)
    config = load_config(args.config)

    # ── Load model ──────────────────────────────────────────────
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    stats = ckpt["normalization_stats"]
    config["model"] = ckpt["model_config"]
    model, _ = build_training_model(config)
    model = model.to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    eps_norm = float(config["normalization"].get("eps", 1e-6))

    # ── Select trainable parameters ─────────────────────────────
    if args.train_mode == "head":
        # Train only fc1 + fc2 (output head)
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.head.parameters():
            p.requires_grad_(True)
    elif args.train_mode == "last_layer":
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.head[-1].parameters():
            p.requires_grad_(True)
    elif args.train_mode == "adapter":
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.head.parameters():
            p.requires_grad_(True)
        for p in model.temporal_mixer.output_proj.parameters():
            p.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} / {sum(p.numel() for p in model.parameters()):,}")

    # ── Load sample ─────────────────────────────────────────────
    dataset = PinoHDF5Dataset(config, [args.sample], normalization_stats=stats, return_normalized=True)
    sample = dataset[0]
    x = sample["input"].unsqueeze(0).to(device)
    y = sample["target"].unsqueeze(0).to(device)
    v_raw = sample["velocity"]
    if v_raw.ndim == 4:
        v_raw = v_raw[..., 0] if v_raw.shape[-1] == 1 else v_raw.squeeze(1)
    elif v_raw.ndim == 3 and v_raw.shape[0] == 1:
        v_raw = v_raw[0]
    velocity = v_raw.unsqueeze(0).to(device)
    s_raw = sample["source_map"]
    if s_raw.ndim == 4:
        s_raw = s_raw[..., 0] if s_raw.shape[-1] == 1 else s_raw.squeeze(1)
    elif s_raw.ndim == 3 and s_raw.shape[0] == 1:
        s_raw = s_raw[0]
    source_map = s_raw.unsqueeze(0).to(device)
    time_coords = sample["time"].to(device)
    wavelet = sample.get("wavelet_saved_t", torch.zeros(x.shape[-1])).to(device)
    metadata = sample.get("metadata", {}) or {}
    dataset.close()

    dx = float(metadata.get("dx", 1.0) or 1.0)
    dz = float(metadata.get("dz", dx) or dx)
    dt_saved = float(torch.median(torch.diff(time_coords)).item())
    source_term = source_map.unsqueeze(-1) * wavelet.view(1, 1, 1, -1)
    t_steps = int(y.shape[-1])
    obs_frames = [int(f) for f in args.observed_frames]

    # ── Pretrained prediction ───────────────────────────────────
    model.eval()
    with torch.no_grad():
        pred_before = model(x)
        phys_before = decode_standard(pred_before, stats["wavefield"], eps=eps_norm)
        phys_target = decode_standard(y, stats["wavefield"], eps=eps_norm)

    before_rel = rel_l2(phys_before, phys_target)
    before_rec = float(receiver_waveform_loss(
        phys_before, phys_target,
        z_indices=list(range(10, min(390, phys_target.shape[2] - 10), 40)),
        x_stride=20,
    ).item())
    r = fast_pde_residual(phys_before, velocity, source_term, dx, dz, dt_saved)
    before_pde = float(r.pow(2).mean().item())

    # ── Hybrid fine-tuning ──────────────────────────────────────
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    obs_idx = torch.tensor(obs_frames, dtype=torch.long, device=device)
    future_frames = [t for t in range(t_steps) if t not in obs_frames]
    ft_log = []

    for step in range(args.steps):
        model.train()
        opt.zero_grad()
        pred = model(x)
        pred_phys = decode_standard(pred, stats["wavefield"], eps=eps_norm)

        # Observed frame MSE
        obs_loss = torch.nn.functional.mse_loss(
            pred.index_select(-1, obs_idx), y.index_select(-1, obs_idx)
        )
        # Fast FD-based PDE residual (no autograd through FD ops)
        residual = fast_pde_residual(pred_phys, velocity, source_term, dx, dz, dt_saved)
        pde_loss = residual.pow(2).mean()
        # Spatial smoothness (penalize high-frequency noise)
        smooth_loss = (pred_phys[:, :, 1:, :] - pred_phys[:, :, :-1, :]).pow(2).mean() + \
                      (pred_phys[:, 1:, :, :] - pred_phys[:, :-1, :, :]).pow(2).mean()

        loss = obs_loss + args.pde_weight * pde_loss + args.smooth_weight * smooth_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 5 == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                pf = decode_standard(model(x), stats["wavefield"], eps=eps_norm)
                ft_log.append({
                    "step": step,
                    "obs_mse": float(obs_loss.item()),
                    "pde_loss": float(pde_loss.item()),
                    "future_l2": rel_l2(pf[..., future_frames], phys_target[..., future_frames]),
                })
                print(f"  step {step:3d}: obs={obs_loss.item():.6f} pde={pde_loss.item():.4f}")

    # ── Metrics after ───────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        pred_after = model(x)
        phys_after = decode_standard(pred_after, stats["wavefield"], eps=eps_norm)

    after_rel = rel_l2(phys_after, phys_target)
    after_rec = float(receiver_waveform_loss(
        phys_after, phys_target,
        z_indices=list(range(10, min(390, phys_target.shape[2] - 10), 40)),
        x_stride=20,
    ).item())
    r = fast_pde_residual(phys_after, velocity, source_term, dx, dz, dt_saved)
    after_pde = float(r.pow(2).mean().item())

    # ── Generate figures ────────────────────────────────────────
    sample_for_plot = {k: (v[0] if isinstance(v, torch.Tensor) else v)
                       for k, v in sample.items()}
    for key in ["velocity", "target", "source_map"]:
        if key in sample:
            sample_for_plot[key] = sample[key]
    sample_for_plot["time"] = time_coords
    sample_for_plot["sample_index"] = torch.tensor([args.sample])
    sample_for_plot["input"] = sample["input"]

    before_dir = output_dir / "before"
    after_dir = output_dir / "after"
    plot_sample(sample_for_plot, pred_before.squeeze(0), before_dir, normalization_stats=stats)
    plot_sample(sample_for_plot, pred_after.squeeze(0), after_dir, normalization_stats=stats)

    # Per-time error
    pt_before = per_time_relative_l2(phys_before, phys_target)[0].cpu().numpy()
    pt_after = per_time_relative_l2(phys_after, phys_target)[0].cpu().numpy()
    t = time_coords.cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, pt_before, "b-", label="Before", alpha=0.7)
    ax.plot(t, pt_after, "r-", label=f"After (FD-{args.fd_order}, head-only)", alpha=0.7)
    for f in obs_frames:
        ax.axvline(t[f], color="gray", ls="--", alpha=0.3)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Relative L2")
    ax.set_title(f"Hybrid FD-NN Adaptation — Sample {args.sample}")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(output_dir / "per_time_error.png", dpi=150)
    plt.close(fig)

    # Receiver waveform
    rx, rz = phys_target.shape[1] // 2, min(390, phys_target.shape[2] - 10)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(t, phys_target[0, rx, rz, :].cpu(), "k-", label="Target", lw=1.5)
    axes[0].plot(t, phys_before[0, rx, rz, :].cpu(), "b--", label="Before", alpha=0.7)
    axes[0].set_title("Pretrained"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, phys_target[0, rx, rz, :].cpu(), "k-", label="Target", lw=1.5)
    axes[1].plot(t, phys_after[0, rx, rz, :].cpu(), "r--", label="After hybrid adapt", alpha=0.7)
    axes[1].set_title("After Hybrid FD-NN Adaptation"); axes[1].set_xlabel("Time (s)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(output_dir / "receiver_waveform.png", dpi=150)
    plt.close(fig)

    # ── Save results ────────────────────────────────────────────
    result = {
        "sample": args.sample, "steps": args.steps, "lr": args.lr,
        "pde_weight": args.pde_weight, "fd_order": args.fd_order,
        "train_mode": args.train_mode, "trainable_params": trainable,
        "before": {"rel_l2": before_rel, "receiver_l2": before_rec, "pde_mse": before_pde},
        "after": {"rel_l2": after_rel, "receiver_l2": after_rec, "pde_mse": after_pde},
        "improvement": {
            "rel_l2": before_rel - after_rel,
            "receiver_l2": before_rec - after_rec,
            "pde_mse": before_pde - after_pde,
        },
        "accepted": after_pde <= before_pde * 1.05,
        "log": ft_log,
        "artifacts": {
            "before": str(before_dir), "after": str(after_dir),
            "per_time": str(output_dir / "per_time_error.png"),
            "receiver": str(output_dir / "receiver_waveform.png"),
        },
    }
    write_json(output_dir / "adaptation_result.json", result)
    print(f"\nBefore: rel={before_rel:.4f} rec={before_rec:.4f} pde={before_pde:.2f}")
    print(f"After:  rel={after_rel:.4f} rec={after_rec:.4f} pde={after_pde:.2f}")
    print(f"Δ:      rel={before_rel-after_rel:+.4f} rec={before_rec-after_rec:+.4f} pde={before_pde-after_pde:+.1f}")


if __name__ == "__main__":
    main()
