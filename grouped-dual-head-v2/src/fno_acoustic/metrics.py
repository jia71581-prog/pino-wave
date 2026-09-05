from __future__ import annotations

import torch
import torch.nn.functional as F


def basic_metrics(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> dict[str, float]:
    pred = pred.detach()
    target = target.detach()
    rel = torch.linalg.norm((pred - target).reshape(pred.shape[0], -1), dim=1) / torch.clamp(
        torch.linalg.norm(target.reshape(target.shape[0], -1), dim=1), min=eps
    )
    return {
        "relative_l2": float(rel.mean().cpu()),
        "mse": float(F.mse_loss(pred, target).cpu()),
        "mae": float(F.l1_loss(pred, target).cpu()),
        "prediction_finite_ratio": float(torch.isfinite(pred).float().mean().cpu()),
        "target_norm": float(torch.linalg.norm(target.reshape(target.shape[0], -1), dim=1).mean().cpu()),
        "prediction_norm": float(torch.linalg.norm(pred.reshape(pred.shape[0], -1), dim=1).mean().cpu()),
    }


def per_time_relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # pred/target are [B, H, W, T].
    diff = pred - target
    num = torch.linalg.norm(diff.permute(0, 3, 1, 2).reshape(pred.shape[0], pred.shape[-1], -1), dim=2)
    den = torch.linalg.norm(target.permute(0, 3, 1, 2).reshape(target.shape[0], target.shape[-1], -1), dim=2)
    return num / torch.clamp(den, min=eps)
