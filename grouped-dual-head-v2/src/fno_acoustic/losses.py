from __future__ import annotations

import torch
import torch.nn.functional as F


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    diff = (pred - target).reshape(pred.shape[0], -1)
    ref = target.reshape(target.shape[0], -1)
    numerator = torch.linalg.norm(diff, dim=1)
    denominator = torch.clamp(torch.linalg.norm(ref, dim=1), min=eps)
    return (numerator / denominator).mean()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    relative_l2_weight: float = 1.0,
    mse_weight: float = 0.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    rel = relative_l2(pred, target, eps=eps)
    mse = F.mse_loss(pred, target)
    loss = relative_l2_weight * rel + mse_weight * mse
    return loss, {"relative_l2": float(rel.detach().cpu()), "mse": float(mse.detach().cpu())}
