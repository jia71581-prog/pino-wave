from __future__ import annotations
import torch


def mionet_contract(medium_rank: torch.Tensor, source_rank: torch.Tensor, trunk_rank: torch.Tensor) -> torch.Tensor:
    """Rank-wise multiple-input DeepONet contraction."""
    if medium_rank.ndim != 2 or source_rank.ndim != 2 or trunk_rank.ndim != 3:
        raise ValueError("MIONet ranks must have shapes [N,R], [N,R], [N,Q,R]")
    if medium_rank.shape != source_rank.shape or medium_rank.shape[0] != trunk_rank.shape[0] or medium_rank.shape[1] != trunk_rank.shape[2]:
        raise ValueError("MIONet rank dimensions do not agree")
    return (medium_rank[:, None, :] * source_rank[:, None, :] * trunk_rank).sum(-1)
