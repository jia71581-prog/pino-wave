from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class HierarchicalLoss:
    total: torch.Tensor
    query: torch.Tensor
    dense: torch.Tensor
    consistency: torch.Tensor
    spectral: torch.Tensor
    family_contributions: torch.Tensor


def hierarchical_reduce(loss: torch.Tensor, *, group_ids=None, family_ids=None) -> HierarchicalLoss:
    """Reduce per-query losses as query -> record -> group -> family.

    ``loss`` is [records, ...]; all records contribute equally within each
    medium family, independent of the number of sources in their group.
    """
    if loss.ndim < 1:
        raise ValueError("loss must have a record dimension")
    per_record = loss.reshape(loss.shape[0], -1).mean(-1)
    if group_ids is None:
        group_ids = list(range(per_record.numel()))
    if family_ids is None:
        family_ids = ["default"] * per_record.numel()
    if len(group_ids) != per_record.numel() or len(family_ids) != per_record.numel():
        raise ValueError("group_ids/family_ids must match record count")
    group_means = []
    group_families = []
    for group in dict.fromkeys(group_ids):
        idx = [i for i, value in enumerate(group_ids) if value == group]
        group_means.append(per_record[idx].mean())
        group_families.append(family_ids[idx[0]])
    families = list(dict.fromkeys(group_families))
    family_contrib = torch.stack([torch.stack([v for v, f in zip(group_means, group_families) if f == fam]).mean() for fam in families])
    total = family_contrib.mean()
    return HierarchicalLoss(total, total, total * 0, total * 0, total * 0, family_contrib)


def _spectral_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[-1] < 2:
        return pred.new_zeros(())
    return (torch.fft.rfft(pred, dim=-1) - torch.fft.rfft(target, dim=-1)).abs().mean()


def grouped_operator_loss(query_pred, query_target, *, dense_pred=None, dense_target=None,
                          consistency_pred=None, spectral_weight=0.05, dense_weight=0.5,
                          consistency_weight=0.1, group_ids=None, family_ids=None):
    query_err = (query_pred - query_target).square()
    q = hierarchical_reduce(query_err, group_ids=group_ids, family_ids=family_ids).total
    d = query_pred.new_zeros(()) if dense_pred is None or dense_target is None else (dense_pred - dense_target).square().mean()
    c = query_pred.new_zeros(()) if consistency_pred is None else (consistency_pred - query_pred.detach()).square().mean()
    s = _spectral_error(query_pred, query_target)
    return q + dense_weight * d + consistency_weight * c + spectral_weight * s, {
        "query": q.detach(), "dense": d.detach(), "consistency": c.detach(), "spectral": s.detach()
    }
