"""B2 harness -- differentiable sequence objective + protocol-exact eval metric.

This is the reusable, GPU-free-testable *trainable core* of any B2 propagator
trainer, deliberately decoupled from the data-loading stack (which is GPU/data
bound and cannot be CPU-verified).  A trainer supplies materialized tensors:

    base_seq : [B, K, 1, Z, X]  frozen warp render on the saved-time grid (the anchor)
    cond     : [B, C, Z, X]     static medium/source/geometry conditioning bundle
    target   : [B, K, 1, Z, X]  ground-truth pressure sequence on the same grid

and this module produces the anchored rollout, a differentiable training loss,
and the SAME-PROTOCOL aggregate metric.

HONESTY CONTRACT (pinned by tests):
  * The eval metric `aggregate_relative_l2` reproduces
    coarse_lwc84.complete_field_relative_l2 EXACTLY (per-record joint space-time
    relative L2 in float64, mean over records).  B2's reported aggregate is thus
    on the identical fixed protocol as warp_r1 / opt16 -- no metric change, no
    eval-difficulty change, no normalization-caliber change.
  * At gate==0 the anchored rollout returns base_seq bit-exactly, so B2's
    aggregate at init == the frozen warp baseline aggregate.  B2 can therefore
    only *improve on* a real, correctly-measured baseline -- never fabricate one.

Two training losses (the R8+22 tension made explicit, not hidden):
  * `joint` (default) == a differentiable form of the reported aggregate; it is
    the honest objective because it optimizes exactly what is scored.  It is
    energy-weighted across time (low-energy late frames get proportionally small
    gradient) -- this is WHY the additive ladder leaves the late bin at ~0.33.
  * `per_frame_uniform` gives every frame's own relative L2 equal weight, so the
    late/pre_onset frames receive real gradient.  Legitimate as a TRAINING choice
    (it does not touch the fixed eval metric), but its effect on the reported
    joint aggregate is an empirical question -- report both, never conflate them.
"""
from __future__ import annotations

import torch

from .propagator import CausalSemigroupPropagator


def _check_seq(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.ndim != 5 or pred.shape[2] != 1:
        raise ValueError("sequence tensors must be [B, K, 1, Z, X]")
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")


@torch.no_grad()
def aggregate_relative_l2(pred: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-16) -> float:
    """Protocol-exact aggregate: mean over records of the per-record joint
    space-time relative L2, computed in float64 (matches
    coarse_lwc84.complete_field_relative_l2 aggregated by merge_record_rows.mean).
    """
    _check_seq(pred, target)
    p = pred.reshape(pred.shape[0], -1).double()
    t = target.reshape(target.shape[0], -1).double()
    num = (p - t).pow(2).sum(dim=1).sqrt()
    den = t.pow(2).sum(dim=1).clamp_min(eps).sqrt()
    return float((num / den).mean().item())


def per_record_relative_l2(pred: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-16) -> torch.Tensor:
    """Differentiable per-record joint relative L2, [B] (native dtype for training)."""
    _check_seq(pred, target)
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    num = (p - t).pow(2).sum(dim=1).clamp_min(0.0).sqrt()
    den = t.pow(2).sum(dim=1).clamp_min(eps).sqrt()
    return num / den


def per_frame_relative_l2(pred: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-16) -> torch.Tensor:
    """Differentiable per-(record,frame) relative L2, [B, K]."""
    _check_seq(pred, target)
    p = pred.reshape(pred.shape[0], pred.shape[1], -1)
    t = target.reshape(target.shape[0], target.shape[1], -1)
    num = (p - t).pow(2).sum(dim=2).clamp_min(0.0).sqrt()
    den = t.pow(2).sum(dim=2).clamp_min(eps).sqrt()
    return num / den


def sequence_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    weighting: str = "joint",
    eps: float = 1e-16,
) -> torch.Tensor:
    """Differentiable training loss.

    weighting='joint'             -> mean over records of joint relative L2 (== eval objective).
    weighting='per_frame_uniform' -> mean over records & frames of per-frame relative L2.
    """
    if weighting == "joint":
        return per_record_relative_l2(pred, target, eps=eps).mean()
    if weighting == "per_frame_uniform":
        return per_frame_relative_l2(pred, target, eps=eps).mean()
    raise ValueError(f"unknown weighting {weighting!r}")


class AnchoredSequenceObjective(torch.nn.Module):
    """Wraps a CausalSemigroupPropagator with the anchored rollout + loss/metric.

    A trainer calls `.step(base_seq, cond, target)` and gets a dict with the
    differentiable loss (for .backward()) and the detached same-protocol
    aggregate (for logging / gating).  No data-loading assumptions leak in here.
    """

    def __init__(self, propagator: CausalSemigroupPropagator, *, weighting: str = "joint") -> None:
        super().__init__()
        if weighting not in ("joint", "per_frame_uniform"):
            raise ValueError(f"unknown weighting {weighting!r}")
        self.propagator = propagator
        self.weighting = weighting

    def rollout(
        self,
        base_seq: torch.Tensor,
        cond: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.propagator.forward_anchored(base_seq, cond, initial_state=initial_state)

    def step(
        self,
        base_seq: torch.Tensor,
        cond: torch.Tensor,
        target: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> dict[str, object]:
        pred = self.rollout(base_seq, cond, initial_state=initial_state)
        loss = sequence_loss(pred, target, weighting=self.weighting)
        return {
            "loss": loss,
            "aggregate_relative_l2": aggregate_relative_l2(pred, target),
            "base_aggregate_relative_l2": aggregate_relative_l2(base_seq, target),
            "prediction": pred,
        }
