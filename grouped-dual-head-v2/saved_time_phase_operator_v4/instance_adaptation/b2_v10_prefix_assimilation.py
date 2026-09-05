"""Causal-prefix, closed-form POD assimilation for B2-v10.

The online solver receives only a true wavefield prefix.  It estimates a small
coefficient vector from the part of that prefix which was not already copied
into the parent as its eight-frame initial condition, then applies the inferred
correction only to the still-unobserved suffix.  Future truth cannot be passed
through this API because ``observed_true`` must end at the declared prefix.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

import torch


def source_cycle_prefix_count(
    window_time_s: torch.Tensor,
    *,
    source_t0_s: float,
    source_f0_hz: float,
    after_peak_cycles: float,
    minimum_frames: int = 8,
    maximum_frames: int | None = None,
) -> int:
    """Return a causal prefix length normalized by source period.

    The prefix includes saved frames through ``t0 + cycles / f0``.  At least
    one frame is always left unobserved for honest forecast evaluation.
    """
    axis = torch.as_tensor(window_time_s, dtype=torch.float64).flatten()
    if axis.ndim != 1 or axis.numel() < 2:
        raise ValueError("window_time_s must contain at least two frames")
    if not torch.isfinite(axis).all() or bool((axis[1:] <= axis[:-1]).any()):
        raise ValueError("window_time_s must be finite and strictly increasing")
    frequency = float(source_f0_hz)
    onset = float(source_t0_s)
    cycles = float(after_peak_cycles)
    if not torch.isfinite(torch.tensor((frequency, onset, cycles))).all():
        raise ValueError("source timing values must be finite")
    if frequency <= 0.0 or cycles < 0.0:
        raise ValueError("source_f0_hz must be positive and cycles nonnegative")
    minimum = int(minimum_frames)
    if not 1 <= minimum < int(axis.numel()):
        raise ValueError("minimum_frames must leave at least one future frame")
    maximum = int(axis.numel() - 1 if maximum_frames is None else maximum_frames)
    if not minimum <= maximum < int(axis.numel()):
        raise ValueError("maximum_frames must be valid and leave future frames")
    stop_time = onset + cycles / frequency
    count = int(
        torch.searchsorted(
            axis,
            torch.tensor(stop_time, dtype=axis.dtype),
            right=True,
        )
    )
    return max(minimum, min(maximum, count))


class CausalPrefixAccessAudit:
    """Record and enforce truth reads within one declared causal prefix."""

    def __init__(self, *, total_frames: int, observed_count: int) -> None:
        total = int(total_frames)
        observed = int(observed_count)
        if not 1 <= observed < total:
            raise ValueError("observed_count must be a proper prefix")
        self.total_frames = total
        self.observed_count = observed
        self.requested_indices: list[int] = []

    def read(self, indices: Iterable[int]) -> tuple[int, ...]:
        requested = tuple(int(value) for value in indices)
        if any(value < 0 or value >= self.observed_count for value in requested):
            raise PermissionError("truth beyond the causal prefix is forbidden")
        for value in requested:
            if value not in self.requested_indices:
                self.requested_indices.append(value)
        return requested

    def payload(self) -> dict[str, object]:
        return {
            "total_frames": self.total_frames,
            "observed_count": self.observed_count,
            "requested_indices": tuple(self.requested_indices),
            "future_truth_used": any(
                value >= self.observed_count for value in self.requested_indices
            ),
        }


@dataclass(frozen=True)
class PrefixAssimilationConfig:
    """Fixed online policy; train-only calibration may replace these values."""

    anchor_frames: int = 8
    ridge_fraction: float = 1.0e-3
    trust_ratio: float = 0.05
    minimum_observed_gain: float = 0.0
    minimum_information_ratio: float = 0.0
    minimum_observed_frames: int | None = None

    def validate(self, *, total_frames: int, observed_count: int) -> None:
        if not 1 <= int(self.anchor_frames) < int(total_frames):
            raise ValueError("anchor_frames must lie inside the parent sequence")
        if not int(self.anchor_frames) <= int(observed_count) < int(total_frames):
            raise ValueError("observed prefix must include anchors and leave a suffix")
        if self.ridge_fraction < 0.0 or not torch.isfinite(
            torch.tensor(self.ridge_fraction)
        ):
            raise ValueError("ridge_fraction must be finite and nonnegative")
        if self.trust_ratio < 0.0 or not torch.isfinite(torch.tensor(self.trust_ratio)):
            raise ValueError("trust_ratio must be finite and nonnegative")
        if not 0.0 <= self.minimum_observed_gain <= 1.0:
            raise ValueError("minimum_observed_gain must lie in [0,1]")
        if not 0.0 <= self.minimum_information_ratio <= 1.0:
            raise ValueError("minimum_information_ratio must lie in [0,1]")
        if self.minimum_observed_frames is not None and not (
            int(self.anchor_frames)
            <= int(self.minimum_observed_frames)
            < int(total_frames)
        ):
            raise ValueError(
                "minimum_observed_frames must include anchors and leave a suffix"
            )


def _validate_inputs(
    parent: torch.Tensor,
    modes: torch.Tensor,
    observed_true: torch.Tensor,
    prior_coefficients: torch.Tensor | None,
) -> tuple[int, int, torch.Tensor]:
    if parent.ndim != 5 or parent.shape[0] != 1:
        raise ValueError("parent must be [1,T,C,Z,X]")
    if modes.ndim != 5 or modes.shape[1:] != parent.shape[1:]:
        raise ValueError("modes must be [R,T,C,Z,X] and match the parent")
    if observed_true.ndim != 5 or observed_true.shape[0] != 1:
        raise ValueError("observed_true must be [1,K,C,Z,X]")
    observed_count = int(observed_true.shape[1])
    if observed_true.shape[2:] != parent.shape[2:]:
        raise ValueError("observed prefix spatial/channel shape mismatch")
    rank = int(modes.shape[0])
    if rank <= 0:
        raise ValueError("at least one POD mode is required")
    if prior_coefficients is None:
        prior = torch.zeros(rank, device=parent.device, dtype=parent.dtype)
    else:
        prior = torch.as_tensor(prior_coefficients, device=parent.device).reshape(-1)
        if prior.numel() != rank:
            raise ValueError("prior coefficient rank mismatch")
        prior = prior.to(dtype=parent.dtype)
    tensors = (parent, modes, observed_true, prior)
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("assimilation inputs must be finite")
    return rank, observed_count, prior


def assimilate_prefix_pod(
    parent: torch.Tensor,
    modes: torch.Tensor,
    observed_true: torch.Tensor,
    *,
    prior_coefficients: torch.Tensor | None = None,
    config: PrefixAssimilationConfig = PrefixAssimilationConfig(),
    access_audit: CausalPrefixAccessAudit | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Infer POD coefficients from a causal prefix and correct only its suffix."""
    started = time.perf_counter()
    parent = parent.detach()
    modes = modes.to(parent).detach()
    observed_true = observed_true.to(parent).detach()
    rank, observed_count, prior = _validate_inputs(
        parent, modes, observed_true, prior_coefficients
    )
    total_frames = int(parent.shape[1])
    config.validate(total_frames=total_frames, observed_count=observed_count)
    if access_audit is None:
        access_audit = CausalPrefixAccessAudit(
            total_frames=total_frames,
            observed_count=observed_count,
        )
    if (
        access_audit.total_frames != total_frames
        or access_audit.observed_count != observed_count
    ):
        raise ValueError("access audit does not match assimilation tensors")
    access_audit.read(range(observed_count))

    anchor = int(config.anchor_frames)
    minimum_observed_frames = (
        anchor
        if config.minimum_observed_frames is None
        else int(config.minimum_observed_frames)
    )
    candidate = parent.clone()
    if observed_count == anchor:
        return candidate, {
            "accepted": False,
            "reason": "no_informative_frames_after_parent_anchor",
            "rank": rank,
            "observed_count": observed_count,
            "minimum_observed_frames": minimum_observed_frames,
            "informative_frames": 0,
            "coefficients": prior.detach().cpu().tolist(),
            "observed_gain": 0.0,
            "information_ratio": 0.0,
            "condition_number": None,
            "correction_ratio": 0.0,
            "elapsed_s": time.perf_counter() - started,
            **access_audit.payload(),
        }
    if observed_count < minimum_observed_frames:
        return candidate, {
            "accepted": False,
            "reason": "insufficient_observed_frames",
            "rank": rank,
            "observed_count": observed_count,
            "minimum_observed_frames": minimum_observed_frames,
            "informative_frames": observed_count - anchor,
            "coefficients": torch.zeros_like(prior).detach().cpu().tolist(),
            "observed_gain": 0.0,
            "information_ratio": 0.0,
            "condition_number": None,
            "correction_ratio": 0.0,
            "elapsed_s": time.perf_counter() - started,
            **access_audit.payload(),
        }

    # A maps coefficients to the newly observed residual.  The copied anchor
    # frames are excluded because their parent residual is exactly zero by
    # construction and therefore carries no information about future error.
    design = modes[:, anchor:observed_count].reshape(rank, -1).double().T
    residual = (
        observed_true[:, anchor:observed_count]
        - parent[:, anchor:observed_count]
    ).reshape(-1).double()
    gram = design.T @ design
    rhs = design.T @ residual
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    largest = float(eigenvalues[-1])
    smallest = float(eigenvalues[0])
    information_ratio = 0.0 if largest <= 0.0 else smallest / largest
    condition_number = None if smallest <= 0.0 else largest / smallest
    gram_scale = float(torch.trace(gram) / rank)
    regularization = float(config.ridge_fraction) * max(gram_scale, 1.0e-16)
    system = gram + regularization * torch.eye(rank, dtype=gram.dtype, device=gram.device)
    rhs = rhs + regularization * prior.double()
    try:
        coefficients = torch.linalg.solve(system, rhs).to(parent)
    except torch.linalg.LinAlgError:
        coefficients = torch.linalg.lstsq(system, rhs[:, None]).solution[:, 0].to(parent)
    if not bool(torch.isfinite(coefficients).all()):
        coefficients = prior.clone()

    predicted_observed_residual = design @ coefficients.double()
    parent_observed_loss = float(residual.square().mean())
    fitted_observed_loss = float(
        (predicted_observed_residual - residual).square().mean()
    )
    if parent_observed_loss <= 1.0e-30:
        observed_gain = 0.0
    else:
        observed_gain = 1.0 - fitted_observed_loss / parent_observed_loss

    future_correction = torch.einsum(
        "r,rtczx->tczx", coefficients, modes[:, observed_count:]
    )[None]
    parent_future = parent[:, observed_count:]
    correction_ratio = float(
        future_correction.norm() / parent_future.norm().clamp_min(1.0e-16)
    )
    if correction_ratio > config.trust_ratio:
        scale = float(config.trust_ratio) / max(correction_ratio, 1.0e-16)
        coefficients = coefficients * scale
        future_correction = future_correction * scale
        correction_ratio = float(
            future_correction.norm() / parent_future.norm().clamp_min(1.0e-16)
        )

    accepted = bool(
        parent_observed_loss > 1.0e-30
        and torch.isfinite(future_correction).all()
        and observed_gain > config.minimum_observed_gain
        and information_ratio >= config.minimum_information_ratio
        and correction_ratio <= config.trust_ratio * 1.000001
    )
    reason = "accepted"
    if not accepted:
        coefficients = torch.zeros_like(coefficients)
        future_correction.zero_()
        correction_ratio = 0.0
        if parent_observed_loss <= 1.0e-30:
            reason = "zero_observed_parent_residual"
        elif observed_gain <= config.minimum_observed_gain:
            reason = "observed_fit_did_not_improve"
        elif information_ratio < config.minimum_information_ratio:
            reason = "prefix_design_not_identifiable"
        else:
            reason = "nonfinite_or_trust_failure"
    candidate[:, observed_count:] = parent_future + future_correction
    if not torch.equal(candidate[:, :observed_count], parent[:, :observed_count]):
        raise AssertionError("online assimilation modified an observed frame")
    return candidate.detach(), {
        "accepted": accepted,
        "reason": reason,
        "rank": rank,
        "observed_count": observed_count,
        "minimum_observed_frames": minimum_observed_frames,
        "informative_frames": observed_count - anchor,
        "coefficients": coefficients.detach().cpu().tolist(),
        "ridge_absolute": regularization,
        "parent_observed_loss": parent_observed_loss,
        "fitted_observed_loss": fitted_observed_loss,
        "observed_gain": observed_gain,
        "information_ratio": information_ratio,
        "condition_number": condition_number,
        "correction_ratio": correction_ratio,
        "elapsed_s": time.perf_counter() - started,
        **access_audit.payload(),
    }


__all__ = [
    "CausalPrefixAccessAudit",
    "PrefixAssimilationConfig",
    "assimilate_prefix_pod",
    "source_cycle_prefix_count",
]
