"""Parent-anchored block-autoregressive correction for saved-time wavefields.

This module deliberately does *not* replace the frequency-domain parent.  The
parent first renders a complete trajectory.  A small shared corrector then
advances in blocks, carrying only the last two corrected residual frames across
block boundaries:

    corrected[B_j] = parent[B_j] + bounded_delta(
        parent[B_j-2:B_j+K], corrected[B_j-2:B_j], public_conditioning
    )

The deployment path never accepts target/future-truth tensors.  Looking ahead
inside the current block is allowed because those frames are sealed parent
predictions, not observations.  Consequently this is causal at the 32-frame
block boundary rather than at every stored frame.

The output head is zero initialized.  At initialization the complete rollout is
bit-exact to the parent (provided the parent's hard free-surface row is zero),
while gradients immediately reach the output head.  A per-record correction
cap and parent re-anchoring prevent recurrent residuals from growing without
bound, the failure mode observed in the historical B2-H free rollout.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from .spectral import FactorizedComplexResidualStack


def render_retained_rfft_frames(
    coefficients: torch.Tensor,
    *,
    time_count: int,
    frame_indices: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Render selected real frames from leading orthonormal-rFFT coefficients.

    Args:
        coefficients: ``[B,F,2,Z,X]`` or ``[F,2,Z,X]`` real/imaginary pairs.
        time_count: length of the real time series represented by the rFFT.
        frame_indices: exact stored-time indices to render.

    Returns:
        ``[B,T,Z,X]`` (or ``[T,Z,X]`` for an unbatched input).

    Unlike materializing a full 401-frame inverse FFT, this evaluates only the
    requested block.  The expression is exactly the real inverse DFT for the
    retained bins, including the DC and even-length Nyquist weights.
    """

    value = torch.as_tensor(coefficients)
    squeeze = False
    if value.ndim == 4:
        value = value.unsqueeze(0)
        squeeze = True
    if value.ndim != 5 or value.shape[2] != 2:
        raise ValueError("coefficients must be [B,F,2,Z,X] or [F,2,Z,X]")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError("coefficients must be finite floating-point pairs")
    count = int(time_count)
    frequency_count = int(value.shape[1])
    if count <= 1 or frequency_count <= 0 or frequency_count > count // 2 + 1:
        raise ValueError("invalid time count or retained frequency count")
    indices = torch.as_tensor(frame_indices, device=value.device, dtype=torch.long)
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("frame_indices must be a non-empty one-dimensional sequence")
    if bool(((indices < 0) | (indices >= count)).any()):
        raise ValueError("frame index lies outside the represented time axis")

    work = value.float()
    frequencies = torch.arange(
        frequency_count, device=value.device, dtype=work.dtype
    )
    phase = (
        2.0
        * math.pi
        * indices.to(work.dtype)[:, None]
        * frequencies[None, :]
        / float(count)
    )
    weights = torch.full_like(frequencies, 2.0)
    weights[0] = 1.0
    if count % 2 == 0 and frequency_count == count // 2 + 1:
        weights[-1] = 1.0
    cosine = torch.cos(phase) * weights[None]
    sine = torch.sin(phase) * weights[None]
    real, imaginary = work[:, :, 0], work[:, :, 1]
    frames = (
        torch.einsum("tf,bfij->btij", cosine, real)
        - torch.einsum("tf,bfij->btij", sine, imaginary)
    ) / math.sqrt(float(count))
    return frames[0] if squeeze else frames


def _group_norm(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if int(channels) % groups == 0:
            return nn.GroupNorm(groups, int(channels))
    return nn.GroupNorm(1, int(channels))


class ParentAnchoredBlockCorrector(nn.Module):
    """Lightweight block-autoregressive residual corrector.

    ``forward_block`` consumes exactly two corrected history frames and a parent
    context containing those same two times plus ``block_size`` future parent
    frames.  No target is part of the model API.
    """

    history_frames = 2

    def __init__(
        self,
        *,
        condition_channels: int,
        block_size: int = 32,
        width: int = 32,
        spectral_rank: int = 16,
        modes: int = 16,
        depth: int = 4,
        maximum_correction_ratio: float = 0.5,
        boundary_blend_frames: int = 4,
        minimum_scale: float = 1.0e-4,
        activation_checkpointing: bool = True,
        hard_free_surface: bool = True,
    ) -> None:
        super().__init__()
        if min(condition_channels, block_size, width, spectral_rank, modes, depth) <= 0:
            raise ValueError("all corrector dimensions must be positive")
        if not 0.0 < float(maximum_correction_ratio) <= 1.0:
            raise ValueError("maximum_correction_ratio must lie in (0,1]")
        if not 0 <= int(boundary_blend_frames) <= int(block_size):
            raise ValueError("boundary_blend_frames must lie in [0, block_size]")
        if float(minimum_scale) <= 0.0:
            raise ValueError("minimum_scale must be positive")

        self.condition_channels = int(condition_channels)
        self.block_size = int(block_size)
        self.width = int(width)
        self.maximum_correction_ratio = float(maximum_correction_ratio)
        self.boundary_blend_frames = int(boundary_blend_frames)
        self.minimum_scale = float(minimum_scale)
        self.hard_free_surface = bool(hard_free_surface)

        # Parent history+future, corrected-minus-parent history, condition, and
        # two absolute block-time coordinates.
        input_channels = self.block_size + self.history_frames + self.history_frames
        input_channels += self.condition_channels + 2
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, self.width, kernel_size=3, padding=1),
            _group_norm(self.width),
            nn.GELU(),
            nn.Conv2d(self.width, self.width, kernel_size=1),
        )
        self.operator = FactorizedComplexResidualStack(
            width=self.width,
            spectral_rank=int(spectral_rank),
            modes=int(modes),
            depth=int(depth),
            activation_checkpointing=bool(activation_checkpointing),
            coupled_axes=True,
        )
        self.output_head = nn.Sequential(
            _group_norm(self.width),
            nn.GELU(),
            nn.Conv2d(self.width, self.block_size, kernel_size=1),
        )
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def _validate_block_inputs(
        self,
        parent_context: torch.Tensor,
        corrected_history: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[int, int, int]:
        expected_parent = self.history_frames + self.block_size
        if parent_context.ndim != 4 or parent_context.shape[1] != expected_parent:
            raise ValueError(
                f"parent_context must be [B,{expected_parent},Z,X]"
            )
        if corrected_history.ndim != 4 or corrected_history.shape[1] != self.history_frames:
            raise ValueError(
                f"corrected_history must be [B,{self.history_frames},Z,X]"
            )
        if condition.ndim != 4 or condition.shape[1] != self.condition_channels:
            raise ValueError(
                f"condition must be [B,{self.condition_channels},Z,X]"
            )
        batch, _, height, width = parent_context.shape
        expected = (batch, height, width)
        if (corrected_history.shape[0], *corrected_history.shape[-2:]) != expected:
            raise ValueError("corrected history batch/spatial shape mismatch")
        if (condition.shape[0], *condition.shape[-2:]) != expected:
            raise ValueError("condition batch/spatial shape mismatch")
        if not all(
            tensor.is_floating_point() and bool(torch.isfinite(tensor).all())
            for tensor in (parent_context, corrected_history, condition)
        ):
            raise ValueError("corrector inputs must be finite floating-point tensors")
        return batch, height, width

    def forward_block(
        self,
        parent_context: torch.Tensor,
        corrected_history: torch.Tensor,
        condition: torch.Tensor,
        *,
        block_start: int,
        total_frames: int,
    ) -> torch.Tensor:
        """Correct one complete block using parent predictions and past state."""

        batch, height, width = self._validate_block_inputs(
            parent_context, corrected_history, condition
        )
        start = int(block_start)
        total = int(total_frames)
        if total <= self.history_frames or not self.history_frames <= start < total:
            raise ValueError("block_start/total_frames are inconsistent")

        parent_history = parent_context[:, : self.history_frames]
        parent_future = parent_context[:, self.history_frames :]
        dynamic_scale = torch.maximum(
            parent_context.float().square().mean((1, 2, 3), keepdim=True).sqrt(),
            corrected_history.float().square().mean((1, 2, 3), keepdim=True).sqrt(),
        ).detach().clamp_min(self.minimum_scale)
        residual_history = corrected_history - parent_history
        time_start = 2.0 * float(start) / float(max(total - 1, 1)) - 1.0
        time_stop = (
            2.0
            * float(min(start + self.block_size - 1, total - 1))
            / float(max(total - 1, 1))
            - 1.0
        )
        time_maps = parent_context.new_empty((batch, 2, height, width))
        time_maps[:, 0].fill_(time_start)
        time_maps[:, 1].fill_(time_stop)
        features = torch.cat(
            (
                parent_context / dynamic_scale.to(parent_context.dtype),
                residual_history / dynamic_scale.to(residual_history.dtype),
                condition,
                time_maps,
            ),
            dim=1,
        )
        hidden = self.operator(self.input_projection(features))
        raw_correction = self.output_head(hidden)
        cap = self.maximum_correction_ratio * dynamic_scale.to(raw_correction.dtype)
        learned = cap * torch.tanh(raw_correction)

        if self.boundary_blend_frames:
            # Carry the previous correction into the next block, then hand over
            # smoothly to the learned block proposal.  The carried value is
            # itself projected into the same hard correction cap.
            carried = cap * torch.tanh(residual_history[:, -1:] / cap.clamp_min(1.0e-12))
            positions = torch.arange(
                self.block_size,
                device=learned.device,
                dtype=learned.dtype,
            )
            blend = (
                1.0 - positions / float(self.boundary_blend_frames)
            ).clamp(0.0, 1.0)[None, :, None, None]
            correction = blend * carried + (1.0 - blend) * learned
        else:
            correction = learned

        corrected = parent_future + correction
        if self.hard_free_surface:
            corrected = corrected.clone()
            corrected[..., 0, :] = 0.0
        return corrected

    def rollout(
        self,
        parent_trajectory: torch.Tensor,
        condition: torch.Tensor,
        *,
        initial_history: torch.Tensor | None = None,
        total_frames: int | None = None,
    ) -> torch.Tensor:
        """Roll over a complete parent trajectory in shared fixed-size blocks.

        ``initial_history`` may contain the two protocol-approved onset
        observations.  If omitted, the first two parent frames are used, so the
        deployment API needs no labels or observations at all.
        """

        parent = torch.as_tensor(parent_trajectory)
        if parent.ndim != 4:
            raise ValueError("parent_trajectory must be [B,T,Z,X]")
        batch, available, height, width = parent.shape
        registered_total = available if total_frames is None else int(total_frames)
        if available <= self.history_frames:
            raise ValueError("parent trajectory is too short")
        if registered_total < available:
            raise ValueError("total_frames cannot be shorter than the supplied trajectory")
        if condition.shape != (batch, self.condition_channels, height, width):
            raise ValueError("condition shape does not match parent trajectory")
        if initial_history is None:
            history = parent[:, : self.history_frames]
        else:
            history = torch.as_tensor(
                initial_history, device=parent.device, dtype=parent.dtype
            )
            if history.shape != (batch, self.history_frames, height, width):
                raise ValueError("initial_history shape mismatch")
        pieces = [history]
        for start in range(self.history_frames, available, self.block_size):
            stop = min(start + self.block_size, available)
            valid = stop - start
            future = parent[:, start:stop]
            if valid < self.block_size:
                future = torch.cat(
                    (
                        future,
                        future[:, -1:].expand(-1, self.block_size - valid, -1, -1),
                    ),
                    dim=1,
                )
            parent_history = parent[:, start - self.history_frames : start]
            context = torch.cat((parent_history, future), dim=1)
            corrected = self.forward_block(
                context,
                history,
                condition,
                block_start=start,
                total_frames=registered_total,
            )
            pieces.append(corrected[:, :valid])
            history = corrected[:, max(0, valid - self.history_frames) : valid]
            if history.shape[1] < self.history_frames:
                history = torch.cat((pieces[-2][:, -1:], history), dim=1)
        return torch.cat(pieces, dim=1)


def _per_record_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    dimensions = tuple(range(1, prediction.ndim))
    # Adding eps inside sqrt keeps the norm differentiable at an exact zero
    # correction.  clamp_min(...).sqrt() still has an undefined 0 * inf chain
    # derivative at zero and poisoned the first zero-initialized smoke update.
    numerator = ((prediction - target).float().square().sum(dimensions) + eps).sqrt()
    denominator = (target.float().square().sum(dimensions) + eps).sqrt()
    return numerator / denominator


def parent_anchored_block_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    parent: torch.Tensor,
    *,
    derivative_weight: float = 0.10,
    spectral_weight: float = 0.10,
    nonworse_weight: float = 0.30,
    correction_weight: float = 0.01,
    spectral_floor_fraction: float = 0.005,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Phase-sensitive train-only objective for one or more contiguous blocks."""

    if prediction.shape != target.shape or prediction.shape != parent.shape:
        raise ValueError("prediction, target, and parent must have identical shapes")
    if prediction.ndim != 4 or prediction.shape[1] < 2:
        raise ValueError("block loss expects [B,T,Z,X] with at least two frames")
    if min(
        derivative_weight,
        spectral_weight,
        nonworse_weight,
        correction_weight,
        spectral_floor_fraction,
    ) < 0.0:
        raise ValueError("loss weights must be nonnegative")

    relative_rows = _per_record_relative_l2(prediction, target)
    relative = relative_rows.mean()
    target_delta = target[:, 1:] - target[:, :-1]
    prediction_delta = prediction[:, 1:] - prediction[:, :-1]
    derivative = _per_record_relative_l2(prediction_delta, target_delta).mean()

    target_spectrum = torch.fft.rfft(target.float(), dim=1, norm="ortho")
    prediction_spectrum = torch.fft.rfft(prediction.float(), dim=1, norm="ortho")
    frequency_count = int(target_spectrum.shape[1])
    edges = (0, max(1, frequency_count // 4), max(2, frequency_count // 2), frequency_count)
    total_energy = target_spectrum.abs().square().sum((1, 2, 3)).clamp_min(1.0e-12)
    spectral_rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        if high <= low:
            continue
        target_band = target_spectrum[:, low:high]
        error_band = prediction_spectrum[:, low:high] - target_band
        numerator = error_band.abs().square().sum((1, 2, 3)).sqrt()
        energy = target_band.abs().square().sum((1, 2, 3))
        denominator = torch.maximum(
            energy, float(spectral_floor_fraction) * total_energy
        ).clamp_min(1.0e-12).sqrt()
        spectral_rows.append(numerator / denominator)
    spectral = torch.stack(spectral_rows, dim=1).mean()

    parent_rows = _per_record_relative_l2(parent, target).detach()
    nonworse = torch.relu(relative_rows - parent_rows).mean()
    correction = _per_record_relative_l2(prediction, parent).mean()
    loss = (
        relative
        + float(derivative_weight) * derivative
        + float(spectral_weight) * spectral
        + float(nonworse_weight) * nonworse
        + float(correction_weight) * correction
    )
    return loss, {
        "relative_l2": relative.detach(),
        "derivative_relative_l2": derivative.detach(),
        "spectral_relative_l2": spectral.detach(),
        "nonworse_hinge": nonworse.detach(),
        "correction_relative_l2": correction.detach(),
        "parent_relative_l2": parent_rows.mean().detach(),
    }


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


__all__ = [
    "ParentAnchoredBlockCorrector",
    "parameter_count",
    "parent_anchored_block_loss",
    "render_retained_rfft_frames",
]
