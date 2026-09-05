"""Fast causal instance adaptation from a pretrained temporal latent subspace."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import torch

from .defect_correction import (
    FactorizedCausalBasis,
    causal_observation_probe_basis,
    causal_smoothstep_envelope,
)


@dataclass(frozen=True)
class CapturedTemporalLatentInputs:
    conditioning: torch.Tensor
    time_s: torch.Tensor
    source_parameters: torch.Tensor
    domain_t_s: float


@dataclass(frozen=True)
class ScalarMultiplierOracle:
    """Train-only response of a sealed correction to scalar damping.

    This diagnostic requires future truth and must never be used by the online
    adapter. It is intentionally separate from the causal basis/solve API.
    """

    multiplier_grid: torch.Tensor
    relative_l2: torch.Tensor
    best_multiplier: torch.Tensor
    parent_relative_l2: torch.Tensor
    best_relative_l2: torch.Tensor


class TemporalLatentInputCapture:
    """Capture query-invariant A3 inputs while the frozen parent is evaluated.

    The parent commonly renders one saved time per block.  Only the first copy
    of the query-invariant conditioning is retained; time blocks are concatenated
    in their original order after the forward pass.
    """

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self._handle = None
        self._conditioning: torch.Tensor | None = None
        self._source_parameters: torch.Tensor | None = None
        self._domain_t_s: float | None = None
        self._time_blocks: list[torch.Tensor] = []

    def __enter__(self) -> "TemporalLatentInputCapture":
        if self._handle is not None:
            raise RuntimeError("temporal latent capture is already active")
        self._handle = self.module.register_forward_pre_hook(
            self._capture, with_kwargs=True
        )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _capture(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if len(args) < 3:
            raise ValueError("temporal latent forward inputs are incomplete")
        conditioning = torch.as_tensor(args[0]).detach()
        time_s = torch.as_tensor(args[1]).detach()
        source_parameters = torch.as_tensor(args[2]).detach()
        domain_t_s = float(kwargs["domain_t_s"])
        if conditioning.ndim != 4 or time_s.ndim != 2:
            raise ValueError("captured temporal latent shapes are invalid")
        if self._conditioning is None:
            self._conditioning = conditioning
            self._source_parameters = source_parameters
            self._domain_t_s = domain_t_s
        else:
            if conditioning.shape != self._conditioning.shape:
                raise ValueError("temporal latent conditioning changed shape")
            if source_parameters.shape != self._source_parameters.shape:
                raise ValueError("temporal latent source parameters changed shape")
            if not math.isclose(
                domain_t_s, float(self._domain_t_s), rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ValueError("temporal latent domain duration changed")
        self._time_blocks.append(time_s)

    def finalize(
        self, *, expected_time_s: torch.Tensor | None = None
    ) -> CapturedTemporalLatentInputs:
        if (
            self._conditioning is None
            or self._source_parameters is None
            or self._domain_t_s is None
            or not self._time_blocks
        ):
            raise RuntimeError("the parent did not invoke its temporal latent module")
        times = torch.cat(self._time_blocks, dim=1)
        if expected_time_s is not None:
            expected = torch.as_tensor(
                expected_time_s, dtype=times.dtype, device=times.device
            )
            if expected.ndim == 1:
                expected = expected.unsqueeze(0).expand(times.shape[0], -1)
            if expected.shape != times.shape or not torch.allclose(
                expected, times, rtol=0.0, atol=1.0e-7
            ):
                raise ValueError("captured temporal blocks do not match the saved axis")
        return CapturedTemporalLatentInputs(
            conditioning=self._conditioning,
            time_s=times,
            source_parameters=self._source_parameters,
            domain_t_s=float(self._domain_t_s),
        )


def temporal_latent_state_sha256(module: torch.nn.Module) -> str:
    """Canonical digest of the pretrained temporal-latent parameters."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def pretrained_temporal_latent_basis(
    module: torch.nn.Module,
    captured: CapturedTemporalLatentInputs,
    parent_field: torch.Tensor,
    observed_wavefield: torch.Tensor,
    observed_indices: tuple[int, int],
    *,
    ramp_steps: int = 4,
) -> FactorizedCausalBasis:
    """Build 32 normalized causal modes from the frozen A3 temporal latent head.

    The spatial anchors and continuous-time coefficients come directly from the
    pretrained weights.  Per-mode rescaling changes only ridge conditioning, not
    the represented correction subspace.  A post-observation envelope makes the
    instance gains an exact no-op at every accessible true frame.
    """

    parent = torch.as_tensor(parent_field)
    observed = torch.as_tensor(
        observed_wavefield, dtype=parent.dtype, device=parent.device
    )
    if parent.ndim != 4:
        raise ValueError("parent field must be [record,time,z,x]")
    records, time_count, height, width = parent.shape
    if observed.shape != (records, 2, height, width):
        raise ValueError("observed wavefield must contain exactly two frames")
    if captured.time_s.shape != (records, time_count):
        raise ValueError("captured time axis does not match the parent field")
    if captured.conditioning.shape[0] != records:
        raise ValueError("captured conditioning does not match the records")
    if tuple(int(value) for value in observed_indices) != tuple(observed_indices):
        raise ValueError("observed indices must be integers")

    with torch.no_grad():
        conditioning = captured.conditioning.to(
            device=parent.device, dtype=parent.dtype
        )
        source_parameters = captured.source_parameters.to(
            device=parent.device, dtype=parent.dtype
        )
        anchors = module.anchor_projection(conditioning)
        features = module.time_features(
            captured.time_s.to(device=parent.device, dtype=parent.dtype),
            source_parameters,
            domain_t_s=float(captured.domain_t_s),
        )
        temporal = module.time_trunk(features).transpose(1, 2)
    if anchors.shape[0] != records or anchors.shape[-2:] != (height, width):
        raise ValueError("pretrained temporal anchors do not match the parent grid")
    if temporal.shape != (records, anchors.shape[1], time_count):
        raise ValueError("pretrained temporal coefficients do not match the anchors")

    envelope = causal_smoothstep_envelope(
        time_count,
        observed_indices,
        ramp_steps=int(ramp_steps),
        dtype=parent.dtype,
        device=parent.device,
    ).unsqueeze(0).expand(records, -1)
    spatial_norm = anchors.square().flatten(2).mean(dim=2).sqrt().clamp_min(1.0e-6)
    spatial = anchors / spatial_norm[:, :, None, None]
    temporal = temporal * spatial_norm[:, :, None]
    post_weight = envelope[:, None].square()
    temporal_norm = (
        (temporal.square() * post_weight).sum(dim=2)
        / post_weight.sum(dim=2).clamp_min(1.0)
    ).sqrt().clamp_min(1.0e-6)
    temporal = temporal / temporal_norm[:, :, None]

    parent_rms = parent.float().square().flatten(1).mean(dim=1).sqrt()
    observed_rms = observed.float().square().flatten(1).mean(dim=1).sqrt()
    field_scale = torch.maximum(parent_rms, observed_rms).clamp_min(1.0e-8).to(
        parent.dtype
    )
    return FactorizedCausalBasis(
        spatial_modes=spatial,
        temporal_modes=temporal,
        causal_envelope=envelope,
        field_scale=field_scale,
        phase_reference=torch.zeros_like(parent),
        trust_fraction=torch.ones(
            records, dtype=parent.dtype, device=parent.device
        ),
        phase_rank=0,
    )


def pretrained_temporal_latent_probe_basis(
    causal_basis: FactorizedCausalBasis,
) -> FactorizedCausalBasis:
    """Expose the same pretrained modes at observations for coefficient fitting.

    This basis is a design-matrix probe only. The deployed correction must still
    be combined with ``causal_basis`` so every observed true frame is preserved.
    """

    return causal_observation_probe_basis(causal_basis)


def train_only_scalar_multiplier_oracle(
    parent_field: torch.Tensor,
    adapted_field: torch.Tensor,
    future_truth: torch.Tensor,
    observed_indices: tuple[int, int],
    *,
    minimum_multiplier: float = -1.0,
    maximum_multiplier: float = 1.0,
    steps: int = 81,
) -> ScalarMultiplierOracle:
    """Evaluate a sealed correction direction over a bounded scalar grid.

    The metric exactly matches the evaluator's mean of per-frame relative L2
    over frames after the final observation. Quadratic sufficient statistics
    avoid materializing one full wavefield for every scalar candidate.
    """

    parent = torch.as_tensor(parent_field).double()
    adapted = torch.as_tensor(
        adapted_field, dtype=parent.dtype, device=parent.device
    )
    truth = torch.as_tensor(future_truth, dtype=parent.dtype, device=parent.device)
    if parent.ndim != 4 or adapted.shape != parent.shape or truth.shape != parent.shape:
        raise ValueError("parent, adapted, and truth must match [record,time,z,x]")
    start = int(observed_indices[1]) + 1
    if start <= int(observed_indices[0]) or start >= parent.shape[1]:
        raise ValueError("observed indices leave no future response support")
    count = int(steps)
    low = float(minimum_multiplier)
    high = float(maximum_multiplier)
    if count < 2 or not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError("scalar multiplier grid is invalid")

    error = (parent[:, start:] - truth[:, start:]).flatten(2)
    correction = (adapted[:, start:] - parent[:, start:]).flatten(2)
    reference = truth[:, start:].flatten(2)
    q0 = error.square().sum(dim=2)
    q1 = 2.0 * (error * correction).sum(dim=2)
    q2 = correction.square().sum(dim=2)
    denominator = reference.square().sum(dim=2).sqrt().clamp_min(1.0e-8)
    grid = torch.linspace(low, high, count, dtype=parent.dtype, device=parent.device)
    squared = (
        q0[:, None]
        + grid[None, :, None] * q1[:, None]
        + grid[None, :, None].square() * q2[:, None]
    ).clamp_min(0.0)
    relative_l2 = (squared.sqrt() / denominator[:, None]).mean(dim=2)
    best_index = relative_l2.argmin(dim=1)
    record_index = torch.arange(parent.shape[0], device=parent.device)
    zero_index = int(torch.argmin(grid.abs()))
    return ScalarMultiplierOracle(
        multiplier_grid=grid,
        relative_l2=relative_l2,
        best_multiplier=grid[best_index],
        parent_relative_l2=relative_l2[:, zero_index],
        best_relative_l2=relative_l2[record_index, best_index],
    )


__all__ = [
    "CapturedTemporalLatentInputs",
    "ScalarMultiplierOracle",
    "TemporalLatentInputCapture",
    "pretrained_temporal_latent_basis",
    "pretrained_temporal_latent_probe_basis",
    "temporal_latent_state_sha256",
    "train_only_scalar_multiplier_oracle",
]
