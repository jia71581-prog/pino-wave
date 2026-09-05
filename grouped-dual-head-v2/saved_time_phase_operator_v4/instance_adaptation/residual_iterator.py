"""Frozen-starter multi-scale residual iteration for fast safe adaptation.

The corrector never owns or mutates the pretrained operator.  It maps the
current calibrated LWC-84 residual to a bounded full-field correction and is
shared across all unrolled steps.  Deployment evaluates a small set of step
sizes, including zero, against one fixed baseline calibration scale; therefore
an accepted step cannot have a worse audited residual than its input.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .losses import lwc84_residual


RESIDUAL_ITERATOR_SCHEMA_VERSION = 1


def _normalization_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if int(channels) % groups == 0:
            return groups
    return 1


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(_normalization_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_normalization_groups(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.main(value) + self.skip(value))


class MultiScaleResidualCorrector(nn.Module):
    """Three-scale spatial corrector with learnable frequency-band gains."""

    def __init__(self, width: int = 24, maximum_correction_fraction: float = 0.25):
        super().__init__()
        if int(width) <= 0 or not math.isfinite(float(maximum_correction_fraction)):
            raise ValueError("residual corrector configuration is invalid")
        if float(maximum_correction_fraction) <= 0.0:
            raise ValueError("maximum correction fraction must be positive")
        self.width = int(width)
        self.maximum_correction_fraction = float(maximum_correction_fraction)
        # Channels: velocity, frozen starter, current state, aligned PDE residual,
        # and normalized saved time.
        self.encoder0 = _ResidualBlock(5, self.width)
        self.encoder1 = _ResidualBlock(self.width, 2 * self.width, stride=2)
        self.bottleneck = _ResidualBlock(2 * self.width, 4 * self.width, stride=2)
        self.decoder1 = _ResidualBlock(6 * self.width, 2 * self.width)
        self.decoder0 = _ResidualBlock(3 * self.width, self.width)
        self.output = nn.Conv2d(self.width, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        # Separate low/middle/high response can be learned offline.  Unit gains
        # initially preserve the exact zero-output/no-op contract.
        self.log_band_gains = nn.Parameter(torch.zeros(3))
        self.step_logit = nn.Parameter(torch.tensor(-1.0986122886681098))  # sigmoid = .25

    @property
    def training_step_size(self) -> torch.Tensor:
        return torch.sigmoid(self.step_logit)

    @staticmethod
    def _frequency_masks(
        height: int, width: int, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fy = torch.fft.fftfreq(height, device=device, dtype=dtype).abs()[:, None]
        fx = torch.fft.rfftfreq(width, device=device, dtype=dtype).abs()[None, :]
        radius = torch.sqrt(fy.square() + fx.square())
        low = torch.sigmoid((0.16 - radius) * 40.0)
        high = torch.sigmoid((radius - 0.42) * 40.0)
        middle = (1.0 - low) * (1.0 - high)
        normalizer = (low + middle + high).clamp_min(torch.finfo(dtype).eps)
        return low / normalizer, middle / normalizer, high / normalizer

    def _mix_frequency_bands(self, value: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-2:]
        spectrum = torch.fft.rfft2(value.float(), norm="ortho")
        masks = self._frequency_masks(
            height, width, device=value.device, dtype=spectrum.real.dtype
        )
        gains = self.log_band_gains.float().clamp(-2.0, 2.0).exp()
        multiplier = sum(gain * mask for gain, mask in zip(gains, masks, strict=True))
        mixed = torch.fft.irfft2(
            spectrum * multiplier[None, None], s=(height, width), norm="ortho"
        )
        return mixed.to(value.dtype)

    def forward(
        self,
        starter: torch.Tensor,
        current: torch.Tensor,
        velocity_mps: torch.Tensor,
        aligned_residual: torch.Tensor,
        time_s: torch.Tensor,
    ) -> torch.Tensor:
        starter_value = torch.as_tensor(starter)
        current_value = torch.as_tensor(
            current, dtype=starter_value.dtype, device=starter_value.device
        )
        residual_value = torch.as_tensor(
            aligned_residual, dtype=starter_value.dtype, device=starter_value.device
        )
        velocity = torch.as_tensor(
            velocity_mps, dtype=starter_value.dtype, device=starter_value.device
        )
        if starter_value.ndim != 4 or current_value.shape != starter_value.shape:
            raise ValueError("starter and current wavefields must match [record,time,z,x]")
        if residual_value.shape != starter_value.shape:
            raise ValueError("aligned residual must match the wavefield")
        records, saved_times, height, width = starter_value.shape
        if velocity.ndim == 3:
            velocity = velocity[:, None]
        if velocity.shape != (records, 1, height, width):
            raise ValueError("velocity must be [record,1,z,x]")
        times = torch.as_tensor(time_s, dtype=starter_value.dtype, device=starter_value.device)
        if times.ndim == 1:
            times = times[None].expand(records, -1)
        if times.shape != (records, saved_times):
            raise ValueError("saved times must be [time] or [record,time]")

        field_scale = starter_value.detach().float().flatten(1).square().mean(dim=1).sqrt()
        field_scale = field_scale.clamp_min(1.0e-8).to(starter_value.dtype)
        velocity_mean = velocity.mean(dim=(-2, -1), keepdim=True)
        velocity_scale = velocity.std(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        time_start = times[:, :1]
        time_span = (times[:, -1:] - time_start).clamp_min(1.0e-8)
        time_coordinate = 2.0 * (times - time_start) / time_span - 1.0

        expand_velocity = velocity.expand(-1, saved_times, -1, -1)
        inputs = torch.stack(
            (
                ((expand_velocity - velocity_mean) / velocity_scale),
                starter_value / field_scale[:, None, None, None],
                current_value / field_scale[:, None, None, None],
                torch.tanh(residual_value),
                time_coordinate[:, :, None, None].expand(-1, -1, height, width),
            ),
            dim=2,
        ).reshape(records * saved_times, 5, height, width)
        level0 = self.encoder0(inputs)
        level1 = self.encoder1(level0)
        center = self.bottleneck(level1)
        up1 = F.interpolate(center, size=level1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.decoder1(torch.cat((up1, level1), dim=1))
        up0 = F.interpolate(up1, size=level0.shape[-2:], mode="bilinear", align_corners=False)
        up0 = self.decoder0(torch.cat((up0, level0), dim=1))
        raw = self.output(up0)
        raw = self._mix_frequency_bands(raw)
        bounded = torch.tanh(raw).reshape(records, saved_times, height, width)
        return (
            bounded
            * field_scale[:, None, None, None]
            * self.maximum_correction_fraction
        )


def align_lwc84_residual(
    residual: torch.Tensor,
    wavefield: torch.Tensor,
    observed_indices: tuple[int, int],
) -> torch.Tensor:
    """Place cropped LWC residuals at their absolute time/spatial centers."""

    value = torch.as_tensor(residual)
    field = torch.as_tensor(wavefield, dtype=value.dtype, device=value.device)
    if value.ndim != 4 or field.ndim != 4 or value.shape[0] != field.shape[0]:
        raise ValueError("residual and field must be batched space-time tensors")
    if value.shape[-2:] == field.shape[-2:]:
        spatial = value
    elif value.shape[-2] + 8 == field.shape[-2] and value.shape[-1] + 8 == field.shape[-1]:
        spatial = F.pad(value, (4, 4, 4, 4))
    else:
        raise ValueError("LWC residual spatial support does not match the field")
    first_center = int(observed_indices[1]) + 1
    stop = first_center + spatial.shape[1]
    if first_center < 0 or stop > field.shape[1]:
        raise ValueError("LWC residual temporal support does not match the field")
    aligned = torch.zeros_like(field)
    aligned[:, first_center:stop] = spatial
    return aligned


@dataclass(frozen=True)
class ResidualIteratorResult:
    field: torch.Tensor
    accepted_steps: int
    residual_history: tuple[tuple[float, ...], ...]
    step_size_history: tuple[tuple[float, ...], ...]
    energy_ratio: tuple[float, ...]
    stopped_reason: str


def _residual_score(residual: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(residual).float().square().flatten(1).mean(dim=1)


def _uniform_saved_dt(time_s: torch.Tensor) -> float:
    source = torch.as_tensor(time_s)
    if not source.is_floating_point():
        raise ValueError("saved times must be floating point")
    source_epsilon = float(torch.finfo(source.dtype).eps)
    times = source.double()
    if times.ndim == 1:
        times = times.unsqueeze(0)
    if times.ndim != 2 or times.shape[1] < 2:
        raise ValueError("saved times must contain at least two values per record")
    deltas = times[:, 1:] - times[:, :-1]
    reference = deltas.median()
    # Saved times are read from float64 HDF5 metadata but the model-facing data
    # contract stores them as float32.  At late times, subtracting adjacent
    # float32 values introduces an O(eps * |t|) delta jitter that is larger than
    # a purely relative-to-dt tolerance.  Account for that representational
    # error while still rejecting an actual missing/duplicated saved frame.
    maximum_time = float(times.abs().max())
    quantization_tolerance = 4.0 * source_epsilon * maximum_time
    tolerance = max(
        1.0e-10,
        abs(float(reference)) * 1.0e-5,
        quantization_tolerance,
    )
    if tolerance >= 0.05 * abs(float(reference)):
        raise ValueError("saved-time precision is insufficient to verify uniform spacing")
    if (
        not bool(torch.isfinite(deltas).all())
        or float(reference) <= 0.0
        or bool((deltas <= 0.0).any())
        or float((deltas - reference).abs().max()) > tolerance
    ):
        raise ValueError("residual iteration requires one uniform saved-time spacing")
    return float(reference)


def refine_with_residual_iterator(
    corrector: nn.Module,
    starter: torch.Tensor,
    velocity_mps: torch.Tensor,
    time_s: torch.Tensor,
    observed_indices: tuple[int, int],
    *,
    iterations: int = 4,
    step_candidates: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    minimum_relative_improvement: float = 1.0e-4,
    minimum_energy_ratio: float = 0.8,
    maximum_energy_ratio: float = 1.25,
    dt: float | None = None,
    dx: float = 10.0,
    dz: float = 10.0,
    frozen_time_indices: Sequence[int] | None = None,
) -> ResidualIteratorResult:
    """Safely refine a frozen starter using only physical residual comparisons."""

    if int(iterations) < 0:
        raise ValueError("residual iterator count cannot be negative")
    candidates = tuple(sorted(set(float(value) for value in step_candidates)))
    if not candidates or candidates[0] != 0.0 or any(
        not math.isfinite(value) or value < 0.0 for value in candidates
    ):
        raise ValueError("step candidates must be finite, nonnegative, and include zero")
    improvement = float(minimum_relative_improvement)
    if not 0.0 <= improvement < 1.0:
        raise ValueError("minimum residual improvement must lie in [0,1)")
    if not 0.0 < float(minimum_energy_ratio) <= float(maximum_energy_ratio):
        raise ValueError("residual iterator energy gate is invalid")

    current = torch.as_tensor(starter).detach().clone()
    velocity = torch.as_tensor(velocity_mps, dtype=current.dtype, device=current.device)
    times = torch.as_tensor(time_s, dtype=current.dtype, device=current.device)
    step_dt = _uniform_saved_dt(times) if dt is None else float(dt)
    initial_residual, calibration_scale = lwc84_residual(
        current,
        velocity,
        dt=step_dt,
        dx=float(dx),
        dz=float(dz),
        observed_indices=observed_indices,
        return_scale=True,
    )
    current_score = _residual_score(initial_residual)
    starter_energy = current.float().square().flatten(1).mean(dim=1).clamp_min(1.0e-12)
    residual_rows: list[list[float]] = [[float(value)] for value in current_score]
    step_rows: list[list[float]] = [[] for _ in range(current.shape[0])]
    accepted_steps = 0
    stopped_reason = "iteration_limit"
    frozen = (
        tuple(int(value) for value in observed_indices)
        if frozen_time_indices is None
        else tuple(int(value) for value in frozen_time_indices)
    )
    if any(value < 0 or value >= current.shape[1] for value in frozen):
        raise ValueError("frozen residual-iterator time index is outside the field")

    for _ in range(int(iterations)):
        residual = lwc84_residual(
            current,
            velocity,
            dt=step_dt,
            dx=float(dx),
            dz=float(dz),
            observed_indices=observed_indices,
            normalization_scale=calibration_scale,
        )
        aligned = align_lwc84_residual(residual, current, observed_indices)
        with torch.no_grad():
            correction = corrector(starter, current, velocity, aligned, times)
            if correction.shape != current.shape or not bool(torch.isfinite(correction).all()):
                stopped_reason = "nonfinite_correction"
                break
            if frozen:
                correction[:, list(frozen)] = 0.0
            best_field = current
            best_score = current_score
            best_alpha = torch.zeros_like(current_score)
            threshold = current_score * (1.0 - improvement)
            for alpha in candidates[1:]:
                candidate = current + float(alpha) * correction
                candidate_energy = candidate.float().square().flatten(1).mean(dim=1)
                energy_ratio = candidate_energy / starter_energy
                candidate_residual = lwc84_residual(
                    candidate,
                    velocity,
                    dt=step_dt,
                    dx=float(dx),
                    dz=float(dz),
                    observed_indices=observed_indices,
                    normalization_scale=calibration_scale,
                )
                candidate_score = _residual_score(candidate_residual)
                valid = (
                    torch.isfinite(candidate_score)
                    & (candidate_score < threshold)
                    & (candidate_score < best_score)
                    & (energy_ratio >= float(minimum_energy_ratio))
                    & (energy_ratio <= float(maximum_energy_ratio))
                )
                best_field = torch.where(valid[:, None, None, None], candidate, best_field)
                best_score = torch.where(valid, candidate_score, best_score)
                best_alpha = torch.where(valid, best_alpha.new_full((), float(alpha)), best_alpha)
            changed = best_alpha > 0.0
            if not bool(changed.any()):
                stopped_reason = "no_contracting_step"
                for row in range(current.shape[0]):
                    residual_rows[row].append(float(current_score[row]))
                    step_rows[row].append(0.0)
                break
            current = best_field
            current_score = best_score
            accepted_steps += 1
            for row in range(current.shape[0]):
                residual_rows[row].append(float(current_score[row]))
                step_rows[row].append(float(best_alpha[row]))

    final_energy = current.float().square().flatten(1).mean(dim=1) / starter_energy
    return ResidualIteratorResult(
        field=current,
        accepted_steps=accepted_steps,
        residual_history=tuple(tuple(row) for row in residual_rows),
        step_size_history=tuple(tuple(row) for row in step_rows),
        energy_ratio=tuple(float(value) for value in final_energy),
        stopped_reason=stopped_reason,
    )


def unroll_residual_iterator(
    corrector: MultiScaleResidualCorrector,
    starter: torch.Tensor,
    velocity_mps: torch.Tensor,
    time_s: torch.Tensor,
    observed_indices: tuple[int, int],
    *,
    iterations: int = 4,
    dt: float | None = None,
    dx: float = 10.0,
    dz: float = 10.0,
    frozen_time_indices: Sequence[int] | None = None,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Differentiable fixed-step unroll used only for offline supervision."""

    current = torch.as_tensor(starter)
    velocity = torch.as_tensor(velocity_mps, dtype=current.dtype, device=current.device)
    times = torch.as_tensor(time_s, dtype=current.dtype, device=current.device)
    step_dt = _uniform_saved_dt(times) if dt is None else float(dt)
    initial_residual, calibration_scale = lwc84_residual(
        current, velocity, dt=step_dt, dx=float(dx), dz=float(dz),
        observed_indices=observed_indices, return_scale=True,
    )
    states: list[torch.Tensor] = [current]
    residuals: list[torch.Tensor] = [initial_residual]
    frozen = (
        tuple(int(value) for value in observed_indices)
        if frozen_time_indices is None
        else tuple(int(value) for value in frozen_time_indices)
    )
    if any(value < 0 or value >= current.shape[1] for value in frozen):
        raise ValueError("frozen residual-iterator time index is outside the field")
    for _ in range(int(iterations)):
        aligned = align_lwc84_residual(residuals[-1], current, observed_indices)
        correction = corrector(starter, current, velocity, aligned, times)
        mask = torch.ones_like(correction)
        if frozen:
            mask[:, list(frozen)] = 0.0
        current = current + corrector.training_step_size * correction * mask
        current_residual = lwc84_residual(
            current, velocity, dt=step_dt, dx=float(dx), dz=float(dz),
            observed_indices=observed_indices,
            normalization_scale=calibration_scale,
        )
        states.append(current)
        residuals.append(current_residual)
    return tuple(states), tuple(residuals)


@dataclass(frozen=True)
class ResidualIteratorLossWeights:
    trajectory: float = 1.0
    spectrum: float = 0.1
    physics: float = 0.1
    contraction: float = 0.1
    fixed_point: float = 0.01
    contraction_target: float = 0.98


def residual_iterator_training_loss(
    states: Sequence[torch.Tensor],
    residuals: Sequence[torch.Tensor],
    target: torch.Tensor,
    *,
    fixed_point_correction: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
    weights: ResidualIteratorLossWeights = ResidualIteratorLossWeights(),
) -> dict[str, torch.Tensor]:
    """Trajectory, progressive-spectrum, physics, and contraction objective."""

    if len(states) < 2 or len(states) != len(residuals):
        raise ValueError("iterator loss requires matching initial-plus-step trajectories")
    truth = torch.as_tensor(target, dtype=states[0].dtype, device=states[0].device)
    if any(state.shape != truth.shape for state in states):
        raise ValueError("every iterator state must match the target")
    batch_size = truth.shape[0]
    if sample_weights is None:
        objective_weights = truth.new_ones((batch_size,), dtype=torch.float32)
    else:
        objective_weights = torch.as_tensor(
            sample_weights, device=truth.device, dtype=torch.float32
        )
        if objective_weights.shape != (batch_size,):
            raise ValueError("iterator sample weights must contain one value per record")
        if (
            not bool(torch.isfinite(objective_weights).all())
            or bool((objective_weights < 0.0).any())
            or float(objective_weights.sum()) <= 0.0
        ):
            raise ValueError("iterator sample weights must be finite and nonnegative")

    def weighted_record_mean(values: torch.Tensor) -> torch.Tensor:
        per_record = torch.as_tensor(values, device=truth.device, dtype=torch.float32)
        if per_record.shape != (batch_size,):
            raise ValueError("iterator objective did not produce one value per record")
        # These are objective multipliers rather than probability weights.  A
        # batch of one must retain its family weight so the common batch=1 path
        # can still emphasize layered/marmousi records.
        return (per_record * objective_weights).mean()

    tiny = torch.finfo(torch.float32).tiny
    truth_norm = truth.float().flatten(1).norm(dim=1).clamp_min(tiny)
    relative_terms = [
        (state.float() - truth.float()).flatten(1).norm(dim=1) / truth_norm
        for state in states[1:]
    ]
    trajectory = torch.stack(
        [weighted_record_mean(value) for value in relative_terms]
    ).mean()

    spectral_terms = []
    for step, state in enumerate(states[1:], start=1):
        error_spectrum = torch.fft.rfft2((state - truth).float(), norm="ortho").abs()
        target_spectrum = torch.fft.rfft2(truth.float(), norm="ortho").abs()
        height, width = state.shape[-2:]
        masks = MultiScaleResidualCorrector._frequency_masks(
            height, width, device=state.device, dtype=error_spectrum.dtype
        )
        progress = step / float(len(states) - 1)
        band_weights = (1.0, 1.0 + progress, 1.0 + 3.0 * progress)
        band_loss = state.new_zeros((batch_size,), dtype=torch.float32)
        for band_weight, mask in zip(band_weights, masks, strict=True):
            numerator = (error_spectrum.square() * mask).sum(dim=(-2, -1)).sqrt()
            denominator = (target_spectrum.square() * mask).sum(dim=(-2, -1)).sqrt().clamp_min(tiny)
            band_loss = band_loss + float(band_weight) * (
                numerator / denominator
            ).mean(dim=1)
        spectral_terms.append(weighted_record_mean(band_loss / sum(band_weights)))
    spectrum = torch.stack(spectral_terms).mean()

    physics_scores = torch.stack(
        [weighted_record_mean(_residual_score(value)) for value in residuals[1:]]
    )
    physics = physics_scores.mean()
    contraction_terms = []
    for previous, current in zip(residuals[:-1], residuals[1:], strict=True):
        previous_score = _residual_score(previous).clamp_min(tiny)
        ratio = _residual_score(current) / previous_score
        contraction_terms.append(
            weighted_record_mean(
                F.relu(ratio - float(weights.contraction_target)).square()
            )
        )
    contraction = torch.stack(contraction_terms).mean()
    fixed_point = (
        states[-1].sum() * 0.0
        if fixed_point_correction is None
        else weighted_record_mean(
            torch.as_tensor(fixed_point_correction).float().square().flatten(1).mean(dim=1)
        )
    )
    total = (
        float(weights.trajectory) * trajectory
        + float(weights.spectrum) * spectrum
        + float(weights.physics) * physics
        + float(weights.contraction) * contraction
        + float(weights.fixed_point) * fixed_point
    )
    return {
        "trajectory": trajectory,
        "spectrum": spectrum,
        "physics": physics,
        "contraction": contraction,
        "fixed_point": fixed_point,
        "total": total,
    }


__all__ = [
    "RESIDUAL_ITERATOR_SCHEMA_VERSION",
    "MultiScaleResidualCorrector",
    "ResidualIteratorLossWeights",
    "ResidualIteratorResult",
    "align_lwc84_residual",
    "refine_with_residual_iterator",
    "residual_iterator_training_loss",
    "unroll_residual_iterator",
]
