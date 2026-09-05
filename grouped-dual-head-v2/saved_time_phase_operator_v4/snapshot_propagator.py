"""Causal wave propagation from observed wavefield snapshots only.

The deployment contract of this module is deliberately narrower than the other
saved-time operators: ``forward`` accepts a short, consecutive pressure history
and a rollout length. Velocity, source parameters, travel times, a numerical
background field, and future truth are not arguments.

The fixed core estimates a stable modal propagator from the observed history.
For an undamped wave mode,

    P[k + 1] = 2 cos(omega * dt) P[k] - P[k - 1].

The modal cosine is estimated by least squares from the snapshots and projected
strictly inside [-1, 1]. This unit-circle projection is the important difference
from an unconstrained autoregressive fit, whose small phase errors can become an
exponential long-rollout instability. A zero-initialized neural closure then
learns heterogeneous mode coupling while remaining bounded by the RMS of the
observed context.

An optional observed-defect memory measures the modal core's one-step errors
entirely inside that same history, then carries the last error and its trend into
the rollout with a fixed exponential decay.  It does not read a medium model,
source metadata, a numerical background field, or any future snapshot.

At least four consecutive, energetic snapshots are required. Two frames specify
pressure and its first time derivative, but cannot identify an unknown propagation
symbol from observations alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import math

import torch
from torch import nn
from torch.nn import functional as F

from .spectral import FactorizedComplexResidualStack


def _group_norm(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if int(channels) % groups == 0:
            return nn.GroupNorm(groups, int(channels))
    return nn.GroupNorm(1, int(channels))


def _validate_history(history: torch.Tensor, *, minimum_frames: int) -> torch.Tensor:
    value = torch.as_tensor(history)
    if value.ndim != 4:
        raise ValueError("wavefield history must be [batch,time,z,x]")
    if value.shape[1] < int(minimum_frames):
        raise ValueError(
            f"at least {int(minimum_frames)} consecutive wavefield snapshots are required"
        )
    if min(int(value.shape[-2]), int(value.shape[-1])) < 4:
        raise ValueError("wavefield snapshots must have spatial dimensions >= 4")
    if not (value.is_floating_point() and torch.isfinite(value).all()):
        raise ValueError("wavefield history must be finite floating-point data")
    return value


@dataclass(frozen=True)
class SnapshotModalSymbol:
    """Stable propagation symbol inferred only from the observed pressure history."""

    cosine: torch.Tensor
    confidence: torch.Tensor
    context_scale: torch.Tensor
    context_features: torch.Tensor


class StableModalCalibrator(nn.Module):
    """Snapshot-conditioned phase/damping correction with Schur-stable roots.

    The final layer is zero initialized.  Consequently the initial phase is
    unchanged and the initial modal radius is exactly ``initial_radius``.  Both
    learned outputs are projected into fixed physical bounds, so calibration
    cannot turn the modal core into an exponentially growing recurrence.
    """

    def __init__(
        self,
        *,
        width: int = 16,
        minimum_radius: float = 0.98,
        maximum_radius: float = 0.9999,
        initial_radius: float = 0.995,
        maximum_phase_scale_deviation: float = 0.02,
        phase_margin: float = 1.0e-3,
        use_context_features: bool = False,
    ) -> None:
        super().__init__()
        if int(width) <= 0:
            raise ValueError("modal calibrator width must be positive")
        if not 0.0 < float(minimum_radius) < float(maximum_radius) <= 1.0:
            raise ValueError("modal radii must satisfy 0 < minimum < maximum <= 1")
        if not float(minimum_radius) < float(initial_radius) < float(maximum_radius):
            raise ValueError("initial modal radius must lie strictly inside its bounds")
        if not 0.0 <= float(maximum_phase_scale_deviation) < 1.0:
            raise ValueError("maximum phase scale deviation must lie in [0,1)")
        self.minimum_radius = float(minimum_radius)
        self.maximum_radius = float(maximum_radius)
        self.maximum_phase_scale_deviation = float(maximum_phase_scale_deviation)
        self.phase_margin = float(phase_margin)
        self.use_context_features = bool(use_context_features)
        fraction = (float(initial_radius) - self.minimum_radius) / (
            self.maximum_radius - self.minimum_radius
        )
        self.register_buffer(
            "initial_radius_logit",
            torch.tensor(math.log(fraction / (1.0 - fraction)), dtype=torch.float32),
        )
        input_channels = 7 if self.use_context_features else 5
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, int(width), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(width), 2, kernel_size=1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    @staticmethod
    def _frequency_coordinates(reference: torch.Tensor) -> tuple[torch.Tensor, ...]:
        z_count, x_frequency_count = int(reference.shape[-2]), int(reference.shape[-1])
        kz = 2.0 * torch.fft.fftfreq(
            z_count, device=reference.device, dtype=reference.dtype
        )
        kx = torch.linspace(
            0.0, 1.0, x_frequency_count,
            device=reference.device, dtype=reference.dtype,
        )
        kz_grid = kz[:, None].expand(z_count, x_frequency_count)
        kx_grid = kx[None, :].expand(z_count, x_frequency_count)
        radial = torch.sqrt(kz_grid.square() + kx_grid.square()).clamp_max(1.0)
        return kz_grid, kx_grid, radial

    def forward(self, symbol: SnapshotModalSymbol) -> tuple[torch.Tensor, torch.Tensor]:
        cosine = symbol.cosine
        kz, kx, radial = self._frequency_coordinates(cosine)
        coordinates = (kz, kx, radial)
        feature_values = [
            cosine,
            symbol.confidence,
            *(value[None].expand_as(cosine) for value in coordinates),
        ]
        if self.use_context_features:
            feature_values.extend(
                symbol.context_features[:, index, None, None].expand_as(cosine)
                for index in range(int(symbol.context_features.shape[1]))
            )
        features = torch.stack(feature_values, dim=1)
        raw = self.network(features)
        radius_fraction = torch.sigmoid(
            raw[:, 0] + self.initial_radius_logit.to(dtype=raw.dtype)
        )
        radius = self.minimum_radius + (
            self.maximum_radius - self.minimum_radius
        ) * radius_fraction
        phase_scale = 1.0 + self.maximum_phase_scale_deviation * torch.tanh(raw[:, 1])
        calibrated_cosine = torch.cos(torch.acos(cosine) * phase_scale).clamp(
            min=-1.0 + self.phase_margin,
            max=1.0 - self.phase_margin,
        )
        return calibrated_cosine, radius


def estimate_snapshot_modal_symbol(
    history: torch.Tensor,
    *,
    minimum_frames: int = 4,
    confidence_floor_fraction: float = 1.0e-8,
    phase_margin: float = 1.0e-3,
    minimum_context_rms: float = 1.0e-14,
) -> SnapshotModalSymbol:
    """Estimate a unit-circle modal recurrence from consecutive snapshots.

    Low-energy modes are shrunk toward ``cos(omega dt)=1`` instead of trusting an
    ill-conditioned ratio. The returned tensors contain no future information.
    """

    value = _validate_history(history, minimum_frames=minimum_frames)
    floor_fraction = float(confidence_floor_fraction)
    margin = float(phase_margin)
    if floor_fraction <= 0.0:
        raise ValueError("confidence_floor_fraction must be positive")
    if not 0.0 < margin < 1.0:
        raise ValueError("phase_margin must lie in (0,1)")
    scale = value.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
    if bool(torch.any(scale <= float(minimum_context_rms))):
        raise ValueError("wavefield context is not energetic enough to infer propagation")
    normalized = value / scale
    frame_rms = normalized.square().mean(dim=(-2, -1)).sqrt()
    tiny = torch.finfo(value.dtype).tiny
    energy_trend = torch.tanh(
        torch.log(frame_rms[:, -1].clamp_min(tiny) / frame_rms[:, 0].clamp_min(tiny))
    )
    temporal_activity = torch.tanh(
        4.0
        * (normalized[:, 1:] - normalized[:, :-1])
        .square()
        .mean(dim=(1, 2, 3))
        .sqrt()
    )
    spectrum = torch.fft.rfft2(normalized, dim=(-2, -1), norm="ortho")
    center = spectrum[:, 1:-1]
    adjacent_sum = spectrum[:, 2:] + spectrum[:, :-2]
    energy = center.abs().square().sum(dim=1)
    numerator = (center.conj() * adjacent_sum).real.sum(dim=1)
    raw_cosine = numerator / (2.0 * energy).clamp_min(torch.finfo(energy.dtype).tiny)
    maximum = energy.amax(dim=(-2, -1), keepdim=True).clamp_min(
        torch.finfo(energy.dtype).tiny
    )
    confidence = energy / (energy + floor_fraction * maximum)
    cosine = confidence * raw_cosine + (1.0 - confidence)
    cosine = cosine.clamp(min=-1.0 + margin, max=1.0 - margin)
    return SnapshotModalSymbol(
        cosine=cosine,
        confidence=confidence,
        context_scale=scale[:, 0, 0, 0],
        context_features=torch.stack((energy_trend, temporal_activity), dim=1),
    )


class SnapshotOnlyWavePropagator(nn.Module):
    """Stable recurrent wave propagator with a snapshot-only inference API.

    The fixed modal core is useful before training. The learned closure sees only
    recent pressure frames, their temporal differences, and the fixed-core proposal.
    Its output projection is zero initialized, making the initial model exactly the
    audited modal baseline while leaving the projection gradient live.
    """

    def __init__(
        self,
        *,
        minimum_history: int = 4,
        memory_frames: int = 4,
        width: int = 48,
        spectral_rank: int = 24,
        modes: int = 16,
        depth: int = 3,
        confidence_floor_fraction: float = 1.0e-8,
        phase_margin: float = 1.0e-3,
        minimum_context_rms: float = 1.0e-14,
        maximum_correction_ratio: float = 0.25,
        activation_checkpointing: bool = True,
        local_differential_residual: bool = False,
        modal_radius: float = 1.0,
        learned_modal_calibration: bool = False,
        modal_calibrator_width: int = 16,
        minimum_modal_radius: float = 0.98,
        maximum_modal_radius: float = 0.9999,
        initial_modal_radius: float = 0.995,
        maximum_phase_scale_deviation: float = 0.02,
        modal_context_features: bool = False,
        causal_backtest_prefix: int = 0,
        observed_defect_memory: bool = False,
        defect_memory_decay: float = 0.55,
        defect_trend_scale: float = 0.75,
        defect_memory_mode: str = "polynomial",
        defect_modal_radius: float = 0.88,
        defect_high_frequency_radius: float | None = None,
        defect_high_frequency_cutoff: float = 0.75,
        minimum_defect_rms_fraction: float = 1.0e-6,
        local_wave_blend_weight: float = 0.0,
        local_wave_blend_ramp_steps: int = 8,
        local_wave_substeps: int = 8,
        local_wave_pool_size: int = 15,
        local_wave_regularization: float = 0.1,
        local_wave_minimum_coefficient: float = 0.02,
        local_wave_maximum_coefficient: float = 1.5,
        local_wave_instance_backtest: bool = False,
        local_wave_adaptive_boost: bool = False,
        local_wave_adaptive_boost_threshold: float = 0.1,
        local_wave_adaptive_boost_width: float = 0.2,
        local_wave_adaptive_boost_maximum_weight: float = 0.25,
        local_wave_adaptive_boost_start_step: int = 8,
        local_wave_adaptive_boost_ramp_steps: int = 4,
    ) -> None:
        super().__init__()
        if int(minimum_history) < 4:
            raise ValueError("minimum_history must be at least four snapshots")
        if int(memory_frames) < 2:
            raise ValueError("memory_frames must be at least two")
        if min(int(width), int(spectral_rank), int(modes), int(depth)) <= 0:
            raise ValueError("closure dimensions must be positive")
        if float(maximum_correction_ratio) <= 0.0:
            raise ValueError("maximum_correction_ratio must be positive")
        if not 0.0 < float(modal_radius) <= 1.0:
            raise ValueError("modal_radius must lie in (0,1]")
        if not 0.0 <= float(defect_memory_decay) < 1.0:
            raise ValueError("defect_memory_decay must lie in [0,1)")
        if not math.isfinite(float(defect_trend_scale)):
            raise ValueError("defect_trend_scale must be finite")
        if str(defect_memory_mode) not in ("polynomial", "stable_modal"):
            raise ValueError(
                "defect_memory_mode must be polynomial or stable_modal"
            )
        if not 0.0 < float(defect_modal_radius) <= 1.0:
            raise ValueError("defect_modal_radius must lie in (0,1]")
        if defect_high_frequency_radius is not None and not (
            0.0 < float(defect_high_frequency_radius) <= 1.0
        ):
            raise ValueError("defect_high_frequency_radius must lie in (0,1]")
        if not 0.0 < float(defect_high_frequency_cutoff) <= math.sqrt(2.0):
            raise ValueError(
                "defect_high_frequency_cutoff must lie in (0,sqrt(2)]"
            )
        if float(minimum_defect_rms_fraction) <= 0.0:
            raise ValueError("minimum_defect_rms_fraction must be positive")
        if not 0.0 <= float(local_wave_blend_weight) <= 1.0:
            raise ValueError("local_wave_blend_weight must lie in [0,1]")
        if int(local_wave_blend_ramp_steps) <= 0:
            raise ValueError("local_wave_blend_ramp_steps must be positive")
        if int(local_wave_substeps) <= 0:
            raise ValueError("local_wave_substeps must be positive")
        if int(local_wave_pool_size) <= 0 or int(local_wave_pool_size) % 2 == 0:
            raise ValueError("local_wave_pool_size must be a positive odd integer")
        if float(local_wave_regularization) <= 0.0:
            raise ValueError("local_wave_regularization must be positive")
        if not (
            0.0
            < float(local_wave_minimum_coefficient)
            <= float(local_wave_maximum_coefficient)
        ):
            raise ValueError(
                "local wave coefficients must satisfy 0 < minimum <= maximum"
            )
        if not math.isfinite(float(local_wave_adaptive_boost_threshold)):
            raise ValueError("local_wave_adaptive_boost_threshold must be finite")
        if float(local_wave_adaptive_boost_width) <= 0.0:
            raise ValueError("local_wave_adaptive_boost_width must be positive")
        if not 0.0 <= float(local_wave_adaptive_boost_maximum_weight) <= 1.0:
            raise ValueError(
                "local_wave_adaptive_boost_maximum_weight must lie in [0,1]"
            )
        if local_wave_adaptive_boost and (
            float(local_wave_adaptive_boost_maximum_weight)
            < float(local_wave_blend_weight)
        ):
            raise ValueError(
                "local_wave_adaptive_boost_maximum_weight must not be below "
                "local_wave_blend_weight when adaptive boost is enabled"
            )
        if int(local_wave_adaptive_boost_start_step) <= 0:
            raise ValueError("local_wave_adaptive_boost_start_step must be positive")
        if int(local_wave_adaptive_boost_ramp_steps) <= 0:
            raise ValueError("local_wave_adaptive_boost_ramp_steps must be positive")
        self.minimum_history = int(minimum_history)
        self.memory_frames = int(memory_frames)
        self.confidence_floor_fraction = float(confidence_floor_fraction)
        self.phase_margin = float(phase_margin)
        self.minimum_context_rms = float(minimum_context_rms)
        self.maximum_correction_ratio = float(maximum_correction_ratio)
        self.modal_radius = float(modal_radius)
        self.learned_modal_calibration = bool(learned_modal_calibration)
        self.causal_backtest_prefix = int(causal_backtest_prefix)
        self.observed_defect_memory = bool(observed_defect_memory)
        self.defect_memory_decay = float(defect_memory_decay)
        self.defect_trend_scale = float(defect_trend_scale)
        self.defect_memory_mode = str(defect_memory_mode)
        self.defect_modal_radius = float(defect_modal_radius)
        self.defect_high_frequency_radius = (
            None
            if defect_high_frequency_radius is None
            else float(defect_high_frequency_radius)
        )
        self.defect_high_frequency_cutoff = float(defect_high_frequency_cutoff)
        self.minimum_defect_rms_fraction = float(minimum_defect_rms_fraction)
        self.local_wave_blend_weight = float(local_wave_blend_weight)
        self.local_wave_blend_ramp_steps = int(local_wave_blend_ramp_steps)
        self.local_wave_substeps = int(local_wave_substeps)
        self.local_wave_pool_size = int(local_wave_pool_size)
        self.local_wave_regularization = float(local_wave_regularization)
        self.local_wave_minimum_coefficient = float(
            local_wave_minimum_coefficient
        )
        self.local_wave_maximum_coefficient = float(
            local_wave_maximum_coefficient
        )
        self.local_wave_instance_backtest = bool(local_wave_instance_backtest)
        self.local_wave_adaptive_boost = bool(local_wave_adaptive_boost)
        self.local_wave_adaptive_boost_threshold = float(
            local_wave_adaptive_boost_threshold
        )
        self.local_wave_adaptive_boost_width = float(
            local_wave_adaptive_boost_width
        )
        self.local_wave_adaptive_boost_maximum_weight = float(
            local_wave_adaptive_boost_maximum_weight
        )
        self.local_wave_adaptive_boost_start_step = int(
            local_wave_adaptive_boost_start_step
        )
        self.local_wave_adaptive_boost_ramp_steps = int(
            local_wave_adaptive_boost_ramp_steps
        )
        if self.local_wave_instance_backtest and not (
            5 <= self.causal_backtest_prefix <= self.minimum_history - 2
        ):
            raise ValueError(
                "local wave instance backtest requires a causal prefix of at least "
                "five frames and at least two held-out history frames"
            )
        if self.local_wave_adaptive_boost and not self.local_wave_instance_backtest:
            raise ValueError(
                "local wave adaptive boost requires local wave instance backtest"
            )
        second_derivative = torch.tensor(
            (
                -1.0 / 560.0,
                8.0 / 315.0,
                -1.0 / 5.0,
                8.0 / 5.0,
                -205.0 / 72.0,
                8.0 / 5.0,
                -1.0 / 5.0,
                8.0 / 315.0,
                -1.0 / 560.0,
            ),
            dtype=torch.float32,
        )
        local_wave_stencil = torch.zeros((1, 1, 9, 9), dtype=torch.float32)
        local_wave_stencil[0, 0, 4, :] = second_derivative
        local_wave_stencil[0, 0, :, 4] += second_derivative
        self.register_buffer(
            "_local_wave_stencil", local_wave_stencil, persistent=False
        )
        if self.causal_backtest_prefix and not (
            4 <= self.causal_backtest_prefix < self.minimum_history
        ):
            raise ValueError(
                "causal_backtest_prefix must be zero or lie in [4, minimum_history)"
            )
        if self.causal_backtest_prefix and not self.learned_modal_calibration:
            raise ValueError("causal backtesting requires learned modal calibration")

        feature_channels = 2 * self.memory_frames + 1
        self.lift = nn.Sequential(
            nn.Conv2d(feature_channels, int(width), kernel_size=3, padding=1),
            _group_norm(int(width)),
            nn.GELU(),
        )
        self.closure = FactorizedComplexResidualStack(
            width=int(width),
            spectral_rank=int(spectral_rank),
            modes=int(modes),
            depth=int(depth),
            activation_checkpointing=bool(activation_checkpointing),
            local_differential_residual=bool(local_differential_residual),
        )
        self.project = nn.Conv2d(int(width), 1, kernel_size=1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        self.correction_gate = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        if self.learned_modal_calibration:
            self.modal_calibrator: StableModalCalibrator | None = StableModalCalibrator(
                width=int(modal_calibrator_width),
                minimum_radius=float(minimum_modal_radius),
                maximum_radius=float(maximum_modal_radius),
                initial_radius=float(initial_modal_radius),
                maximum_phase_scale_deviation=float(maximum_phase_scale_deviation),
                phase_margin=self.phase_margin,
                use_context_features=bool(modal_context_features),
            )
        else:
            self.modal_calibrator = None

    def infer_symbol(self, history: torch.Tensor) -> SnapshotModalSymbol:
        return estimate_snapshot_modal_symbol(
            history,
            minimum_frames=self.minimum_history,
            confidence_floor_fraction=self.confidence_floor_fraction,
            phase_margin=self.phase_margin,
            minimum_context_rms=self.minimum_context_rms,
        )

    def _recent(self, frames: list[torch.Tensor]) -> torch.Tensor:
        recent = frames[-self.memory_frames :]
        if len(recent) < self.memory_frames:
            recent = [recent[0]] * (self.memory_frames - len(recent)) + recent
        return torch.stack(recent, dim=1)

    def _closure_features(
        self,
        frames: list[torch.Tensor],
        proposal: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        recent = self._recent(frames) / scale[:, None, None, None]
        differences = recent[:, 1:] - recent[:, :-1]
        proposal_n = proposal / scale[:, None, None]
        acceleration = proposal_n - 2.0 * recent[:, -1] + recent[:, -2]
        return torch.cat(
            (recent, differences, proposal_n[:, None], acceleration[:, None]), dim=1
        )

    @staticmethod
    def _modal_step(
        previous_spectrum: torch.Tensor,
        current_spectrum: torch.Tensor,
        cosine: torch.Tensor,
        radius: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        radius_squared = radius.square() if torch.is_tensor(radius) else float(radius) ** 2
        return (
            2.0 * radius * cosine * current_spectrum
            - radius_squared * previous_spectrum
        )

    @staticmethod
    def _high_frequency_modal_radius(
        spatial_shape: tuple[int, int],
        *,
        base_radius: float,
        high_frequency_radius: float,
        cutoff: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a fixed radial rFFT damping map on normalized Nyquist axes.

        The map is independent of the medium, source, targets, and future wavefield.
        It is applied only after the first forecast defect step, preserving the
        audited one-step causal correction.
        """

        height, width = (int(value) for value in spatial_shape)
        vertical = torch.fft.fftfreq(height, device=device, dtype=dtype)
        horizontal = torch.fft.rfftfreq(width, device=device, dtype=dtype)
        epsilon = torch.finfo(dtype).eps
        vertical = vertical / vertical.abs().amax().clamp_min(epsilon)
        horizontal = horizontal / horizontal.abs().amax().clamp_min(epsilon)
        radial = torch.sqrt(vertical[:, None].square() + horizontal[None].square())
        return torch.where(
            radial <= float(cutoff),
            torch.as_tensor(base_radius, dtype=dtype, device=device),
            torch.as_tensor(high_frequency_radius, dtype=dtype, device=device),
        )

    def _local_wave_laplacian(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the generator's radius-four spatial stencil without grid metadata."""

        if value.ndim != 3:
            raise ValueError("local wavefield must be [batch,z,x]")
        kernel = self._local_wave_stencil.to(dtype=value.dtype, device=value.device)
        return F.conv2d(
            F.pad(value[:, None], (4, 4, 4, 4)), kernel
        )[:, 0]

    def _snapshot_local_wave_rollout(
        self,
        history: torch.Tensor,
        *,
        steps: int,
        minimum_frames: int | None = None,
    ) -> torch.Tensor:
        """Infer and roll out a stable local wave recurrence from snapshots only.

        The effective squared-Courant field is fitted from observed temporal
        accelerations and radius-four Laplacians.  A pooled ridge estimate falls
        back to the record-level coefficient outside the illuminated wavefront.
        Substepping keeps the explicit recurrence stable despite the coarse saved
        snapshot interval.  No velocity model, source, physical grid metadata, or
        future field is used.
        """

        value = _validate_history(
            history,
            minimum_frames=(
                self.minimum_history
                if minimum_frames is None
                else int(minimum_frames)
            ),
        )
        if min(int(value.shape[-2]), int(value.shape[-1])) < 9:
            raise ValueError("local wave branch requires spatial dimensions >= 9")
        middle = value[:, 1:-1]
        laplacian = self._local_wave_laplacian(
            middle.reshape(-1, int(value.shape[-2]), int(value.shape[-1]))
        ).reshape_as(middle)
        acceleration = value[:, 2:] - 2.0 * middle + value[:, :-2]
        numerator = (laplacian * acceleration).sum(dim=1)
        denominator = laplacian.square().sum(dim=1)
        epsilon = torch.finfo(value.dtype).tiny
        global_coefficient = (
            numerator.sum(dim=(-2, -1))
            / denominator.sum(dim=(-2, -1)).clamp_min(epsilon)
        ).clamp(
            self.local_wave_minimum_coefficient,
            self.local_wave_maximum_coefficient,
        )
        pool_size = min(
            self.local_wave_pool_size,
            int(value.shape[-2]),
            int(value.shape[-1]),
        )
        if pool_size % 2 == 0:
            pool_size -= 1
        numerator = F.avg_pool2d(
            numerator[:, None], pool_size, stride=1, padding=pool_size // 2
        )[:, 0]
        denominator = F.avg_pool2d(
            denominator[:, None], pool_size, stride=1, padding=pool_size // 2
        )[:, 0]
        ridge_scale = denominator.mean(dim=(-2, -1), keepdim=True)
        ridge = self.local_wave_regularization * ridge_scale
        coefficient = (
            (numerator + ridge * global_coefficient[:, None, None])
            / (denominator + ridge).clamp_min(epsilon)
        ).clamp(
            self.local_wave_minimum_coefficient,
            self.local_wave_maximum_coefficient,
        )

        substeps = self.local_wave_substeps
        current = value[:, -1]
        acceleration_now = coefficient * self._local_wave_laplacian(current)
        coarse_velocity = current - value[:, -2] + 0.5 * acceleration_now
        previous = (
            current
            - coarse_velocity / float(substeps)
            + 0.5 * acceleration_now / float(substeps * substeps)
        )
        substep_coefficient = coefficient / float(substeps * substeps)
        outputs: list[torch.Tensor] = []
        for _ in range(int(steps)):
            for _ in range(substeps):
                following = (
                    2.0 * current
                    - previous
                    + substep_coefficient * self._local_wave_laplacian(current)
                )
                previous, current = current, following
            outputs.append(current)
        return torch.stack(outputs, dim=1)

    def _local_wave_backtest_score(self, history: torch.Tensor) -> torch.Tensor:
        """Fit an unconstrained blend score on held-out input-history frames."""

        value = _validate_history(history, minimum_frames=self.minimum_history)
        prefix_count = self.causal_backtest_prefix
        prefix = value[:, :prefix_count]
        held_out = value[:, prefix_count:]
        prefix_symbol = estimate_snapshot_modal_symbol(
            prefix,
            minimum_frames=4,
            confidence_floor_fraction=self.confidence_floor_fraction,
            phase_margin=self.phase_margin,
            minimum_context_rms=self.minimum_context_rms,
        )
        fixed = self._modal_sequence(
            prefix,
            cosine=prefix_symbol.cosine,
            radius=self.modal_radius,
            steps=int(held_out.shape[1]),
        )
        if self.modal_calibrator is None:
            modal = fixed
        else:
            calibrated_cosine, calibrated_radius = self.modal_calibrator(
                prefix_symbol
            )
            calibrated = self._modal_sequence(
                prefix,
                cosine=calibrated_cosine,
                radius=calibrated_radius,
                steps=int(held_out.shape[1]),
            )
            fixed_error = (fixed - held_out).double().square().sum(
                dim=(1, 2, 3)
            )
            calibrated_error = (calibrated - held_out).double().square().sum(
                dim=(1, 2, 3)
            )
            modal = torch.where(
                (calibrated_error < fixed_error)[:, None, None, None],
                calibrated,
                fixed,
            )
        local = self._snapshot_local_wave_rollout(
            prefix,
            steps=int(held_out.shape[1]),
            minimum_frames=5,
        )
        direction = (local - modal).double()
        residual = (held_out - modal).double()
        numerator = (direction * residual).sum(dim=(1, 2, 3))
        denominator = direction.square().sum(dim=(1, 2, 3)).clamp_min(1.0e-30)
        return (numerator / denominator).to(dtype=value.dtype).detach()

    def _local_wave_backtest_weight(self, history: torch.Tensor) -> torch.Tensor:
        """Fit a bounded blend weight on held-out frames inside the input history."""

        return self._local_wave_backtest_score(history).clamp(
            0.0, self.local_wave_blend_weight
        )

    def _local_wave_blend_fraction(
        self,
        rollout_index: int,
        maximum_weight: torch.Tensor | float | None = None,
        backtest_score: torch.Tensor | None = None,
    ) -> torch.Tensor | float:
        weight = (
            self.local_wave_blend_weight
            if maximum_weight is None
            else maximum_weight
        )
        ramp_fraction = min(
            1.0,
            float(rollout_index) / float(self.local_wave_blend_ramp_steps),
        )
        baseline_weight = weight * ramp_fraction
        if (
            not self.local_wave_adaptive_boost
            or backtest_score is None
            or int(rollout_index) < self.local_wave_adaptive_boost_start_step
        ):
            return baseline_weight
        gate = (
            (backtest_score - self.local_wave_adaptive_boost_threshold)
            / self.local_wave_adaptive_boost_width
        ).clamp(0.0, 1.0)
        target_weight = weight + gate * (
            self.local_wave_adaptive_boost_maximum_weight - weight
        )
        late_fraction = min(
            1.0,
            float(
                int(rollout_index)
                - self.local_wave_adaptive_boost_start_step
                + 1
            )
            / float(self.local_wave_adaptive_boost_ramp_steps),
        )
        return baseline_weight + late_fraction * (target_weight - baseline_weight)

    @classmethod
    def _modal_sequence(
        cls,
        history: torch.Tensor,
        *,
        cosine: torch.Tensor,
        radius: torch.Tensor | float,
        steps: int,
    ) -> torch.Tensor:
        spatial_shape = (int(history.shape[-2]), int(history.shape[-1]))
        previous = torch.fft.rfft2(history[:, -2], norm="ortho")
        current = torch.fft.rfft2(history[:, -1], norm="ortho")
        outputs = []
        for _ in range(int(steps)):
            next_spectrum = cls._modal_step(previous, current, cosine, radius)
            outputs.append(
                torch.fft.irfft2(next_spectrum, s=spatial_shape, norm="ortho")
            )
            previous, current = current, next_spectrum
        return torch.stack(outputs, dim=1)

    @classmethod
    def _observed_modal_defects(
        cls,
        history: torch.Tensor,
        *,
        cosine: torch.Tensor,
        radius: torch.Tensor | float,
    ) -> torch.Tensor:
        """Return one-step modal errors measured only on observed snapshots."""

        spatial_shape = (int(history.shape[-2]), int(history.shape[-1]))
        previous = torch.fft.rfft2(history[:, 0], norm="ortho")
        current = torch.fft.rfft2(history[:, 1], norm="ortho")
        defects: list[torch.Tensor] = []
        for target_index in range(2, int(history.shape[1])):
            proposal_spectrum = cls._modal_step(previous, current, cosine, radius)
            proposal = torch.fft.irfft2(
                proposal_spectrum, s=spatial_shape, norm="ortho"
            )
            defects.append(history[:, target_index] - proposal)
            previous = current
            current = torch.fft.rfft2(
                history[:, target_index], norm="ortho"
            )
        return torch.stack(defects, dim=1)

    def causal_backtest_decision(self, wavefield_history: torch.Tensor) -> torch.Tensor:
        """Select calibration using only a prefix/held-out split inside the history."""

        value = _validate_history(
            wavefield_history, minimum_frames=self.minimum_history
        )
        if not self.causal_backtest_prefix:
            return torch.ones(
                int(value.shape[0]), dtype=torch.bool, device=value.device
            )
        assert self.modal_calibrator is not None
        prefix = value[:, : self.causal_backtest_prefix]
        held_out = value[:, self.causal_backtest_prefix :]
        prefix_symbol = estimate_snapshot_modal_symbol(
            prefix,
            minimum_frames=4,
            confidence_floor_fraction=self.confidence_floor_fraction,
            phase_margin=self.phase_margin,
            minimum_context_rms=self.minimum_context_rms,
        )
        calibrated_cosine, calibrated_radius = self.modal_calibrator(prefix_symbol)
        fixed = self._modal_sequence(
            prefix,
            cosine=prefix_symbol.cosine,
            radius=self.modal_radius,
            steps=int(held_out.shape[1]),
        )
        calibrated = self._modal_sequence(
            prefix,
            cosine=calibrated_cosine,
            radius=calibrated_radius,
            steps=int(held_out.shape[1]),
        )
        fixed_error = (fixed - held_out).double().square().sum(dim=(1, 2, 3))
        calibrated_error = (
            (calibrated - held_out).double().square().sum(dim=(1, 2, 3))
        )
        return (calibrated_error < fixed_error).detach()

    def _rollout(
        self,
        history: torch.Tensor,
        *,
        steps: int,
        learned_closure: bool,
        modal_calibration: bool = True,
        defect_memory: bool = True,
    ) -> torch.Tensor:
        value = _validate_history(history, minimum_frames=self.minimum_history)
        if int(steps) <= 0:
            raise ValueError("steps must be positive")
        local_wave_prediction = (
            self._snapshot_local_wave_rollout(value, steps=int(steps))
            if self.local_wave_blend_weight > 0.0
            else None
        )
        local_wave_backtest_score = (
            self._local_wave_backtest_score(value)
            if local_wave_prediction is not None
            and self.local_wave_instance_backtest
            else None
        )
        local_wave_maximum_weight: torch.Tensor | float = (
            local_wave_backtest_score.clamp(0.0, self.local_wave_blend_weight)
            if local_wave_backtest_score is not None
            else self.local_wave_blend_weight
        )
        symbol = self.infer_symbol(value)
        if self.modal_calibrator is None or not modal_calibration:
            modal_cosine = symbol.cosine
            modal_radius: torch.Tensor | float = self.modal_radius
        else:
            calibrated_cosine, calibrated_radius = self.modal_calibrator(symbol)
            if self.causal_backtest_prefix:
                decision = self.causal_backtest_decision(value)[:, None, None]
                modal_cosine = torch.where(
                    decision, calibrated_cosine, symbol.cosine
                )
                modal_radius = torch.where(
                    decision,
                    calibrated_radius,
                    torch.full_like(calibrated_radius, self.modal_radius),
                )
            else:
                modal_cosine, modal_radius = calibrated_cosine, calibrated_radius
        last_defect: torch.Tensor | None = None
        defect_trend: torch.Tensor | None = None
        defect_active: torch.Tensor | None = None
        defect_cosine: torch.Tensor | None = None
        previous_defect_spectrum: torch.Tensor | None = None
        current_defect_spectrum: torch.Tensor | None = None
        defect_radius_map: torch.Tensor | None = None
        if self.observed_defect_memory and defect_memory:
            observed_defects = self._observed_modal_defects(
                value,
                cosine=modal_cosine,
                radius=modal_radius,
            )
            last_defect = observed_defects[:, -1]
            defect_trend = last_defect - observed_defects[:, -2]
            if self.defect_memory_mode == "stable_modal":
                defect_scale = observed_defects.square().mean(
                    dim=(1, 2, 3)
                ).sqrt()
                defect_active = defect_scale > (
                    self.minimum_defect_rms_fraction * symbol.context_scale
                )
                safe_defects = torch.where(
                    defect_active[:, None, None, None],
                    observed_defects,
                    value[:, -int(observed_defects.shape[1]) :],
                )
                defect_symbol = estimate_snapshot_modal_symbol(
                    safe_defects,
                    minimum_frames=4,
                    confidence_floor_fraction=self.confidence_floor_fraction,
                    phase_margin=self.phase_margin,
                    minimum_context_rms=self.minimum_context_rms,
                )
                defect_cosine = defect_symbol.cosine
                previous_defect_spectrum = torch.fft.rfft2(
                    observed_defects[:, -2], norm="ortho"
                )
                current_defect_spectrum = torch.fft.rfft2(
                    observed_defects[:, -1], norm="ortho"
                )
                if self.defect_high_frequency_radius is not None:
                    defect_radius_map = self._high_frequency_modal_radius(
                        (int(value.shape[-2]), int(value.shape[-1])),
                        base_radius=self.defect_modal_radius,
                        high_frequency_radius=self.defect_high_frequency_radius,
                        cutoff=self.defect_high_frequency_cutoff,
                        dtype=value.dtype,
                        device=value.device,
                    )
        spatial_shape = (int(value.shape[-2]), int(value.shape[-1]))
        frames = [value[:, index] for index in range(int(value.shape[1]))]
        previous_spectrum = torch.fft.rfft2(frames[-2], norm="ortho")
        current_spectrum = torch.fft.rfft2(frames[-1], norm="ortho")
        outputs: list[torch.Tensor] = []
        fixed_scale = symbol.context_scale.to(dtype=value.dtype, device=value.device)
        closure_active = bool(learned_closure)
        if closure_active and not self.training:
            closure_active = bool(torch.count_nonzero(self.project.weight)) or bool(
                torch.count_nonzero(self.project.bias)
            )
        for rollout_index in range(int(steps)):
            proposal_spectrum = self._modal_step(
                previous_spectrum, current_spectrum, modal_cosine, modal_radius
            )
            proposal = torch.fft.irfft2(
                proposal_spectrum, s=spatial_shape, norm="ortho"
            )
            if last_defect is not None and defect_trend is not None:
                if self.defect_memory_mode == "stable_modal":
                    assert defect_active is not None
                    assert defect_cosine is not None
                    assert previous_defect_spectrum is not None
                    assert current_defect_spectrum is not None
                    defect_radius: torch.Tensor | float = self.defect_modal_radius
                    if rollout_index > 0 and defect_radius_map is not None:
                        defect_radius = defect_radius_map
                    next_defect_spectrum = self._modal_step(
                        previous_defect_spectrum,
                        current_defect_spectrum,
                        defect_cosine,
                        defect_radius,
                    )
                    defect_correction = torch.fft.irfft2(
                        next_defect_spectrum, s=spatial_shape, norm="ortho"
                    ) * defect_active[:, None, None]
                    previous_defect_spectrum = current_defect_spectrum
                    current_defect_spectrum = next_defect_spectrum
                else:
                    defect_correction = (
                        self.defect_memory_decay ** rollout_index
                    ) * (
                        last_defect
                        + (rollout_index + 1)
                        * self.defect_trend_scale
                        * defect_trend
                    )
                proposal = proposal + defect_correction
            if closure_active:
                features = self._closure_features(frames, proposal, fixed_scale)
                correction = torch.tanh(
                    self.project(self.closure(self.lift(features)))
                )[:, 0]
                correction = (
                    torch.tanh(self.correction_gate)
                    * self.maximum_correction_ratio
                    * fixed_scale[:, None, None]
                    * correction
                )
                next_frame = proposal + correction
            else:
                next_frame = proposal
            frames.append(next_frame)
            previous_spectrum = current_spectrum
            current_spectrum = torch.fft.rfft2(next_frame, norm="ortho")
            if local_wave_prediction is None:
                output_frame = next_frame
            else:
                blend = self._local_wave_blend_fraction(
                    rollout_index,
                    local_wave_maximum_weight,
                    local_wave_backtest_score,
                )
                if torch.is_tensor(blend):
                    blend = blend[:, None, None]
                output_frame = next_frame + blend * (
                    local_wave_prediction[:, rollout_index] - next_frame
                )
            outputs.append(output_frame)
        return torch.stack(outputs, dim=1)

    def forward(self, wavefield_history: torch.Tensor, steps: int) -> torch.Tensor:
        """Predict future pressure using only consecutive observed snapshots."""

        return self._rollout(
            wavefield_history,
            steps=int(steps),
            learned_closure=True,
            modal_calibration=True,
            defect_memory=True,
        )

    def calibrated_modal_rollout(
        self, wavefield_history: torch.Tensor, steps: int
    ) -> torch.Tensor:
        """Differentiable stable modal rollout used for calibration-only training."""

        return self._rollout(
            wavefield_history,
            steps=int(steps),
            learned_closure=False,
            modal_calibration=True,
            defect_memory=True,
        )

    @torch.no_grad()
    def modal_baseline(self, wavefield_history: torch.Tensor, steps: int) -> torch.Tensor:
        """Return the fixed stable snapshot recurrence without neural correction."""

        return self._rollout(
            wavefield_history,
            steps=int(steps),
            learned_closure=False,
            modal_calibration=False,
            defect_memory=False,
        )


@torch.no_grad()
def transfer_pretrained_decoder_stack(
    model: SnapshotOnlyWavePropagator,
    parent_state: Mapping[str, torch.Tensor],
    *,
    prefix: str = "dense_decoder.stack.",
) -> dict[str, object]:
    """Warm-start the closure filters from a compatible pretrained A3 decoder.

    The snapshot lift and zero output projection remain untouched, so the transfer
    preserves the exact stable modal baseline at initialization. Only weights are
    copied; no parent optimizer state or runtime medium/source input is inherited.
    """

    candidate = model.closure.state_dict()
    transferred: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    mismatched: list[str] = []
    for name, target in candidate.items():
        parent_name = f"{prefix}{name}"
        if parent_name not in parent_state:
            missing.append(parent_name)
            continue
        source = torch.as_tensor(parent_state[parent_name])
        if source.shape != target.shape:
            mismatched.append(
                f"{parent_name}:{tuple(source.shape)}!={tuple(target.shape)}"
            )
            continue
        transferred[name] = source.to(dtype=target.dtype, device=target.device)
    if missing or mismatched or len(transferred) != len(candidate):
        raise ValueError(
            "pretrained decoder stack is incompatible with snapshot closure: "
            f"missing={missing[:3]}, mismatched={mismatched[:3]}, "
            f"copied={len(transferred)}/{len(candidate)}"
        )
    model.closure.load_state_dict(transferred, strict=True)
    if bool(torch.count_nonzero(model.project.weight)) or bool(
        torch.count_nonzero(model.project.bias)
    ):
        raise RuntimeError("snapshot output projection must remain zero after transfer")
    return {
        "source_prefix": str(prefix),
        "transferred_tensors": len(transferred),
        "transferred_parameters": int(sum(value.numel() for value in transferred.values())),
        "modal_warmstart_preserved": True,
    }


__all__ = [
    "SnapshotModalSymbol",
    "StableModalCalibrator",
    "SnapshotOnlyWavePropagator",
    "estimate_snapshot_modal_symbol",
    "transfer_pretrained_decoder_stack",
]
