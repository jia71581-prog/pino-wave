"""Meta-learned causal error bases with a small convex deployment solve.

The shared basis generator is trained offline.  At deployment the pretrained
operator and basis generator are frozen; only the coefficients of a field-
linear correction are solved.  Because the acoustic LWC-84 defect is affine in
the pressure field, the online problem is a regularized weighted least-squares
system rather than an iterative neural-network fine-tune.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .forced_defect import (
    LAPLACIAN8_COEFFICIENTS,
    SavedGridCPMLConfig,
    lwc84_discrete_defect,
)
from .losses import PhysicsPointSet, sample_fixed_physics_residual


CPADC_SCHEMA_VERSION = 10


def causal_smoothstep_envelope(
    time_count: int,
    observed_indices: tuple[int, int],
    *,
    ramp_steps: int = 4,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return a C1 onset envelope that is zero through the last observed frame."""

    count = int(time_count)
    ramp = int(ramp_steps)
    if count <= 0 or ramp <= 0:
        raise ValueError("time_count and ramp_steps must be positive")
    last_observed = int(observed_indices[1])
    if last_observed < 0 or last_observed >= count:
        raise ValueError("the last observed index is outside the time axis")
    index = torch.arange(count, dtype=dtype, device=device)
    position = ((index - float(last_observed)) / float(ramp)).clamp(0.0, 1.0)
    return position.square() * (3.0 - 2.0 * position)


@dataclass(frozen=True)
class FactorizedCausalBasis:
    """A memory-efficient linear correction basis.

    The first ``rank - phase_rank`` modes are separable amplitude/scattering
    modes.  The final ``phase_rank`` modes multiply a normalized temporal
    derivative of the parent field and therefore provide explicit first-order
    arrival-time correction directions.
    """

    spatial_modes: torch.Tensor  # [record,rank,z,x]
    temporal_modes: torch.Tensor  # [record,rank,time]
    causal_envelope: torch.Tensor  # [record,time]
    field_scale: torch.Tensor  # [record]
    phase_reference: torch.Tensor  # [record,time,z,x], normalized
    trust_fraction: torch.Tensor  # [record], learned fraction of the hard cap
    phase_rank: int

    def __post_init__(self) -> None:
        spatial = self.spatial_modes
        temporal = self.temporal_modes
        envelope = self.causal_envelope
        scale = self.field_scale
        phase = self.phase_reference
        trust = self.trust_fraction
        if spatial.ndim != 4 or temporal.ndim != 3:
            raise ValueError("basis factors must be [record,rank,z,x] and [record,rank,time]")
        if spatial.shape[:2] != temporal.shape[:2]:
            raise ValueError("spatial and temporal basis ranks do not match")
        records, rank, height, width = spatial.shape
        if envelope.shape != (records, temporal.shape[2]):
            raise ValueError("causal envelope shape does not match the basis")
        if scale.shape != (records,):
            raise ValueError("field scale must contain one value per record")
        if trust.shape != (records,):
            raise ValueError("trust fraction must contain one value per record")
        if phase.shape != (records, temporal.shape[2], height, width):
            raise ValueError("phase reference shape does not match the basis")
        if int(self.phase_rank) < 0 or int(self.phase_rank) > rank:
            raise ValueError("phase_rank must lie within the basis rank")
        tensors = (spatial, temporal, envelope, scale, phase, trust)
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("basis factors must be finite")
        if bool(torch.any(scale <= 0.0)):
            raise ValueError("field scales must be positive")
        if bool(torch.any((trust <= 0.0) | (trust > 1.0))):
            raise ValueError("trust fractions must lie in (0,1]")

    @property
    def rank(self) -> int:
        return int(self.spatial_modes.shape[1])

    @property
    def time_count(self) -> int:
        return int(self.temporal_modes.shape[2])

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return int(self.spatial_modes.shape[2]), int(self.spatial_modes.shape[3])

    def detached(self) -> "FactorizedCausalBasis":
        """Return factors detached for a memory-efficient first-order inner solve."""

        return FactorizedCausalBasis(
            spatial_modes=self.spatial_modes.detach(),
            temporal_modes=self.temporal_modes.detach(),
            causal_envelope=self.causal_envelope.detach(),
            field_scale=self.field_scale.detach(),
            phase_reference=self.phase_reference.detach(),
            trust_fraction=self.trust_fraction.detach(),
            phase_rank=self.phase_rank,
        )

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> "FactorizedCausalBasis":
        """Move every frozen basis factor to one explicit adaptation device."""

        target = torch.device(device)

        def move(value: torch.Tensor) -> torch.Tensor:
            return value.to(
                device=target,
                dtype=value.dtype if dtype is None else dtype,
            )

        return FactorizedCausalBasis(
            spatial_modes=move(self.spatial_modes),
            temporal_modes=move(self.temporal_modes),
            causal_envelope=move(self.causal_envelope),
            field_scale=move(self.field_scale),
            phase_reference=move(self.phase_reference),
            trust_fraction=move(self.trust_fraction),
            phase_rank=self.phase_rank,
        )

    def _indices(self, time_indices: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if time_indices is None:
            return torch.arange(
                self.time_count,
                dtype=torch.long,
                device=self.temporal_modes.device,
            )
        result = torch.as_tensor(
            time_indices, dtype=torch.long, device=self.temporal_modes.device
        ).flatten()
        if bool(((result < 0) | (result >= self.time_count)).any()):
            raise ValueError("basis time index is outside the saved axis")
        return result

    def materialize(
        self,
        *,
        rank_start: int = 0,
        rank_stop: int | None = None,
        time_indices: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Materialize a rank/time block as ``[record,rank,time,z,x]``."""

        start = int(rank_start)
        stop = self.rank if rank_stop is None else int(rank_stop)
        if start < 0 or stop <= start or stop > self.rank:
            raise ValueError("invalid basis rank slice")
        indices = self._indices(time_indices)
        spatial = self.spatial_modes[:, start:stop]
        temporal = self.temporal_modes[:, start:stop, indices]
        values = spatial[:, :, None] * temporal[:, :, :, None, None]
        phase_start = self.rank - int(self.phase_rank)
        local_phase_start = max(phase_start, start) - start
        local_phase_stop = stop - start if stop > phase_start else local_phase_start
        if local_phase_stop > local_phase_start:
            phase_values = values[:, local_phase_start:local_phase_stop]
            phase_values = phase_values * self.phase_reference[:, None, indices]
            values = torch.cat(
                (
                    values[:, :local_phase_start],
                    phase_values,
                    values[:, local_phase_stop:],
                ),
                dim=1,
            )
        return (
            values
            * self.causal_envelope[:, None, indices, None, None]
            * self.field_scale[:, None, None, None, None]
        )

    def combine(
        self,
        coefficients: torch.Tensor,
        *,
        time_indices: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply coefficients without materializing the rank dimension."""

        coefficient = torch.as_tensor(
            coefficients,
            dtype=self.spatial_modes.dtype,
            device=self.spatial_modes.device,
        )
        if coefficient.shape != self.spatial_modes.shape[:2]:
            raise ValueError("coefficients must be [record,rank]")
        indices = self._indices(time_indices)
        amplitude_rank = self.rank - int(self.phase_rank)
        height, width = self.spatial_shape
        result = torch.zeros(
            (coefficient.shape[0], len(indices), height, width),
            dtype=self.spatial_modes.dtype,
            device=self.spatial_modes.device,
        )
        if amplitude_rank:
            result = result + torch.einsum(
                "rk,rkt,rkzx->rtzx",
                coefficient[:, :amplitude_rank],
                self.temporal_modes[:, :amplitude_rank, indices],
                self.spatial_modes[:, :amplitude_rank],
            )
        if self.phase_rank:
            phase = torch.einsum(
                "rk,rkt,rkzx->rtzx",
                coefficient[:, amplitude_rank:],
                self.temporal_modes[:, amplitude_rank:, indices],
                self.spatial_modes[:, amplitude_rank:],
            )
            result = result + phase * self.phase_reference[:, indices]
        return (
            result
            * self.causal_envelope[:, indices, None, None]
            * self.field_scale[:, None, None, None]
        )


def causal_observation_probe_basis(
    causal_basis: FactorizedCausalBasis,
) -> FactorizedCausalBasis:
    """Expose causal modes at observed frames only for coefficient fitting.

    The returned basis is a design-matrix probe.  Deployment must continue to
    combine coefficients with ``causal_basis`` so the correction stays exactly
    zero through the final true observation.
    """

    return FactorizedCausalBasis(
        spatial_modes=causal_basis.spatial_modes,
        temporal_modes=causal_basis.temporal_modes,
        causal_envelope=torch.ones_like(causal_basis.causal_envelope),
        field_scale=causal_basis.field_scale,
        phase_reference=causal_basis.phase_reference,
        trust_fraction=causal_basis.trust_fraction,
        phase_rank=causal_basis.phase_rank,
    )


def _normalization_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if int(channels) % groups == 0:
            return groups
    return 1


class CausalErrorBasisGenerator(nn.Module):
    """Generate amplitude and phase error modes from causal deployment inputs."""

    def __init__(
        self,
        *,
        rank: int = 64,
        phase_rank: int = 16,
        width: int = 32,
        ramp_steps: int = 4,
    ) -> None:
        super().__init__()
        if int(rank) <= 0 or not 0 <= int(phase_rank) <= int(rank):
            raise ValueError("basis rank configuration is invalid")
        if int(width) <= 0 or int(ramp_steps) <= 0:
            raise ValueError("basis width and causal ramp must be positive")
        self.rank = int(rank)
        self.phase_rank = int(phase_rank)
        self.width = int(width)
        self.ramp_steps = int(ramp_steps)
        groups = _normalization_groups(self.width)
        # velocity, two observations, two parent-observation mismatches,
        # aligned defect RMS, and parent-field RMS.
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(7, self.width, 5, padding=2),
            nn.GroupNorm(groups, self.width),
            nn.GELU(),
            nn.Conv2d(self.width, self.width, 3, padding=1),
            nn.GroupNorm(groups, self.width),
            nn.GELU(),
            nn.Conv2d(self.width, self.rank, 1),
        )
        # normalized time, parent RMS, aligned defect RMS, Ricker wavelet,
        # and normalized causal distance from the final observation.
        self.temporal_encoder = nn.Sequential(
            nn.Linear(5, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.rank),
        )
        self.trust_head = nn.Sequential(
            nn.Linear(12, self.width),
            nn.GELU(),
            nn.Linear(self.width, 1),
        )
        nn.init.zeros_(self.trust_head[-1].weight)
        nn.init.constant_(self.trust_head[-1].bias, math.log(0.2 / 0.8))

    @staticmethod
    def _field_scale(parent: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        parent_rms = parent.detach().float().square().flatten(1).mean(dim=1).sqrt()
        observed_rms = observed.detach().float().square().flatten(1).mean(dim=1).sqrt()
        return torch.maximum(parent_rms, observed_rms).clamp_min(1.0e-8).to(parent.dtype)

    @staticmethod
    def _phase_reference(parent: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        phase = torch.zeros_like(parent)
        phase[:, 1:-1] = 0.5 * (parent[:, 2:] - parent[:, :-2])
        phase[:, 0] = parent[:, 1] - parent[:, 0]
        phase[:, -1] = parent[:, -1] - parent[:, -2]
        return phase / scale[:, None, None, None]

    @staticmethod
    def _ricker_feature(
        source_parameters: torch.Tensor,
        time_s: torch.Tensor,
    ) -> torch.Tensor:
        parameters = torch.as_tensor(
            source_parameters, dtype=time_s.dtype, device=time_s.device
        )
        if parameters.ndim != 2 or parameters.shape[1] != 5:
            raise ValueError("source parameters must be [record,5]")
        frequency = parameters[:, 2:3]
        onset = parameters[:, 3:4]
        tau = time_s - onset
        value = math.pi**2 * frequency.square()
        return (1.0 - 2.0 * value * tau.square()) * torch.exp(-value * tau.square())

    def forward(
        self,
        parent_field: torch.Tensor,
        velocity_mps: torch.Tensor,
        observed_wavefield: torch.Tensor,
        source_parameters: torch.Tensor,
        time_s: torch.Tensor,
        observed_indices: tuple[int, int],
        *,
        parent_defect: torch.Tensor,
        defect_scale: torch.Tensor,
    ) -> FactorizedCausalBasis:
        parent = torch.as_tensor(parent_field)
        velocity = torch.as_tensor(
            velocity_mps, dtype=parent.dtype, device=parent.device
        )
        observed = torch.as_tensor(
            observed_wavefield, dtype=parent.dtype, device=parent.device
        )
        if parent.ndim != 4:
            raise ValueError("parent field must be [record,time,z,x]")
        records, time_count, height, width = parent.shape
        if velocity.ndim == 3:
            velocity = velocity[:, None]
        if velocity.shape != (records, 1, height, width):
            raise ValueError("velocity shape does not match the parent field")
        if observed.shape != (records, 2, height, width):
            raise ValueError("observed wavefield must contain two full-grid frames")
        defect = torch.as_tensor(
            parent_defect, dtype=parent.dtype, device=parent.device
        )
        if defect.ndim != 4 or defect.shape[0] != records:
            raise ValueError("parent defect must be [record,time,z,x]")
        missing_z = height - defect.shape[-2]
        missing_x = width - defect.shape[-1]
        if missing_z == 0 and missing_x == 0:
            defect_margin = 0
        elif (
            missing_z != missing_x
            or missing_z < 8
            or missing_z % 2
        ):
            raise ValueError(
                "parent defect must use either full CPML support or a symmetric crop"
            )
        else:
            defect_margin = missing_z // 2
        scale = torch.as_tensor(
            defect_scale, dtype=parent.dtype, device=parent.device
        ).flatten()
        if scale.shape != (records,) or bool(torch.any(scale <= 0.0)):
            raise ValueError("defect scale must contain one positive value per record")

        times = torch.as_tensor(time_s, dtype=parent.dtype, device=parent.device)
        if times.ndim == 1:
            times = times.unsqueeze(0).expand(records, -1)
        if times.shape != (records, time_count):
            raise ValueError("time_s shape does not match the parent field")
        selected = list(observed_indices)
        if len(selected) != 2 or min(selected) < 0 or max(selected) >= time_count:
            raise ValueError("observed indices are invalid")

        field_scale = self._field_scale(parent, observed)
        velocity_mean = velocity.mean(dim=(-2, -1), keepdim=True)
        velocity_std = velocity.std(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        normalized_velocity = (velocity - velocity_mean) / velocity_std
        normalized_observed = observed / field_scale[:, None, None, None]
        parent_observed_error = (
            parent[:, selected] - observed
        ) / field_scale[:, None, None, None]
        defect_normalized = defect / scale[:, None, None, None]
        defect_rms = defect_normalized.square().mean(dim=1, keepdim=True).sqrt()
        if defect_margin:
            defect_rms = F.pad(
                defect_rms,
                (defect_margin, defect_margin, defect_margin, defect_margin),
            )
        parent_rms = (
            parent.square().mean(dim=1, keepdim=True).sqrt()
            / field_scale[:, None, None, None]
        )
        spatial_input = torch.cat(
            (
                normalized_velocity,
                normalized_observed,
                parent_observed_error,
                defect_rms,
                parent_rms,
            ),
            dim=1,
        )
        spatial_modes = self.spatial_encoder(spatial_input)
        spatial_modes = spatial_modes / spatial_modes.square().flatten(2).mean(dim=2).sqrt().clamp_min(
            1.0e-5
        )[:, :, None, None]

        time_start = times[:, :1]
        time_span = (times[:, -1:] - time_start).clamp_min(1.0e-8)
        normalized_time = 2.0 * (times - time_start) / time_span - 1.0
        parent_energy = parent.square().flatten(2).mean(dim=2).sqrt() / field_scale[:, None]
        first_center = int(observed_indices[1]) + 1
        aligned_defect = torch.zeros(
            (records, time_count), dtype=parent.dtype, device=parent.device
        )
        defect_energy = defect_normalized.square().flatten(2).mean(dim=2).sqrt()
        stop = min(first_center + defect_energy.shape[1], time_count)
        aligned_defect[:, first_center:stop] = defect_energy[:, : stop - first_center]
        ricker = self._ricker_feature(source_parameters, times)
        causal_distance = (
            torch.arange(time_count, dtype=parent.dtype, device=parent.device)[None]
            - float(observed_indices[1])
        ).clamp_min(0.0) / float(max(time_count - int(observed_indices[1]) - 1, 1))
        temporal_input = torch.stack(
            (normalized_time, parent_energy, aligned_defect, ricker, causal_distance.expand(records, -1)),
            dim=-1,
        )
        temporal_modes = self.temporal_encoder(temporal_input).transpose(1, 2)
        trust_features = torch.cat(
            (
                spatial_input.mean(dim=(-2, -1)),
                temporal_input.mean(dim=1),
            ),
            dim=1,
        )
        trust_fraction = torch.sigmoid(self.trust_head(trust_features)).flatten()
        envelope = causal_smoothstep_envelope(
            time_count,
            observed_indices,
            ramp_steps=self.ramp_steps,
            dtype=parent.dtype,
            device=parent.device,
        ).unsqueeze(0).expand(records, -1)
        post_weight = envelope[:, None].square()
        temporal_norm = (
            (temporal_modes.square() * post_weight).sum(dim=2)
            / post_weight.sum(dim=2).clamp_min(1.0)
        ).sqrt().clamp_min(1.0e-5)
        temporal_modes = temporal_modes / temporal_norm[:, :, None]
        return FactorizedCausalBasis(
            spatial_modes=spatial_modes,
            temporal_modes=temporal_modes,
            causal_envelope=envelope,
            field_scale=field_scale,
            phase_reference=self._phase_reference(parent, field_scale),
            trust_fraction=trust_fraction,
            phase_rank=self.phase_rank,
        )


@dataclass(frozen=True)
class WeightedRidgeResult:
    coefficients: torch.Tensor
    objective: torch.Tensor
    data_objective: torch.Tensor
    prior_objective: torch.Tensor
    condition_number: torch.Tensor
    effective_rank: torch.Tensor
    cholesky_jitter: float


def solve_weighted_ridge(
    design: torch.Tensor,
    target: torch.Tensor,
    *,
    row_weights: torch.Tensor | None = None,
    prior_precision: torch.Tensor | float = 1.0e-4,
    initial_jitter: float = 1.0e-8,
    maximum_attempts: int = 6,
    solve_dtype: torch.dtype | None = None,
) -> WeightedRidgeResult:
    """Solve a differentiable batched regularized least-squares problem."""

    matrix = torch.as_tensor(design)
    right = torch.as_tensor(target, dtype=matrix.dtype, device=matrix.device)
    if matrix.ndim != 3 or right.shape != matrix.shape[:2]:
        raise ValueError("design and target must be [record,row,rank] and [record,row]")
    if matrix.shape[1] == 0 or matrix.shape[2] == 0:
        raise ValueError("weighted ridge requires nonempty rows and rank")
    if not bool(torch.isfinite(matrix).all()) or not bool(torch.isfinite(right).all()):
        raise ValueError("weighted ridge inputs must be finite")
    dtype = solve_dtype or (
        torch.float64 if matrix.device.type == "cpu" else matrix.dtype
    )
    work_matrix = matrix.to(dtype)
    work_right = right.to(dtype)
    if row_weights is None:
        weight = torch.ones_like(work_right)
    else:
        weight = torch.as_tensor(row_weights, dtype=dtype, device=matrix.device)
        if weight.ndim == 1:
            weight = weight.unsqueeze(0).expand(matrix.shape[0], -1)
        if weight.shape != work_right.shape:
            raise ValueError("row weights must be [row] or [record,row]")
        if not bool(torch.isfinite(weight).all()) or bool(torch.any(weight < 0.0)):
            raise ValueError("row weights must be finite and nonnegative")
    root_weight = weight.sqrt()
    weighted_matrix = work_matrix * root_weight[:, :, None]
    weighted_right = work_right * root_weight

    prior = torch.as_tensor(prior_precision, dtype=dtype, device=matrix.device)
    if prior.ndim == 0:
        prior = prior.expand(matrix.shape[0], matrix.shape[2])
    elif prior.ndim == 1:
        if prior.shape != (matrix.shape[2],):
            raise ValueError("prior precision vector must match the basis rank")
        prior = prior.unsqueeze(0).expand(matrix.shape[0], -1)
    if prior.shape != (matrix.shape[0], matrix.shape[2]):
        raise ValueError("prior precision must be scalar, [rank], or [record,rank]")
    if not bool(torch.isfinite(prior).all()) or bool(torch.any(prior < 0.0)):
        raise ValueError("prior precision must be finite and nonnegative")

    data_normal = weighted_matrix.transpose(1, 2) @ weighted_matrix
    eigenvalues = torch.linalg.eigvalsh(
        0.5 * (data_normal.detach() + data_normal.detach().transpose(1, 2))
    ).clamp_min(0.0)
    rank_tolerance = (
        eigenvalues[:, -1:]
        * max(int(matrix.shape[1]), int(matrix.shape[2]))
        * torch.finfo(dtype).eps
    )
    effective_rank = (eigenvalues > rank_tolerance).sum(dim=1)
    normal = data_normal + torch.diag_embed(prior)
    normal = 0.5 * (normal + normal.transpose(1, 2))
    normal_right = weighted_matrix.transpose(1, 2) @ weighted_right.unsqueeze(-1)
    identity = torch.eye(matrix.shape[2], dtype=dtype, device=matrix.device)[None]
    jitter = max(float(initial_jitter), 0.0)
    factor = None
    for _ in range(max(1, int(maximum_attempts))):
        factor_candidate, info = torch.linalg.cholesky_ex(normal + jitter * identity)
        if bool((info == 0).all()):
            factor = factor_candidate
            break
        jitter = 1.0e-12 if jitter == 0.0 else jitter * 10.0
    if factor is None:
        raise RuntimeError("regularized normal matrix is not positive definite")
    coefficient_work = torch.cholesky_solve(normal_right, factor).squeeze(-1)
    residual = torch.einsum("rmk,rk->rm", work_matrix, coefficient_work) - work_right
    data_objective = (weight * residual.square()).sum(dim=1)
    prior_objective = (prior * coefficient_work.square()).sum(dim=1)
    condition = torch.linalg.cond((normal + jitter * identity).detach())
    coefficients = coefficient_work.to(matrix.dtype)
    return WeightedRidgeResult(
        coefficients=coefficients,
        objective=(data_objective + prior_objective).to(matrix.dtype),
        data_objective=data_objective.to(matrix.dtype),
        prior_objective=prior_objective.to(matrix.dtype),
        condition_number=condition.to(matrix.dtype),
        effective_rank=effective_rank,
        cholesky_jitter=float(jitter),
    )


def sampled_basis_defect_design(
    basis: FactorizedCausalBasis,
    velocity_mps: torch.Tensor,
    points: torch.Tensor | PhysicsPointSet,
    observed_indices: tuple[int, int],
    *,
    dt: float,
    dx: float,
    dz: float,
    time_order: int = 4,
    rank_chunk_size: int = 4,
    cpml_config: SavedGridCPMLConfig | dict[str, object] | None = None,
) -> torch.Tensor:
    """Evaluate the homogeneous LWC-84 operator on basis modes in rank chunks."""

    chunk_size = int(rank_chunk_size)
    if chunk_size <= 0:
        raise ValueError("rank_chunk_size must be positive")
    velocity = torch.as_tensor(
        velocity_mps,
        dtype=basis.spatial_modes.dtype,
        device=basis.spatial_modes.device,
    )
    if velocity.ndim == 3:
        velocity = velocity[:, None]
    records = basis.spatial_modes.shape[0]
    if velocity.shape != (records, 1, *basis.spatial_shape):
        raise ValueError("velocity shape does not match the basis")
    blocks: list[torch.Tensor] = []
    for start in range(0, basis.rank, chunk_size):
        stop = min(start + chunk_size, basis.rank)
        modes = basis.materialize(rank_start=start, rank_stop=stop)
        local_rank = stop - start
        flattened = modes.reshape(
            records * local_rank,
            basis.time_count,
            *basis.spatial_shape,
        )
        repeated_velocity = velocity.repeat_interleave(local_rank, dim=0)
        defect = lwc84_discrete_defect(
            flattened,
            repeated_velocity,
            dt=float(dt),
            dx=float(dx),
            dz=float(dz),
            observed_indices=observed_indices,
            time_order=int(time_order),
            normalize=False,
            cpml_config=cpml_config,
        )
        sampled = sample_fixed_physics_residual(defect, points, observed_indices)
        blocks.append(
            sampled.reshape(records, local_rank, -1).transpose(1, 2)
        )
    return torch.cat(blocks, dim=2)


def _basis_values_at_points(
    basis: FactorizedCausalBasis,
    time_index: torch.Tensor,
    z_index: torch.Tensor,
    x_index: torch.Tensor,
) -> torch.Tensor:
    """Evaluate every factorized mode at selected absolute grid points."""

    times = torch.as_tensor(
        time_index, dtype=torch.long, device=basis.spatial_modes.device
    ).flatten()
    z = torch.as_tensor(
        z_index, dtype=torch.long, device=basis.spatial_modes.device
    ).flatten()
    x = torch.as_tensor(
        x_index, dtype=torch.long, device=basis.spatial_modes.device
    ).flatten()
    if not (times.shape == z.shape == x.shape) or times.numel() == 0:
        raise ValueError("basis point indices must be nonempty matching vectors")
    height, width = basis.spatial_shape
    if bool(
        ((times < 0) | (times >= basis.time_count)).any()
        or ((z < 0) | (z >= height)).any()
        or ((x < 0) | (x >= width)).any()
    ):
        raise ValueError("basis point index is outside the field support")

    # Advanced indexing retains [record, rank, point], which avoids ever
    # materializing [record, rank, time, z, x] during online adaptation.
    spatial = basis.spatial_modes[:, :, z, x]
    temporal = basis.temporal_modes[:, :, times]
    values = spatial * temporal
    amplitude_rank = basis.rank - int(basis.phase_rank)
    if basis.phase_rank:
        phase = basis.phase_reference[:, times, z, x]
        values = torch.cat(
            (
                values[:, :amplitude_rank],
                values[:, amplitude_rank:] * phase[:, None],
            ),
            dim=1,
        )
    values = (
        values
        * basis.causal_envelope[:, None, times]
        * basis.field_scale[:, None, None]
    )
    return values.permute(0, 2, 1)


def sampled_basis_defect_design_sparse(
    basis: FactorizedCausalBasis,
    velocity_mps: torch.Tensor,
    points: torch.Tensor | PhysicsPointSet,
    observed_indices: tuple[int, int],
    *,
    dt: float,
    dx: float,
    dz: float,
) -> torch.Tensor:
    """Apply the second-order interior LWC operator only at sampled rows.

    This is the hyper-reduced deployment counterpart of
    :func:`sampled_basis_defect_design`.  It is exact for ``time_order=2`` and
    interior cells, but requires only O(point_count * rank) storage instead of
    materializing every rank mode over the complete space-time field.  CPML
    physics remains an offline loss; this online design deliberately excludes
    the boundary closure and is therefore registered as a distinct contract.
    """

    if min(float(dt), float(dx), float(dz)) <= 0.0:
        raise ValueError("sparse defect spacings must be positive")
    raw_points = points.points if isinstance(points, PhysicsPointSet) else points
    selected = torch.as_tensor(raw_points, device=basis.spatial_modes.device)
    if selected.ndim != 2 or selected.shape[1] != 3 or selected.shape[0] == 0:
        raise ValueError("sparse defect points must be nonempty [point,3]")
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("sparse defect points must be finite")
    if bool(((selected[:, 1:] < 0.0) | (selected[:, 1:] > 1.0)).any()):
        raise ValueError("sparse defect spatial fractions must lie in [0,1]")

    time_index = selected[:, 0].long()
    first_center = int(observed_indices[1]) + 1
    if bool(
        ((time_index < first_center) | (time_index >= basis.time_count - 1)).any()
    ):
        raise ValueError("sparse defect time is outside centered support")
    margin = 4
    height, width = basis.spatial_shape
    support_height = height - 2 * margin
    support_width = width - 2 * margin
    if min(support_height, support_width) <= 0:
        raise ValueError("basis grid is too small for the radius-four stencil")
    z_index = margin + torch.round(
        selected[:, 1] * float(support_height - 1)
    ).long()
    x_index = margin + torch.round(
        selected[:, 2] * float(support_width - 1)
    ).long()

    center = _basis_values_at_points(basis, time_index, z_index, x_index)
    temporal = (
        _basis_values_at_points(basis, time_index + 1, z_index, x_index)
        - 2.0 * center
        + _basis_values_at_points(basis, time_index - 1, z_index, x_index)
    ) / float(dt) ** 2
    dxx = torch.zeros_like(center)
    dzz = torch.zeros_like(center)
    for offset, coefficient in enumerate(LAPLACIAN8_COEFFICIENTS):
        shift = int(offset) - 4
        if shift == 0:
            continue
        dxx = dxx + float(coefficient) * (
            _basis_values_at_points(
                basis, time_index, z_index, x_index + shift
            )
            - center
        )
        dzz = dzz + float(coefficient) * (
            _basis_values_at_points(
                basis, time_index, z_index + shift, x_index
            )
            - center
        )
    laplacian = dxx / float(dx) ** 2 + dzz / float(dz) ** 2
    velocity = torch.as_tensor(
        velocity_mps,
        dtype=basis.spatial_modes.dtype,
        device=basis.spatial_modes.device,
    )
    if velocity.ndim == 3:
        velocity = velocity[:, None]
    if velocity.shape != (
        basis.spatial_modes.shape[0],
        1,
        height,
        width,
    ):
        raise ValueError("velocity shape does not match the sparse basis design")
    speed_squared = velocity[:, 0, z_index, x_index].square()
    return temporal - speed_squared[:, :, None] * laplacian


def _local_box_test_points(
    points: torch.Tensor | PhysicsPointSet,
    *,
    support_height: int,
    support_width: int,
    width: int,
) -> tuple[torch.Tensor, int]:
    """Expand collocation rows into a replicated local box test function.

    The returned rows are ordered ``[point, box_cell]`` so callers can apply
    the strong-form operator first and then average each local test-function
    moment.  Replication at the cropped residual boundary keeps every original
    collocation row at equal mass and, crucially, makes ``width=1`` an exact
    identity rather than a numerically similar special case.
    """

    test_width = int(width)
    if test_width < 1 or test_width % 2 == 0:
        raise ValueError("defect test-function width must be a positive odd integer")
    if test_width > min(int(support_height), int(support_width)):
        raise ValueError("defect test-function width exceeds the residual support")
    raw_points = points.points if isinstance(points, PhysicsPointSet) else points
    selected = torch.as_tensor(raw_points)
    if selected.ndim != 2 or selected.shape[1] != 3 or selected.shape[0] == 0:
        raise ValueError("defect test-function points must be nonempty [point,3]")
    if not selected.is_floating_point():
        selected = selected.float()
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("defect test-function points must be finite")
    if bool(((selected[:, 1:] < 0.0) | (selected[:, 1:] > 1.0)).any()):
        raise ValueError("defect test-function spatial fractions must lie in [0,1]")
    if test_width == 1:
        return selected, 1

    height = int(support_height)
    width_cells = int(support_width)
    radius = test_width // 2
    offsets = torch.arange(
        -radius, radius + 1, device=selected.device, dtype=torch.long
    )
    dz, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    dz = dz.reshape(1, -1)
    dx = dx.reshape(1, -1)
    z_center = torch.round(selected[:, 1] * float(height - 1)).long()[:, None]
    x_center = torch.round(selected[:, 2] * float(width_cells - 1)).long()[:, None]
    z_index = (z_center + dz).clamp(0, height - 1)
    x_index = (x_center + dx).clamp(0, width_cells - 1)
    expanded = selected[:, None, :].expand(-1, test_width**2, -1).clone()
    expanded[:, :, 1] = z_index.to(selected.dtype) / float(max(height - 1, 1))
    expanded[:, :, 2] = x_index.to(selected.dtype) / float(
        max(width_cells - 1, 1)
    )
    return expanded.reshape(-1, 3), test_width**2


@dataclass(frozen=True)
class DefectCorrectionWeights:
    defect: float = 1.0
    observed: float = 1.0
    bridge: float = 0.5
    prior: float = 1.0e-4

    def __post_init__(self) -> None:
        values = (self.defect, self.observed, self.bridge, self.prior)
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
            raise ValueError("defect-correction weights must be finite and nonnegative")


@dataclass(frozen=True)
class ConvexDefectCorrectionResult:
    field: torch.Tensor
    coefficients: torch.Tensor
    accepted: torch.Tensor
    rollback_reasons: tuple[str | None, ...]
    objective_before: torch.Tensor
    objective_after: torch.Tensor
    condition_number: torch.Tensor
    effective_design_rank: torch.Tensor
    design_row_count: int
    correction_ratio: torch.Tensor
    unconstrained_correction_ratio: torch.Tensor
    projection_scale: torch.Tensor
    trust_fraction: torch.Tensor
    effective_correction_ratio_limit: torch.Tensor
    cholesky_jitter: float
    coefficient_solve_elapsed_s: float
    coefficient_solve_device: str
    coefficient_objective_device: str
    correction_materialization_device: str
    future_truth_used: bool = False


def _field_design_block(
    basis: FactorizedCausalBasis,
    parent: torch.Tensor,
    target: torch.Tensor,
    indices: Sequence[int],
    *,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = tuple(int(value) for value in indices)
    if not selected:
        raise ValueError("field design block requires at least one time index")
    modes = basis.materialize(time_indices=selected)
    step = max(1, int(stride))
    modes = modes[..., ::step, ::step]
    reference = torch.as_tensor(
        target, dtype=parent.dtype, device=parent.device
    )[..., ::step, ::step]
    if reference.shape != (parent.shape[0], len(selected), modes.shape[-2], modes.shape[-1]):
        raise ValueError("field target shape does not match its time indices")
    residual_target = reference - parent[:, list(selected), ::step, ::step]
    design = modes.permute(0, 2, 3, 4, 1).reshape(parent.shape[0], -1, basis.rank)
    right = residual_target.reshape(parent.shape[0], -1)
    scale = basis.field_scale[:, None]
    return design / scale[:, :, None], right / scale


def solve_causal_defect_correction(
    basis: FactorizedCausalBasis,
    parent_field: torch.Tensor,
    velocity_mps: torch.Tensor,
    source_parameters: torch.Tensor,
    source_map: torch.Tensor,
    time_s: torch.Tensor,
    observed_indices: tuple[int, int],
    physics_points: torch.Tensor | PhysicsPointSet,
    *,
    field_scale_pa: torch.Tensor | float | None = None,
    observed_wavefield: torch.Tensor | None = None,
    observed_design_basis: FactorizedCausalBasis | None = None,
    bridge_wavefield: torch.Tensor | None = None,
    bridge_indices: Sequence[int] = (),
    weights: DefectCorrectionWeights = DefectCorrectionWeights(),
    dt: float | None = None,
    dx: float = 10.0,
    dz: float = 10.0,
    time_order: int = 4,
    spatial_stride: int = 4,
    rank_chunk_size: int = 4,
    defect_design: str = "materialized",
    defect_test_function_width: int = 1,
    minimum_relative_improvement: float = 1.0e-4,
    maximum_condition_number: float = 1.0e8,
    minimum_effective_design_rank: int = 1,
    maximum_correction_ratio: float = 0.25,
    minimum_unconstrained_correction_ratio: float = 0.0,
    cpml_config: SavedGridCPMLConfig | dict[str, object] | None = None,
    solve_device: torch.device | str | None = None,
) -> ConvexDefectCorrectionResult:
    """Perform the leakage-free online CPADC coefficient solve.

    No future truth is accepted by this API.  The only optional field targets
    are the two audited observations and explicitly supplied synthetic bridge
    frames.
    """

    parent = torch.as_tensor(parent_field)
    if parent.ndim != 4 or parent.shape[0] != basis.spatial_modes.shape[0]:
        raise ValueError("parent field shape does not match the basis")
    times = torch.as_tensor(time_s, dtype=parent.dtype, device=parent.device)
    if times.ndim == 1:
        times = times.unsqueeze(0).expand(parent.shape[0], -1)
    if times.shape != parent.shape[:2]:
        raise ValueError("time_s shape does not match the parent field")
    step_dt = float((times[:, 1:] - times[:, :-1]).mean()) if dt is None else float(dt)
    design_blocks: list[torch.Tensor] = []
    target_blocks: list[torch.Tensor] = []
    weight_blocks: list[torch.Tensor] = []
    test_function_width = int(defect_test_function_width)
    if test_function_width < 1 or test_function_width % 2 == 0:
        raise ValueError("defect test-function width must be a positive odd integer")
    if float(weights.defect) > 0.0:
        design_kind = str(defect_design).strip().lower()
        if design_kind not in {"materialized", "sparse_interior"}:
            raise ValueError("unknown CPADC defect design")
        if design_kind == "sparse_interior" and (
            int(time_order) != 2 or cpml_config is not None
        ):
            raise ValueError(
                "sparse interior defect design requires time_order=2 without CPML"
            )
        parent_defect, defect_scale = lwc84_discrete_defect(
            parent,
            velocity_mps,
            dt=step_dt,
            dx=float(dx),
            dz=float(dz),
            observed_indices=observed_indices,
            source_parameters=source_parameters,
            source_map=source_map,
            time_s=times,
            field_scale_pa=field_scale_pa,
            time_order=int(time_order),
            normalize=False,
            return_scale=True,
            cpml_config=cpml_config,
        )
        test_points, test_cell_count = _local_box_test_points(
            physics_points,
            support_height=int(parent_defect.shape[-2]),
            support_width=int(parent_defect.shape[-1]),
            width=test_function_width,
        )
        test_l2_normalization = math.sqrt(float(test_cell_count))
        sampled_parent = sample_fixed_physics_residual(
            parent_defect, test_points, observed_indices
        ).reshape(parent.shape[0], -1, test_cell_count).mean(dim=2)
        sampled_parent = sampled_parent * test_l2_normalization
        point_count = int(sampled_parent.shape[1])
        if point_count <= 0:
            raise ValueError("defect test function produced no collocation rows")
        if test_cell_count > 1 and point_count * test_cell_count != len(test_points):
            raise RuntimeError("defect test-function row grouping is inconsistent")
        if test_cell_count == 1 and point_count != len(test_points):
            raise RuntimeError("defect collocation row count is inconsistent")
        if design_kind == "sparse_interior":
            physics_design = sampled_basis_defect_design_sparse(
                basis,
                velocity_mps,
                test_points,
                observed_indices,
                dt=step_dt,
                dx=float(dx),
                dz=float(dz),
            )
        else:
            physics_design = sampled_basis_defect_design(
                basis,
                velocity_mps,
                test_points,
                observed_indices,
                dt=step_dt,
                dx=float(dx),
                dz=float(dz),
                time_order=int(time_order),
                rank_chunk_size=int(rank_chunk_size),
                cpml_config=cpml_config,
            )
        physics_design = physics_design.reshape(
            parent.shape[0], point_count, test_cell_count, basis.rank
        ).mean(dim=2) * test_l2_normalization
        design_blocks.append(physics_design / defect_scale[:, None, None])
        target_blocks.append(-sampled_parent / defect_scale[:, None])
        point_weights = (
            physics_points.weights.to(device=parent.device, dtype=parent.dtype)
            if isinstance(physics_points, PhysicsPointSet)
            else torch.full(
                (sampled_parent.shape[1],),
                1.0 / float(sampled_parent.shape[1]),
                dtype=parent.dtype,
                device=parent.device,
            )
        )
        weight_blocks.append(
            point_weights[None].expand(parent.shape[0], -1)
            * float(weights.defect)
        )

    if observed_wavefield is not None and float(weights.observed) > 0.0:
        observed = torch.as_tensor(
            observed_wavefield, dtype=parent.dtype, device=parent.device
        )
        design_basis = basis if observed_design_basis is None else observed_design_basis
        if (
            design_basis.spatial_modes.shape != basis.spatial_modes.shape
            or design_basis.temporal_modes.shape != basis.temporal_modes.shape
            or design_basis.field_scale.shape != basis.field_scale.shape
        ):
            raise ValueError("observed design basis is incompatible with output basis")
        observed_design, observed_target = _field_design_block(
            design_basis,
            parent,
            observed,
            observed_indices,
            stride=int(spatial_stride),
        )
        design_blocks.append(observed_design)
        target_blocks.append(observed_target)
        weight_blocks.append(
            torch.full_like(
                observed_target,
                float(weights.observed) / float(observed_target.shape[1]),
            )
        )

    if bridge_wavefield is not None and float(weights.bridge) > 0.0:
        bridge = torch.as_tensor(
            bridge_wavefield, dtype=parent.dtype, device=parent.device
        )
        bridge_design, bridge_target = _field_design_block(
            basis,
            parent,
            bridge,
            bridge_indices,
            stride=int(spatial_stride),
        )
        design_blocks.append(bridge_design)
        target_blocks.append(bridge_target)
        weight_blocks.append(
            torch.full_like(
                bridge_target,
                float(weights.bridge) / float(bridge_target.shape[1]),
            )
        )

    if not design_blocks:
        raise ValueError("coefficient solve has no positive-weight causal objective")
    design = torch.cat(design_blocks, dim=1)
    target = torch.cat(target_blocks, dim=1)
    row_weights = torch.cat(weight_blocks, dim=1)
    coefficient_device = (
        design.device if solve_device is None else torch.device(solve_device)
    )
    solve_started = time.perf_counter()
    solve_design = design.to(coefficient_device)
    solve_target = target.to(coefficient_device)
    solve_row_weights = row_weights.to(coefficient_device)
    ridge = solve_weighted_ridge(
        solve_design,
        solve_target,
        row_weights=solve_row_weights,
        prior_precision=float(weights.prior),
    )
    coefficient_solve_elapsed = time.perf_counter() - solve_started
    objective_before = (solve_row_weights * solve_target.square()).sum(dim=1)
    raw_coefficients_on_field_device = ridge.coefficients.to(parent.device)
    raw_correction = basis.combine(raw_coefficients_on_field_device)
    parent_rms = (
        parent.float().square().flatten(1).mean(dim=1).sqrt().clamp_min(1.0e-8)
    )
    raw_correction_rms = (
        raw_correction.float().square().flatten(1).mean(dim=1).sqrt()
    )
    raw_correction_ratio = (raw_correction_rms / parent_rms).to(coefficient_device)
    hard_ratio_limit = float(maximum_correction_ratio)
    if not math.isfinite(hard_ratio_limit) or hard_ratio_limit <= 0.0:
        raise ValueError("maximum_correction_ratio must be finite and positive")
    strength_floor = float(minimum_unconstrained_correction_ratio)
    if not math.isfinite(strength_floor) or strength_floor < 0.0:
        raise ValueError(
            "minimum_unconstrained_correction_ratio must be finite and nonnegative"
        )
    effective_ratio_limit = basis.trust_fraction.to(
        device=coefficient_device, dtype=raw_correction_ratio.dtype
    ) * hard_ratio_limit
    # The ridge direction is the unconstrained convex minimizer.  Intersect its
    # ray with the convex correction-energy ball so training and deployment see
    # the same bounded candidate.  Along this ray the quadratic objective is
    # monotone down to the ridge minimizer, hence this projection cannot make
    # the causal objective worse than the zero-correction parent.
    projection_scale = torch.minimum(
        torch.ones_like(raw_correction_ratio),
        effective_ratio_limit / raw_correction_ratio.clamp_min(1.0e-12),
    )
    proposed_coefficients = ridge.coefficients * projection_scale[:, None]
    proposed_correction = basis.combine(proposed_coefficients.to(parent.device))
    correction_rms = proposed_correction.float().square().flatten(1).mean(dim=1).sqrt()
    correction_ratio = (correction_rms / parent_rms).to(coefficient_device)
    proposed_residual = torch.einsum(
        "rmk,rk->rm", solve_design, proposed_coefficients
    ) - solve_target
    proposed_objective = (
        solve_row_weights * proposed_residual.square()
    ).sum(dim=1) + float(weights.prior) * proposed_coefficients.square().sum(dim=1)
    relative_improvement = (
        objective_before - proposed_objective
    ) / objective_before.clamp_min(1.0e-12)
    finite = (
        torch.isfinite(proposed_coefficients).all(dim=1)
        & torch.isfinite(proposed_objective)
        & torch.isfinite(ridge.condition_number)
        & torch.isfinite(correction_ratio)
        & torch.isfinite(raw_correction_ratio)
    )
    accepted = (
        finite
        & (raw_correction_ratio >= strength_floor)
        & (relative_improvement >= float(minimum_relative_improvement))
        & (ridge.condition_number <= float(maximum_condition_number))
        & (ridge.effective_rank >= int(minimum_effective_design_rank))
        & (correction_ratio <= effective_ratio_limit * (1.0 + 1.0e-5))
    )
    coefficients = torch.where(
        accepted[:, None], proposed_coefficients, torch.zeros_like(proposed_coefficients)
    )
    field = parent + basis.combine(coefficients.to(parent.device))
    reasons: list[str | None] = []
    for record in range(parent.shape[0]):
        if not bool(finite[record]):
            reasons.append("nonfinite_solve")
        elif float(raw_correction_ratio[record]) < strength_floor:
            reasons.append("calibrated_causal_strength_abstention")
        elif float(ridge.condition_number[record]) > float(maximum_condition_number):
            reasons.append("condition_number_limit")
        elif int(ridge.effective_rank[record]) < int(minimum_effective_design_rank):
            reasons.append("effective_design_rank_limit")
        elif float(relative_improvement[record]) < float(minimum_relative_improvement):
            reasons.append("causal_objective_not_improved")
        else:
            reasons.append(None)
    return ConvexDefectCorrectionResult(
        field=field,
        coefficients=coefficients,
        accepted=accepted,
        rollback_reasons=tuple(reasons),
        objective_before=objective_before,
        objective_after=torch.where(accepted, proposed_objective, objective_before),
        condition_number=ridge.condition_number,
        effective_design_rank=ridge.effective_rank,
        design_row_count=int(design.shape[1]),
        correction_ratio=torch.where(accepted, correction_ratio, torch.zeros_like(correction_ratio)),
        unconstrained_correction_ratio=raw_correction_ratio,
        projection_scale=projection_scale,
        trust_fraction=basis.trust_fraction,
        effective_correction_ratio_limit=effective_ratio_limit,
        cholesky_jitter=ridge.cholesky_jitter,
        coefficient_solve_elapsed_s=float(coefficient_solve_elapsed),
        coefficient_solve_device=str(coefficient_device),
        coefficient_objective_device=str(design.device),
        correction_materialization_device=str(basis.spatial_modes.device),
    )


@dataclass(frozen=True)
class MetaDefectLossWeights:
    full_field: float = 1.0
    late: float = 0.5
    temporal_difference: float = 0.1
    spectrum: float = 0.1
    coefficient: float = 1.0e-5


def meta_defect_correction_loss(
    candidate: torch.Tensor,
    target: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    weights: MetaDefectLossWeights = MetaDefectLossWeights(),
    late_start_fraction: float = 0.5,
    future_start_index: int = 0,
) -> dict[str, torch.Tensor]:
    """Future-only outer truth loss; never call this function at deployment."""

    prediction = torch.as_tensor(candidate)
    truth = torch.as_tensor(target, dtype=prediction.dtype, device=prediction.device)
    if prediction.shape != truth.shape or prediction.ndim != 4:
        raise ValueError("candidate and target must match [record,time,z,x]")
    future_start = int(future_start_index)
    if future_start < 0 or future_start >= prediction.shape[1] - 1:
        raise ValueError("future_start_index must leave at least two future frames")
    prediction = prediction[:, future_start:]
    truth = truth[:, future_start:]
    scale = truth.detach().float().flatten(1).norm(dim=1).clamp_min(1.0e-8)
    full = ((prediction - truth).float().flatten(1).norm(dim=1) / scale).mean()
    late_start = min(
        prediction.shape[1] - 1,
        max(0, int(math.floor(prediction.shape[1] * float(late_start_fraction)))),
    )
    late_truth = truth[:, late_start:]
    late_scale = late_truth.detach().float().flatten(1).norm(dim=1).clamp_min(1.0e-8)
    late = (
        (prediction[:, late_start:] - late_truth).float().flatten(1).norm(dim=1)
        / late_scale
    ).mean()
    prediction_dt = prediction[:, 1:] - prediction[:, :-1]
    truth_dt = truth[:, 1:] - truth[:, :-1]
    temporal_scale = truth_dt.detach().float().flatten(1).norm(dim=1).clamp_min(1.0e-8)
    temporal = ((prediction_dt - truth_dt).float().flatten(1).norm(dim=1) / temporal_scale).mean()
    prediction_spectrum = torch.fft.rfft2(prediction.float(), norm="ortho")
    truth_spectrum = torch.fft.rfft2(truth.float(), norm="ortho")
    spectrum_scale = truth_spectrum.abs().flatten(1).norm(dim=1).clamp_min(1.0e-8)
    spectrum = (
        (prediction_spectrum - truth_spectrum).abs().flatten(1).norm(dim=1)
        / spectrum_scale
    ).mean()
    coefficient = torch.as_tensor(coefficients).square().mean()
    total = (
        float(weights.full_field) * full
        + float(weights.late) * late
        + float(weights.temporal_difference) * temporal
        + float(weights.spectrum) * spectrum
        + float(weights.coefficient) * coefficient
    )
    return {
        "full_field": full,
        "late": late,
        "temporal_difference": temporal,
        "spectrum": spectrum,
        "coefficient": coefficient,
        "total": total,
    }


__all__ = [
    "CPADC_SCHEMA_VERSION",
    "CausalErrorBasisGenerator",
    "ConvexDefectCorrectionResult",
    "DefectCorrectionWeights",
    "FactorizedCausalBasis",
    "MetaDefectLossWeights",
    "WeightedRidgeResult",
    "causal_observation_probe_basis",
    "causal_smoothstep_envelope",
    "meta_defect_correction_loss",
    "sampled_basis_defect_design",
    "sampled_basis_defect_design_sparse",
    "solve_causal_defect_correction",
    "solve_weighted_ridge",
]
