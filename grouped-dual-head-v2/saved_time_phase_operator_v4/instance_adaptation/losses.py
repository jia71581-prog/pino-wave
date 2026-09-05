"""Finite, dimensionless losses for two-frame causal adaptation."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F
from saved_time_phase_operator_v4.losses import band_limited_residual_loss

from .bridge import BridgeResult
from .forced_defect import SavedGridCPMLConfig, lwc84_discrete_defect


_D2_8 = (
    -1.0 / 560.0, 8.0 / 315.0, -1.0 / 5.0, 8.0 / 5.0,
    -205.0 / 72.0, 8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0,
)


def _laplacian8(field: torch.Tensor, *, dx_m: float, dz_m: float) -> torch.Tensor:
    padded = F.pad(field, (4, 4, 4, 4))
    height, width = field.shape[-2:]
    dxx = torch.zeros_like(field)
    dzz = torch.zeros_like(field)
    center = field
    for index, coefficient in enumerate(_D2_8):
        if index == 4:
            continue
        dxx = dxx + coefficient * (
            padded[..., 4 : 4 + height, index : index + width] - center
        )
        dzz = dzz + coefficient * (
            padded[..., index : index + height, 4 : 4 + width] - center
        )
    return dxx / float(dx_m) ** 2 + dzz / float(dz_m) ** 2


@dataclass(frozen=True)
class LossWeights:
    observed: float = 1.0
    bridge: float = 0.5
    pde: float = 0.25
    phase: float = 0.1
    energy: float = 0.05
    anchor: float = 1.0e-5


@dataclass(frozen=True)
class PhysicsSampling:
    """Configuration for how PDE collocation points are drawn each closure.

    ``method='uniform'`` reproduces the historical fixed uniform-random points
    (``build_fixed_physics_points``) and is the backward-compatible default.

    ``method='rad'`` uses residual-based adaptive distribution sampling (Wu et
    al., CMAME 2023): the per-cell draw probability is

        p(t,z,x) ∝ ( |r(t,z,x)|^k / E[|r|^k] + c ) · tilt(t)

    concentrating collocation on high-residual regions — which here are exactly
    the confirmed failure modes (sharp moving wavefront + late frames).  ``tilt``
    is a causal time weight ``exp(-time_tilt · (t - t0)/(T - t0))`` favouring the
    earliest unobserved times so the fit propagates forward from the observed
    frames instead of letting early error accumulate into late frames.
    """

    method: str = "uniform"
    k: float = 1.0
    c: float = 1.0
    time_tilt: float = 1.5
    resample_every: int = 4
    hard_quantile: float = 0.9
    release_quantile: float = 0.7
    uniform_fraction: float = 0.2
    rams_fraction: float = 0.2
    rams_steps: int = 3

    def __post_init__(self) -> None:
        method = str(self.method).strip().lower().replace("-", "_")
        if method not in {"uniform", "rad", "r3_rams"}:
            raise ValueError("physics sampling method must be uniform, rad, or r3_rams")
        if not math.isfinite(float(self.k)) or float(self.k) < 0.0:
            raise ValueError("RAD exponent must be finite and nonnegative")
        if not math.isfinite(float(self.c)) or float(self.c) < 0.0:
            raise ValueError("RAD floor must be finite and nonnegative")
        if int(self.resample_every) <= 0 or int(self.rams_steps) < 0:
            raise ValueError("physics resampling intervals must be positive")
        fractions = (
            float(self.hard_quantile),
            float(self.release_quantile),
            float(self.uniform_fraction),
            float(self.rams_fraction),
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("physics sampling fractions must lie in [0,1]")
        if float(self.release_quantile) > float(self.hard_quantile):
            raise ValueError("R3 release quantile cannot exceed the hard quantile")


@dataclass(frozen=True)
class PhysicsPointSet:
    """Unique collocation points and normalized quadrature weights.

    The first ``hard_count`` entries contain every deterministic per-frame
    high-residual point.  Remaining entries provide uniform coverage and RAMS
    exploration without replacement.  ``weights`` always sums to one, so the
    sampled PDE objective is independent of the changing hard-set size.
    """

    points: torch.Tensor
    weights: torch.Tensor
    hard_count: int
    uniform_count: int
    rams_count: int

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("physics point set must be [point,3]")
        if self.weights.shape != (self.points.shape[0],):
            raise ValueError("physics point weights must match the point count")
        if self.points.shape[0] != self.hard_count + self.uniform_count + self.rams_count:
            raise ValueError("physics point stratum counts do not match the point tensor")
        if not bool(torch.isfinite(self.weights).all()) or bool(torch.any(self.weights < 0.0)):
            raise ValueError("physics point weights must be finite and nonnegative")
        if not torch.isclose(
            self.weights.sum(), self.weights.new_tensor(1.0), rtol=1.0e-5, atol=1.0e-7
        ):
            raise ValueError("physics point weights must sum to one")


def build_fixed_physics_points(
    time_count: int,
    observed_indices: tuple[int, int],
    *,
    count: int = 512,
    seed: int = 17,
) -> torch.Tensor:
    """Create deterministic [time,z,x] fractions for a fixed optimizer closure."""
    if int(time_count) < 3 or int(count) <= 0:
        raise ValueError("time_count and point count must be positive")
    generator = torch.Generator().manual_seed(int(seed))
    points = torch.rand((int(count), 3), generator=generator)
    # A centered second temporal derivative is only defined through T - 2.
    low = int(observed_indices[1]) + 1
    high = int(time_count) - 1
    if low >= high:
        raise ValueError("no post-observation temporal center is available")
    points[:, 0] = torch.randint(low, high, (int(count),), generator=generator).float()
    return points


def sample_fixed_physics_residual(
    residual: torch.Tensor,
    points: torch.Tensor | PhysicsPointSet,
    observed_indices: tuple[int, int],
) -> torch.Tensor:
    """Select residual values at deterministic absolute-time/fractional-space points."""
    value = torch.as_tensor(residual)
    raw_points = points.points if isinstance(points, PhysicsPointSet) else points
    selected_points = torch.as_tensor(raw_points, device=value.device)
    if value.ndim != 4:
        raise ValueError("residual must be [record,time,z,x]")
    if selected_points.ndim != 2 or selected_points.shape[1] != 3:
        raise ValueError("physics points must be [point,3]")
    if not bool(torch.isfinite(selected_points).all()):
        raise ValueError("physics points must be finite")
    if bool(((selected_points[:, 1:] < 0.0) | (selected_points[:, 1:] > 1.0)).any()):
        raise ValueError("physics spatial fractions must lie in [0,1]")

    first_center = int(observed_indices[1]) + 1
    time_index = selected_points[:, 0].long() - first_center
    if bool(((time_index < 0) | (time_index >= value.shape[1])).any()):
        raise ValueError("physics point time is outside the residual support")
    z_index = torch.round(selected_points[:, 1] * (value.shape[2] - 1)).long()
    x_index = torch.round(selected_points[:, 2] * (value.shape[3] - 1)).long()
    return value[:, time_index, z_index, x_index]


def build_rad_physics_points(
    residual: torch.Tensor,
    observed_indices: tuple[int, int],
    *,
    count: int = 512,
    k: float = 1.0,
    c: float = 1.0,
    time_tilt: float = 1.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Residual-based adaptive distribution (RAD) collocation points.

    ``residual`` is the full ``[record,time,z,x]`` LWC-84 residual field
    (detached).  Points are drawn with probability proportional to
    ``|r|^k / E[|r|^k] + c`` modulated by the causal time tilt
    ``exp(-time_tilt · (t - t0)/(T - t0))``, then returned in the SAME
    ``[point, 3]`` ``[time_abs, z_frac, x_frac]`` layout as
    ``build_fixed_physics_points`` so ``sample_fixed_physics_residual`` gathers
    them unchanged.  ``time_abs`` is the absolute saved-time index; residual
    index zero corresponds to absolute time ``observed_indices[1] + 1``.
    """
    value = torch.as_tensor(residual).detach()
    if value.ndim != 4:
        raise ValueError("residual must be [record,time,z,x]")
    if int(count) <= 0:
        raise ValueError("point count must be positive")
    frames, height, width = value.shape[1], value.shape[2], value.shape[3]
    if frames < 1 or min(height, width) < 1:
        raise ValueError("residual field is empty")

    # Aggregate |r|^k over records so the density is a single field.
    magnitude = value.abs().mean(dim=0)                       # (time,z,x)
    powered = magnitude.clamp_min(0.0).pow(max(0.0, float(k)))
    mean_pow = powered.mean().clamp_min(1.0e-12)
    density = powered / mean_pow + max(0.0, float(c))         # RAD density

    # Causal time tilt: earliest unobserved time carries the largest weight.
    first_center = int(observed_indices[1]) + 1
    if float(time_tilt) != 0.0 and frames > 1:
        span = torch.arange(frames, device=value.device, dtype=value.dtype) / float(frames - 1)
        tilt = torch.exp(-float(time_tilt) * span)            # (time,)
        density = density * tilt[:, None, None]

    probabilities = density.reshape(-1)
    total = probabilities.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0.0:
        # Degenerate residual (all-zero / non-finite) → fall back to uniform.
        probabilities = torch.ones_like(probabilities)
    flat_index = torch.multinomial(
        probabilities, int(count), replacement=True, generator=generator
    )
    t_idx = (flat_index // (height * width)).long()
    rem = flat_index % (height * width)
    z_idx = (rem // width).long()
    x_idx = (rem % width).long()

    points = torch.empty((int(count), 3), device=value.device, dtype=torch.float32)
    points[:, 0] = (t_idx + first_center).float()             # absolute saved-time index
    denom_z = max(height - 1, 1)
    denom_x = max(width - 1, 1)
    points[:, 1] = z_idx.float() / float(denom_z)             # z fraction in [0,1]
    points[:, 2] = x_idx.float() / float(denom_x)             # x fraction in [0,1]
    return points


def _points_from_flat_indices(
    flat_index: torch.Tensor,
    *,
    frames: int,
    height: int,
    width: int,
    first_center: int,
) -> torch.Tensor:
    del frames
    indices = torch.as_tensor(flat_index, dtype=torch.long)
    t_idx = indices // (height * width)
    remainder = indices % (height * width)
    z_idx = remainder // width
    x_idx = remainder % width
    points = torch.empty((len(indices), 3), device=indices.device, dtype=torch.float32)
    points[:, 0] = (t_idx + int(first_center)).float()
    points[:, 1] = z_idx.float() / float(max(height - 1, 1))
    points[:, 2] = x_idx.float() / float(max(width - 1, 1))
    return points


def _flat_indices_from_points(
    points: torch.Tensor,
    *,
    frames: int,
    height: int,
    width: int,
    first_center: int,
) -> torch.Tensor:
    value = torch.as_tensor(points)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError("retained physics points must be [point,3]")
    t_idx = value[:, 0].long() - int(first_center)
    z_idx = torch.round(value[:, 1] * max(height - 1, 1)).long()
    x_idx = torch.round(value[:, 2] * max(width - 1, 1)).long()
    valid = (
        (t_idx >= 0) & (t_idx < frames)
        & (z_idx >= 0) & (z_idx < height)
        & (x_idx >= 0) & (x_idx < width)
    )
    return (t_idx[valid] * height * width + z_idx[valid] * width + x_idx[valid]).unique()


def _discrete_rams_ascent(
    seeds: torch.Tensor,
    magnitude: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    """Move discrete seeds to a local residual maximum without interpolation."""

    frames, height, width = magnitude.shape
    current = torch.as_tensor(seeds, dtype=torch.long, device=magnitude.device)
    if not len(current) or int(steps) <= 0:
        return current
    offsets = torch.tensor(
        ((0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1),
         (0, 1), (1, -1), (1, 0), (1, 1)),
        dtype=torch.long,
        device=magnitude.device,
    )
    for _ in range(int(steps)):
        t_idx = current // (height * width)
        remainder = current % (height * width)
        z_idx = remainder // width
        x_idx = remainder % width
        neighbor_z = (z_idx[:, None] + offsets[None, :, 0]).clamp(0, height - 1)
        neighbor_x = (x_idx[:, None] + offsets[None, :, 1]).clamp(0, width - 1)
        scores = magnitude[t_idx[:, None], neighbor_z, neighbor_x]
        best = scores.argmax(dim=1)
        rows = torch.arange(len(current), device=current.device)
        current = t_idx * height * width + neighbor_z[rows, best] * width + neighbor_x[rows, best]
    return current.unique(sorted=True)


def build_r3_rams_physics_points(
    residual: torch.Tensor,
    observed_indices: tuple[int, int],
    *,
    count: int = 512,
    hard_quantile: float = 0.9,
    release_quantile: float = 0.7,
    uniform_fraction: float = 0.2,
    rams_fraction: float = 0.2,
    k: float = 1.0,
    c: float = 1.0,
    time_tilt: float = 1.5,
    rams_steps: int = 3,
    retained_points: torch.Tensor | PhysicsPointSet | None = None,
    generator: torch.Generator | None = None,
) -> PhysicsPointSet:
    """Build an all-hotspot R3 + discrete-RAMS collocation set.

    Every frame contributes exactly ``ceil((1-hard_quantile) * H * W)`` largest
    residual cells.  Previously selected cells are retained while they remain
    above ``release_quantile``.  Uniform and RAMS points are drawn without
    replacement and deduplicated against the complete hard set.
    """

    value = torch.as_tensor(residual).detach()
    if value.ndim != 4:
        raise ValueError("residual must be [record,time,z,x]")
    if int(count) <= 0:
        raise ValueError("point count must be positive")
    frames, height, width = value.shape[1:]
    if min(frames, height, width) <= 0:
        raise ValueError("residual field is empty")
    quantile = float(hard_quantile)
    release = float(release_quantile)
    if not 0.0 <= release <= quantile <= 1.0:
        raise ValueError("R3 quantiles must satisfy 0 <= release <= hard <= 1")
    uniform_share = float(uniform_fraction)
    rams_share = float(rams_fraction)
    if not 0.0 <= uniform_share <= 1.0 or not 0.0 <= rams_share <= 1.0:
        raise ValueError("R3/RAMS fractions must lie in [0,1]")

    magnitude = value.abs().mean(dim=0)
    if not bool(torch.isfinite(magnitude).all()):
        finite = magnitude[torch.isfinite(magnitude)]
        replacement = finite.max() if finite.numel() else magnitude.new_tensor(1.0)
        magnitude = torch.nan_to_num(magnitude, nan=0.0, posinf=float(replacement), neginf=0.0)
    cells_per_frame = height * width
    hard_per_frame = max(1, min(cells_per_frame, int(math.ceil((1.0 - quantile) * cells_per_frame))))
    frame_top = magnitude.flatten(1).topk(
        hard_per_frame, dim=1, largest=True, sorted=True
    ).indices
    time_offsets = torch.arange(frames, device=value.device)[:, None] * cells_per_frame
    hard_indices = (frame_top + time_offsets).reshape(-1).unique(sorted=True)

    first_center = int(observed_indices[1]) + 1
    if retained_points is not None:
        old_points = retained_points.points if isinstance(retained_points, PhysicsPointSet) else retained_points
        old_indices = _flat_indices_from_points(
            torch.as_tensor(old_points, device=value.device),
            frames=frames,
            height=height,
            width=width,
            first_center=first_center,
        )
        release_threshold = torch.quantile(magnitude.flatten(1), release, dim=1)
        old_t = old_indices // cells_per_frame
        keep = magnitude.reshape(-1)[old_indices] >= release_threshold[old_t]
        hard_indices = torch.cat((hard_indices, old_indices[keep])).unique(sorted=True)

    total_cells = frames * cells_per_frame
    occupied = torch.zeros(total_cells, dtype=torch.bool, device=value.device)
    occupied[hard_indices] = True
    remaining_count = int((~occupied).sum())
    uniform_count = min(remaining_count, max(0, int(round(int(count) * uniform_share))))

    def draw_without_replacement(probability: torch.Tensor, draws: int) -> torch.Tensor:
        draws = min(int(draws), int(torch.count_nonzero(probability > 0)))
        if draws <= 0:
            return torch.empty(0, dtype=torch.long, device=value.device)
        return torch.multinomial(probability, draws, replacement=False, generator=generator)

    uniform_probability = (~occupied).to(dtype=torch.float32)
    uniform_indices = draw_without_replacement(uniform_probability, uniform_count)
    occupied[uniform_indices] = True

    desired_rams = min(
        int((~occupied).sum()), max(0, int(round(int(count) * rams_share)))
    )
    density = magnitude.clamp_min(0.0).pow(max(0.0, float(k)))
    density = density / density.mean().clamp_min(1.0e-12) + max(0.0, float(c))
    if float(time_tilt) != 0.0 and frames > 1:
        time_axis = torch.arange(frames, device=value.device, dtype=value.dtype) / float(frames - 1)
        density = density * torch.exp(-float(time_tilt) * time_axis)[:, None, None]
    seed_probability = density.reshape(-1).float()
    seed_probability[occupied] = 0.0
    seeds = draw_without_replacement(seed_probability, desired_rams)
    moved = _discrete_rams_ascent(seeds, magnitude, steps=int(rams_steps))
    moved = moved[~occupied[moved]]
    moved = moved[:desired_rams]
    occupied[moved] = True
    if len(moved) < desired_rams:
        fill_probability = density.reshape(-1).float()
        fill_probability[occupied] = 0.0
        fill = draw_without_replacement(fill_probability, desired_rams - len(moved))
        moved = torch.cat((moved, fill)).unique(sorted=True)
        occupied[fill] = True

    hard_count = len(hard_indices)
    uniform_count = len(uniform_indices)
    rams_count = len(moved)
    all_indices = torch.cat((hard_indices, uniform_indices, moved))
    if len(all_indices) != len(all_indices.unique()):
        raise AssertionError("R3/RAMS sampler produced duplicate points")
    points = _points_from_flat_indices(
        all_indices,
        frames=frames,
        height=height,
        width=width,
        first_center=first_center,
    )

    hard_mass = hard_count / float(total_cells)
    remaining_mass = max(0.0, 1.0 - hard_mass)
    effective_rams_share = rams_share if rams_count else 0.0
    effective_uniform_share = 1.0 - effective_rams_share if uniform_count else 0.0
    if not uniform_count and rams_count:
        effective_rams_share = 1.0
    weights = torch.empty(len(points), device=value.device, dtype=torch.float32)
    if hard_count:
        weights[:hard_count] = hard_mass / hard_count
    if uniform_count:
        start = hard_count
        weights[start : start + uniform_count] = (
            remaining_mass * effective_uniform_share / uniform_count
        )
    if rams_count:
        weights[-rams_count:] = remaining_mass * effective_rams_share / rams_count
    weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
    return PhysicsPointSet(
        points=points,
        weights=weights,
        hard_count=hard_count,
        uniform_count=uniform_count,
        rams_count=rams_count,
    )



def lwc84_residual(
    field: torch.Tensor,
    velocity_mps: torch.Tensor,
    *,
    dt: float,
    dx: float,
    dz: float,
    observed_indices: tuple[int, int],
    time_order: int = 4,
    normalization_scale: torch.Tensor | None = None,
    return_scale: bool = False,
    cpml_config: SavedGridCPMLConfig | dict[str, object] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute a homogeneous saved-time LWC-84-compatible residual.

    The data-generating solver (``src/fno_acoustic/data_generation/lwc84.py``) advances
    the field with the LWC-84 scheme

        p_{n+1} = 2 p_n - p_{n-1} + dt^2 L(p_n) + dt^4/12 L^2(p_n),   L = c^2 nabla^2_8

    i.e. the discrete field satisfies the *modified equation*

        (p_{n+1} - 2 p_n + p_{n-1}) / dt^2 = L(p_n) + dt^2/12 L^2(p_n).

    ``time_order=4`` (default) includes the ``dt^2/12 L^2(p_n)`` Lax-Wendroff term so the
    residual matches the solver's fourth-order-in-time operator; on a true solver field
    it is ~4x smaller than the plain 3-point form (verified on marmousi: 0.0018 vs 0.0079
    relative), because the 3-point second difference alone carries the field's O(dt^2)
    time-truncation error as a spurious residual that would mislead fine-tuning. The
    spatial operator is already the solver's 8th-order ``_laplacian8``. ``time_order=2``
    recovers the legacy plain second-difference residual (for ablation/comparison).
    """

    if cpml_config is not None:
        return lwc84_discrete_defect(
            field,
            velocity_mps,
            dt=float(dt),
            dx=float(dx),
            dz=float(dz),
            observed_indices=observed_indices,
            time_order=int(time_order),
            normalization_scale=normalization_scale,
            normalize=True,
            return_scale=bool(return_scale),
            cpml_config=cpml_config,
        )

    value = torch.as_tensor(field)
    velocity = torch.as_tensor(velocity_mps, device=value.device, dtype=value.dtype)
    if value.ndim != 4 or value.shape[1] < 3:
        raise ValueError("field must be [record,time,z,x] with at least three times")
    if velocity.ndim == 3:
        velocity = velocity[:, None]
    if velocity.shape[:2] != (value.shape[0], 1) or velocity.shape[-2:] != value.shape[-2:]:
        raise ValueError("velocity shape does not match field")
    if min(value.shape[-2:]) <= 8:
        raise ValueError("field grid is too small for LWC-84 stencil")
    dt_value = float(dt)
    if dt_value <= 0.0 or float(dx) <= 0.0 or float(dz) <= 0.0:
        raise ValueError("LWC-84 spacings must be positive")
    if int(time_order) not in (2, 4):
        raise ValueError("time_order must be 2 (plain 3-point) or 4 (LWC modified eqn)")
    second_time = (
        value[:, 2:] - 2.0 * value[:, 1:-1] + value[:, :-2]
    ) / dt_value**2
    laplace = _laplacian8(
        value[:, 1:-1], dx_m=float(dx), dz_m=float(dz)
    )
    # L(p_n) = c^2 nabla^2_8 p_n on the full grid (kept uncropped so the fourth-order
    # term can apply a second laplacian to it before the shared radius-four crop).
    c2 = velocity.square()
    spatial_full = c2 * laplace
    if int(time_order) == 4:
        # Lax-Wendroff fourth-order-in-time correction: the solver's discrete field
        # satisfies (p_{n+1}-2p_n+p_{n-1})/dt^2 = L(p_n) + dt^2/12 L^2(p_n).
        # L^2(p_n) = c^2 nabla^2_8 ( c^2 nabla^2_8 p_n ).
        l_squared = c2 * _laplacian8(spatial_full, dx_m=float(dx), dz_m=float(dz))
        spatial_full = spatial_full + (dt_value * dt_value / 12.0) * l_squared
    # The stored 201 x 201 grid excludes CPML memory variables.  Evaluate only
    # where the radius-four stencil is supported by physical samples instead
    # of inventing four zero-valued boundaries through padding.
    temporal_acceleration = second_time[..., 4:-4, 4:-4]
    spatial_acceleration = spatial_full[..., 4:-4, 4:-4]
    # Residual index zero is centered at saved time one.  Start strictly after
    # the second observed frame k1.
    start = max(int(observed_indices[1]), 0)
    temporal_acceleration = temporal_acceleration[:, start:]
    spatial_acceleration = spatial_acceleration[:, start:]
    residual = temporal_acceleration - spatial_acceleration
    if normalization_scale is None:
        scale = (
            temporal_acceleration.detach().square()
            + spatial_acceleration.detach().square()
        ).flatten(1).mean(dim=1).sqrt().clamp_min(1.0e-8)
    else:
        scale = torch.as_tensor(
            normalization_scale, dtype=value.dtype, device=value.device
        ).flatten()
        if scale.shape != (value.shape[0],) or not bool(torch.isfinite(scale).all()):
            raise ValueError("LWC-84 normalization scale must contain one finite value per record")
        scale = scale.clamp_min(1.0e-8)
    normalized = residual / scale[:, None, None, None]
    return (normalized, scale) if bool(return_scale) else normalized


def instance_loss_terms(
    *,
    raw_prediction: torch.Tensor,
    target_observed: torch.Tensor,
    bridge: BridgeResult | None,
    velocity: torch.Tensor,
    source: torch.Tensor,
    points: torch.Tensor | PhysicsPointSet,
    weights: LossWeights = LossWeights(),
    dt: float = 0.0025,
    dx: float = 10.0,
    dz: float = 10.0,
    observed_indices: tuple[int, int] = (0, 1),
    anchor: torch.Tensor | None = None,
    pde_normalization_scale: torch.Tensor | None = None,
    cpml_config: SavedGridCPMLConfig | dict[str, object] | None = None,
) -> dict[str, torch.Tensor]:
    """Return separately normalized terms and their weighted sum."""
    del source
    observed = raw_prediction[:, list(observed_indices)]
    observed_scale = target_observed.detach().flatten(1).norm(dim=1).clamp_min(1.0e-8)
    observed_loss = ((observed - target_observed).flatten(1).norm(dim=1) / observed_scale).mean()
    bridge_loss = raw_prediction.sum() * 0.0
    if bridge is not None and bridge.valid and bridge.frames.numel():
        predicted_bridge = raw_prediction[:, list(bridge.time_indices)]
        bridge_loss = band_limited_residual_loss(predicted_bridge, bridge.frames.to(raw_prediction.device))
    residual = lwc84_residual(
        raw_prediction,
        velocity,
        dt=float(dt),
        dx=float(dx),
        dz=float(dz),
        observed_indices=observed_indices,
        normalization_scale=pde_normalization_scale,
        cpml_config=cpml_config,
    )
    sampled_residual = sample_fixed_physics_residual(
        residual, points, observed_indices
    )
    if isinstance(points, PhysicsPointSet):
        weights_on_device = points.weights.to(
            device=sampled_residual.device, dtype=sampled_residual.dtype
        )
        pde_loss = (sampled_residual.square() * weights_on_device[None]).sum(dim=1).mean()
    else:
        pde_loss = sampled_residual.square().mean()
    phase_loss = band_limited_residual_loss(
        raw_prediction[:, list(observed_indices)], target_observed
    )
    predicted_energy = raw_prediction.square().flatten(2).mean(dim=-1)
    energy_loss = (predicted_energy[:, 1:] - predicted_energy[:, :-1]).abs().mean()
    anchor_loss = raw_prediction.sum() * 0.0 if anchor is None else anchor.square().mean()
    total = (
        float(weights.observed) * observed_loss
        + float(weights.bridge) * bridge_loss
        + float(weights.pde) * pde_loss
        + float(weights.phase) * phase_loss
        + float(weights.energy) * energy_loss
        + float(weights.anchor) * anchor_loss
    )
    return {
        "observed": observed_loss,
        "bridge": bridge_loss,
        "pde": pde_loss,
        "phase": phase_loss,
        "energy": energy_loss,
        "anchor": anchor_loss,
        "total": total,
    }


__all__ = [
    "LossWeights",
    "PhysicsPointSet",
    "PhysicsSampling",
    "build_fixed_physics_points",
    "build_rad_physics_points",
    "build_r3_rams_physics_points",
    "instance_loss_terms",
    "lwc84_residual",
    "sample_fixed_physics_residual",
]
