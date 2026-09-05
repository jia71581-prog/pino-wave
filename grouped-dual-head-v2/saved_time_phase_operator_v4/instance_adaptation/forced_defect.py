"""Source-consistent effective saved-grid LWC-84 defects for causal adaptation.

The historical instance-adaptation residual is intentionally homogeneous.  It
is useful on a source-free tail, but it omits the Ricker source while it is
active.  This module implements the complete *effective saved-grid* forced
recurrence without changing the legacy residual implementation.

The dataset was integrated with 20 internal steps per saved frame and then
restricted by a binomial5 filter.  Those hidden fine states cannot be recovered
from the saved field.  Consequently this is deliberately a source-consistent
coarse defect, not an exact residual of the fine-grid data generator; the
offline meta-learned basis models that composition/restriction closure gap.

For a pressure field ``p`` the generator advances

    p[n+1] = 2 p[n] - p[n-1]
             + dt**2 * (L p[n] + q[n])
             + dt**4 / 12 * (L(L p[n] + q[n]) + q_tt[n]),

where ``L p = c**2 laplacian_8(p)``.  The defect below is the left hand side
minus the right hand side after division by ``dt**2``.  It is affine in the
field and its correction operator (obtained by omitting the source) is linear,
which is the property needed by the small convex online solve.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math

import torch
from torch.nn import functional as F

from src.fno_acoustic.data_generation.cpml import (
    CFSCPMLOperator,
    build_cfs_cpml_profiles,
)
from src.fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig
from src.fno_acoustic.data_generation.lwc84 import apply_l_operator


_D2_8 = (
    -1.0 / 560.0,
    8.0 / 315.0,
    -1.0 / 5.0,
    8.0 / 5.0,
    -205.0 / 72.0,
    8.0 / 5.0,
    -1.0 / 5.0,
    8.0 / 315.0,
    -1.0 / 560.0,
)

# Public read-only stencil identity used by the hyper-reduced CPADC deployment
# design.  Keeping one coefficient source prevents the sparse point operator
# from drifting away from ``laplacian8``.
LAPLACIAN8_COEFFICIENTS = _D2_8


@dataclass(frozen=True)
class SavedGridCPMLConfig:
    """Auditable saved-grid equivalent of the production three-sided CPML.

    The stored 201 x 201 field excludes the fine-grid CPML pressure and memory
    variables.  The registered causal closure injects each supplied physical
    frame, keeps the unavailable exterior pressure at its zero initial value,
    and recursively evolves the four CFS memory fields.  Twenty 10 m cells preserve the exact
    200 m physical thickness of the generator's forty 5 m cells.  The CFS
    coefficients use the same reflection target, polynomial, kappa, alpha and
    6750 m/s reference speed as the production solver.

    This is an effective saved-grid closure, not a claim that the omitted 20
    fine substeps or binomial restriction can be inverted.
    """

    npml: int = 20
    c_ref_mps: float = 6750.0
    target_reflection: float = 1.0e-8
    polynomial_order: int = 3
    kappa_max: float = 3.0
    minimum_frequency_hz: float = 8.0
    internal_dt_s: float = 1.25e-4
    internal_substeps_per_saved_frame: int = 20
    memory_time_integration: str = "internal_substep_exact_linear_causal_vectorized"
    exterior_initialization: str = "zero"
    physical_state_injection: str = "hard_each_saved_frame"

    def __post_init__(self) -> None:
        if int(self.npml) < 4:
            raise ValueError("saved-grid CPML requires at least four cells")
        positive = (
            self.c_ref_mps,
            self.target_reflection,
            self.kappa_max,
            self.minimum_frequency_hz,
            self.internal_dt_s,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("saved-grid CPML parameters must be finite and positive")
        if not 0.0 < float(self.target_reflection) < 1.0:
            raise ValueError("saved-grid CPML reflection target must lie in (0,1)")
        if int(self.polynomial_order) < 1:
            raise ValueError("saved-grid CPML polynomial order must be positive")
        if int(self.internal_substeps_per_saved_frame) < 1:
            raise ValueError("saved-grid CPML internal substep count must be positive")
        if self.exterior_initialization != "zero":
            raise ValueError("only causal zero CPML exterior initialization is supported")
        if self.physical_state_injection != "hard_each_saved_frame":
            raise ValueError("unsupported saved-grid CPML physical-state closure")
        if self.memory_time_integration not in {
            "internal_substep_exact_linear_causal_vectorized",
            "internal_substep_exact_zoh_vectorized",
        }:
            raise ValueError("unsupported saved-grid CPML memory integration")

    def as_dict(self) -> dict[str, object]:
        return dict(asdict(self))


def saved_grid_cpml_config(
    value: SavedGridCPMLConfig | Mapping[str, object] | None,
) -> SavedGridCPMLConfig | None:
    """Normalize a checkpoint/config CPML mapping without silent defaults."""

    if value is None:
        return None
    if isinstance(value, SavedGridCPMLConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("pde_cpml must be a mapping or SavedGridCPMLConfig")
    enabled = bool(value.get("enabled", True))
    if not enabled:
        return None
    allowed = {field.name for field in SavedGridCPMLConfig.__dataclass_fields__.values()}
    unknown = set(str(key) for key in value) - allowed - {"enabled"}
    if unknown:
        raise ValueError(f"unknown saved-grid CPML settings: {sorted(unknown)}")
    return SavedGridCPMLConfig(
        **{str(key): item for key, item in value.items() if str(key) != "enabled"}
    )


def laplacian8(field: torch.Tensor, *, dx_m: float, dz_m: float) -> torch.Tensor:
    """Apply the data generator's radius-four spatial Laplacian."""

    value = torch.as_tensor(field)
    if value.ndim < 2 or min(value.shape[-2:]) <= 8:
        raise ValueError("field grid is too small for the radius-four stencil")
    if float(dx_m) <= 0.0 or float(dz_m) <= 0.0:
        raise ValueError("LWC-84 spatial steps must be positive")
    padded = F.pad(value, (4, 4, 4, 4))
    height, width = value.shape[-2:]
    dxx = torch.zeros_like(value)
    dzz = torch.zeros_like(value)
    for index, coefficient in enumerate(_D2_8):
        if index == 4:
            continue
        dxx = dxx + coefficient * (
            padded[..., 4 : 4 + height, index : index + width] - value
        )
        dzz = dzz + coefficient * (
            padded[..., index : index + height, 4 : 4 + width] - value
        )
    return dxx / float(dx_m) ** 2 + dzz / float(dz_m) ** 2


def _batched_time_axis(
    time_s: torch.Tensor,
    *,
    records: int,
    dtype: torch.dtype,
    device: torch.device,
    minimum_count: int = 3,
) -> torch.Tensor:
    axis = torch.as_tensor(time_s, dtype=dtype, device=device)
    if axis.ndim == 1:
        axis = axis.unsqueeze(0).expand(int(records), -1)
    if axis.ndim != 2 or axis.shape[0] != int(records):
        raise ValueError("time_s must be [time] or [record,time]")
    if axis.shape[1] < int(minimum_count) or not bool(torch.isfinite(axis).all()):
        raise ValueError(
            f"time_s must contain at least {int(minimum_count)} finite samples"
        )
    return axis


def _validate_uniform_time_spacing(axis: torch.Tensor, *, dt: float) -> None:
    """Validate a saved time axis without rejecting float32 roundoff."""

    deltas = axis[:, 1:] - axis[:, :-1]
    dtype_epsilon = (
        torch.finfo(axis.dtype).eps
        if axis.dtype.is_floating_point
        else torch.finfo(torch.float64).eps
    )
    tolerance = max(
        1.0e-10,
        abs(float(dt)) * 1.0e-5,
        4.0 * float(dtype_epsilon) * float(axis.detach().abs().max()),
    )
    if (
        not bool(torch.isfinite(deltas).all())
        or bool(torch.any(deltas <= 0.0))
        or float((deltas - float(dt)).abs().max()) > tolerance
    ):
        raise ValueError("time_s must be uniformly spaced by dt")


def lwc84_source_terms(
    source_parameters: torch.Tensor,
    source_map: torch.Tensor,
    time_s: torch.Tensor,
    *,
    dx_m: float,
    dz_m: float,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    field_scale_pa: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``q`` and ``q_tt`` in the same units as the supplied field.

    ``source_parameters`` use the physical ``[x,z,f0,t0,amplitude]`` contract
    and ``source_map`` is the saved-grid bilinear point-source map.  If the
    field has been encoded by ``PhysicalNormalizer.encode_pressure``, callers
    must pass ``field_scale_pa = pressure_scale_pa * amplitude``.  Omitting the
    scale means that the field is in physical pressure units.
    """

    inferred = torch.as_tensor(source_parameters)
    target_dtype = dtype or (
        inferred.dtype if inferred.is_floating_point() else torch.float32
    )
    target_device = torch.device(device) if device is not None else inferred.device
    parameters = torch.as_tensor(
        source_parameters, dtype=target_dtype, device=target_device
    )
    if parameters.ndim != 2 or parameters.shape[1] != 5:
        raise ValueError("source parameters must be [record,5]")
    if not bool(torch.isfinite(parameters).all()):
        raise ValueError("source parameters must be finite")
    if bool(torch.any(parameters[:, 2] <= 0.0)):
        raise ValueError("source frequencies must be positive")

    spatial = torch.as_tensor(source_map, dtype=target_dtype, device=target_device)
    if spatial.ndim == 3:
        spatial = spatial[:, None]
    if spatial.ndim != 4 or spatial.shape[:2] != (parameters.shape[0], 1):
        raise ValueError("source map must be [record,1,z,x]")
    if not bool(torch.isfinite(spatial).all()) or bool(torch.any(spatial < 0.0)):
        raise ValueError("source map must be finite and nonnegative")

    axis = _batched_time_axis(
        time_s,
        records=parameters.shape[0],
        dtype=target_dtype,
        device=target_device,
        minimum_count=1,
    )
    frequency = parameters[:, 2:3]
    onset = parameters[:, 3:4]
    amplitude = parameters[:, 4:5]
    tau = axis - onset
    ricker_a = math.pi**2 * frequency.square()
    exponential = torch.exp(-ricker_a * tau.square())
    wavelet = (1.0 - 2.0 * ricker_a * tau.square()) * exponential
    wavelet_tt = (
        -6.0 * ricker_a
        + 24.0 * ricker_a.square() * tau.square()
        - 8.0 * ricker_a.pow(3) * tau.pow(4)
    ) * exponential
    delta = spatial[:, 0] / (float(dx_m) * float(dz_m))
    source = amplitude[:, :, None, None] * wavelet[:, :, None, None] * delta[:, None]
    source_tt = (
        amplitude[:, :, None, None]
        * wavelet_tt[:, :, None, None]
        * delta[:, None]
    )

    if field_scale_pa is not None:
        scale = torch.as_tensor(
            field_scale_pa, dtype=target_dtype, device=target_device
        ).flatten()
        if scale.numel() == 1:
            scale = scale.expand(parameters.shape[0])
        if (
            scale.shape != (parameters.shape[0],)
            or not bool(torch.isfinite(scale).all())
            or bool(torch.any(scale <= 0.0))
        ):
            raise ValueError("field_scale_pa must contain one finite positive value per record")
        source = source / scale[:, None, None, None]
        source_tt = source_tt / scale[:, None, None, None]
    return source, source_tt


def lwc84_step(
    p_nm1: torch.Tensor,
    p_n: torch.Tensor,
    velocity_mps: torch.Tensor,
    *,
    source: torch.Tensor,
    source_tt: torch.Tensor,
    dt_s: float,
    dx_m: float,
    dz_m: float,
) -> torch.Tensor:
    """Advance one complete forced LWC-84 step on the saved grid."""

    previous = torch.as_tensor(p_nm1)
    current = torch.as_tensor(p_n, dtype=previous.dtype, device=previous.device)
    velocity = torch.as_tensor(
        velocity_mps, dtype=previous.dtype, device=previous.device
    )
    if previous.shape != current.shape or velocity.shape != current.shape:
        raise ValueError("step fields and velocity must have matching [record,z,x] shapes")
    forcing = torch.as_tensor(source, dtype=current.dtype, device=current.device)
    forcing_tt = torch.as_tensor(source_tt, dtype=current.dtype, device=current.device)
    if forcing.shape != current.shape or forcing_tt.shape != current.shape:
        raise ValueError("source terms must match the step field")
    operator = velocity.square() * laplacian8(
        current, dx_m=float(dx_m), dz_m=float(dz_m)
    )
    acceleration = operator + forcing
    fourth = velocity.square() * laplacian8(
        acceleration, dx_m=float(dx_m), dz_m=float(dz_m)
    ) + forcing_tt
    step = float(dt_s)
    if step <= 0.0:
        raise ValueError("LWC-84 time step must be positive")
    return (
        2.0 * current
        - previous
        + step**2 * acceleration
        + step**4 * fourth / 12.0
    )


def _physical_embedding(field: torch.Tensor, *, npml: int) -> torch.Tensor:
    """Embed a physical [z,x] field in left/right/bottom exterior cells."""

    return F.pad(field, (int(npml), int(npml), 0, int(npml)))


def _free_surface_physical(field: torch.Tensor) -> torch.Tensor:
    height = int(field.shape[-2])
    z_mask = torch.ones((height,), dtype=field.dtype, device=field.device)
    z_mask[0] = 0.0
    return field * z_mask[:, None]


def _inject_physical_state(
    exterior_state: torch.Tensor,
    physical_state: torch.Tensor,
    *,
    npml: int,
) -> torch.Tensor:
    embedded = _physical_embedding(
        _free_surface_physical(physical_state), npml=int(npml)
    )
    physical_mask = _physical_embedding(
        torch.ones_like(physical_state), npml=int(npml)
    )
    return exterior_state * (1.0 - physical_mask) + embedded


def _outer_dirichlet_mask(field: torch.Tensor) -> torch.Tensor:
    """Match the production solver's top/far-side explicit zero clamps."""

    height, width = field.shape[-2:]
    z = torch.ones((height,), dtype=field.dtype, device=field.device)
    x = torch.ones((width,), dtype=field.dtype, device=field.device)
    z[0] = 0.0
    z[-1] = 0.0
    x[0] = 0.0
    x[-1] = 0.0
    return z[:, None] * x[None, :]


def _lwc84_cpml_discrete_defect(
    value: torch.Tensor,
    velocity: torch.Tensor,
    *,
    dt: float,
    dx: float,
    dz: float,
    observed_indices: tuple[int, int],
    source_parameters: torch.Tensor | None,
    source_map: torch.Tensor | None,
    time_s: torch.Tensor | None,
    field_scale_pa: torch.Tensor | float | None,
    time_order: int,
    normalization_scale: torch.Tensor | None,
    normalize: bool,
    return_scale: bool,
    cpml_config: SavedGridCPMLConfig,
    cpml_gradient_start_index: int,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Causally close omitted CPML states and evaluate the LWC recurrence.

    At every saved center the supplied physical field is injected exactly and
    the four CFS memory variables are evolved recursively.  The omitted
    exterior pressure follows the checkpointed zero closure, so boundary
    residuals never use future truth or an unregistered halo.  The top row is
    governed by the explicit free-surface condition.
    """

    records, time_count, height, width = value.shape
    npml = int(cpml_config.npml)
    grid = AcousticGrid(
        nx=int(width),
        nz=int(height),
        dx_m=float(dx),
        dz_m=float(dz),
        centering="node",
    )
    boundaries = BoundaryConfig(
        top="free_surface_dirichlet",
        left="cpml",
        right="cpml",
        bottom="cpml",
        npml=npml,
        cpml_target_reflection=float(cpml_config.target_reflection),
        cpml_polynomial_order=int(cpml_config.polynomial_order),
        cpml_outside_physical_domain=True,
    )
    internal_step = float(cpml_config.internal_dt_s)
    internal_substeps = int(cpml_config.internal_substeps_per_saved_frame)
    if not math.isclose(
        internal_step * internal_substeps,
        float(dt),
        rel_tol=1.0e-6,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "CPML internal_dt_s * internal_substeps_per_saved_frame must equal dt"
        )
    profiles = build_cfs_cpml_profiles(
        grid,
        boundaries,
        dt_s=internal_step,
        c_ref_mps=float(cpml_config.c_ref_mps),
        target_reflection=float(cpml_config.target_reflection),
        polynomial_order=int(cpml_config.polynomial_order),
        kappa_max=float(cpml_config.kappa_max),
        minimum_frequency_hz=float(cpml_config.minimum_frequency_hz),
        device=value.device,
        dtype=value.dtype,
    )
    cpml = CFSCPMLOperator(profiles, dx_m=float(dx), dz_m=float(dz))
    extended_velocity = F.pad(
        velocity[:, 0], (npml, npml, 0, npml), mode="replicate"
    )
    boundary_mask = _outer_dirichlet_mask(extended_velocity)
    physical_z, physical_x = boundaries.physical_slices(grid)

    supplied = (
        source_parameters is not None,
        source_map is not None,
        time_s is not None,
    )
    if any(supplied) and not all(supplied):
        raise ValueError(
            "source_parameters, source_map, and time_s must be supplied together"
        )
    if not all(supplied) and field_scale_pa is not None:
        raise ValueError("field_scale_pa is only valid for a forced defect")
    if all(supplied):
        assert source_parameters is not None
        assert source_map is not None
        assert time_s is not None
        axis = _batched_time_axis(
            time_s,
            records=records,
            dtype=value.dtype,
            device=value.device,
            minimum_count=3,
        )
        if axis.shape[1] != time_count:
            raise ValueError("time_s length must match the field time dimension")
        _validate_uniform_time_spacing(axis, dt=float(dt))
        source, source_tt = lwc84_source_terms(
            source_parameters,
            source_map,
            axis,
            dx_m=float(dx),
            dz_m=float(dz),
            dtype=value.dtype,
            device=value.device,
            field_scale_pa=field_scale_pa,
        )
    else:
        source = value.new_zeros((records, time_count, height, width))
        source_tt = torch.zeros_like(source)

    defects: list[torch.Tensor] = []
    step = float(dt)
    gradient_start = int(cpml_gradient_start_index)
    if gradient_start < 0 or gradient_start >= time_count - 1:
        raise ValueError("CPML gradient start index is outside recurrence support")

    for center in range(1, time_count - 1):
        forcing = _physical_embedding(source[:, center], npml=npml)
        forcing_tt = _physical_embedding(source_tt[:, center], npml=npml)
        previous_physical = _free_surface_physical(value[:, center - 1])
        current_physical = _free_surface_physical(value[:, center])
        # The CPML state through ``gradient_start`` is a causal warm-up value,
        # not part of the differentiable loss window.  Detaching both endpoints
        # here prevents the final warm-up update from pulling one additional
        # pre-window frame into the graph.  The first retained update still
        # depends on frame ``gradient_start`` as its previous endpoint.
        if center <= gradient_start:
            previous_physical = previous_physical.detach()
            current_physical = current_physical.detach()
        previous = _physical_embedding(
            previous_physical, npml=npml
        ) * boundary_mask
        current = _physical_embedding(
            current_physical, npml=npml
        ) * boundary_mask
        if (
            cpml_config.memory_time_integration
            == "internal_substep_exact_linear_causal_vectorized"
        ):
            cpml_acceleration = cpml.apply_linear_trajectory(
                previous,
                current,
                extended_velocity,
                substeps=internal_substeps,
                update_memory=True,
            )
        else:
            # Schema-7 compatibility only.  New checkpoints must register the
            # causal linear reconstruction above.
            cpml_acceleration = cpml.apply_repeated_zero_order_hold(
                current,
                extended_velocity,
                substeps=internal_substeps,
                update_memory=True,
            )
        acceleration = cpml_acceleration + forcing
        right_hand_side = acceleration
        if int(time_order) == 4:
            fourth = apply_l_operator(
                acceleration,
                extended_velocity,
                dx_m=float(dx),
                dz_m=float(dz),
                boundary="free_surface",
            ) + forcing_tt
            right_hand_side = right_hand_side + step**2 * fourth / 12.0
        temporal = (
            value[:, center + 1]
            - 2.0 * value[:, center]
            + value[:, center - 1]
        ) / step**2
        physical_rhs = right_hand_side[..., physical_z, physical_x]
        local_defect = temporal - physical_rhs
        # Production clamps the physical top row to p=0 after every step.
        free_surface_defect = value[:, center, :1, :] / step**2
        local_defect = torch.cat(
            (free_surface_defect, local_defect[:, 1:, :]), dim=-2
        )
        defects.append(local_defect)


    defect = torch.stack(defects, dim=1)
    start = max(int(observed_indices[1]), 0)
    defect = defect[:, start:]
    temporal_support = (
        value[:, 2:] - 2.0 * value[:, 1:-1] + value[:, :-2]
    )[:, start:] / step**2
    if normalization_scale is None:
        scale = (
            temporal_support.detach().square()
            + (temporal_support.detach() - defect.detach()).square()
        ).flatten(1).mean(dim=1).sqrt().clamp_min(1.0e-8)
    else:
        scale = torch.as_tensor(
            normalization_scale, dtype=value.dtype, device=value.device
        ).flatten()
        if scale.shape != (records,) or not bool(torch.isfinite(scale).all()):
            raise ValueError("normalization_scale must contain one finite value per record")
        scale = scale.clamp_min(1.0e-8)
    result = defect / scale[:, None, None, None] if bool(normalize) else defect
    return (result, scale) if bool(return_scale) else result


def lwc84_discrete_defect(
    field: torch.Tensor,
    velocity_mps: torch.Tensor,
    *,
    dt: float,
    dx: float,
    dz: float,
    observed_indices: tuple[int, int],
    source_parameters: torch.Tensor | None = None,
    source_map: torch.Tensor | None = None,
    time_s: torch.Tensor | None = None,
    field_scale_pa: torch.Tensor | float | None = None,
    time_order: int = 4,
    normalization_scale: torch.Tensor | None = None,
    normalize: bool = True,
    return_scale: bool = False,
    cpml_config: SavedGridCPMLConfig | Mapping[str, object] | None = None,
    cpml_gradient_start_index: int = 0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute a source-consistent LWC-84 defect on supported interior cells.

    Passing no source arguments returns the *linear homogeneous correction
    operator*.  Passing all of ``source_parameters``, ``source_map`` and
    ``time_s`` returns the affine forced defect of a candidate solution.  The
    fourth-order form crops an eight-cell spatial halo because ``L(Lp)`` has
    radius eight.  The second-order form crops four cells.  This is stricter
    than the legacy residual and makes patch interiors independent of the
    artificial patch boundary.
    """

    value = torch.as_tensor(field)
    velocity = torch.as_tensor(velocity_mps, dtype=value.dtype, device=value.device)
    if value.ndim != 4 or value.shape[1] < 3:
        raise ValueError("field must be [record,time,z,x] with at least three times")
    if velocity.ndim == 3:
        velocity = velocity[:, None]
    if velocity.shape != (value.shape[0], 1, value.shape[2], value.shape[3]):
        raise ValueError("velocity shape does not match field")
    spatial_margin = 8 if int(time_order) == 4 else 4
    if min(value.shape[-2:]) <= 2 * spatial_margin:
        raise ValueError("field grid is too small for the effective defect support")
    if float(dt) <= 0.0 or float(dx) <= 0.0 or float(dz) <= 0.0:
        raise ValueError("LWC-84 spacings must be positive")
    if int(time_order) not in (2, 4):
        raise ValueError("time_order must be 2 or 4")

    normalized_cpml = saved_grid_cpml_config(cpml_config)
    if normalized_cpml is not None:
        return _lwc84_cpml_discrete_defect(
            value,
            velocity,
            dt=float(dt),
            dx=float(dx),
            dz=float(dz),
            observed_indices=observed_indices,
            source_parameters=source_parameters,
            source_map=source_map,
            time_s=time_s,
            field_scale_pa=field_scale_pa,
            time_order=int(time_order),
            normalization_scale=normalization_scale,
            normalize=bool(normalize),
            return_scale=bool(return_scale),
            cpml_config=normalized_cpml,
            cpml_gradient_start_index=int(cpml_gradient_start_index),
        )
    if int(cpml_gradient_start_index) != 0:
        raise ValueError("cpml_gradient_start_index requires an enabled CPML")

    supplied = (
        source_parameters is not None,
        source_map is not None,
        time_s is not None,
    )
    if any(supplied) and not all(supplied):
        raise ValueError(
            "source_parameters, source_map, and time_s must be supplied together"
        )
    if not all(supplied) and field_scale_pa is not None:
        raise ValueError("field_scale_pa is only valid for a forced defect")

    second_time = (
        value[:, 2:] - 2.0 * value[:, 1:-1] + value[:, :-2]
    ) / float(dt) ** 2
    c2 = velocity.square()
    operator = c2 * laplacian8(value[:, 1:-1], dx_m=float(dx), dz_m=float(dz))

    if all(supplied):
        assert source_parameters is not None
        assert source_map is not None
        assert time_s is not None
        axis = _batched_time_axis(
            time_s,
            records=value.shape[0],
            dtype=value.dtype,
            device=value.device,
            minimum_count=3,
        )
        if axis.shape[1] != value.shape[1]:
            raise ValueError("time_s length must match the field time dimension")
        _validate_uniform_time_spacing(axis, dt=float(dt))
        source, source_tt = lwc84_source_terms(
            source_parameters,
            source_map,
            axis[:, 1:-1],
            dx_m=float(dx),
            dz_m=float(dz),
            dtype=value.dtype,
            device=value.device,
            field_scale_pa=field_scale_pa,
        )
        if source.shape != operator.shape:
            raise ValueError("source map spatial shape does not match the field")
    else:
        source = torch.zeros_like(operator)
        source_tt = torch.zeros_like(operator)

    acceleration = operator + source
    right_hand_side = acceleration
    if int(time_order) == 4:
        fourth = c2 * laplacian8(
            acceleration, dx_m=float(dx), dz_m=float(dz)
        ) + source_tt
        right_hand_side = right_hand_side + float(dt) ** 2 * fourth / 12.0

    spatial_slice = slice(spatial_margin, -spatial_margin)
    temporal_acceleration = second_time[..., spatial_slice, spatial_slice]
    spatial_acceleration = right_hand_side[..., spatial_slice, spatial_slice]
    start = max(int(observed_indices[1]), 0)
    temporal_acceleration = temporal_acceleration[:, start:]
    spatial_acceleration = spatial_acceleration[:, start:]
    defect = temporal_acceleration - spatial_acceleration

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
            raise ValueError("normalization_scale must contain one finite value per record")
        scale = scale.clamp_min(1.0e-8)
    result = defect / scale[:, None, None, None] if bool(normalize) else defect
    return (result, scale) if bool(return_scale) else result


__all__ = [
    "LAPLACIAN8_COEFFICIENTS",
    "SavedGridCPMLConfig",
    "laplacian8",
    "lwc84_discrete_defect",
    "lwc84_source_terms",
    "lwc84_step",
    "saved_grid_cpml_config",
]
