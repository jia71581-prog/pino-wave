"""Short synthetic onset bridge using the same LWC-84 recurrence as the teacher."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from .contracts import SyntheticBridgeProvenance


_D2_8 = (
    -1.0 / 560.0, 8.0 / 315.0, -1.0 / 5.0, 8.0 / 5.0,
    -205.0 / 72.0, 8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0,
)


def _laplacian8(field: torch.Tensor, *, dx_m: float, dz_m: float) -> torch.Tensor:
    if min(field.shape[-2:]) <= 8:
        raise ValueError("grid is too small for the radius-four stencil")
    padded = F.pad(field, (4, 4, 4, 4))
    dxx = torch.zeros_like(field)
    dzz = torch.zeros_like(field)
    height, width = field.shape[-2:]
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


def _lwc84_step(
    p_nm1,
    p_n,
    velocity_mps,
    *,
    source,
    source_tt,
    dt_s,
    dx_m,
    dz_m,
):
    acceleration = (
        velocity_mps.square() * _laplacian8(p_n, dx_m=dx_m, dz_m=dz_m)
        + source
    )
    fourth = (
        velocity_mps.square()
        * _laplacian8(acceleration, dx_m=dx_m, dz_m=dz_m)
        + source_tt
    )
    return 2.0 * p_n - p_nm1 + dt_s**2 * acceleration + dt_s**4 * fourth / 12.0


def _source_terms(
    source_parameters: torch.Tensor,
    source_map: torch.Tensor | None,
    time_s: torch.Tensor,
    *,
    dx_m: float,
    dz_m: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    parameters = torch.as_tensor(
        source_parameters, dtype=torch.float32, device=device
    )
    if parameters.ndim != 2 or parameters.shape[1] != 5:
        raise ValueError("source parameters must be [record,5]")
    if source_map is None:
        shape = (parameters.shape[0], len(time_s), 1, 1)
        zeros = torch.zeros(shape, dtype=parameters.dtype, device=device)
        return zeros, zeros
    spatial = torch.as_tensor(source_map, dtype=torch.float32, device=device)
    if spatial.ndim == 3:
        spatial = spatial[:, None]
    if spatial.ndim != 4 or spatial.shape[:2] != (parameters.shape[0], 1):
        raise ValueError("source map must be [record,1,z,x]")
    if not bool(torch.isfinite(spatial).all()) or bool(torch.any(spatial < 0.0)):
        raise ValueError("source map must be finite and nonnegative")
    axis = torch.as_tensor(time_s, dtype=parameters.dtype, device=device)
    frequency = parameters[:, 2:3]
    onset = parameters[:, 3:4]
    amplitude = parameters[:, 4:5]
    tau = axis[None] - onset
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
    return source, source_tt


@dataclass(frozen=True)
class BridgeResult:
    frames: torch.Tensor
    time_indices: tuple[int, ...]
    provenance: SyntheticBridgeProvenance
    valid: bool
    failure_reason: str | None = None
    substeps_per_saved_step: int = 1


def make_onset_bridge(
    velocity_mps: torch.Tensor,
    source_parameters: torch.Tensor,
    observed_wavefield: torch.Tensor,
    observed_indices: tuple[int, int],
    time_s: torch.Tensor,
    *,
    source_map: torch.Tensor | None = None,
    steps: int = 4,
    dx_m: float = 10.0,
    dz_m: float = 10.0,
    device: str | torch.device = "cpu",
) -> BridgeResult:
    """Generate only synthetic frames after k1; no HDF5 access occurs here."""
    count = int(steps)
    raw_axis = torch.as_tensor(time_s)
    axis = raw_axis.to(dtype=torch.float64)
    observed = torch.as_tensor(observed_wavefield, dtype=torch.float32, device=device)
    velocity = torch.as_tensor(velocity_mps, dtype=torch.float32, device=device)
    if observed.ndim != 4 or observed.shape[1] != 2:
        raise ValueError("observed_wavefield must be [record,2,z,x]")
    if velocity.ndim == 3:
        velocity = velocity[:, None]
    if velocity.ndim != 4 or velocity.shape[0] != observed.shape[0]:
        raise ValueError("velocity must be [record,1,z,x]")
    if axis.ndim != 1 or axis.numel() < 2 or not torch.isfinite(axis).all():
        raise ValueError("time_s must be a finite saved axis")
    deltas = axis[1:] - axis[:-1]
    dt = float((axis[-1] - axis[0]) / max(int(axis.numel()) - 1, 1))
    dtype_epsilon = (
        torch.finfo(raw_axis.dtype).eps
        if raw_axis.dtype.is_floating_point
        else torch.finfo(torch.float64).eps
    )
    roundoff_tolerance = max(
        1.0e-10,
        8.0
        * float(dtype_epsilon)
        * max(float(axis.detach().abs().max()), abs(dt)),
    )
    if (
        dt <= 0.0
        or bool(torch.any(deltas <= 0.0))
        or float((deltas - dt).abs().max()) > roundoff_tolerance
    ):
        return BridgeResult(
            frames=observed[:, :0],
            time_indices=(),
            provenance=SyntheticBridgeProvenance.from_tensor(observed[:, :0], observed_indices),
            valid=False,
            failure_reason="saved time axis is not uniform",
        )
    saved_step_qmax = (
        dt**2
        * float(velocity.detach().abs().max()) ** 2
        * (2048.0 / 315.0)
        * (1.0 / float(dx_m) ** 2 + 1.0 / float(dz_m) ** 2)
    )
    substeps = max(1, int(math.ceil(math.sqrt(saved_step_qmax / 8.0))))
    if substeps > 64:
        return BridgeResult(
            frames=observed[:, :0],
            time_indices=(),
            provenance=SyntheticBridgeProvenance.from_tensor(
                observed[:, :0], observed_indices
            ),
            valid=False,
            failure_reason="required LWC-84 bridge substeps exceed safety cap",
            substeps_per_saved_step=substeps,
        )
    if count < 0 or int(observed_indices[1]) + count >= int(axis.numel()):
        return BridgeResult(
            frames=observed[:, :0],
            time_indices=(),
            provenance=SyntheticBridgeProvenance.from_tensor(observed[:, :0], observed_indices),
            valid=False,
            failure_reason="bridge exceeds the saved time axis",
        )
    try:
        previous_saved, p_n = observed[:, 0], observed[:, 1]
        dt_substep = dt / float(substeps)
        current_time = float(axis[int(observed_indices[1])])
        substep_axis = (
            torch.arange(
                max(1, count * substeps),
                dtype=torch.float64,
                device=torch.device(device),
            )
            * dt_substep
            + current_time
        )
        source, source_tt = _source_terms(
            source_parameters,
            source_map,
            substep_axis,
            dx_m=float(dx_m),
            dz_m=float(dz_m),
            device=torch.device(device),
        )
        time_derivative = (p_n - previous_saved) / dt
        acceleration = (
            velocity[:, 0].square()
            * _laplacian8(p_n, dx_m=float(dx_m), dz_m=float(dz_m))
            + source[:, 0]
        )
        p_nm1 = (
            p_n
            - dt_substep * time_derivative
            + 0.5 * dt_substep**2 * acceleration
        )
        generated: list[torch.Tensor] = []
        substep_index = 0
        for offset in range(count):
            del offset
            for _ in range(substeps):
                p_next = _lwc84_step(
                    p_nm1,
                    p_n,
                    velocity[:, 0],
                    source=source[:, substep_index],
                    source_tt=source_tt[:, substep_index],
                    dt_s=dt_substep,
                    dx_m=float(dx_m),
                    dz_m=float(dz_m),
                )
                p_next[:, 0, :] = 0.0
                p_nm1, p_n = p_n, p_next
                substep_index += 1
            generated.append(p_n)
        frames = torch.stack(generated, dim=1) if generated else observed[:, :0]
    except (RuntimeError, ValueError, FloatingPointError) as error:
        return BridgeResult(
            frames=observed[:, :0],
            time_indices=(),
            provenance=SyntheticBridgeProvenance.from_tensor(observed[:, :0], observed_indices),
            valid=False,
            failure_reason=str(error),
        )
    if not torch.isfinite(frames).all():
        return BridgeResult(
            frames=observed[:, :0],
            time_indices=(),
            provenance=SyntheticBridgeProvenance.from_tensor(observed[:, :0], observed_indices),
            valid=False,
            failure_reason="bridge produced nonfinite values",
        )
    indices = tuple(range(int(observed_indices[1]) + 1, int(observed_indices[1]) + count + 1))
    return BridgeResult(
        frames=frames,
        time_indices=indices,
        provenance=SyntheticBridgeProvenance.from_tensor(frames, observed_indices),
        valid=True,
        substeps_per_saved_step=substeps,
    )


__all__ = ["BridgeResult", "make_onset_bridge"]
