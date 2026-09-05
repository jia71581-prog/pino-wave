"""Training losses that emphasize unresolved spatial frequency bands."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


FAMILY_CLASS_ORDER = ("uniform", "layered", "marmousi")


def family_route_targets(
    medium_labels,
    *,
    medium_count: int,
    record_to_medium: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Resolve record labels to one registered class index per encoded medium."""

    labels = tuple(str(label) for label in medium_labels)
    count = int(medium_count)
    if count <= 0 or not labels or any(
        label not in FAMILY_CLASS_ORDER for label in labels
    ):
        raise ValueError("family router labels must use the registered families")
    if record_to_medium is None:
        if len(labels) != count:
            raise ValueError("family router labels require record_to_medium")
        targets = labels
    else:
        mapping = torch.as_tensor(record_to_medium, dtype=torch.long).cpu()
        if (
            mapping.shape != (len(labels),)
            or mapping.numel() == 0
            or int(mapping.min()) < 0
            or int(mapping.max()) >= count
        ):
            raise ValueError("family router record_to_medium is invalid")
        resolved: list[str | None] = [None] * count
        for label, medium_index in zip(labels, mapping.tolist(), strict=True):
            current = resolved[int(medium_index)]
            if current is not None and current != label:
                raise ValueError("records sharing a medium have conflicting family labels")
            resolved[int(medium_index)] = label
        if any(label is None for label in resolved):
            raise ValueError("family router mapping omits an encoded medium")
        targets = tuple(str(label) for label in resolved)
    class_index = {name: index for index, name in enumerate(FAMILY_CLASS_ORDER)}
    return torch.tensor(
        [class_index[label] for label in targets],
        dtype=torch.long,
        device=device,
    )


def family_router_loss(
    logits: torch.Tensor,
    medium_labels,
    *,
    record_to_medium: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise one velocity-derived route per unique encoded medium."""

    values = torch.as_tensor(logits)
    if values.ndim != 2 or values.shape[1] != len(FAMILY_CLASS_ORDER):
        raise ValueError("family router logits must have shape [medium,3]")
    target = family_route_targets(
        medium_labels,
        medium_count=values.shape[0],
        record_to_medium=record_to_medium,
        device=values.device,
    )
    loss = F.cross_entropy(values.float(), target)
    probabilities = values.float().softmax(dim=-1)
    entropy = -(
        probabilities * probabilities.clamp_min(1.0e-12).log()
    ).sum(dim=-1).mean()
    report = {
        "accuracy": float((probabilities.argmax(dim=-1) == target).float().mean()),
        "entropy": float(entropy),
    }
    for index, family in enumerate(FAMILY_CLASS_ORDER):
        report[f"route_probability_{family}"] = float(
            probabilities[:, index].mean()
        )
    return loss, report


def band_limited_residual_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    energy_floor_fraction: float = 0.05,
    band_weights: tuple[float, float, float] = (1.0, 1.5, 2.0),
) -> torch.Tensor:
    """Relative complex-spectrum error with a stable per-record energy floor."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("spectrum loss requires matching [record,time,z,x] fields")
    if not 0.0 < energy_floor_fraction <= 1.0 or any(value <= 0 for value in band_weights):
        raise ValueError("spectrum loss weights and energy floor must be positive")
    predicted = torch.fft.rfft2(prediction.float(), norm="ortho")
    reference = torch.fft.rfft2(target.float(), norm="ortho")
    height, width = target.shape[-2:]
    z_frequency = torch.fft.fftfreq(height, device=target.device).abs()
    x_frequency = torch.fft.rfftfreq(width, device=target.device).abs()
    radius = torch.sqrt(z_frequency[:, None].square() + x_frequency[None, :].square())
    radius = radius / radius.max().clamp_min(torch.finfo(radius.dtype).eps)
    masks = (
        radius <= 1.0 / 3.0,
        (radius > 1.0 / 3.0) & (radius <= 2.0 / 3.0),
        radius > 2.0 / 3.0,
    )
    total = reference.flatten(start_dim=1).norm(dim=-1)
    terms = []
    for weight, mask in zip(band_weights, masks, strict=True):
        error_norm = (predicted - reference)[..., mask].flatten(start_dim=1).norm(dim=-1)
        target_norm = reference[..., mask].flatten(start_dim=1).norm(dim=-1)
        denominator = torch.maximum(target_norm, energy_floor_fraction * total).clamp_min(1.0e-8)
        terms.append(float(weight) * (error_norm / denominator).mean())
    return sum(terms) / float(sum(band_weights))


@dataclass(frozen=True)
class RelativeEnergySquaredReference:
    """Full selected-time target energies reused by streamed loss blocks."""

    target_square: torch.Tensor
    spectrum_denominator_square: torch.Tensor
    spatial_shape: tuple[int, int]
    energy_floor_fraction: float
    band_weights: tuple[float, float, float]
    frame_target_square: torch.Tensor | None = None


@dataclass(frozen=True)
class RelativeEnergySquaredBlockLoss:
    """Additive squared-relative objective for one exact-time block."""

    total: torch.Tensor
    frame: torch.Tensor
    spectrum: torch.Tensor


def _spectrum_band_masks(
    height: int, width: int, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_frequency = torch.fft.fftfreq(int(height), device=device).abs()
    x_frequency = torch.fft.rfftfreq(int(width), device=device).abs()
    radius = torch.sqrt(
        z_frequency[:, None].square() + x_frequency[None, :].square()
    )
    radius = radius / radius.max().clamp_min(torch.finfo(radius.dtype).eps)
    return (
        radius <= 1.0 / 3.0,
        (radius > 1.0 / 3.0) & (radius <= 2.0 / 3.0),
        radius > 2.0 / 3.0,
    )


def relative_energy_squared_reference(
    target: torch.Tensor,
    *,
    energy_floor_fraction: float = 0.05,
    band_weights: tuple[float, float, float] = (1.0, 1.5, 2.0),
) -> RelativeEnergySquaredReference:
    """Build full-time denominators for an exactly decomposable stream loss."""

    reference = torch.as_tensor(target)
    floor = float(energy_floor_fraction)
    weights = tuple(float(value) for value in band_weights)
    if reference.ndim != 4:
        raise ValueError("relative-energy target must be [record,time,z,x]")
    if not 0.0 < floor <= 1.0 or len(weights) != 3 or any(
        value <= 0.0 for value in weights
    ):
        raise ValueError("relative-energy spectrum configuration is invalid")
    height, width = (int(value) for value in reference.shape[-2:])
    target_square = reference.float().square().flatten(1).sum(dim=-1).detach()
    target_spectrum = torch.fft.rfft2(reference.float(), norm="ortho")
    masks = _spectrum_band_masks(height, width, reference.device)
    band_square = torch.stack(
        tuple(
            target_spectrum[..., mask]
            .abs()
            .square()
            .flatten(start_dim=1)
            .sum(dim=-1)
            for mask in masks
        ),
        dim=0,
    ).detach()
    denominator = torch.maximum(
        band_square,
        floor**2 * target_square[None],
    ).clamp_min(1.0e-16)
    # Per-frame target energy for optional per-frame (time-reweighted) frame loss.
    # Floored at ``floor**2`` of the record's mean frame energy so near-zero late
    # frames cannot produce an exploding relative denominator (the V72 failure mode).
    frames = int(reference.shape[1])
    per_frame = reference.float().square().flatten(2).sum(dim=-1).detach()  # (records,frames)
    mean_frame = per_frame.mean(dim=1, keepdim=True).clamp_min(1.0e-16)
    frame_target_square = torch.maximum(per_frame, floor**2 * mean_frame).clamp_min(1.0e-16)
    return RelativeEnergySquaredReference(
        target_square=target_square.clamp_min(1.0e-16),
        spectrum_denominator_square=denominator,
        spatial_shape=(height, width),
        energy_floor_fraction=floor,
        band_weights=weights,
        frame_target_square=frame_target_square,
    )


def relative_energy_squared_block_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    reference: RelativeEnergySquaredReference,
    spectrum_weight: float,
    frame_time_weights: torch.Tensor | None = None,
    frame_index_range: tuple[int, int] | None = None,
) -> RelativeEnergySquaredBlockLoss:
    """Return one additive block of a full-time squared-relative objective.

    When ``frame_time_weights`` is None the frame term is the legacy
    record-energy-normalized mean (bit-identical to before).  When provided
    (a 1-D tensor over ALL selected frames, matched to this block via
    ``frame_index_range=(start, stop)``), the frame term instead normalizes each
    frame by its OWN energy (``reference.frame_target_square``) and weights it by
    ``frame_time_weights`` — so low-energy late frames become visible to the
    gradient.  Weights are renormalized to mean 1 over the full axis, keeping the
    overall loss scale comparable to the unweighted objective.
    """

    predicted = torch.as_tensor(prediction)
    expected = torch.as_tensor(target, device=predicted.device)
    if predicted.shape != expected.shape or predicted.ndim != 4:
        raise ValueError("relative-energy block fields must match [record,time,z,x]")
    records = int(predicted.shape[0])
    if (
        tuple(int(value) for value in predicted.shape[-2:]) != reference.spatial_shape
        or reference.target_square.shape != (records,)
        or reference.spectrum_denominator_square.shape != (3, records)
    ):
        raise ValueError("relative-energy reference does not match the loss block")
    spectral_weight = float(spectrum_weight)
    if not math.isfinite(spectral_weight) or spectral_weight < 0.0:
        raise ValueError("relative-energy spectrum weight must be nonnegative")

    difference = predicted.float() - expected.float()
    if frame_time_weights is None:
        error_square = difference.square().flatten(1).sum(dim=-1)
        frame = (error_square / reference.target_square).mean()
    else:
        if reference.frame_target_square is None:
            raise ValueError("per-frame weighting requires reference.frame_target_square")
        if frame_index_range is None:
            raise ValueError("frame_time_weights requires frame_index_range")
        start, stop = int(frame_index_range[0]), int(frame_index_range[1])
        if stop - start != int(predicted.shape[1]):
            raise ValueError("frame_index_range does not match the block frame count")
        weights = torch.as_tensor(frame_time_weights, dtype=torch.float32, device=predicted.device)
        if weights.ndim != 1:
            raise ValueError("frame_time_weights must be 1-D over all selected frames")
        # per-frame relative error, normalized by each frame's own energy
        error_per_frame = difference.square().flatten(2).sum(dim=-1)  # (records,block_frames)
        denom = reference.frame_target_square[:, start:stop]          # (records,block_frames)
        rel_per_frame = error_per_frame / denom
        block_weights = weights[start:stop]                           # (block_frames,)
        # mean 1 over the FULL axis so scale matches the unweighted loss
        norm = weights.mean().clamp_min(1.0e-12)
        frame = (rel_per_frame * (block_weights / norm)[None, :]).mean()
    if spectral_weight > 0.0:
        spectrum_error = torch.fft.rfft2(difference, norm="ortho")
        masks = _spectrum_band_masks(*reference.spatial_shape, predicted.device)
        terms = []
        for index, (weight, mask) in enumerate(
            zip(reference.band_weights, masks, strict=True)
        ):
            band_error_square = (
                spectrum_error[..., mask]
                .abs()
                .square()
                .flatten(start_dim=1)
                .sum(dim=-1)
            )
            terms.append(
                weight
                * (
                    band_error_square
                    / reference.spectrum_denominator_square[index]
                ).mean()
            )
        spectrum = sum(terms) / float(sum(reference.band_weights))
    else:
        spectrum = predicted.new_zeros(())
    return RelativeEnergySquaredBlockLoss(
        total=frame + spectral_weight * spectrum,
        frame=frame,
        spectrum=spectrum,
    )


@dataclass(frozen=True)
class ResidualRecoveryLoss:
    total: torch.Tensor
    frame: torch.Tensor
    delta: torch.Tensor
    temporal: torch.Tensor
    gradient: torch.Tensor
    spectrum: torch.Tensor
    pde: torch.Tensor | None = None
    pml: torch.Tensor | None = None


def source_causality_onset_s(
    source_parameters: torch.Tensor,
    *,
    lead_cycles: float = 0.0,
) -> torch.Tensor:
    """Return a causal cutoff a fixed number of cycles before the Ricker peak."""

    source = torch.as_tensor(source_parameters)
    cycles = float(lead_cycles)
    if source.ndim != 2 or source.shape[1] != 5:
        raise ValueError("causality source parameters must have shape [record,5]")
    if not math.isfinite(cycles) or cycles < 0.0:
        raise ValueError("hard-causality lead cycles must be finite and nonnegative")
    frequency = source[:, 2]
    if not torch.isfinite(frequency).all() or torch.any(frequency <= 0.0):
        raise ValueError("hard-causality source frequency must be finite and positive")
    return source[:, 3] - cycles / frequency


def apply_hard_causality(
    field: torch.Tensor,
    time_s: torch.Tensor,
    source_onset_s: torch.Tensor,
) -> torch.Tensor:
    """Enforce a zero acoustic field before each source onset."""

    values = torch.as_tensor(field)
    times = torch.as_tensor(time_s, dtype=values.dtype, device=values.device)
    onset = torch.as_tensor(source_onset_s, dtype=values.dtype, device=values.device)
    if values.ndim != 4 or times.shape != values.shape[:2] or onset.shape != (values.shape[0],):
        raise ValueError("causality inputs must be field[R,T,Z,X], time[R,T], onset[R]")
    active = times >= onset[:, None]
    return values * active[:, :, None, None].to(values.dtype)


def wave_pde_residual_loss(
    field: torch.Tensor,
    velocity_mps: torch.Tensor,
    time_s: torch.Tensor,
    *,
    dz_m: float,
    dx_m: float,
    source_free_mask: torch.Tensor | None = None,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Relative, source-free acoustic wave-equation residual on a dense field.

    Penalizes departure of the predicted field ``u`` from ``u_tt = c^2 * lap(u)``
    on the SOURCE-FREE interior (where the forcing ``f`` vanishes), directly
    targeting wrong wavefront position/phase via the differential operator rather
    than only pointwise amplitude.  All derivatives are finite differences on the
    already-dense ``[record, time, z, x]`` field — no autograd or re-evaluation.

    ``u_tt`` uses a NON-uniform 3-point second difference from the actual
    ``time_s`` spacing, so adaptively (non-uniformly) sampled frames are handled
    correctly.  The residual is normalized RELATIVE to the local operator scale
    (``mean(u_tt^2) + mean((c^2 lap)^2)``) so the term is dimensionless, bounded,
    and scale-stable across families — it never dominates the data loss by units.

    Returns a scalar in roughly ``[0, 1]``.  With fewer than 3 frames (no interior
    time point) or an all-false ``source_free_mask`` it returns 0 (no constraint).
    """

    u = torch.as_tensor(field).float()
    if u.ndim != 4:
        raise ValueError("wave PDE field must be [record,time,z,x]")
    records, frames, height, width = u.shape
    c = torch.as_tensor(velocity_mps, device=u.device).float()
    if c.shape != (records, height, width):
        raise ValueError("velocity must be [record,z,x] matching the field grid")
    times = torch.as_tensor(time_s, device=u.device).float()
    if times.shape != (records, frames):
        raise ValueError("time_s must be [record,time] matching the field")
    if not math.isfinite(dz_m) or not math.isfinite(dx_m) or dz_m <= 0.0 or dx_m <= 0.0:
        raise ValueError("grid spacings dz_m, dx_m must be positive and finite")
    if frames < 3 or height < 3 or width < 3:
        return u.new_zeros(())

    # --- non-uniform second time derivative on interior frames [1 .. T-2] ---
    t_prev, t_cur, t_nxt = times[:, :-2], times[:, 1:-1], times[:, 2:]
    h1 = (t_cur - t_prev).clamp_min(eps)[:, :, None, None]   # (R, T-2, 1, 1)
    h2 = (t_nxt - t_cur).clamp_min(eps)[:, :, None, None]
    u0, u1, u2 = u[:, :-2], u[:, 1:-1], u[:, 2:]
    # standard 3-point non-uniform d2u/dt2
    u_tt = 2.0 * (h2 * u0 - (h1 + h2) * u1 + h1 * u2) / (h1 * h2 * (h1 + h2))

    # --- spatial Laplacian on interior nodes, aligned to the same interior frames ---
    u_c = u1  # field at the interior time points
    lap = torch.zeros_like(u_c)
    lap[:, :, 1:-1, :] += (u_c[:, :, 2:, :] - 2.0 * u_c[:, :, 1:-1, :] + u_c[:, :, :-2, :]) / (dz_m * dz_m)
    lap[:, :, :, 1:-1] += (u_c[:, :, :, 2:] - 2.0 * u_c[:, :, :, 1:-1] + u_c[:, :, :, :-2]) / (dx_m * dx_m)

    c2 = (c * c)[:, None, :, :]                              # (R, 1, Z, X) broadcast over interior time
    forcing_term = c2 * lap
    residual = u_tt - forcing_term

    # valid interior in space (drop the 1-cell border where lap is one-sided/zero)
    valid = torch.zeros(height, width, dtype=torch.bool, device=u.device)
    valid[1:-1, 1:-1] = True
    mask = valid[None, None, :, :].expand_as(residual)
    if source_free_mask is not None:
        sfm = torch.as_tensor(source_free_mask, device=u.device).bool()
        if sfm.shape == (records, height, width):
            sfm = sfm[:, None, :, :]
        elif sfm.shape != (records, frames - 2, height, width):
            raise ValueError("source_free_mask must be [record,z,x] or [record,T-2,z,x]")
        mask = mask & sfm.expand_as(residual)

    count = mask.sum().clamp_min(1)
    num = (residual.square() * mask).sum() / count
    scale = (
        (u_tt.square() * mask).sum() / count
        + (forcing_term.square() * mask).sum() / count
    )
    return num / (scale + eps)


def pml_interface_residual_loss(
    field: torch.Tensor,
    velocity_mps: torch.Tensor,
    time_s: torch.Tensor,
    *,
    dz_m: float,
    dx_m: float,
    boundary_band_cells: int = 4,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Three-sided outgoing-wave residual at the saved-grid CPML interface.

    The production CPML lives outside the saved 201 x 201 physical crop, so its
    pressure and memory variables are unavailable to the neural operator.  This
    loss therefore enforces the local first-order radiation condition at the
    left, right, and bottom interfaces of that crop.  The free-surface top is
    deliberately excluded.  Each side is normalized by its local time/spatial
    derivative energy, making the result dimensionless and scale stable.
    """

    u = torch.as_tensor(field).float()
    if u.ndim != 4:
        raise ValueError("PML interface field must be [record,time,z,x]")
    records, frames, height, width = u.shape
    c = torch.as_tensor(velocity_mps, dtype=u.dtype, device=u.device)
    if c.shape != (records, height, width):
        raise ValueError("PML interface velocity must match [record,z,x]")
    times = torch.as_tensor(time_s, dtype=u.dtype, device=u.device)
    if times.shape != (records, frames):
        raise ValueError("PML interface times must match [record,time]")
    band = int(boundary_band_cells)
    if (
        frames < 3
        or band < 1
        or band >= height
        or 2 * band >= width
    ):
        if frames < 3:
            return u.new_zeros(())
        raise ValueError("PML interface boundary band is invalid")
    if not math.isfinite(dz_m) or not math.isfinite(dx_m) or dz_m <= 0.0 or dx_m <= 0.0:
        raise ValueError("PML interface spacings must be positive and finite")

    dt = (times[:, 2:] - times[:, :-2]).clamp_min(float(eps))[:, :, None, None]
    u_t = (u[:, 2:] - u[:, :-2]) / dt
    center = u[:, 1:-1]

    left_dx = (center[..., 1 : band + 1] - center[..., :band]) / float(dx_m)
    right_dx = (center[..., -band:] - center[..., -band - 1 : -1]) / float(dx_m)
    bottom_dz = (
        center[..., -band:, :] - center[..., -band - 1 : -1, :]
    ) / float(dz_m)

    left_ut = u_t[..., :band]
    right_ut = u_t[..., -band:]
    bottom_ut = u_t[..., -band:, :]
    left_c = c[:, None, :, :band]
    right_c = c[:, None, :, -band:]
    bottom_c = c[:, None, -band:, :]

    def relative_radiation(residual, temporal, normal):
        numerator = residual.square().mean()
        denominator = temporal.square().mean() + normal.square().mean()
        return numerator / (denominator + float(eps))

    left_normal = left_c * left_dx
    right_normal = right_c * right_dx
    bottom_normal = bottom_c * bottom_dz
    left = relative_radiation(left_ut - left_normal, left_ut, left_normal)
    right = relative_radiation(right_ut + right_normal, right_ut, right_normal)
    bottom = relative_radiation(bottom_ut + bottom_normal, bottom_ut, bottom_normal)
    return (left + right + bottom) / 3.0


def zero_initial_condition_loss(
    initial_field: torch.Tensor,
    *,
    reference_rms: torch.Tensor,
    first_step_weight: float = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Dimensionless zero initial pressure and zero first-step change loss."""

    value = torch.as_tensor(initial_field).float()
    if value.ndim != 4 or value.shape[1] != 2:
        raise ValueError("initial field must be [record,2,z,x]")
    reference = torch.as_tensor(reference_rms, dtype=value.dtype, device=value.device)
    if reference.shape != (value.shape[0],) or not bool(torch.isfinite(reference).all()):
        raise ValueError("initial-condition reference RMS must be finite per record")
    step_weight = float(first_step_weight)
    if not math.isfinite(step_weight) or step_weight < 0.0:
        raise ValueError("initial first-step weight must be finite and nonnegative")
    scale_square = reference.square().clamp_min(float(eps))
    pressure_square = value[:, 0].square().flatten(1).mean(dim=1)
    step_square = (value[:, 1] - value[:, 0]).square().flatten(1).mean(dim=1)
    return ((pressure_square + step_weight * step_square) / scale_square).mean()



def _relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    energy_reference: torch.Tensor | None = None,
    energy_floor_fraction: float = 0.0,
) -> torch.Tensor:
    error = (prediction.float() - target.float()).flatten(1).norm(dim=-1)
    scale = target.float().flatten(1).norm(dim=-1)
    floor = float(energy_floor_fraction)
    if not 0.0 <= floor <= 1.0:
        raise ValueError("relative L2 energy floor must lie in [0, 1]")
    if energy_reference is not None:
        reference = torch.as_tensor(energy_reference)
        if reference.shape != target.shape:
            raise ValueError("relative L2 energy reference must match its target")
        reference_scale = reference.float().flatten(1).norm(dim=-1)
        scale = torch.maximum(scale, floor * reference_scale)
    return (error / scale.clamp_min(1.0e-8)).mean()


def _global_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    energy_reference: torch.Tensor | None = None,
    energy_floor_fraction: float = 0.0,
) -> torch.Tensor:
    """Relative L2 after pooling every record, matching scattering evaluation."""

    error = (prediction.float() - target.float()).norm()
    scale = target.float().norm()
    floor = float(energy_floor_fraction)
    if not 0.0 <= floor <= 1.0:
        raise ValueError("relative L2 energy floor must lie in [0, 1]")
    if energy_reference is not None:
        reference = torch.as_tensor(energy_reference)
        if reference.shape != target.shape:
            raise ValueError("relative L2 energy reference must match its target")
        scale = torch.maximum(scale, floor * reference.float().norm())
    return error / scale.clamp_min(1.0e-8)


def frame_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    energy_floor_fraction: float = 0.0,
    frame_time_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Relative L2 normalized per ``(record, time)`` frame instead of per record.

    ``_relative_l2`` flattens every stored time into one per-record vector, so
    its denominator is the whole-record energy, which is dominated by the early
    high-amplitude wavefront; late low-amplitude frames then receive almost no
    gradient weight.  This variant divides each frame by its own spatial energy
    so every stored time contributes equally to the mean.  Frames whose energy
    falls below ``energy_floor_fraction`` of their record's peak-frame energy are
    floored to that fraction, which keeps near-zero pre-onset frames from
    dividing by ~0 and dominating the objective.

    ``frame_time_weights`` (a 1-D tensor over the ``time`` axis) optionally
    reweights frames before the mean — e.g. a causal late-frame ramp so long-time
    frames drive more gradient.  It is renormalized to mean 1 so the overall loss
    scale is unchanged; ``None`` (default) is the uniform per-frame mean.
    """

    if prediction.shape != target.shape or target.ndim != 4:
        raise ValueError("frame relative L2 requires matching [record,time,z,x] fields")
    floor = float(energy_floor_fraction)
    if not 0.0 <= floor <= 1.0:
        raise ValueError("frame relative L2 energy floor must lie in [0, 1]")
    error = (prediction.float() - target.float()).flatten(start_dim=2).norm(dim=-1)
    scale = target.float().flatten(start_dim=2).norm(dim=-1)
    if floor > 0.0:
        peak = scale.amax(dim=1, keepdim=True)
        scale = torch.maximum(scale, floor * peak)
    per_frame = error / scale.clamp_min(1.0e-8)
    if frame_time_weights is None:
        return per_frame.mean()
    weights = torch.as_tensor(frame_time_weights, dtype=torch.float32, device=per_frame.device)
    if weights.ndim != 1 or weights.numel() != per_frame.shape[1]:
        raise ValueError("frame_time_weights must be 1-D over the time axis")
    weights = weights / weights.mean().clamp_min(1.0e-12)
    return (per_frame * weights[None, :]).mean()


def residual_recovery_loss(
    prediction: torch.Tensor,
    coarse: torch.Tensor,
    target: torch.Tensor,
    *,
    time_indices: torch.Tensor,
    frame_weight: float = 1.0,
    delta_weight: float,
    delta_reference: str = "model_coarse",
    delta_reduction: str = "per_record",
    delta_global_target_square: float | torch.Tensor | None = None,
    delta_piece_weight: float = 1.0,
    temporal_weight: float,
    gradient_weight: float,
    spectrum_weight: float,
    delta_energy_floor_fraction: float = 0.0,
    per_frame_frame: bool = False,
    frame_energy_floor_fraction: float = 0.0,
    frame_time_weights: torch.Tensor | None = None,
    pde_weight: float = 0.0,
    velocity_mps: torch.Tensor | None = None,
    time_values_s: torch.Tensor | None = None,
    pde_dz_m: float | None = None,
    pde_dx_m: float | None = None,
    pde_source_free_mask: torch.Tensor | None = None,
    pml_weight: float = 0.0,
    pml_boundary_band_cells: int = 4,
) -> ResidualRecoveryLoss:
    """Supervise both the full field and the correction of a transferred field.

    When ``per_frame_frame`` is set the full-field frame term is normalized per
    stored time (:func:`frame_relative_l2`) rather than per record, so late
    low-amplitude frames are trained as hard as the early wavefront.
    ``frame_time_weights`` optionally tilts that per-frame mean toward late times.
    """

    if prediction.shape != target.shape or coarse.shape != target.shape or target.ndim != 4:
        raise ValueError("recovery loss requires matching [record,time,z,x] fields")
    indices = torch.as_tensor(time_indices, dtype=torch.long, device=prediction.device)
    if indices.shape != prediction.shape[:2]:
        raise ValueError("time indices must match recovery record/time dimensions")
    weights = tuple(
        float(value)
        for value in (
            frame_weight,
            delta_weight,
            temporal_weight,
            gradient_weight,
            spectrum_weight,
        )
    )
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("recovery loss weights must be nonnegative")
    delta_floor = float(delta_energy_floor_fraction)
    if not 0.0 <= delta_floor <= 1.0:
        raise ValueError("delta energy floor fraction must lie in [0, 1]")

    if frame_weight > 0.0:
        if per_frame_frame:
            frame = frame_relative_l2(
                prediction,
                target,
                energy_floor_fraction=frame_energy_floor_fraction,
                frame_time_weights=frame_time_weights,
            )
        else:
            frame = _relative_l2(prediction, target)
    else:
        frame = prediction.new_zeros(())
    reference = str(delta_reference)
    if reference == "model_coarse":
        target_delta = (target.float() - coarse.float()).detach()
        predicted_delta = prediction.float() - coarse.float()
    elif reference == "target":
        # Background-cache recovery retargets ``target`` to the physical
        # scattering field (wavefield - P_bg).  The background conditioner is
        # part of ``coarse`` itself, so subtracting model coarse from prediction
        # would algebraically cancel the very branch being trained.  In that
        # protocol the correction target is already ``target`` and the predicted
        # correction is the complete prediction.
        target_delta = target.float().detach()
        predicted_delta = prediction.float()
    else:
        raise ValueError("delta reference must be 'model_coarse' or 'target'")
    reduction = str(delta_reduction)
    if reduction == "per_record":
        delta_loss = _relative_l2
    elif reduction == "global":
        delta_loss = _global_relative_l2
    elif reduction == "global_squared":
        if reference != "target":
            raise ValueError("global-squared delta reduction requires target reference")
        if delta_floor != 0.0:
            raise ValueError("global-squared delta reduction requires zero energy floor")
        if delta_global_target_square is None:
            raise ValueError("global-squared delta reduction requires target energy")
        piece_weight = float(delta_piece_weight)
        if not math.isfinite(piece_weight) or piece_weight <= 0.0:
            raise ValueError("global-squared delta piece weight must be positive")
        denominator = torch.as_tensor(
            delta_global_target_square,
            dtype=torch.float64,
            device=prediction.device,
        ).clamp_min(1.0e-16)
        delta = (
            (predicted_delta.double() - target_delta.double()).square().sum()
            / denominator
            / piece_weight
        ).to(prediction.dtype)
        delta_loss = None
    else:
        raise ValueError(
            "delta reduction must be 'per_record', 'global', or 'global_squared'"
        )
    if delta_weight <= 0.0:
        delta = prediction.new_zeros(())
    elif delta_loss is not None:
        delta = delta_loss(
            predicted_delta,
            target_delta,
            energy_reference=target,
            energy_floor_fraction=delta_floor,
        )

    if temporal_weight > 0.0:
        consecutive = indices[:, 1:] == indices[:, :-1] + 1
        if bool(consecutive.any()):
            predicted_dt = prediction[:, 1:].float() - prediction[:, :-1].float()
            target_dt = target[:, 1:].float() - target[:, :-1].float()
            mask = consecutive[:, :, None, None].expand_as(predicted_dt)
            temporal_error = (predicted_dt - target_dt)[mask].norm()
            temporal_scale = target_dt[mask].norm()
            temporal = temporal_error / temporal_scale.clamp_min(1.0e-8)
        else:
            temporal = prediction.new_zeros(())
    else:
        temporal = prediction.new_zeros(())

    if gradient_weight > 0.0:
        pred_dx = prediction[..., 1:] - prediction[..., :-1]
        true_dx = target[..., 1:] - target[..., :-1]
        pred_dz = prediction[..., 1:, :] - prediction[..., :-1, :]
        true_dz = target[..., 1:, :] - target[..., :-1, :]
        gradient_error = (pred_dx.float() - true_dx.float()).square().mean().sqrt()
        gradient_error = gradient_error + (
            pred_dz.float() - true_dz.float()
        ).square().mean().sqrt()
        gradient_scale = true_dx.float().square().mean().sqrt()
        gradient_scale = gradient_scale + true_dz.float().square().mean().sqrt()
        gradient = gradient_error / gradient_scale.clamp_min(1.0e-8)
    else:
        gradient = prediction.new_zeros(())

    spectrum = (
        band_limited_residual_loss(prediction, target)
        if spectrum_weight > 0.0
        else prediction.new_zeros(())
    )
    pde_w = float(pde_weight)
    if not math.isfinite(pde_w) or pde_w < 0.0:
        raise ValueError("recovery loss pde_weight must be finite and nonnegative")
    if pde_w > 0.0:
        if velocity_mps is None or time_values_s is None or pde_dz_m is None or pde_dx_m is None:
            raise ValueError(
                "pde_weight > 0 requires velocity_mps, time_values_s, pde_dz_m, pde_dx_m"
            )
        pde = wave_pde_residual_loss(
            prediction,
            velocity_mps,
            time_values_s,
            dz_m=float(pde_dz_m),
            dx_m=float(pde_dx_m),
            source_free_mask=pde_source_free_mask,
        )
    else:
        pde = prediction.new_zeros(())
    pml_w = float(pml_weight)
    if not math.isfinite(pml_w) or pml_w < 0.0:
        raise ValueError("recovery loss PML weight must be finite and nonnegative")
    if pml_w > 0.0:
        if velocity_mps is None or time_values_s is None or pde_dz_m is None or pde_dx_m is None:
            raise ValueError(
                "pml_weight > 0 requires velocity_mps, time_values_s, pde_dz_m, pde_dx_m"
            )
        pml = pml_interface_residual_loss(
            prediction,
            velocity_mps,
            time_values_s,
            dz_m=float(pde_dz_m),
            dx_m=float(pde_dx_m),
            boundary_band_cells=int(pml_boundary_band_cells),
        )
    else:
        pml = prediction.new_zeros(())
    # Do not attach disabled terms to autograd as ``0 * term``.  Norms have an
    # undefined derivative at an exact zero residual, so multiplying a disabled
    # zero-valued term can otherwise turn a valid active-term gradient into NaN.
    total = prediction.new_zeros(())
    for weight, term in zip(
        weights,
        (frame, delta, temporal, gradient, spectrum),
        strict=True,
    ):
        if weight > 0.0:
            total = total + weight * term
    if pde_w > 0.0:
        total = total + pde_w * pde
    if pml_w > 0.0:
        total = total + pml_w * pml
    return ResidualRecoveryLoss(total, frame, delta, temporal, gradient, spectrum, pde, pml)


__all__ = [
    "FAMILY_CLASS_ORDER",
    "RelativeEnergySquaredBlockLoss",
    "RelativeEnergySquaredReference",
    "ResidualRecoveryLoss",
    "apply_hard_causality",
    "band_limited_residual_loss",
    "family_router_loss",
    "family_route_targets",
    "frame_relative_l2",
    "pml_interface_residual_loss",
    "relative_energy_squared_block_loss",
    "relative_energy_squared_reference",
    "residual_recovery_loss",
    "source_causality_onset_s",
    "zero_initial_condition_loss",
]
