"""Leakage-safe dense training primitives for the Patch-DeepONet baseline."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from grouped_ufno_mionet_v3.model.travel_time import dense_query_coordinates
from grouped_ufno_mionet_v3.normalization import PhysicalNormalizer
from saved_time_phase_operator_v4.losses import (
    apply_hard_causality,
    band_limited_residual_loss,
    source_causality_onset_s,
)
from saved_time_phase_operator_v4.instance_adaptation.forced_defect import (
    lwc84_discrete_defect,
)

from .features import build_query_descriptors, build_static_features
from .model import PatchDeepONet


@dataclass(frozen=True)
class PatchDeepONetLoss:
    total: torch.Tensor
    record_relative_l2: torch.Tensor
    temporal_difference: torch.Tensor
    spatial_gradient: torch.Tensor
    spectrum: torch.Tensor


@dataclass(frozen=True)
class PIDeepONetResidual:
    """Dimensionless discrete-physics loss and its collocation count."""

    loss: torch.Tensor
    triplet_count: int


def _record_tensors(batch, device: torch.device) -> dict[str, torch.Tensor]:
    if len(batch.sample_id) <= 0 or batch.dense_travel_time_s is None:
        raise ValueError("Patch-DeepONet training requires records and cached travel time")
    values = {
        name: getattr(batch, name).to(device, non_blocking=device.type == "cuda")
        for name in (
            "velocity_mps",
            "record_to_medium",
            "source_parameters",
            "source_map",
            "requested_time_s",
            "dense_target_physical",
            "left_index",
            "x_m",
            "z_m",
        )
    }
    values["travel_time_s"] = batch.dense_travel_time_s.to(
        device, non_blocking=device.type == "cuda"
    )
    if values["record_to_medium"].shape != (len(batch.sample_id),):
        raise ValueError("Patch-DeepONet record-to-medium mapping changed")
    values["velocity_mps"] = values["velocity_mps"].index_select(
        0, values["record_to_medium"].long()
    )
    return values


def dense_training_pair(
    model: PatchDeepONet,
    batch,
    normalizer: PhysicalNormalizer,
    device: torch.device,
    *,
    time_block: int,
    query_chunk: int,
    hard_causality_lead_cycles: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized prediction, target and exact saved-time indices.

    The spatial branch is encoded once per source--medium record. Query blocks
    share that encoding, so changing the time block changes memory but not the
    mathematical prediction.
    """

    if int(time_block) <= 0 or int(query_chunk) <= 0:
        raise ValueError("time block and query chunk must be positive")
    tensors = _record_tensors(batch, device)
    record_count = len(batch.sample_id)
    velocity = tensors["velocity_mps"]
    source = tensors["source_parameters"]
    source_n = normalizer.encode_source(source)
    travel = tensors["travel_time_s"]
    static = build_static_features(
        normalizer.encode_velocity(velocity),
        tensors["source_map"],
        travel,
        velocity,
        domain_t_s=1.0,
    )
    encoded = model.encode_static(static)
    x_m = tensors["x_m"]
    z_m = tensors["z_m"]
    coordinates = dense_query_coordinates(x_m, z_m, records=record_count)
    spatial_points = int(coordinates.shape[1])
    outputs: list[torch.Tensor] = []
    times_all = tensors["requested_time_s"]
    for start in range(0, times_all.shape[1], int(time_block)):
        times = times_all[:, start : start + int(time_block)]
        count = times.shape[1]
        xyz = torch.cat(
            (
                coordinates[:, None].expand(-1, count, -1, -1),
                times[:, :, None, None].expand(-1, -1, spatial_points, 1),
            ),
            dim=-1,
        ).reshape(record_count, count * spatial_points, 3)
        travel_block = travel.reshape(record_count, -1)[:, None].expand(-1, count, -1).reshape(record_count, -1)
        descriptors = build_query_descriptors(
            xyz,
            source_n,
            source,
            travel_block,
            domain_x_m=2000.0,
            domain_z_m=2000.0,
            domain_t_s=1.0,
        )
        queried = torch.cat(
            [
                model.query_encoded(
                    static, encoded, descriptors[:, offset : offset + int(query_chunk)]
                )
                for offset in range(0, descriptors.shape[1], int(query_chunk))
            ],
            dim=1,
        )
        outputs.append(queried.reshape(record_count, count, len(z_m), len(x_m)))
    prediction = torch.cat(outputs, dim=1)
    # Apply the same known free surface and source-onset contract as the proposed
    # model. This is not a learned or target-derived advantage for either method.
    surface = torch.ones_like(prediction)
    surface[..., 0, :] = 0.0
    prediction = prediction * surface
    onset = source_causality_onset_s(
        source, lead_cycles=float(hard_causality_lead_cycles)
    )
    prediction = apply_hard_causality(prediction, tensors["requested_time_s"], onset)
    target = normalizer.encode_pressure(tensors["dense_target_physical"], source[:, 4])
    if prediction.shape != target.shape or not torch.isfinite(prediction).all():
        raise RuntimeError("Patch-DeepONet dense training prediction is invalid")
    return prediction, target, tensors["left_index"]


def patch_deeponet_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    time_indices: torch.Tensor,
    *,
    temporal_weight: float = 0.1,
    gradient_weight: float = 0.1,
    spectrum_weight: float = 0.1,
) -> PatchDeepONetLoss:
    """Architecture-appropriate counterpart of the proposed composite loss.

    The primary per-record relative norm and its temporal, spatial-gradient and
    spectral regularizers are shared. The proposed model's coarse-residual delta
    term is omitted because Patch-DeepONet has no coarse/correction decomposition.
    """

    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[0] <= 0:
        raise ValueError("Patch-DeepONet loss requires matching [record,time,z,x] tensors")
    weights = tuple(float(value) for value in (temporal_weight, gradient_weight, spectrum_weight))
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("Patch-DeepONet loss weights must be finite and nonnegative")
    error = (prediction.float() - target.float()).flatten(1).norm(dim=-1)
    scale = target.float().flatten(1).norm(dim=-1).clamp_min(1.0e-8)
    relative = (error / scale).mean()

    indices = torch.as_tensor(time_indices, device=prediction.device)
    if indices.shape != prediction.shape[:2]:
        raise ValueError("time indices must match the selected frames")
    consecutive = indices[:, 1:] == indices[:, :-1] + 1
    if bool(consecutive.any()):
        predicted_dt = prediction[:, 1:] - prediction[:, :-1]
        target_dt = target[:, 1:] - target[:, :-1]
        mask = consecutive[:, :, None, None].expand_as(predicted_dt)
        temporal = (
            (predicted_dt.float() - target_dt.float())[mask].norm()
            / target_dt.float()[mask].norm().clamp_min(1.0e-8)
        )
    else:
        temporal = prediction.new_zeros(())

    pred_dx, true_dx = prediction[..., 1:] - prediction[..., :-1], target[..., 1:] - target[..., :-1]
    pred_dz, true_dz = prediction[..., 1:, :] - prediction[..., :-1, :], target[..., 1:, :] - target[..., :-1, :]
    gradient = (
        (pred_dx.float() - true_dx.float()).square().mean().sqrt()
        + (pred_dz.float() - true_dz.float()).square().mean().sqrt()
    ) / (
        true_dx.float().square().mean().sqrt()
        + true_dz.float().square().mean().sqrt()
    ).clamp_min(1.0e-8)
    spectrum = (
        band_limited_residual_loss(prediction, target)
        if weights[2] > 0.0
        else prediction.new_zeros(())
    )
    total = relative
    for weight, value in zip(weights, (temporal, gradient, spectrum), strict=True):
        if weight > 0.0:
            total = total + weight * value
    return PatchDeepONetLoss(total, relative, temporal, gradient, spectrum)


def pi_deeponet_lwc84_residual_loss(
    field: torch.Tensor,
    velocity_mps: torch.Tensor,
    source_parameters: torch.Tensor,
    source_map: torch.Tensor,
    time_s: torch.Tensor,
    time_indices: torch.Tensor,
    *,
    pressure_scale_pa: float,
    dt_s: float,
    dx_m: float,
    dz_m: float,
    maximum_triplets_per_record: int = 2,
    denominator_epsilon: float = 1.0e-8,
) -> PIDeepONetResidual:
    """Return a generator-matched physics penalty on exact saved-frame triplets.

    ``appearance16`` mixes separated time windows, so applying a time derivative
    across every adjacent tensor position would be wrong.  This routine selects
    only triples whose exact saved-time indices are consecutive.  It then applies
    the forced fourth-order-in-time/eighth-order-in-space LWC84 defect used by the
    data generator.  The first and last valid triples are retained by default,
    providing early/source and late/propagation collocation at bounded cost.

    The supplied field is in the model's pressure-normalized units.  Source terms
    are divided by the same per-record ``pressure_scale_pa * amplitude`` before the
    defect is formed.  The underlying routine normalizes each triplet by its
    detached temporal/spatial operator RMS, clamped by ``denominator_epsilon``.
    """

    value = torch.as_tensor(field)
    if value.ndim != 4 or value.shape[0] <= 0:
        raise ValueError("PI-DeepONet field must be [record,time,z,x]")
    records, frames, height, width = value.shape
    indices = torch.as_tensor(time_indices, device=value.device)
    times = torch.as_tensor(time_s, dtype=value.dtype, device=value.device)
    if indices.shape != (records, frames) or times.shape != (records, frames):
        raise ValueError("PI-DeepONet time values and indices must match the field")
    if int(maximum_triplets_per_record) <= 0:
        raise ValueError("PI-DeepONet requires a positive triplet cap")
    if not math.isclose(float(denominator_epsilon), 1.0e-8, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("the audited LWC84 defect uses denominator epsilon 1e-8")
    spacings = (float(dt_s), float(dx_m), float(dz_m), float(pressure_scale_pa))
    if any(not math.isfinite(item) or item <= 0.0 for item in spacings):
        raise ValueError("PI-DeepONet physical spacings and pressure scale must be positive")

    velocity = torch.as_tensor(velocity_mps, dtype=value.dtype, device=value.device)
    if velocity.ndim == 4 and velocity.shape[1] == 1:
        velocity = velocity[:, 0]
    if velocity.shape != (records, height, width):
        raise ValueError("PI-DeepONet velocity must be [record,z,x]")
    sources = torch.as_tensor(
        source_parameters, dtype=value.dtype, device=value.device
    )
    spatial_source = torch.as_tensor(source_map, dtype=value.dtype, device=value.device)
    if sources.shape != (records, 5):
        raise ValueError("PI-DeepONet source parameters must be [record,5]")
    if spatial_source.ndim == 3:
        spatial_source = spatial_source[:, None]
    if spatial_source.shape != (records, 1, height, width):
        raise ValueError("PI-DeepONet source map must be [record,1,z,x]")

    consecutive = (
        (indices[:, 1:-1] == indices[:, :-2] + 1)
        & (indices[:, 2:] == indices[:, 1:-1] + 1)
    )
    selected_records: list[int] = []
    selected_starts: list[int] = []
    cap = int(maximum_triplets_per_record)
    for record in range(records):
        candidates = torch.where(consecutive[record])[0]
        if candidates.numel() == 0:
            raise ValueError(
                "PI-DeepONet appearance contains no consecutive saved-frame triplet"
            )
        if candidates.numel() > cap:
            picks = torch.linspace(
                0,
                candidates.numel() - 1,
                steps=cap,
                device=candidates.device,
            ).round().long()
            candidates = candidates.index_select(0, picks).unique(sorted=True)
        selected_records.extend([record] * int(candidates.numel()))
        selected_starts.extend(int(item) for item in candidates.tolist())

    record_ids = torch.tensor(selected_records, dtype=torch.long, device=value.device)
    starts = torch.tensor(selected_starts, dtype=torch.long, device=value.device)
    positions = torch.stack((starts, starts + 1, starts + 2), dim=1)
    triplets = value[record_ids[:, None], positions]
    triplet_times = times[record_ids[:, None], positions]
    triplet_sources = sources.index_select(0, record_ids)
    field_scale = float(pressure_scale_pa) * triplet_sources[:, 4]
    defect = lwc84_discrete_defect(
        triplets,
        velocity.index_select(0, record_ids),
        dt=float(dt_s),
        dx=float(dx_m),
        dz=float(dz_m),
        observed_indices=(0, 0),
        source_parameters=triplet_sources,
        source_map=spatial_source.index_select(0, record_ids),
        time_s=triplet_times,
        field_scale_pa=field_scale,
        time_order=4,
        normalize=True,
    )
    if not isinstance(defect, torch.Tensor) or not torch.isfinite(defect).all():
        raise FloatingPointError("PI-DeepONet LWC84 defect is non-finite")
    return PIDeepONetResidual(
        loss=defect.float().square().mean(),
        triplet_count=int(record_ids.numel()),
    )


__all__ = [
    "PIDeepONetResidual",
    "PatchDeepONetLoss",
    "dense_training_pair",
    "patch_deeponet_loss",
    "pi_deeponet_lwc84_residual_loss",
]
