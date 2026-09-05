#!/usr/bin/env python3
"""Capacity-ladder 3-record overfit probe.

Question this answers
---------------------
v54 showed the current architecture (width=64, dense_depth=8, plus every additive
head) cannot memorize three training records below aggregate relative L2 ~0.28.
Is that floor caused by (a) insufficient raw decoder/backbone capacity, or (b) a
structural limit of the global-spectral correction operator?

This probe rebuilds a *fresh* SavedTimePhaseOperatorV4 at an arbitrary
``width`` / ``dense_depth`` / ``dense_spectral_rank`` / ``dense_modes`` and tries to
memorize the same deterministic uniform/layered/marmousi triplet the v51-v55
diagnostics used, scored with the identical accumulator and target
(agg < 0.10 and every family < 0.12). Because a three-record microbatch is tiny,
memory is not the training-time constraint it is for the full dataset, so we can
probe architectures far larger than full training could fit.

Initialization regimes (chosen per the capacity-ladder design):
  * ``--warmstart-frontend`` (only valid at width=64): transfer the trained
    medium/source/travel/fusion front-end from a parent checkpoint and leave the
    (wider/deeper) dense decoder freshly initialized. Isolates decoder capacity
    from front-end optimization difficulty.
  * ``--expanded-frequency-init-checkpoint`` + ``--expanded-frequency-init-identity``
    (requires --helmholtz-synthesis --helmholtz-rank 0): strictly expand the
    independent direct-frequency head from a parent trained with fewer Helmholtz
    bins, copying its [cos(0:F_old), sin(0:F_old)] channels into the matching
    positions of the [cos(0:F_new), sin(0:F_new)] layout and zeroing every newly
    introduced frequency channel. All other tensors must match key-for-key and
    shape-for-shape.
  * no warm start: the whole model is randomly initialized. Used for the width
    ladder, where the front-end shapes no longer match the width-64 parent.

The additive heads (temporal-basis, family-expert, band-adapter) are held OFF
across every rung so the ladder isolates the true capacity levers (width, depth,
spectral rank) rather than re-testing the v49-v71 adapter family that is already
known to be dead.

This script trains on a single CUDA device; run one rung per GPU to fill a node.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import math
import time
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts
from saved_time_phase_operator_v4.probe import ProbeVariant
from saved_time_phase_operator_v4.data import split_pilot_batch
from saved_time_phase_operator_v4.full_support import FullSupportStepSpec  # noqa: F401

from scripts.train_saved_time_v4_probe import _model, _atomic_json, _digest
from scripts.train_saved_time_v4_full_support import (
    _train_update,
    _gradient_report,
    clip_trainable_gradients,
    temporal_basis_gradient_norms,
)
from scripts.train_grouped_v3_pilot import _to_device, load_normalizer  # noqa: F401
from scripts.diagnose_saved_time_temporal_three_record_overfit import (
    FAMILIES,
    select_one_index_per_family,
    build_repeat_schedule,
    _dataset,
    _evaluate_triplet,
    _retarget_batch_to_scattering,
    _meets_target,
    build_overfit_terminal_report,
    save_overfit_checkpoint,
    restore_best_overfit_checkpoint,
)
from scripts.refine_saved_time_v4_lbfgs import _append_jsonl, _gpu_snapshot


BASE_CONFIG = "configs/grouped_v3/continuous_pilot.yaml"
FRONT_END_PREFIXES = (
    "coordinate_encoder",
    "fusion",
    "medium_encoder",
    "source_encoder",
    "travel_branch",
    "saved_time_values_s",
)


def load_exact_model_initialization(
    model,
    checkpoint_path: Path,
    identity_path: Path,
    *,
    manifest_digest: str,
    device: torch.device,
) -> dict[str, object]:
    """Strictly load one same-architecture model without optimizer/RNG state."""

    raw = checkpoint_path.read_bytes()
    payload = torch.load(io.BytesIO(raw), map_location=device)
    identity = json.loads(identity_path.read_text())
    if not isinstance(payload, dict) or not isinstance(identity, dict):
        raise ValueError("exact initialization artifacts must contain mappings")
    if payload.get("manifest_digest") != manifest_digest:
        raise ValueError("exact initialization manifest digest mismatch")
    if payload.get("config_digest") != identity.get("run_digest"):
        raise ValueError("exact initialization run digest mismatch")
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("exact initialization checkpoint has no model state")
    model.load_state_dict(state, strict=True)
    return {
        "schema": "exact_model_initialization_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
        "identity": str(identity_path),
        "identity_sha256": hashlib.sha256(identity_path.read_bytes()).hexdigest(),
        "parent_run_digest": identity["run_digest"],
        "parent_global_step": int(payload.get("global_step", -1)),
        "optimizer_state_restored": False,
        "rng_state_restored": False,
        "strict_state_dict": True,
    }


def load_continue_frequency_initialization(
    model,
    checkpoint_path: Path,
    identity_path: Path,
    *,
    manifest_digest: str,
    device: torch.device,
) -> dict[str, object]:
    """Load a same-frequency parent for continue-pretraining, tolerating ONLY
    parameters that are absent from the parent (new zero-init gates such as the
    Helmholtz frequency-softmax conditioner).  Every tensor present in the parent must
    match shape-for-shape; the parent may not carry tensors the child dropped.

    This is the honest continue-pretraining path when a rung adds an identity
    gate but keeps the frequency count fixed: update-0 reproduces the parent's
    trained field bit-for-bit (all new params are exact identities), and only the
    new gate is learned on top.
    """

    raw = checkpoint_path.read_bytes()
    payload = torch.load(io.BytesIO(raw), map_location=device)
    identity = json.loads(identity_path.read_text())
    if not isinstance(payload, dict) or not isinstance(identity, dict):
        raise ValueError("continue-frequency initialization artifacts must contain mappings")
    if payload.get("manifest_digest") != manifest_digest:
        raise ValueError("continue-frequency initialization manifest digest mismatch")
    if payload.get("config_digest") != identity.get("run_digest"):
        raise ValueError("continue-frequency initialization run digest mismatch")
    source = payload.get("model_state")
    if not isinstance(source, dict):
        raise ValueError("continue-frequency checkpoint has no model state")
    target = model.state_dict()
    unexpected = set(source) - set(target)
    if unexpected:
        raise ValueError(
            f"continue-frequency parent carries tensors the model does not have: {sorted(unexpected)[:8]}"
        )
    mismatched = [
        key for key in (set(source) & set(target))
        if tuple(source[key].shape) != tuple(target[key].shape)
    ]
    if mismatched:
        raise ValueError(
            f"continue-frequency shape mismatch on: {sorted(mismatched)[:8]}"
        )
    new_parameters = sorted(set(target) - set(source))
    nonzero_new = [
        key for key in new_parameters if bool(torch.count_nonzero(target[key]).item())
    ]
    if nonzero_new:
        raise ValueError(
            "continue-frequency identity parameters must be zero initialized: "
            f"{nonzero_new[:8]}"
        )
    model.load_state_dict(source, strict=False)
    return {
        "schema": "continue_frequency_initialization_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
        "identity": str(identity_path),
        "identity_sha256": hashlib.sha256(identity_path.read_bytes()).hexdigest(),
        "parent_run_digest": identity["run_digest"],
        "parent_global_step": int(payload.get("global_step", -1)),
        "new_identity_parameters": new_parameters,
        "optimizer_state_restored": False,
        "rng_state_restored": False,
        "strict_non_new_state": True,
    }


def load_expanded_frequency_initialization(
    model,
    checkpoint_path: Path,
    identity_path: Path,
    *,
    manifest_digest: str,
    device: torch.device,
) -> dict[str, object]:
    """Strictly expand an independent [cos, sin] frequency output head."""

    raw = checkpoint_path.read_bytes()
    payload = torch.load(io.BytesIO(raw), map_location=device)
    identity = json.loads(identity_path.read_text())
    if payload.get("manifest_digest") != manifest_digest:
        raise ValueError("expanded-frequency initialization manifest digest mismatch")
    if payload.get("config_digest") != identity.get("run_digest"):
        raise ValueError("expanded-frequency initialization run digest mismatch")
    source = payload.get("model_state")
    if not isinstance(source, dict):
        raise ValueError("expanded-frequency checkpoint has no model state")
    target = model.state_dict()
    if set(source) != set(target):
        raise ValueError("expanded-frequency initialization state keys differ")
    weight_key = "local_field.helmholtz_synthesis.head.weight"
    bias_key = "local_field.helmholtz_synthesis.head.bias"
    if weight_key not in source or bias_key not in source:
        raise ValueError("expanded-frequency initialization requires an independent head")
    for key in source:
        if key not in {weight_key, bias_key} and source[key].shape != target[key].shape:
            raise ValueError(f"expanded-frequency non-head shape mismatch: {key}")
    old_channels = int(source[weight_key].shape[0])
    new_channels = int(target[weight_key].shape[0])
    if old_channels % 2 or new_channels % 2 or new_channels <= old_channels:
        raise ValueError("expanded-frequency head must strictly increase an even channel count")
    if source[weight_key].shape[1:] != target[weight_key].shape[1:]:
        raise ValueError("expanded-frequency head input shape changed")
    old_frequencies, new_frequencies = old_channels // 2, new_channels // 2
    expanded = dict(source)
    weight = torch.zeros_like(target[weight_key])
    bias = torch.zeros_like(target[bias_key])
    weight[:old_frequencies] = source[weight_key][:old_frequencies]
    weight[new_frequencies : new_frequencies + old_frequencies] = source[weight_key][
        old_frequencies:
    ]
    bias[:old_frequencies] = source[bias_key][:old_frequencies]
    bias[new_frequencies : new_frequencies + old_frequencies] = source[bias_key][
        old_frequencies:
    ]
    expanded[weight_key] = weight
    expanded[bias_key] = bias
    model.load_state_dict(expanded, strict=True)
    return {
        "schema": "expanded_frequency_initialization_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
        "identity": str(identity_path),
        "identity_sha256": hashlib.sha256(identity_path.read_bytes()).hexdigest(),
        "parent_run_digest": identity["run_digest"],
        "parent_global_step": int(payload.get("global_step", -1)),
        "source_frequencies": old_frequencies,
        "target_frequencies": new_frequencies,
        "new_frequency_channels_zero": True,
        "optimizer_state_restored": False,
        "rng_state_restored": False,
        "strict_non_head_state": True,
    }


@torch.no_grad()
def zero_initialize_direct_frequency_head(model) -> dict[str, object]:
    synthesis = model.local_field.helmholtz_synthesis
    if synthesis is None or int(synthesis.rank) != 0 or not hasattr(synthesis, "head"):
        raise ValueError("zero initialization requires an independent direct-frequency head")
    synthesis.head.weight.zero_()
    synthesis.head.bias.zero_()
    return {
        "schema": "direct_frequency_zero_initialization_v1",
        "weight_nonzero": int(torch.count_nonzero(synthesis.head.weight)),
        "bias_nonzero": int(torch.count_nonzero(synthesis.head.bias)),
        "zero_field_at_initialization": True,
    }


def direct_frequency_target_coefficients(
    target: torch.Tensor,
    count: int,
    *,
    arrival_time_s: torch.Tensor | None = None,
    saved_time_s: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the cosine/sine fields used by the fixed direct synthesis."""

    if target.ndim != 4 or count <= 0:
        raise ValueError("direct-frequency target must be [record,time,z,x]")
    spectrum = torch.fft.rfft(target.float(), dim=1, norm="forward")[:, :count]
    if spectrum.shape[1] != count:
        raise ValueError("requested direct-frequency count exceeds the time grid")
    cosine = 2.0 * spectrum.real
    sine = -2.0 * spectrum.imag
    cosine[:, 0] = spectrum[:, 0].real
    sine[:, 0] = 0.0
    if arrival_time_s is not None:
        if saved_time_s is None or saved_time_s.ndim != 1 or saved_time_s.numel() != target.shape[1]:
            raise ValueError("WKB coefficient targets require the complete saved-time axis")
        arrival = torch.as_tensor(
            arrival_time_s, device=target.device, dtype=target.dtype
        )
        if arrival.shape != (target.shape[0], target.shape[2], target.shape[3]):
            raise ValueError("arrival map shape does not match coefficient target")
        dt = (saved_time_s[-1] - saved_time_s[0]) / float(saved_time_s.numel() - 1)
        omega = (
            2.0
            * torch.pi
            * torch.arange(count, device=target.device, dtype=target.dtype)
            / (float(saved_time_s.numel()) * dt.to(target.dtype))
        )
        phase = omega[None, :, None, None] * arrival[:, None]
        phase_cos = torch.cos(phase)
        phase_sin = torch.sin(phase)
        raw_cosine = cosine
        raw_sine = sine
        cosine = raw_cosine * phase_cos + raw_sine * phase_sin
        sine = -raw_cosine * phase_sin + raw_sine * phase_cos
    return torch.cat((cosine, sine), dim=1)


def direct_frequency_relative_l2_squared(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("direct-frequency prediction and target shapes differ")
    error = (prediction.float() - target.float()).square().sum()
    denominator = target.float().square().sum().clamp_min(1.0e-20)
    value = error / denominator
    if not bool(torch.isfinite(value)):
        raise FloatingPointError("non-finite direct-frequency objective")
    return value


def direct_frequency_balanced_relative_l2_squared(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    frequency_count: int,
    energy_floor_fraction: float,
) -> torch.Tensor:
    """Mean per-frequency relative error with a record-local energy floor."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("balanced direct-frequency tensors must match [R,2F,Z,X]")
    if prediction.shape[1] != 2 * int(frequency_count):
        raise ValueError("balanced direct-frequency channel count changed")
    floor_fraction = float(energy_floor_fraction)
    if not 0.0 < floor_fraction <= 1.0:
        raise ValueError("coefficient frequency energy floor must lie in (0,1]")
    records = prediction.shape[0]
    error = (prediction.float() - target.float()).reshape(
        records, 2, int(frequency_count), -1
    ).square().sum(dim=(1, 3))
    energy = target.float().reshape(
        records, 2, int(frequency_count), -1
    ).square().sum(dim=(1, 3))
    floor = floor_fraction * energy.amax(dim=1, keepdim=True)
    value = (error / torch.maximum(energy, floor).clamp_min(1.0e-20)).mean()
    if not bool(torch.isfinite(value)):
        raise FloatingPointError("non-finite balanced direct-frequency objective")
    return value


def coefficient_microbatch_backward_weights(
    microbatches,
    family_gradient_weights: dict[str, float] | None,
) -> tuple[float, ...]:
    """Normalize family weights across coefficient-supervision microbatches.

    The diagnostic uses one record per microbatch for family-focused runs.  Equal
    weights preserve the historical ``loss / len(micros)`` update exactly.  A
    mixed-family microbatch is only accepted when all of its records have the same
    registered weight, which prevents silently assigning one scalar weight to a
    heterogeneous loss reduction.
    """

    configured = {
        family: 1.0 for family in FAMILIES
    } if family_gradient_weights is None else {
        family: float(family_gradient_weights.get(family, 1.0))
        for family in FAMILIES
    }
    raw: list[float] = []
    for micro in microbatches:
        values = tuple(configured[str(family)] for family in micro.medium_type)
        if not values:
            raise ValueError("coefficient microbatch has no records")
        if any(not math.isclose(value, values[0]) for value in values[1:]):
            raise ValueError(
                "family-weighted coefficient supervision requires homogeneous "
                "microbatches (use --microbatch-records 1)"
            )
        raw.append(float(values[0]))
    total = float(sum(raw))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("coefficient family weights must have positive finite sum")
    return tuple(value / total for value in raw)


def metrics_meet_registered_target(
    metrics: dict[str, object],
    *,
    aggregate_maximum: float,
    family_maximum: float,
) -> bool:
    families = metrics.get("family_relative_l2")
    if not isinstance(families, dict):
        return False
    return float(metrics["aggregate_relative_l2"]) <= float(aggregate_maximum) and all(
        float(families.get(family, math.inf)) <= float(family_maximum)
        for family in FAMILIES
    )


def family_gate_score(metrics: dict[str, object]) -> float:
    """Worst aggregate/family error used for checkpoint selection."""

    families = metrics.get("family_relative_l2")
    if not isinstance(families, dict):
        raise ValueError("family metrics are required for gate-aware selection")
    return max(
        float(metrics["aggregate_relative_l2"]),
        *(float(families[family]) for family in FAMILIES),
    )


def coefficient_prediction_for_supervision(
    synthesis,
    raw_coefficients: torch.Tensor,
    rendered: torch.Tensor,
) -> torch.Tensor:
    """Return the exact coefficient tensor consumed by time synthesis."""

    if synthesis.frequency_softmax:
        return synthesis.apply_frequency_gate_to_coefficients(
            raw_coefficients,
            rendered,
        )
    return raw_coefficients


def coefficient_arrival_time_s(
    model,
    dense_travel_time_s: torch.Tensor,
    source_parameters: torch.Tensor,
) -> torch.Tensor:
    """Return the phase arrival used by both WKB targets and synthesis."""

    arrival = torch.as_tensor(dense_travel_time_s)
    source = torch.as_tensor(
        source_parameters,
        dtype=arrival.dtype,
        device=arrival.device,
    )
    if arrival.ndim != 3 or source.shape != (arrival.shape[0], 5):
        raise ValueError("coefficient arrival inputs must be [record,z,x] and [record,5]")
    if bool(getattr(model.local_field, "helmholtz_source_onset_phase", False)):
        arrival = arrival + source[:, 3][:, None, None]
    return arrival


def _direct_frequency_coefficient_update(
    model,
    optimizer,
    batch,
    normalizer,
    device: torch.device,
    *,
    full_targets: dict[str, np.ndarray],
    frequency_count: int,
    microbatch_records: int,
    saved_time_s: torch.Tensor,
    frequency_energy_floor_fraction: float,
    family_gradient_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Backpropagate exact full-trace Fourier labels without a 401-frame graph."""

    synthesis = model.local_field.helmholtz_synthesis
    head = synthesis.head
    optimizer.zero_grad(set_to_none=True)
    micros = tuple(
        split_pilot_batch(batch, microbatch_records=int(microbatch_records))
    )
    backward_weights = coefficient_microbatch_backward_weights(
        micros, family_gradient_weights
    )
    relative_values: list[float] = []
    for micro, backward_weight in zip(micros, backward_weights, strict=True):
        tensors = _to_device(micro, device)
        source = tensors["source_parameters"]
        prepared = model.prepare_sources(
            model.encode_medium(tensors["velocity_mps"], normalizer),
            source,
            tensors["source_map"],
            normalizer,
            record_to_medium=tensors["record_to_medium"],
        )
        dense_grid = model.prepare_dense_grid(
            prepared,
            x_m=tensors["x_m"],
            z_m=tensors["z_m"],
            travel_time_s=(
                None
                if micro.dense_travel_time_s is None
                else micro.dense_travel_time_s.to(
                    device, non_blocking=device.type == "cuda"
                )
            ),
        )
        captured: list[tuple[torch.Tensor, torch.Tensor]] = []
        hook = head.register_forward_hook(
            lambda _module, inputs, output: captured.append((inputs[0], output))
        )
        try:
            model.dense_normalized_with_coarse(
                prepared,
                tensors["requested_time_s"][:, :1],
                dense_grid=dense_grid,
                time_block=1,
            )
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("direct-frequency head was not captured exactly once")
        rendered, predicted_coefficients = captured[0]
        # Coefficient supervision must optimize the same gated coefficients
        # consumed by time synthesis.  Supervising the raw head output would
        # bypass the new gate and leave every gate gradient at None.
        predicted_coefficients = coefficient_prediction_for_supervision(
            synthesis,
            predicted_coefficients,
            rendered,
        )
        physical = torch.stack(
            [
                torch.from_numpy(full_targets[sample_id])
                for sample_id in micro.sample_id
            ],
            dim=0,
        ).to(device=device, dtype=tensors["dense_target_physical"].dtype)
        normalized = normalizer.encode_pressure(physical, source[:, 4])
        target_coefficients = direct_frequency_target_coefficients(
            normalized,
            int(frequency_count),
            arrival_time_s=(
                coefficient_arrival_time_s(
                    model,
                    micro.dense_travel_time_s.to(device),
                    source,
                )
                if model.local_field.helmholtz_synthesis.wkb_phase
                else None
            ),
            saved_time_s=(
                saved_time_s.to(device)
                if model.local_field.helmholtz_synthesis.wkb_phase
                else None
            ),
        )
        if float(frequency_energy_floor_fraction) > 0.0:
            loss = direct_frequency_balanced_relative_l2_squared(
                predicted_coefficients,
                target_coefficients,
                frequency_count=int(frequency_count),
                energy_floor_fraction=float(frequency_energy_floor_fraction),
            )
        else:
            loss = direct_frequency_relative_l2_squared(
                predicted_coefficients, target_coefficients
            )
        (loss * float(backward_weight)).backward()
        relative_values.append(math.sqrt(max(float(loss.detach()), 0.0)))
    return {
        "coefficient_relative_l2": float(sum(relative_values) / len(relative_values)),
        "total": float(sum(value * value for value in relative_values) / len(relative_values)),
        "full_trace_label_frames": 401.0,
    }


def build_probe_config(
    *, dense_lr: float, backbone_lr: float, temporal_lr: float, seed: int, travel_time_h5: str,
    family_gradient_weights: dict | None = None,
    per_frame_frame: bool = False,
    frame_energy_floor_fraction: float = 0.0,
    late_frame_gain: float = 0.0,
    late_frame_start_fraction: float = 0.4,
    full_field_frame_weight: float = 1.0,
    delta_loss_weight: float = 0.5,
    delta_energy_floor_fraction: float = 0.1,
    delta_reference: str = "model_coarse",
    delta_reduction: str = "per_record",
    spatial_gradient_loss_weight: float = 0.1,
) -> dict:
    """Mirror the v54 diagnostic loss/optimizer contract (known-good for _train_update).

    ``delta`` in the loss selects the coarse+correction recovery path, matching the
    real architecture and the v51-v55 comparison baseline. No family_experts,
    band adapter, or numerical teacher keys, so the clean dense path is taken.

    ``family_gradient_weights`` (default all 1.0) scales the per-family backward so a
    stubborn family (e.g. marmousi) can be focused without touching the others.

    ``per_frame_frame`` switches the full-field frame term from record-normalized to
    per-frame-normalized so low-amplitude late frames are trained as hard as the early
    wavefront. ``frame_energy_floor_fraction`` floors near-zero pre-onset frames so
    they do not divide by ~0. ``late_frame_gain`` > 0 tilts the per-frame mean toward
    late times via w(t)=1+gain*ramp(t) starting at ``late_frame_start_fraction`` of the
    time axis. All default off so the record-normalized A+1 baseline reproduces exactly.
    """

    weights = (
        {family: 1.0 for family in FAMILIES}
        if family_gradient_weights is None
        else {family: float(family_gradient_weights.get(family, 1.0)) for family in FAMILIES}
    )

    loss_cfg = {
        "full_field_frame": float(full_field_frame_weight),
        "delta": float(delta_loss_weight),
        "delta_reference": str(delta_reference),
        "delta_reduction": str(delta_reduction),
        "delta_energy_floor_fraction": float(delta_energy_floor_fraction),
        "hard_causality": True,
        "hard_causality_lead_cycles": 1.0,
        "spatial_gradient": float(spatial_gradient_loss_weight),
        "spectrum": 0.0,
        "temporal_difference": 0.0,
        "per_frame_frame": bool(per_frame_frame),
        "frame_energy_floor_fraction": float(frame_energy_floor_fraction),
        "late_frame_gain": float(late_frame_gain),
        "late_frame_start_fraction": float(late_frame_start_fraction),
    }

    return {
        "seed": int(seed),
        "energy_floor_fraction": 0.01,
        "travel_time_h5": str(travel_time_h5),
        "loss": loss_cfg,
        "optimizer": {
            "dense_learning_rate": float(dense_lr),
            "backbone_learning_rate": float(backbone_lr),
            "geometry_learning_rate": float(backbone_lr),
            "temporal_basis_learning_rate": float(temporal_lr),
            "gradient_clip": 1.0,
            "gradient_clip_mode": "prefix_limits",
            "gradient_clip_prefix_limits": {
                "coordinate_encoder": 5.0,
                "default": 1.0,
                "dense_decoder": 20.0,
                "local_field": 20.0,
                "fusion": 5.0,
                "medium_encoder": 2.0,
                "source_encoder": 5.0,
                "travel_branch": 5.0,
            },
            "full_forward_checkpointing": True,
            "weight_decay": 0.0,
        },
        "family_gradient_weights": weights,
    }


def build_base_config(width: int, *, base_config: str = BASE_CONFIG) -> V3Config:
    base = V3Config.from_yaml(str(base_config))
    if int(width) != int(base.model.width):
        if int(width) % int(base.model.heads) != 0:
            raise ValueError(
                f"width {width} must be divisible by attention heads {base.model.heads}"
            )
        model_cfg = dataclasses.replace(base.model, width=int(width))
        base = dataclasses.replace(base, model=model_cfg)
    return base


def transfer_front_end(model, checkpoint_path: Path) -> dict:
    """Load only the front-end tensors from a parent checkpoint; leave decoder fresh."""

    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = payload["model_state"] if "model_state" in payload else payload
    front_end = {
        key: value
        for key, value in state.items()
        if not key.startswith("dense_decoder.")
    }
    target_keys = set(model.state_dict().keys())
    usable = {k: v for k, v in front_end.items() if k in target_keys}
    result = model.load_state_dict(usable, strict=False)
    missing = [k for k in result.missing_keys]
    unexpected = list(result.unexpected_keys)
    transferred = sorted(usable.keys())
    non_decoder_missing = [k for k in missing if not k.startswith("dense_decoder.")]
    if non_decoder_missing:
        raise ValueError(
            "front-end transfer left non-decoder tensors uninitialized: "
            + ", ".join(non_decoder_missing[:5])
        )
    if unexpected:
        raise ValueError(f"front-end transfer had unexpected keys: {unexpected[:5]}")
    return {
        "checkpoint": str(checkpoint_path),
        "transferred_tensor_count": len(transferred),
        "decoder_tensors_left_fresh": sum(
            1 for k in missing if k.startswith("dense_decoder.")
        ),
    }


def warmstart_full_checkpoint(model, checkpoint_path: Path) -> dict:
    """Load every shape-matching tensor from a full parent checkpoint.

    ControlNet-style warm start: the parent (e.g. W2_w128_d12) holds the trained
    MIONet front-end AND dense decoder but has no ``local_field.*`` tensors. We
    load all of it strictly by shape and leave only the zero-initialised
    ``local_field`` generator fresh. Because the residual output conv is zero-init,
    the local field contributes exactly nothing at update-0, so the baseline
    evaluation reproduces the parent's trained metric. Any missing tensor that is
    not ``local_field.*``, any unexpected source key, or any shape mismatch is a
    hard error (the architecture would not actually match the parent).
    """

    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), dict):
        raise ValueError("warm-start checkpoint is missing model_state")
    source = payload["model_state"]
    target = model.state_dict()

    loaded: dict[str, torch.Tensor] = {}
    shape_mismatch: list[dict] = []
    for key, value in source.items():
        if key not in target:
            continue  # tallied as unexpected below
        if tuple(target[key].shape) != tuple(value.shape):
            shape_mismatch.append(
                {"key": key, "source": list(value.shape), "target": list(target[key].shape)}
            )
            continue
        loaded[key] = value

    result = model.load_state_dict(loaded, strict=False)
    unexpected_source = sorted(k for k in source if k not in target)
    missing = sorted(result.missing_keys)
    non_local_missing = [k for k in missing if not k.startswith("local_field.")]

    if shape_mismatch:
        raise ValueError(
            "warm start architecture differs from parent (shape mismatch): "
            + ", ".join(f"{m['key']} src{m['source']} dst{m['target']}" for m in shape_mismatch[:4])
        )
    if unexpected_source:
        raise ValueError(
            f"warm start had source keys not present in target: {unexpected_source[:8]}"
        )
    if non_local_missing:
        raise ValueError(
            "warm start left non-local_field tensors uninitialised: "
            + ", ".join(non_local_missing[:8])
        )

    return {
        "kind": "full_checkpoint_warmstart",
        "checkpoint": str(checkpoint_path),
        "source_format": payload.get("format"),
        "source_global_step": payload.get("global_step"),
        "source_config_digest": payload.get("config_digest"),
        "source_manifest_digest": payload.get("manifest_digest"),
        "source_metrics": payload.get("metrics"),
        "loaded_tensor_count": len(loaded),
        "target_tensor_count": len(target),
        "loaded_keys_sample": sorted(loaded.keys())[:16],
        "fresh_tensor_count": len(missing),
        "fresh_keys": missing,
        "unexpected_source_keys": unexpected_source,
        "shape_mismatches": shape_mismatch,
    }


def build_capacity_optimizer(
    model, *, dense_lr: float, backbone_lr: float, local_field_lr: float | None = None,
    optimizer_name: str = "adamw",
) -> torch.optim.Optimizer:
    """Optimizer covering every trainable parameter (guaranteed coverage).

    When ``local_field_lr`` is None the local field shares the dense group (legacy
    two-group behaviour, preserved for the from-scratch ladder runs). When set, the
    ``local_field.*`` tensors get their own group at ``local_field_lr`` so a fresh
    (zero-init) residual generator warm-started on top of a trained decoder can
    learn faster than the already-converged decoder is nudged.

    ``optimizer_name`` selects the algorithm:
      * ``adamw`` (default) -- torch AdamW, betas (0.9, 0.99), wd 0. Legacy behaviour.
      * ``soap``            -- Shampoo-style second-order preconditioner (pytorch-optimizer);
        motivated by the project finding that spectral bias is dynamical and a
        second-order optimizer reorders spectral/rank learning to reach high-k / high-rank
        content the first-order AdamW learns only slowly.
    """

    dense_params, local_field_params, other_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("local_field."):
            (local_field_params if local_field_lr is not None else dense_params).append(param)
        elif name.startswith("dense_decoder."):
            dense_params.append(param)
        else:
            other_params.append(param)
    groups = []
    if dense_params:
        groups.append({"params": dense_params, "lr": float(dense_lr)})
    if local_field_params:
        groups.append({"params": local_field_params, "lr": float(local_field_lr)})
    if other_params:
        groups.append({"params": other_params, "lr": float(backbone_lr)})
    if not groups:
        raise ValueError("no trainable parameters found")
    name = str(optimizer_name).lower()
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=float(dense_lr), weight_decay=0.0, betas=(0.9, 0.99))
    if name == "soap":
        from pytorch_optimizer import SOAP

        # precondition_1d so bias/1-D tensors are also preconditioned; the small
        # precondition_frequency keeps the second-order estimate fresh during a short
        # continue-pretraining run. weight_decay kept 0 to match the AdamW baseline.
        return SOAP(
            groups,
            lr=float(dense_lr),
            betas=(0.95, 0.95),
            weight_decay=0.0,
            precondition_frequency=10,
            precondition_1d=True,
        )
    raise ValueError(f"unsupported optimizer_name: {optimizer_name!r}")


TRAINABLE_PREFIXES = (
    "dense_decoder",
    "medium_encoder",
    "source_encoder",
    "fusion",
    "coordinate_encoder",
    "travel_branch",
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument(
        "--base-config",
        default=BASE_CONFIG,
        help="dataset/model base YAML; use the repaired manifest-bound config for new work",
    )
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--dense-depth", type=int, default=8)
    parser.add_argument("--dense-spectral-rank", type=int, default=112)
    parser.add_argument("--dense-modes", type=int, default=32)
    parser.add_argument(
        "--local-field",
        action="store_true",
        help="replace the global low-rank MIONet coarse field with the translation-"
        "equivariant local-propagation U-Net generator",
    )
    parser.add_argument(
        "--local-field-channels",
        default="1,1,2,2",
        help="comma-separated per-level channel multipliers of width for the local-field U-Net",
    )
    parser.add_argument("--local-field-causal-width-s", type=float, default=0.005)
    parser.add_argument(
        "--helmholtz-synthesis",
        action="store_true",
        help="use the temporal-frequency (Helmholtz) coarse-field parametrization: the "
        "U-Net renders once per record and a complex-field head predicts nf static "
        "Helmholtz fields synthesized to any saved time by a fixed inverse transform "
        "(query-invariant; attacks the position/phase floor the per-frame forms cannot). "
        "Requires --local-field.",
    )
    parser.add_argument(
        "--helmholtz-frequencies",
        type=int,
        default=64,
        help="number of lowest rfft temporal frequency bins the Helmholtz head predicts "
        "(oracle bound: 64 bins reach <5%% on all families)",
    )
    parser.add_argument(
        "--helmholtz-no-wkb",
        action="store_true",
        help="disable the WKB retarded-time ansatz and predict raw (oscillatory) cos/sin "
        "coefficient fields directly; default uses WKB so the network only outputs smooth "
        "amplitude envelopes and the eikonal travel time supplies the fast oscillation",
    )
    parser.add_argument(
        "--helmholtz-rank",
        type=int,
        default=0,
        help="low-rank cross-frequency factorization: R shared smooth complex basis "
        "fields mixed per frequency (amplitude probe measured cross-frequency rank <=6). "
        "0 = independent 2*nf coefficient head (temporally underdetermined per the G2 probe)",
    )
    parser.add_argument(
        "--helmholtz-frequency-softmax",
        action="store_true",
        help="add a zero-init (identity) per-record softmax gate over the Helmholtz "
        "frequency bins; lets each record re-weight the band emphasis of the "
        "synthesized field (spectrum audit: mid/high bands fit worst) without "
        "changing the analytic time basis or query invariance",
    )
    parser.add_argument(
        "--helmholtz-source-onset-phase",
        action="store_true",
        help="use tau+t0 in the WKB retarded phase so source onset is removed "
        "analytically from coefficient targets and inverse synthesis",
    )
    parser.add_argument(
        "--helmholtz-source-relative-coordinates",
        action="store_true",
        help="add a zero-init source-centered (dx,dz,r) spatial feature projection "
        "to the Helmholtz record mapper",
    )
    parser.add_argument(
        "--helmholtz-coefficient-supervision",
        action="store_true",
        help="supervise the direct raw 64-bin head from the complete train-trace rFFT "
        "while forwarding one sentinel frame; Parseval-equivalent without a 401-frame graph",
    )
    parser.add_argument(
        "--helmholtz-zero-init-head",
        action="store_true",
        help="zero the independent direct-frequency output affine map so the initial field is exactly zero",
    )
    parser.add_argument(
        "--helmholtz-disable-causal-gate",
        action="store_true",
        help="return the complete coefficient-synthesized waveform without the legacy post-synthesis arrival sigmoid",
    )
    parser.add_argument(
        "--helmholtz-disable-free-surface-factor",
        action="store_true",
        help="do not re-taper a complete-field coefficient output by the legacy 20 m free-surface factor",
    )
    parser.add_argument(
        "--helmholtz-disable-hard-causality",
        action="store_true",
        help="do not hard-zero the complete Fourier waveform again in loss/evaluation post-processing",
    )
    parser.add_argument(
        "--coefficient-frequency-energy-floor-fraction",
        type=float,
        default=0.0,
        help="if positive, average per-frequency coefficient relative errors using "
        "this fraction of each record's peak-bin energy as the denominator floor",
    )
    parser.add_argument(
        "--background-cache",
        help="Direction A+1: NumericalTeacherCache-format smoothed-velocity background "
        "P_bg (scripts/build_smoothed_background_cache.py). Retargets the model onto the "
        "scattering residual encode(scat)=encode(wf)-encode(P_bg); metrics add P_bg back. "
        "encode_pressure is linear so this is exact. Oracle floor <1%% (layered/marmousi).",
    )
    parser.add_argument(
        "--local-field-residual",
        action="store_true",
        help="keep the global MIONet coarse field and ADD the local-propagation "
        "U-Net as a zero-initialised residual (ControlNet-style), so training "
        "starts on the global-coarse trajectory instead of relearning it from "
        "scratch; the MIONet front-end stays active and trainable",
    )
    parser.add_argument(
        "--local-field-grad-clip",
        type=float,
        default=20.0,
        help="per-prefix gradient-norm clip limit for the fresh local-field U-Net; "
        "raise above 20 to stop throttling the coarse-field generator early in training",
    )
    parser.add_argument(
        "--normalization-json",
        default=None,
        help="override the normalizer JSON (e.g. the legacy pinned digest that matches "
        "the current dataset after the TGRS-ablation digest drift); default keeps the "
        "base config's path",
    )
    parser.add_argument("--warmstart-frontend")
    parser.add_argument("--exact-init-checkpoint")
    parser.add_argument("--exact-init-identity")
    parser.add_argument("--expanded-frequency-init-checkpoint")
    parser.add_argument("--expanded-frequency-init-identity")
    parser.add_argument("--continue-frequency-init-checkpoint")
    parser.add_argument("--continue-frequency-init-identity")
    parser.add_argument(
        "--warmstart-checkpoint",
        help="ControlNet-style full warm start: load every shape-matching tensor "
        "(MIONet front-end AND dense decoder) from this parent checkpoint, leaving "
        "only the zero-init local_field generator fresh. Valid at any width; the "
        "architecture (width/depth/rank/modes) must match the parent so update-0 "
        "reproduces the parent's trained metric. Use with --local-field-residual.",
    )
    parser.add_argument(
        "--local-field-learning-rate",
        type=float,
        default=None,
        help="if set, the fresh local_field generator gets its own AdamW group at "
        "this learning rate, decoupled from the (already-trained, warm-started) "
        "dense decoder which stays on --dense-learning-rate",
    )
    parser.add_argument(
        "--family-gradient-weights",
        default=None,
        help="per-family backward weight overrides, e.g. 'marmousi:2.0,layered:1.0'; "
        "focuses optimization on a stubborn family without touching the others "
        "(unspecified families default to 1.0)",
    )
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--evaluate-every", type=int, default=25)
    parser.add_argument("--dense-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--temporal-basis-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--validation-frames", type=int, default=32)
    parser.add_argument("--training-frames", type=int, default=16)
    parser.add_argument("--microbatch-records", type=int, default=3)
    parser.add_argument("--target-aggregate-relative-l2", type=float, default=0.10)
    parser.add_argument("--target-family-relative-l2", type=float, default=0.12)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument(
        "--travel-time-h5",
        default="/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5",
    )
    parser.add_argument("--seed", type=int, default=372)
    parser.add_argument(
        "--per-frame-frame",
        action="store_true",
        help="normalize the full-field frame loss per (record,time) frame instead of "
        "per record, so low-amplitude late frames are trained as hard as the early "
        "wavefront (fixes the layered late-frame short-fall in the A+1 hybrid solver)",
    )
    parser.add_argument(
        "--frame-energy-floor-fraction",
        type=float,
        default=0.05,
        help="floor each frame's energy denominator at this fraction of the record's "
        "peak-frame energy so near-zero pre-onset frames do not divide by ~0 "
        "(only used when --per-frame-frame is set)",
    )
    parser.add_argument(
        "--late-frame-gain",
        type=float,
        default=0.0,
        help="tilt the per-frame mean toward late times via w(t)=1+gain*ramp(t); "
        "0 = uniform per-frame mean (requires --per-frame-frame)",
    )
    parser.add_argument(
        "--late-frame-start-fraction",
        type=float,
        default=0.4,
        help="fraction of the time axis where the late-frame gain ramp begins",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="few updates and skip the expensive all-saved-times final evaluation",
    )
    args = parser.parse_args(argv)
    if args.updates <= 0 or args.evaluate_every <= 0:
        raise ValueError("updates and evaluation cadence must be positive")
    if args.target_aggregate_relative_l2 <= 0.0 or args.target_family_relative_l2 <= 0.0:
        raise ValueError("registered target thresholds must be positive")
    local_field_channels = tuple(
        int(token) for token in str(args.local_field_channels).split(",") if token.strip()
    )
    if args.local_field and (len(local_field_channels) < 2 or local_field_channels[0] != 1):
        raise ValueError("local-field channels need >=2 levels and a leading multiplier of 1")
    if args.local_field_residual and not args.local_field:
        raise ValueError("--local-field-residual requires --local-field")
    if args.helmholtz_synthesis and not args.local_field:
        raise ValueError("--helmholtz-synthesis requires --local-field")
    if args.helmholtz_synthesis and args.local_field_residual:
        raise ValueError(
            "--helmholtz-synthesis is a primary coarse-field form; it is mutually "
            "exclusive with --local-field-residual (per-frame residual on MIONet)"
        )
    if args.helmholtz_frequencies <= 0:
        raise ValueError("--helmholtz-frequencies must be positive")
    if args.helmholtz_coefficient_supervision:
        if not args.helmholtz_synthesis:
            raise ValueError("coefficient supervision requires Helmholtz synthesis")
        if int(args.helmholtz_rank) != 0:
            raise ValueError("coefficient supervision requires the independent head")
        if args.background_cache:
            raise ValueError("coefficient supervision forbids numerical backgrounds")
    if args.helmholtz_zero_init_head and int(args.helmholtz_rank) != 0:
        raise ValueError("zero head initialization requires the independent head")
    if args.helmholtz_disable_causal_gate and not args.helmholtz_coefficient_supervision:
        raise ValueError("disabling the Helmholtz causal gate requires complete coefficient supervision")
    if (
        args.helmholtz_disable_free_surface_factor
        or args.helmholtz_disable_hard_causality
    ) and not args.helmholtz_coefficient_supervision:
        raise ValueError("disabling Helmholtz output projections requires complete coefficient supervision")
    if not 0.0 <= float(args.coefficient_frequency_energy_floor_fraction) <= 1.0:
        raise ValueError("coefficient frequency energy floor must lie in [0,1]")
    if (
        float(args.coefficient_frequency_energy_floor_fraction) > 0.0
        and not args.helmholtz_coefficient_supervision
    ):
        raise ValueError("coefficient frequency balancing requires coefficient supervision")
    if args.late_frame_gain < 0.0:
        raise ValueError("--late-frame-gain must be nonnegative")
    if args.late_frame_gain > 0.0 and not args.per_frame_frame:
        raise ValueError("--late-frame-gain>0 requires --per-frame-frame")
    if not 0.0 <= args.frame_energy_floor_fraction <= 1.0:
        raise ValueError("--frame-energy-floor-fraction must lie in [0, 1]")
    if not 0.0 <= args.late_frame_start_fraction < 1.0:
        raise ValueError("--late-frame-start-fraction must lie in [0, 1)")

    family_gradient_weights = None
    if args.family_gradient_weights:
        family_gradient_weights = {}
        for token in str(args.family_gradient_weights).split(","):
            token = token.strip()
            if not token:
                continue
            family, _, value = token.partition(":")
            family = family.strip()
            if family not in FAMILIES:
                raise ValueError(f"unknown family in --family-gradient-weights: {family!r}")
            weight = float(value)
            if not weight > 0.0:
                raise ValueError("family gradient weights must be positive")
            family_gradient_weights[family] = weight

    root = Path(args.artifact_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    terminal_path = root / "terminal.json"
    if terminal_path.exists():
        print(terminal_path.read_text().strip())
        return 0

    warmstart = None if args.warmstart_frontend is None else Path(args.warmstart_frontend).resolve()
    if warmstart is not None:
        if int(args.width) != 64:
            raise ValueError("front-end warm start is only valid at width=64")
        if not warmstart.is_file():
            raise FileNotFoundError(f"warmstart checkpoint missing: {warmstart}")

    warmstart_checkpoint = (
        None if args.warmstart_checkpoint is None else Path(args.warmstart_checkpoint).resolve()
    )
    exact_init_checkpoint = (
        None
        if args.exact_init_checkpoint is None
        else Path(args.exact_init_checkpoint).resolve()
    )
    exact_init_identity = (
        None
        if args.exact_init_identity is None
        else Path(args.exact_init_identity).resolve()
    )
    expanded_init_checkpoint = (
        None if args.expanded_frequency_init_checkpoint is None
        else Path(args.expanded_frequency_init_checkpoint).resolve()
    )
    expanded_init_identity = (
        None if args.expanded_frequency_init_identity is None
        else Path(args.expanded_frequency_init_identity).resolve()
    )
    continue_init_checkpoint = (
        None if args.continue_frequency_init_checkpoint is None
        else Path(args.continue_frequency_init_checkpoint).resolve()
    )
    continue_init_identity = (
        None if args.continue_frequency_init_identity is None
        else Path(args.continue_frequency_init_identity).resolve()
    )
    if (exact_init_checkpoint is None) != (exact_init_identity is None):
        raise ValueError("exact initialization requires checkpoint and identity")
    if (expanded_init_checkpoint is None) != (expanded_init_identity is None):
        raise ValueError("expanded-frequency initialization requires checkpoint and identity")
    if (continue_init_checkpoint is None) != (continue_init_identity is None):
        raise ValueError("continue-frequency initialization requires checkpoint and identity")
    init_count = sum(
        candidate is not None
        for candidate in (exact_init_checkpoint, expanded_init_checkpoint, continue_init_checkpoint)
    )
    if init_count > 1:
        raise ValueError("exact, expanded-frequency, and continue-frequency initialization are mutually exclusive")
    if continue_init_checkpoint is not None:
        if warmstart is not None or warmstart_checkpoint is not None:
            raise ValueError("continue-frequency initialization is mutually exclusive with warm starts")
        if not continue_init_checkpoint.is_file() or not continue_init_identity.is_file():
            raise FileNotFoundError("continue-frequency initialization artifact is missing")
    if exact_init_checkpoint is not None:
        if warmstart is not None or warmstart_checkpoint is not None:
            raise ValueError("exact initialization is mutually exclusive with warm starts")
        if not exact_init_checkpoint.is_file() or not exact_init_identity.is_file():
            raise FileNotFoundError("exact initialization artifact is missing")
    if expanded_init_checkpoint is not None:
        if warmstart is not None or warmstart_checkpoint is not None:
            raise ValueError("expanded-frequency initialization is mutually exclusive with warm starts")
        if not expanded_init_checkpoint.is_file() or not expanded_init_identity.is_file():
            raise FileNotFoundError("expanded-frequency initialization artifact is missing")
    if warmstart_checkpoint is not None:
        if warmstart is not None:
            raise ValueError(
                "--warmstart-checkpoint (full) and --warmstart-frontend are mutually exclusive"
            )
        if not warmstart_checkpoint.is_file():
            raise FileNotFoundError(
                f"warmstart checkpoint missing: {warmstart_checkpoint}"
            )
        if not (args.local_field and args.local_field_residual):
            raise ValueError(
                "--warmstart-checkpoint expects --local-field --local-field-residual so "
                "the only fresh tensors are the zero-init local_field generator"
            )

    device = torch.device("cuda")
    seed = int(args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    base = build_base_config(int(args.width), base_config=str(args.base_config))
    if args.normalization_json is not None:
        data_cfg = dataclasses.replace(
            base.data, normalization_json=str(args.normalization_json)
        )
        base = dataclasses.replace(base, data=data_cfg)
    manifest = build_manifest(base.data.source_h5)
    validate_expected_counts(
        manifest,
        {
            "train": base.data.expected_train_records,
            "validation": base.data.expected_validation_records,
        },
    )
    normalizer = load_normalizer(base, manifest.digest)

    variant = ProbeVariant(
        depth=int(args.dense_depth),
        use_local_phase=True,
        spectral_rank=int(args.dense_spectral_rank),
        modes=int(args.dense_modes),
        temporal_basis_rank=0,
        family_expert_rank=0,
        local_field=bool(args.local_field),
        local_field_channel_multipliers=local_field_channels,
        local_field_causal_width_s=float(args.local_field_causal_width_s),
        local_field_residual=bool(args.local_field_residual),
        local_field_helmholtz_synthesis=bool(args.helmholtz_synthesis),
        local_field_helmholtz_synthesis_frequencies=int(args.helmholtz_frequencies),
        local_field_helmholtz_synthesis_wkb_phase=not bool(args.helmholtz_no_wkb),
        local_field_helmholtz_synthesis_rank=int(args.helmholtz_rank),
        local_field_helmholtz_synthesis_frequency_softmax=bool(
            args.helmholtz_frequency_softmax
        ),
        local_field_helmholtz_synthesis_source_onset_phase=bool(
            args.helmholtz_source_onset_phase
        ),
        local_field_helmholtz_source_relative_coordinates=bool(
            args.helmholtz_source_relative_coordinates
        ),
    )
    model = _model(base, manifest, variant).to(device)
    if args.helmholtz_disable_causal_gate:
        model.local_field.helmholtz_apply_causal_gate = False
    if args.helmholtz_disable_free_surface_factor:
        model.dense_apply_free_surface_factor = False

    # In local-field REPLACE mode the global MIONet coarse path (fusion /
    # coordinate_encoder / travel_branch) is bypassed and receives no gradient;
    # the local_field U-Net is the sole coarse generator. In RESIDUAL mode the
    # MIONet path stays active (it produces the base coarse) and the zero-init
    # local_field is trained on top of it, so every trainable prefix is live.
    if args.local_field and not args.local_field_residual:
        active_prefixes = (
            "local_field",
            "medium_encoder",
            "source_encoder",
        )
        if not args.helmholtz_coefficient_supervision:
            active_prefixes = ("dense_decoder",) + active_prefixes
    elif args.local_field:
        active_prefixes = TRAINABLE_PREFIXES + ("local_field",)
    else:
        active_prefixes = TRAINABLE_PREFIXES

    transfer_report = None
    if exact_init_checkpoint is not None:
        transfer_report = load_exact_model_initialization(
            model,
            exact_init_checkpoint,
            exact_init_identity,
            manifest_digest=manifest.digest,
            device=device,
        )
    elif expanded_init_checkpoint is not None:
        transfer_report = load_expanded_frequency_initialization(
            model,
            expanded_init_checkpoint,
            expanded_init_identity,
            manifest_digest=manifest.digest,
            device=device,
        )
    elif continue_init_checkpoint is not None:
        transfer_report = load_continue_frequency_initialization(
            model,
            continue_init_checkpoint,
            continue_init_identity,
            manifest_digest=manifest.digest,
            device=device,
        )
    elif warmstart is not None:
        transfer_report = transfer_front_end(model, warmstart)
    elif warmstart_checkpoint is not None:
        transfer_report = warmstart_full_checkpoint(model, warmstart_checkpoint)
    zero_init_report = None
    if args.helmholtz_zero_init_head:
        if transfer_report is not None:
            raise ValueError("zero head initialization cannot overwrite transferred weights")
        zero_init_report = zero_initialize_direct_frequency_head(model)

    for param in model.parameters():
        param.requires_grad_(True)
    parameter_count = int(sum(p.numel() for p in model.parameters()))

    config = build_probe_config(
        dense_lr=float(args.dense_learning_rate),
        backbone_lr=float(args.backbone_learning_rate),
        temporal_lr=float(args.temporal_basis_learning_rate),
        seed=seed,
        travel_time_h5=str(args.travel_time_h5),
        family_gradient_weights=family_gradient_weights,
        per_frame_frame=bool(args.per_frame_frame),
        frame_energy_floor_fraction=float(args.frame_energy_floor_fraction),
        late_frame_gain=float(args.late_frame_gain),
        late_frame_start_fraction=float(args.late_frame_start_fraction),
    )
    if args.helmholtz_disable_hard_causality:
        config["loss"]["hard_causality"] = False
    if args.local_field:
        config["optimizer"]["gradient_clip_prefix_limits"]["local_field"] = float(
            args.local_field_grad_clip
        )
    optimizer = build_capacity_optimizer(
        model,
        dense_lr=float(args.dense_learning_rate),
        backbone_lr=float(args.backbone_learning_rate),
        local_field_lr=(
            None if args.local_field_learning_rate is None else float(args.local_field_learning_rate)
        ),
    )

    indices = select_one_index_per_family(manifest.records, split=args.split)
    selected_records = tuple(
        record for record in manifest.records if record.split == args.split
    )
    sample_ids = tuple(selected_records[index].sample_id for index in indices)

    updates = 8 if args.smoke else int(args.updates)
    evaluate_every = 4 if args.smoke else int(args.evaluate_every)
    schedule = build_repeat_schedule(indices, updates=updates)
    effective_training_frames = (
        4 if args.helmholtz_coefficient_supervision else int(args.training_frames)
    )
    train_data = _dataset(
        config,
        base,
        manifest,
        indices,
        split=args.split,
        schedule=schedule,
        time_policy="appearance16",
        frames_per_record=effective_training_frames,
    )

    identity = {
        "schema": "saved_time_capacity_ladder_overfit_v1",
        "base_config": str(args.base_config),
        "width": int(args.width),
        "dense_depth": int(args.dense_depth),
        "dense_spectral_rank": int(args.dense_spectral_rank),
        "dense_modes": int(args.dense_modes),
        "local_field": bool(args.local_field),
        "local_field_channel_multipliers": list(local_field_channels) if args.local_field else None,
        "local_field_causal_width_s": float(args.local_field_causal_width_s) if args.local_field else None,
        "local_field_grad_clip": float(args.local_field_grad_clip) if args.local_field else None,
        "local_field_residual": bool(args.local_field_residual),
        "additive_heads": {"temporal_basis_rank": 0, "family_expert_rank": 0, "band_adapter_rank": 0},
        "parameter_count": parameter_count,
        "front_end_transfer": transfer_report,
        "direct_frequency_zero_initialization": zero_init_report,
        "helmholtz_apply_causal_gate": bool(
            model.local_field.helmholtz_apply_causal_gate
            if args.helmholtz_synthesis
            else True
        ),
        "dense_apply_free_surface_factor": bool(
            model.dense_apply_free_surface_factor
        ),
        "hard_causality_postprocessing": bool(config["loss"]["hard_causality"]),
        "warmstart_frontend": None if warmstart is None else str(warmstart),
        "warmstart_checkpoint": None if warmstart_checkpoint is None else str(warmstart_checkpoint),
        "exact_init_checkpoint": (
            None if exact_init_checkpoint is None else str(exact_init_checkpoint)
        ),
        "exact_init_identity": (
            None if exact_init_identity is None else str(exact_init_identity)
        ),
        "expanded_frequency_init_checkpoint": (
            None if expanded_init_checkpoint is None else str(expanded_init_checkpoint)
        ),
        "expanded_frequency_init_identity": (
            None if expanded_init_identity is None else str(expanded_init_identity)
        ),
        "continue_frequency_init_checkpoint": (
            None if continue_init_checkpoint is None else str(continue_init_checkpoint)
        ),
        "continue_frequency_init_identity": (
            None if continue_init_identity is None else str(continue_init_identity)
        ),
        "helmholtz_frequency_softmax": bool(args.helmholtz_frequency_softmax),
        "helmholtz_source_onset_phase": bool(args.helmholtz_source_onset_phase),
        "helmholtz_source_relative_coordinates": bool(
            args.helmholtz_source_relative_coordinates
        ),
        "local_field_learning_rate": (
            None if args.local_field_learning_rate is None else float(args.local_field_learning_rate)
        ),
        "family_gradient_weights": family_gradient_weights,
        "manifest_digest": manifest.digest,
        "split": str(args.split),
        "record_indices": indices,
        "sample_ids": sample_ids,
        "families": FAMILIES,
        "updates": updates,
        "evaluate_every": evaluate_every,
        "training_frames_per_record": int(args.training_frames),
        "dataset_forward_frames_per_record": effective_training_frames,
        "helmholtz_coefficient_supervision": bool(
            args.helmholtz_coefficient_supervision
        ),
        "coefficient_target_frames": (
            len(manifest.time_s) if args.helmholtz_coefficient_supervision else None
        ),
        "coefficient_frequency_energy_floor_fraction": float(
            args.coefficient_frequency_energy_floor_fraction
        ),
        "validation_frames_per_record": int(args.validation_frames),
        "microbatch_records": int(args.microbatch_records),
        "target_aggregate_relative_l2": float(args.target_aggregate_relative_l2),
        "target_family_relative_l2": float(args.target_family_relative_l2),
        "dense_learning_rate": float(args.dense_learning_rate),
        "backbone_learning_rate": float(args.backbone_learning_rate),
        "seed": seed,
        "smoke": bool(args.smoke),
    }
    identity["run_digest"] = _digest(identity)
    _atomic_json(identity, root / "run_identity.json")

    background_provider = None
    if args.background_cache:
        from saved_time_phase_operator_v4.background_field import BackgroundFieldProvider
        background_provider = BackgroundFieldProvider(str(args.background_cache))
        if not background_provider.covers(sample_ids):
            raise ValueError(
                "background cache does not cover the selected records: "
                f"need {sample_ids}, have {background_provider.sample_ids}"
            )

    full_targets: dict[str, np.ndarray] = {}
    if args.helmholtz_coefficient_supervision:
        records_by_id = {record.sample_id: record for record in manifest.records}
        with h5py.File(base.data.source_h5, "r", swmr=True) as handle:
            for sample_id in sample_ids:
                record = records_by_id[sample_id]
                value = np.asarray(handle["wavefield"][record.source_index])
                if value.shape != (len(manifest.time_s), 201, 201):
                    raise ValueError("full-trace coefficient target shape changed")
                if not np.isfinite(value).all():
                    raise FloatingPointError("non-finite full-trace coefficient target")
                full_targets[sample_id] = value

    baseline = _evaluate_triplet(
        model,
        base,
        manifest,
        normalizer,
        device,
        config,
        indices,
        split=args.split,
        time_policy="validation_fixed",
        frames_per_record=int(args.validation_frames),
        background_provider=background_provider,
    )
    _append_jsonl(root / "metrics.jsonl", {"event": "baseline", "update": 0, "metrics": baseline})
    baseline_checkpoint = save_overfit_checkpoint(
        root,
        model=model,
        update=0,
        manifest_digest=manifest.digest,
        config_digest=identity["run_digest"],
        aggregate_relative_l2=float(baseline["aggregate_relative_l2"]),
    )

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    best_score = float(baseline["aggregate_relative_l2"])
    best_selection_score = family_gate_score(baseline)
    best_checkpoint = str(baseline_checkpoint)
    last_update = 0
    for update, batch in enumerate(train_data, start=1):
        model.train()
        batch = _retarget_batch_to_scattering(batch, background_provider)
        if args.helmholtz_coefficient_supervision:
            components = _direct_frequency_coefficient_update(
                model,
                optimizer,
                batch,
                normalizer,
                device,
                full_targets=full_targets,
                frequency_count=int(args.helmholtz_frequencies),
                microbatch_records=int(args.microbatch_records),
                saved_time_s=torch.as_tensor(manifest.time_s),
                frequency_energy_floor_fraction=float(
                    args.coefficient_frequency_energy_floor_fraction
                ),
                family_gradient_weights=family_gradient_weights,
            )
        else:
            components = _train_update(
                model,
                optimizer,
                batch,
                normalizer,
                device,
                config,
                microbatch_records=int(args.microbatch_records),
            )
        gradient_prefixes = active_prefixes
        if args.helmholtz_zero_init_head and update == 1:
            # The exact-zero final affine map intentionally blocks upstream
            # gradients on the first backward.  After optimizer.step makes the
            # head nonzero, every live encoder prefix is required as usual.
            gradient_prefixes = ("local_field",)
        gradients = _gradient_report(model, gradient_prefixes)
        gradients.update(temporal_basis_gradient_norms(model))
        gradient_norm, clipping = clip_trainable_gradients(
            model,
            maximum_norm=float(config["optimizer"]["gradient_clip"]),
            mode=str(config["optimizer"]["gradient_clip_mode"]),
            prefix_limits=config["optimizer"]["gradient_clip_prefix_limits"],
            return_report=True,
        )
        optimizer.step()
        last_update = update
        _append_jsonl(
            root / "updates.jsonl",
            {
                "event": "optimizer_update",
                "update": update,
                "loss_components": components,
                "gradient_norm_before_clip": gradient_norm,
                "gradient_clipping": clipping,
                "gpu": _gpu_snapshot(),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        if update % evaluate_every and update != updates:
            continue
        metrics = _evaluate_triplet(
            model,
            base,
            manifest,
            normalizer,
            device,
            config,
            indices,
            split=args.split,
            time_policy="validation_fixed",
            frames_per_record=int(args.validation_frames),
            background_provider=background_provider,
        )
        checkpoint_path = save_overfit_checkpoint(
            root,
            model=model,
            update=update,
            manifest_digest=manifest.digest,
            config_digest=identity["run_digest"],
            aggregate_relative_l2=float(metrics["aggregate_relative_l2"]),
        )
        score = float(metrics["aggregate_relative_l2"])
        selection_score = family_gate_score(metrics)
        if selection_score <= best_selection_score:
            best_score = score
            best_selection_score = selection_score
            best_checkpoint = str(checkpoint_path)
        report = {
            "event": "evaluation",
            "update": update,
            "metrics": metrics,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "elapsed_seconds": time.monotonic() - started,
            "checkpoint": str(checkpoint_path),
        }
        _append_jsonl(root / "metrics.jsonl", report)
        print(json.dumps({"update": update, "agg": score, "family": metrics.get("family_relative_l2")}, sort_keys=True), flush=True)
        if metrics_meet_registered_target(
            metrics,
            aggregate_maximum=float(args.target_aggregate_relative_l2),
            family_maximum=float(args.target_family_relative_l2),
        ):
            break

    if args.smoke:
        terminal = {
            "status": "smoke_complete",
            "updates_completed": last_update,
            "best_fixed_aggregate_relative_l2": best_score,
            "parameter_count": parameter_count,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "front_end_transfer": transfer_report,
        }
        _atomic_json(terminal, terminal_path)
        print(json.dumps(terminal, sort_keys=True), flush=True)
        return 0

    restore_best_overfit_checkpoint(
        best_checkpoint,
        model=model,
        manifest_digest=manifest.digest,
        config_digest=identity["run_digest"],
        map_location=device,
    )
    full_metrics = _evaluate_triplet(
        model,
        base,
        manifest,
        normalizer,
        device,
        config,
        indices,
        split=args.split,
        time_policy="all_saved",
        frames_per_record=len(manifest.time_s),
        background_provider=background_provider,
    )
    _append_jsonl(
        root / "metrics.jsonl",
        {"event": "all_saved_evaluation", "checkpoint": best_checkpoint, "metrics": full_metrics},
    )
    terminal = build_overfit_terminal_report(
        updates_completed=last_update,
        baseline_relative_l2=float(baseline["aggregate_relative_l2"]),
        best_fixed_relative_l2=best_score,
        best_checkpoint=best_checkpoint,
        all_saved_metrics=full_metrics,
    )
    terminal["parameter_count"] = parameter_count
    terminal["peak_cuda_bytes"] = int(torch.cuda.max_memory_allocated())
    terminal["architecture"] = {
        "width": int(args.width),
        "dense_depth": int(args.dense_depth),
        "dense_spectral_rank": int(args.dense_spectral_rank),
        "dense_modes": int(args.dense_modes),
    }
    terminal["front_end_transfer"] = transfer_report
    terminal["registered_target"] = {
        "aggregate_relative_l2": float(args.target_aggregate_relative_l2),
        "family_relative_l2": float(args.target_family_relative_l2),
    }
    terminal["best_fixed_family_gate_score"] = best_selection_score
    terminal["can_memorize_below_target"] = metrics_meet_registered_target(
        full_metrics,
        aggregate_maximum=float(args.target_aggregate_relative_l2),
        family_maximum=float(args.target_family_relative_l2),
    )
    terminal["interpretation"] = (
        "capacity_is_sufficient"
        if terminal["can_memorize_below_target"]
        else "capacity_or_optimization_is_insufficient"
    )
    _atomic_json(terminal, terminal_path)
    print(json.dumps(terminal, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
