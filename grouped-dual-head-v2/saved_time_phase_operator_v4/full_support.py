"""Deterministic full-support schedules and their coverage audits."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn


DENSE_PREFIXES = ("dense_decoder", "local_field", "source_encoder", "fusion")
GEOMETRY_PREFIXES = ("coordinate_encoder", "travel_branch")
BACKBONE_PREFIXES = ("medium_encoder",)


@dataclass(frozen=True)
class FullSupportStepSpec:
    """One macro batch with per-record lifetime appearance counters."""

    step: int
    epoch: int
    record_indices: tuple[int, ...]
    appearance_indices: tuple[int, ...]


@dataclass(frozen=True)
class EpochScheduleAudit:
    epoch: int
    record_count: int
    appearances: int
    minimum_appearances: int
    maximum_appearances: int
    macro_count: int
    optimizer_updates: int


@dataclass(frozen=True)
class FamilyCurriculumEpochAudit(EpochScheduleAudit):
    family_appearances: dict[str, int]


@dataclass(frozen=True)
class TrainableStage:
    epoch: int
    trainable_prefixes: tuple[str, ...]
    trainable_parameters: int
    frozen_parameters: int


def _positive(name: str, value: int) -> int:
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def rad_record_weights(
    record_errors: Sequence[float],
    *,
    k: float = 1.0,
    c: float = 1.0,
    floor: float = 0.0,
) -> np.ndarray:
    """Residual-adaptive per-record sampling weights (normalized to sum 1).

    ``p_rec ∝ (e_rec / mean(e))^k + c`` — the same RAD density used on the frame
    axis, so high-error records (e.g. marmousi) are drawn more often.  ``c`` keeps
    every record's probability non-zero (guards against forgetting easy records /
    overfitting hard ones); ``floor`` optionally raises the minimum probability
    mass.  Deterministic given the inputs.
    """

    errors = np.asarray(record_errors, dtype=np.float64)
    if errors.ndim != 1 or errors.size == 0:
        raise ValueError("record_errors must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(errors)) or np.any(errors < 0.0):
        raise ValueError("record errors must be finite and nonnegative")
    mean = float(errors.mean())
    if mean <= 0.0:
        weights = np.ones_like(errors)
    else:
        weights = np.power(errors / mean, max(0.0, float(k))) + max(0.0, float(c))
    weights = weights / weights.sum()
    floor = float(floor)
    if floor > 0.0:
        if floor * errors.size >= 1.0:
            raise ValueError("record weight floor is too large for the record count")
        weights = (1.0 - floor * errors.size) * weights + floor
    return weights / weights.sum()


def build_full_support_schedule(
    record_count: int,
    *,
    epochs: int,
    macro_records: int,
    macros_per_update: int,
    seed: int,
    appearance_offset: int = 0,
    epoch_offset: int = 0,
    record_weights: np.ndarray | None = None,
    record_oversample: float = 1.0,
) -> tuple[FullSupportStepSpec, ...]:
    """Build epoch-distinct schedules padded only to complete optimizer updates.

    ``record_weights`` (a normalized length-``record_count`` vector) optionally
    replaces the uniform per-epoch permutation with residual-adaptive weighted
    sampling, so high-error records appear more often.  Because the base schedule
    is exactly one appearance per record per epoch (no padding slack), real
    reweighting requires ``record_oversample`` > 1.0: each epoch then leads with a
    full uniform permutation (coverage contract) followed by
    ``(record_oversample - 1) * record_count`` extra weighted draws, lengthening
    the epoch so hard records recur.  ``record_weights=None`` or
    ``record_oversample=1.0`` keeps the legacy uniform permutation bit-for-bit.
    """

    records = _positive("record_count", record_count)
    epoch_count = _positive("epochs", epochs)
    macro_size = _positive("macro_records", macro_records)
    accumulation = _positive("macros_per_update", macros_per_update)
    offset = int(appearance_offset)
    if offset < 0:
        raise ValueError("appearance offset must be nonnegative")
    inherited_epochs = int(epoch_offset)
    if inherited_epochs < 0:
        raise ValueError("epoch offset must be nonnegative")
    update_records = macro_size * accumulation
    oversample = float(record_oversample)
    if not math.isfinite(oversample) or oversample < 1.0:
        raise ValueError("record_oversample must be >= 1.0")
    target_records = int(round(records * oversample))
    padded_records = math.ceil(target_records / update_records) * update_records
    weights = None
    if record_weights is not None:
        weights = np.asarray(record_weights, dtype=np.float64)
        if weights.shape != (records,):
            raise ValueError("record_weights must have one entry per record")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or weights.sum() <= 0.0:
            raise ValueError("record_weights must be finite, nonnegative, and sum > 0")
        weights = weights / weights.sum()

    def _epoch_order(global_epoch: int) -> np.ndarray:
        rng = np.random.default_rng(int(seed) + global_epoch * 104729)
        if weights is None:
            return rng.permutation(records)
        # Coverage contract: lead with a full uniform permutation so every record
        # appears at least once, then let residual-adaptive weighted draws fill the
        # remaining (oversampled) padded slots so high-error records recur more.
        guaranteed = rng.permutation(records)
        if padded_records <= records:
            return guaranteed
        extra = rng.choice(records, size=padded_records - records, replace=True, p=weights)
        return np.concatenate([guaranteed, extra])

    appearances = np.full(records, offset, dtype=np.int64)
    schedule: list[FullSupportStepSpec] = []
    for inherited_epoch in range(inherited_epochs):
        inherited_order = _epoch_order(inherited_epoch)
        for value in np.resize(inherited_order, padded_records):
            appearances[int(value)] += 1
    step = inherited_epochs * (padded_records // macro_size)
    for epoch in range(epoch_count):
        global_epoch = inherited_epochs + epoch
        order = _epoch_order(global_epoch)
        padded = np.resize(order, padded_records)
        for start in range(0, padded_records, macro_size):
            selected = tuple(int(value) for value in padded[start : start + macro_size])
            current = tuple(int(appearances[value]) for value in selected)
            for value in selected:
                appearances[value] += 1
            schedule.append(
                FullSupportStepSpec(
                    step=step,
                    epoch=epoch,
                    record_indices=selected,
                    appearance_indices=current,
                )
            )
            step += 1
    return tuple(schedule)


def build_family_curriculum_schedule(
    record_families: Sequence[str],
    *,
    stages: Sequence[Mapping[str, object]],
    epochs: int,
    macro_records: int,
    macros_per_update: int,
    seed: int,
    appearance_offset: int = 0,
) -> tuple[FullSupportStepSpec, ...]:
    """Build fixed-cost easy-to-hard family macros with explicit replay patterns."""

    families = tuple(str(value) for value in record_families)
    if not families:
        raise ValueError("family curriculum requires training records")
    allowed = ("uniform", "layered", "marmousi")
    unknown_records = sorted(set(families) - set(allowed))
    if unknown_records:
        raise ValueError(f"family curriculum has unknown family: {unknown_records}")
    epoch_count = _positive("epochs", epochs)
    macro_size = _positive("macro_records", macro_records)
    accumulation = _positive("macros_per_update", macros_per_update)
    offset = int(appearance_offset)
    if offset < 0:
        raise ValueError("appearance offset must be nonnegative")
    stage_values = tuple(stages)
    if not stage_values:
        raise ValueError("family curriculum requires at least one stage")

    stage_by_epoch: list[tuple[str, ...]] = []
    for stage in stage_values:
        stage_epochs = _positive("family curriculum stage epochs", int(stage.get("epochs", 0)))
        pattern = tuple(str(value) for value in stage.get("macro_pattern", ()))
        if not pattern:
            raise ValueError("family curriculum macro pattern cannot be empty")
        unknown = sorted(set(pattern) - set(allowed))
        if unknown:
            raise ValueError(f"family curriculum has unknown family: {unknown}")
        stage_by_epoch.extend((pattern,) * stage_epochs)
    if len(stage_by_epoch) != epoch_count:
        raise ValueError("family curriculum stage epoch count does not match epochs")

    indices_by_family = {
        family: np.asarray(
            [index for index, value in enumerate(families) if value == family],
            dtype=np.int64,
        )
        for family in allowed
    }
    used = set(value for pattern in stage_by_epoch for value in pattern)
    missing = sorted(family for family in used if not len(indices_by_family[family]))
    if missing:
        raise ValueError(f"family curriculum has no records for family: {missing}")

    update_records = macro_size * accumulation
    padded_records = math.ceil(len(families) / update_records) * update_records
    macros_per_epoch = padded_records // macro_size
    appearances = np.full(len(families), offset, dtype=np.int64)
    schedule: list[FullSupportStepSpec] = []
    step = 0
    family_seed = {"uniform": 11, "layered": 23, "marmousi": 37}
    for epoch, pattern in enumerate(stage_by_epoch):
        macro_families = tuple(
            pattern[index % len(pattern)] for index in range(macros_per_epoch)
        )
        selected_by_family: dict[str, np.ndarray] = {}
        cursor_by_family: dict[str, int] = {}
        for family in set(macro_families):
            required = macro_families.count(family) * macro_size
            generator = np.random.default_rng(
                int(seed) + epoch * 104729 + family_seed[family]
            )
            order = generator.permutation(indices_by_family[family])
            selected_by_family[family] = np.resize(order, required)
            cursor_by_family[family] = 0
        for family in macro_families:
            start = cursor_by_family[family]
            stop = start + macro_size
            selected = tuple(
                int(value) for value in selected_by_family[family][start:stop]
            )
            cursor_by_family[family] = stop
            current = tuple(int(appearances[value]) for value in selected)
            for value in selected:
                appearances[value] += 1
            schedule.append(
                FullSupportStepSpec(
                    step=step,
                    epoch=epoch,
                    record_indices=selected,
                    appearance_indices=current,
                )
            )
            step += 1
    return tuple(schedule)


def audit_epoch_schedule(
    schedule: Sequence[FullSupportStepSpec],
    *,
    epoch: int,
    record_count: int,
    macros_per_update: int = 4,
    allow_oversample: bool = False,
) -> EpochScheduleAudit:
    """Reject incomplete, imbalanced, malformed, or update-misaligned epochs.

    ``allow_oversample`` relaxes the balanced-coverage check (max-min<=1): with
    residual-adaptive record oversampling the appearance counts are deliberately
    unequal (hard records recur more), so only full coverage (min>=1) is required.
    """

    records = _positive("record_count", record_count)
    accumulation = _positive("macros_per_update", macros_per_update)
    selected = tuple(spec for spec in schedule if int(spec.epoch) == int(epoch))
    if not selected:
        raise ValueError(f"schedule has no macros for epoch {epoch}")
    if len(selected) % accumulation:
        raise ValueError("epoch macro count does not align with optimizer updates")
    macro_size = len(selected[0].record_indices)
    if macro_size <= 0:
        raise ValueError("schedule macro cannot be empty")
    counts = np.zeros(records, dtype=np.int64)
    for spec in selected:
        if len(spec.record_indices) != macro_size:
            raise ValueError("epoch schedule has inconsistent macro sizes")
        if len(spec.appearance_indices) != macro_size:
            raise ValueError("appearance indices do not match macro records")
        for record_index in spec.record_indices:
            index = int(record_index)
            if not 0 <= index < records:
                raise ValueError(f"record index is outside the split: {index}")
            counts[index] += 1
    minimum = int(counts.min())
    maximum = int(counts.max())
    if minimum < 1:
        raise ValueError("epoch schedule does not cover every record")
    if not allow_oversample and maximum - minimum > 1:
        raise ValueError("epoch schedule padding is imbalanced")
    return EpochScheduleAudit(
        epoch=int(epoch),
        record_count=int(np.count_nonzero(counts)),
        appearances=int(counts.sum()),
        minimum_appearances=minimum,
        maximum_appearances=maximum,
        macro_count=len(selected),
        optimizer_updates=len(selected) // accumulation,
    )


def audit_family_curriculum_epoch_schedule(
    schedule: Sequence[FullSupportStepSpec],
    *,
    epoch: int,
    record_families: Sequence[str],
    macros_per_update: int = 4,
) -> FamilyCurriculumEpochAudit:
    """Audit update alignment, index validity, and homogeneous family macros."""

    families = tuple(str(value) for value in record_families)
    if not families:
        raise ValueError("family curriculum audit requires records")
    accumulation = _positive("macros_per_update", macros_per_update)
    selected = tuple(spec for spec in schedule if int(spec.epoch) == int(epoch))
    if not selected:
        raise ValueError(f"schedule has no macros for epoch {epoch}")
    if len(selected) % accumulation:
        raise ValueError("epoch macro count does not align with optimizer updates")
    macro_size = len(selected[0].record_indices)
    if macro_size <= 0:
        raise ValueError("schedule macro cannot be empty")
    counts = np.zeros(len(families), dtype=np.int64)
    family_counts: dict[str, int] = {}
    for spec in selected:
        if len(spec.record_indices) != macro_size:
            raise ValueError("epoch schedule has inconsistent macro sizes")
        if len(spec.appearance_indices) != macro_size:
            raise ValueError("appearance indices do not match macro records")
        macro_families: set[str] = set()
        for record_index in spec.record_indices:
            index = int(record_index)
            if not 0 <= index < len(families):
                raise ValueError(f"record index is outside the split: {index}")
            counts[index] += 1
            macro_families.add(families[index])
        if len(macro_families) != 1:
            raise ValueError("family curriculum macro must contain exactly one family")
        family = next(iter(macro_families))
        family_counts[family] = family_counts.get(family, 0) + macro_size
    active = counts[counts > 0]
    return FamilyCurriculumEpochAudit(
        epoch=int(epoch),
        record_count=int(np.count_nonzero(counts)),
        appearances=int(counts.sum()),
        minimum_appearances=int(active.min()),
        maximum_appearances=int(active.max()),
        macro_count=len(selected),
        optimizer_updates=len(selected) // accumulation,
        family_appearances=dict(sorted(family_counts.items())),
    )


def schedule_digest(schedule: Sequence[FullSupportStepSpec]) -> str:
    """Bind a complete schedule without storing it in the run identity."""

    payload = [asdict(spec) for spec in schedule]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf8")
    return hashlib.sha256(encoded).hexdigest()


def _parameter_prefix(name: str) -> str:
    return str(name).split(".", 1)[0]


def configure_trainable_stage(model: nn.Module, *, epoch: int) -> TrainableStage:
    """Apply the registered 1-2, 3-5, 6+ epoch unfreezing stages."""

    current_epoch = _positive("epoch", epoch)
    if current_epoch <= 2:
        active = DENSE_PREFIXES
    elif current_epoch <= 5:
        active = (*DENSE_PREFIXES, *GEOMETRY_PREFIXES)
    else:
        active = (*DENSE_PREFIXES, *GEOMETRY_PREFIXES, *BACKBONE_PREFIXES)
    known = set((*DENSE_PREFIXES, *GEOMETRY_PREFIXES, *BACKBONE_PREFIXES))
    trainable = 0
    frozen = 0
    for name, parameter in model.named_parameters():
        prefix = _parameter_prefix(name)
        if prefix not in known:
            raise ValueError(f"unregistered top-level parameter prefix: {prefix}")
        parameter.requires_grad_(prefix in active)
        if parameter.requires_grad:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()
    if trainable <= 0:
        raise ValueError("trainable stage selected no parameters")
    return TrainableStage(
        epoch=current_epoch,
        trainable_prefixes=tuple(active),
        trainable_parameters=int(trainable),
        frozen_parameters=int(frozen),
    )


def configure_pinned_stage(
    model: nn.Module,
    *,
    prefixes: Sequence[str],
) -> TrainableStage:
    """Hold a fixed trainable prefix set for every epoch (no progressive unfreeze).

    Used by staged-adapter warm-start runs (e.g. the zero-init temporal propagation
    operator): the parent must stay frozen so the adapter develops on a stable
    optimum, and freezing the geometry/backbone paths keeps peak activation memory at
    the proven decoder-only high-water mark instead of the geometry-unfrozen one that
    OOMs the 24 GiB card once the temporal operator's recompute is added.
    """

    active = tuple(str(prefix) for prefix in prefixes)
    known = set((*DENSE_PREFIXES, *GEOMETRY_PREFIXES, *BACKBONE_PREFIXES))
    if not active:
        raise ValueError("pinned stage needs at least one trainable prefix")
    unknown = set(active) - known
    if unknown:
        raise ValueError(f"pinned stage has unregistered prefixes: {sorted(unknown)}")
    trainable = 0
    frozen = 0
    for name, parameter in model.named_parameters():
        prefix = _parameter_prefix(name)
        if prefix not in known:
            raise ValueError(f"unregistered top-level parameter prefix: {prefix}")
        parameter.requires_grad_(prefix in active)
        if parameter.requires_grad:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()
    if trainable <= 0:
        raise ValueError("pinned stage selected no trainable parameters")
    return TrainableStage(
        epoch=0,
        trainable_prefixes=active,
        trainable_parameters=int(trainable),
        frozen_parameters=int(frozen),
    )


def configure_recovery_stage(
    model: nn.Module,
    *,
    epoch: int,
    decoder_only_epochs: int = 2,
) -> TrainableStage:
    """Warm a reset residual decoder before progressively unfreezing its parent."""

    current_epoch = _positive("epoch", epoch)
    decoder_epochs = _positive("decoder_only_epochs", decoder_only_epochs)
    if current_epoch <= decoder_epochs:
        active = ("dense_decoder",)
    elif current_epoch <= decoder_epochs + 2:
        active = DENSE_PREFIXES
    elif current_epoch <= decoder_epochs + 5:
        active = (*DENSE_PREFIXES, *GEOMETRY_PREFIXES)
    else:
        active = (*DENSE_PREFIXES, *GEOMETRY_PREFIXES, *BACKBONE_PREFIXES)
    known = set((*DENSE_PREFIXES, *GEOMETRY_PREFIXES, *BACKBONE_PREFIXES))
    trainable = 0
    frozen = 0
    for name, parameter in model.named_parameters():
        prefix = _parameter_prefix(name)
        if prefix not in known:
            raise ValueError(f"unregistered top-level parameter prefix: {prefix}")
        parameter.requires_grad_(prefix in active)
        if name == "dense_decoder.correction_scale":
            parameter.requires_grad_(False)
        if parameter.requires_grad:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()
    if trainable <= 0:
        raise ValueError("recovery stage selected no trainable parameters")
    return TrainableStage(
        epoch=current_epoch,
        trainable_prefixes=tuple(active),
        trainable_parameters=int(trainable),
        frozen_parameters=int(frozen),
    )


def configure_family_expert_stage(
    model: nn.Module,
    *,
    epoch: int,
    head_only_epochs: int = 1,
    dense_unfreeze_epoch: int | None = None,
    shared_unfreeze_epoch: int | None = None,
    geometry_unfreeze_epoch: int | None = None,
    backbone_unfreeze_epoch: int | None = None,
) -> TrainableStage:
    """Warm experts, then progressively unfreeze every conditioned operator path."""

    current_epoch = _positive("epoch", epoch)
    head_epochs = _positive("head_only_epochs", head_only_epochs)
    dense_epoch = (
        None
        if dense_unfreeze_epoch is None
        else _positive("dense_unfreeze_epoch", dense_unfreeze_epoch)
    )
    shared_epoch = (
        None
        if shared_unfreeze_epoch is None
        else _positive("shared_unfreeze_epoch", shared_unfreeze_epoch)
    )
    geometry_epoch = (
        None
        if geometry_unfreeze_epoch is None
        else _positive("geometry_unfreeze_epoch", geometry_unfreeze_epoch)
    )
    backbone_epoch = (
        None
        if backbone_unfreeze_epoch is None
        else _positive("backbone_unfreeze_epoch", backbone_unfreeze_epoch)
    )
    if dense_epoch is not None and dense_epoch <= head_epochs:
        raise ValueError("dense_unfreeze_epoch must follow the expert head warmup")
    registered = (dense_epoch, shared_epoch, geometry_epoch, backbone_epoch)
    last_registered = max(
        (index for index, value in enumerate(registered) if value is not None),
        default=-1,
    )
    present = tuple(registered[: last_registered + 1])
    if present and (
        any(value is None for value in present)
        or tuple(sorted(present)) != present
        or len(set(present)) != len(present)
        or present[0] <= head_epochs
    ):
        raise ValueError(
            "family expert unfreeze epochs must be fully registered and strictly ordered"
        )
    prefix_stage: tuple[str, ...] | None = None
    if backbone_epoch is not None and current_epoch >= backbone_epoch:
        prefix_stage = (*DENSE_PREFIXES, *GEOMETRY_PREFIXES, *BACKBONE_PREFIXES)
    elif geometry_epoch is not None and current_epoch >= geometry_epoch:
        prefix_stage = (*DENSE_PREFIXES, *GEOMETRY_PREFIXES)
    elif shared_epoch is not None and current_epoch >= shared_epoch:
        prefix_stage = DENSE_PREFIXES
    elif dense_epoch is not None and current_epoch >= dense_epoch:
        prefix_stage = ("dense_decoder",)
    trainable = 0
    frozen = 0
    for name, parameter in model.named_parameters():
        is_expert = name.startswith("dense_decoder.family_experts.")
        if current_epoch <= head_epochs:
            active = is_expert and (
                ".router." in name
                or ".output." in name
                or ".time." in name
            )
        elif prefix_stage is not None:
            active = _parameter_prefix(name) in prefix_stage
        else:
            active = is_expert or name.startswith("dense_decoder.temporal_basis.")
        parameter.requires_grad_(active)
        if name == "dense_decoder.correction_scale":
            parameter.requires_grad_(False)
        if parameter.requires_grad:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()
    if trainable <= 0:
        raise ValueError("family expert stage selected no trainable parameters")
    if current_epoch <= head_epochs:
        prefixes = ("dense_decoder.family_experts",)
    elif prefix_stage is not None:
        prefixes = prefix_stage
    else:
        prefixes = (
            "dense_decoder.family_experts",
            "dense_decoder.temporal_basis",
        )
    return TrainableStage(
        epoch=current_epoch,
        trainable_prefixes=prefixes,
        trainable_parameters=int(trainable),
        frozen_parameters=int(frozen),
    )


def adamw_backend_options(implementation: str) -> dict[str, bool]:
    """Return an explicit PyTorch AdamW execution backend."""

    backend = str(implementation)
    options = {
        "single_tensor": {"foreach": False},
        "foreach": {"foreach": True},
        "fused": {"fused": True},
    }
    if backend not in options:
        raise ValueError("AdamW implementation must be single_tensor, foreach, or fused")
    return options[backend]


def build_staged_adamw(
    model: nn.Module,
    *,
    dense_lr: float,
    geometry_lr: float,
    backbone_lr: float,
    weight_decay: float,
    temporal_basis_lr: float | None = None,
    expert_lr: float | None = None,
    temporal_operator_lr: float | None = None,
    warp_lr: float | None = None,
    green_kernel_lr: float | None = None,
    temporal_latent_lr: float | None = None,
    multi_arrival_lr: float | None = None,
    dispersive_modal_lr: float | None = None,
    windowed_propagation_lr: float | None = None,
    local_field_lr: float | None = None,
    implementation: str = "single_tensor",
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1.0e-8,
) -> torch.optim.AdamW:
    """Build mutually exclusive optimizer groups before staged unfreezing."""

    backend_options = adamw_backend_options(implementation)
    learning_rates = {
        "dense": float(dense_lr),
        "geometry": float(geometry_lr),
        "backbone": float(backbone_lr),
    }
    if temporal_basis_lr is not None:
        learning_rates["temporal"] = float(temporal_basis_lr)
    if expert_lr is not None:
        learning_rates["expert"] = float(expert_lr)
    if temporal_operator_lr is not None:
        learning_rates["temporal_operator"] = float(temporal_operator_lr)
    if warp_lr is not None:
        learning_rates["warp"] = float(warp_lr)
    if green_kernel_lr is not None:
        learning_rates["green_kernel"] = float(green_kernel_lr)
    if temporal_latent_lr is not None:
        learning_rates["temporal_latent"] = float(temporal_latent_lr)
    if multi_arrival_lr is not None:
        learning_rates["multi_arrival"] = float(multi_arrival_lr)
    if dispersive_modal_lr is not None:
        learning_rates["dispersive_modal"] = float(dispersive_modal_lr)
    if windowed_propagation_lr is not None:
        learning_rates["windowed_propagation"] = float(windowed_propagation_lr)
    if local_field_lr is not None:
        learning_rates["local_field"] = float(local_field_lr)
    if any(value <= 0.0 for value in learning_rates.values()) or float(weight_decay) < 0.0:
        raise ValueError("learning rates must be positive and weight decay nonnegative")
    beta1, beta2 = (float(value) for value in betas)
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("AdamW betas must be in [0, 1)")
    if float(eps) <= 0.0:
        raise ValueError("AdamW eps must be positive")
    prefix_to_group = {
        **{prefix: "dense" for prefix in DENSE_PREFIXES},
        **{prefix: "geometry" for prefix in GEOMETRY_PREFIXES},
        **{prefix: "backbone" for prefix in BACKBONE_PREFIXES},
    }
    grouped: dict[str, dict[str, list[nn.Parameter]]] = {
        name: {"decay": [], "no_decay": []} for name in learning_rates
    }
    for name, parameter in model.named_parameters():
        prefix = _parameter_prefix(name)
        if prefix not in prefix_to_group:
            raise ValueError(f"unregistered optimizer parameter prefix: {prefix}")
        decay_class = (
            "no_decay"
            if parameter.ndim <= 1 or name.endswith("bias") or "embedding" in name
            else "decay"
        )
        group_name = prefix_to_group[prefix]
        if expert_lr is not None and name.startswith(
            "dense_decoder.family_experts."
        ):
            group_name = "expert"
        elif temporal_basis_lr is not None and name.startswith(
            "dense_decoder.temporal_basis."
        ):
            group_name = "temporal"
        elif temporal_operator_lr is not None and name.startswith(
            "local_field.temporal_operator."
        ):
            group_name = "temporal_operator"
        elif warp_lr is not None and name.startswith("local_field.warp."):
            group_name = "warp"
        elif green_kernel_lr is not None and name.startswith("local_field.green_kernel."):
            group_name = "green_kernel"
        elif temporal_latent_lr is not None and name.startswith("local_field.temporal_latent."):
            group_name = "temporal_latent"
        elif multi_arrival_lr is not None and name.startswith("local_field.multi_arrival."):
            group_name = "multi_arrival"
        elif dispersive_modal_lr is not None and name.startswith("local_field.dispersive_modal."):
            group_name = "dispersive_modal"
        elif windowed_propagation_lr is not None and name.startswith("local_field.windowed_propagation."):
            group_name = "windowed_propagation"
        elif local_field_lr is not None and name.startswith("local_field."):
            group_name = "local_field"
        grouped[group_name][decay_class].append(parameter)
    if any(not any(parts.values()) for parts in grouped.values()):
        missing = [name for name, parts in grouped.items() if not any(parts.values())]
        raise ValueError(f"optimizer parameter groups are empty: {missing}")
    parameter_groups = []
    group_order = tuple(
        name
        for name in ("dense", "local_field", "temporal", "temporal_operator", "warp", "green_kernel", "temporal_latent", "multi_arrival", "dispersive_modal", "windowed_propagation", "expert", "geometry", "backbone")
        if name in learning_rates
    )
    for name in group_order:
        for decay_class in ("decay", "no_decay"):
            parameters = grouped[name][decay_class]
            if not parameters:
                continue
            parameter_groups.append(
                {
                    "params": parameters,
                    "lr": learning_rates[name],
                    "initial_lr": learning_rates[name],
                    "group_name": f"{name}_{decay_class}",
                    "weight_decay": (
                        float(weight_decay) if decay_class == "decay" else 0.0
                    ),
                }
            )
    return torch.optim.AdamW(
        parameter_groups,
        weight_decay=0.0,
        betas=(beta1, beta2),
        eps=float(eps),
        **backend_options,
    )


def warmup_cosine_factor(
    epoch_index: int,
    *,
    total_epochs: int,
    warmup_epochs: int,
    minimum_factor: float,
) -> float:
    """Return an epoch-level factor that reaches its exact floor at the last epoch."""

    total = _positive("total_epochs", total_epochs)
    warmup = _positive("warmup_epochs", warmup_epochs)
    index = int(epoch_index)
    floor = float(minimum_factor)
    if warmup >= total or not 0.0 < floor <= 1.0:
        raise ValueError("warmup/floor configuration is invalid")
    if not 0 <= index < total:
        raise ValueError("epoch index is outside the registered run")
    if index < warmup:
        return float(index + 1) / float(warmup)
    denominator = total - warmup - 1
    if denominator <= 0:
        return floor
    progress = float(index - warmup) / float(denominator)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (1.0 - floor) * cosine


__all__ = [
    "adamw_backend_options",
    "EpochScheduleAudit",
    "FamilyCurriculumEpochAudit",
    "FullSupportStepSpec",
    "TrainableStage",
    "audit_epoch_schedule",
    "audit_family_curriculum_epoch_schedule",
    "build_staged_adamw",
    "build_family_curriculum_schedule",
    "build_full_support_schedule",
    "configure_trainable_stage",
    "configure_pinned_stage",
    "configure_recovery_stage",
    "configure_family_expert_stage",
    "schedule_digest",
    "warmup_cosine_factor",
]
