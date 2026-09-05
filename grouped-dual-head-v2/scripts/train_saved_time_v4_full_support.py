#!/usr/bin/env python
"""Resumable full-support training for the stored-time V4 operator."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Mapping, Sequence

import h5py
import numpy as np
import torch
import yaml
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts
from grouped_ufno_mionet_v3.data.pilot import make_pilot_loader
from grouped_ufno_mionet_v3.training.checkpoint import load_checkpoint, save_checkpoint_atomic
from saved_time_phase_operator_v4.data import (
    ExactStoredTimeBatchDataset,
    merge_pilot_batches,
    split_pilot_batch,
)
from saved_time_phase_operator_v4.band_adapter import (
    activate_band_adapter_output,
    build_band_adapter_adamw,
    configure_band_adapter_stage,
    registered_band_adapter_learning_rates,
)
from saved_time_phase_operator_v4.evaluation import time_axis_sha256
from saved_time_phase_operator_v4.full_support import (
    FullSupportStepSpec,
    audit_epoch_schedule,
    audit_family_curriculum_epoch_schedule,
    build_family_curriculum_schedule,
    build_full_support_schedule,
    rad_record_weights,
    build_staged_adamw,
    configure_family_expert_stage,
    configure_pinned_stage,
    configure_recovery_stage,
    configure_trainable_stage,
    schedule_digest,
    warmup_cosine_factor,
)
from saved_time_phase_operator_v4.family_gradients import (
    homogeneous_family_scale,
)
from saved_time_phase_operator_v4.gradient_control import (
    clip_gradients_by_prefix,
)
from saved_time_phase_operator_v4.losses import (
    apply_hard_causality,
    family_route_targets,
    family_router_loss,
    relative_energy_squared_block_loss,
    relative_energy_squared_reference,
    residual_recovery_loss,
    source_causality_onset_s,
    zero_initial_condition_loss,
)
from saved_time_phase_operator_v4.multifidelity import (
    NumericalTeacherCache,
    multifidelity_distillation_block_loss,
    multifidelity_energy_reference,
    numerical_teacher_pool_indices,
)
from saved_time_phase_operator_v4.muon import build_staged_muon_adamw
from saved_time_phase_operator_v4.probe import ProbeVariant
from saved_time_phase_operator_v4.residual_activation import (
    activate_residual_head,
    residual_activation_config,
)
from saved_time_phase_operator_v4.sampling import (
    appearance_time_indices,
    rad_bin_budget,
    validation_time_indices,
)
from saved_time_phase_operator_v4.spectral import transfer_expanded_spectral_modes
from saved_time_phase_operator_v4.streaming_metrics import ExactWavefieldMetricAccumulator
from scripts.refine_saved_time_v4_lbfgs import _append_jsonl, _gpu_snapshot
from scripts.train_grouped_v3_pilot import _to_device, load_normalizer
from scripts.train_saved_time_v4_probe import (
    _atomic_hardlink,
    _atomic_json,
    _digest,
    _loss,
    _model,
)


def epoch_step_ranges(*, total_macros: int, macros_per_epoch: int) -> tuple[tuple[int, int], ...]:
    total = int(total_macros)
    per_epoch = int(macros_per_epoch)
    if total <= 0 or per_epoch <= 0 or total % per_epoch:
        raise ValueError("total macros must divide into complete epochs")
    return tuple((start, start + per_epoch) for start in range(0, total, per_epoch))


def run_epochs_for_mode(
    *,
    registered_epochs: int,
    pilot_epochs: int,
    pilot: bool,
    smoke_updates: int,
) -> int:
    """Resolve the schedule horizon used by full, pilot, and smoke modes."""

    if bool(pilot) or int(smoke_updates) > 0:
        return int(pilot_epochs)
    return int(registered_epochs)


def checkpoint_is_eligible(
    config: Mapping[str, object],
    *,
    metrics: Mapping[str, object],
    validation_scope: str,
    pilot_or_smoke: bool,
) -> bool:
    """Select checkpoints without treating a jointly trained coarse path as fixed."""

    eligible = bool(pilot_or_smoke) or validation_scope == "fixed_full_time_panel"
    if not eligible:
        return False
    if config.get("residual_recovery") and not config.get("family_experts"):
        try:
            improvement = float(metrics["relative_improvement_vs_coarse"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("residual checkpoint lacks coarse-improvement evidence") from error
        return math.isfinite(improvement) and improvement > 0.0
    return True


def prune_epoch_checkpoints(
    checkpoints_dir,
    *,
    keep_last: int | None,
    best_epoch: int | None,
) -> list[int]:
    """Delete old ``epoch_NNNN.pt`` snapshots, keeping the most recent ``keep_last`` plus
    the current best epoch. Returns the epoch numbers removed (sorted).

    Disk-hoarding guard: without this, a 40-epoch run keeps 40 x ~0.32 GiB of per-epoch
    checkpoints. ``best.pt``/``latest.pt`` are hardlinks INTO this directory, so removing a
    named ``epoch_NNNN.pt`` never destroys the best/latest data (the inode survives while any
    hardlink remains); the current best epoch's named file is preserved anyway for
    discoverability. ``keep_last=None`` (or <=0) disables pruning -> byte-compatible with the
    prior always-keep-all behavior."""

    if keep_last is None or int(keep_last) <= 0:
        return []
    keep_last = int(keep_last)
    from pathlib import Path

    directory = Path(checkpoints_dir)
    epochs: list[int] = []
    for path in directory.glob("epoch_*.pt"):
        stem = path.stem.split("_", 1)[-1]
        try:
            epochs.append(int(stem))
        except ValueError:
            continue  # ignore non-epoch files
    if not epochs:
        return []
    epochs.sort()
    keep = set(epochs[-keep_last:])
    if best_epoch is not None:
        keep.add(int(best_epoch))
    removed: list[int] = []
    for epoch in epochs:
        if epoch in keep:
            continue
        target = directory / f"epoch_{epoch:04d}.pt"
        try:
            target.unlink()
            removed.append(epoch)
        except FileNotFoundError:
            pass
    return sorted(removed)


def family_expert_stage_epoch(
    config: Mapping[str, object], *, epoch: int
) -> int:
    """Resolve an explicit stage-only offset used by full-unfreeze memory probes."""

    current = int(epoch)
    experts = config.get("family_experts", {})
    if current <= 0 or not isinstance(experts, Mapping):
        raise ValueError("family expert stage configuration is invalid")
    raw_offset = experts.get("stage_epoch_offset", 0)
    if isinstance(raw_offset, bool):
        raise ValueError("family expert stage epoch offset is invalid")
    offset = int(raw_offset)
    if offset < 0:
        raise ValueError("family expert stage epoch offset is invalid")
    return current + offset


def distributed_context() -> dict[str, int | bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if enabled and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return {
        "enabled": enabled,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_main": rank == 0,
    }


def distributed_barrier(ctx: Mapping[str, int | bool]) -> None:
    if bool(ctx["enabled"]):
        dist.barrier()


def distributed_cleanup(ctx: Mapping[str, int | bool]) -> None:
    if bool(ctx["enabled"]) and dist.is_initialized():
        dist.destroy_process_group()


def distributed_max_int(
    value: int, *, device: torch.device, ctx: Mapping[str, int | bool]
) -> int:
    if not bool(ctx["enabled"]):
        return int(value)
    tensor = torch.tensor([int(value)], dtype=torch.long, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def distributed_broadcast_float(
    value: float, *, device: torch.device, ctx: Mapping[str, int | bool]
) -> float:
    """Broadcast one rank-zero validation scalar to every training rank."""

    if not bool(ctx["enabled"]):
        return float(value)
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    dist.broadcast(tensor, src=0)
    return float(tensor.item())


def distributed_average_gradients(model: torch.nn.Module, ctx: Mapping[str, int | bool]) -> None:
    if not bool(ctx["enabled"]):
        return
    scale = 1.0 / float(ctx["world_size"])
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.mul_(scale)


def ddp_update_specs(
    specs: Sequence[FullSupportStepSpec],
    *,
    macros_per_update: int,
    rank: int,
    world_size: int,
) -> tuple[FullSupportStepSpec, ...]:
    accumulation = int(macros_per_update)
    size = int(world_size)
    if accumulation <= 0 or size <= 0 or accumulation % size:
        raise ValueError("macros_per_update must be divisible by the DDP world size")
    selected: list[FullSupportStepSpec] = []
    for start in range(0, len(specs), accumulation):
        chunk = specs[start : start + accumulation]
        if len(chunk) != accumulation:
            raise ValueError("epoch schedule ended with an incomplete update")
        selected.extend(chunk[int(rank) :: size])
    return tuple(selected)


def build_update_report(
    *,
    epoch: int,
    update_index: int,
    updates_per_epoch: int,
    global_step: int,
    loss_components: Mapping[str, float],
    gradient_norm: float,
    gradient_norms: Mapping[str, float],
    learning_rates: Mapping[str, float],
    physical_microbatch_records: int,
    gpu: Mapping[str, float | None],
    elapsed_seconds: float,
    gradient_clipping: Mapping[str, object] | None = None,
    attempt: int = 1,
) -> dict[str, object]:
    """Build one durable optimizer-update metric record."""

    current = int(update_index)
    total = int(updates_per_epoch)
    if (
        int(epoch) <= 0
        or int(attempt) <= 0
        or current <= 0
        or total <= 0
        or current > total
    ):
        raise ValueError("optimizer update position is invalid")
    report: dict[str, object] = {
        "event": "optimizer_update",
        "epoch": int(epoch),
        "attempt": int(attempt),
        "update": current,
        "updates_per_epoch": total,
        "global_step": int(global_step),
        "loss_components": {
            str(name): float(value) for name, value in loss_components.items()
        },
        "gradient_norm_before_clip": float(gradient_norm),
        "gradient_norms": {
            str(name): float(value) for name, value in gradient_norms.items()
        },
        "learning_rates": {
            str(name): float(value) for name, value in learning_rates.items()
        },
        "physical_microbatch_records": int(physical_microbatch_records),
        "gpu": dict(gpu),
        "elapsed_seconds": float(elapsed_seconds),
    }
    if gradient_clipping:
        report["gradient_clipping"] = dict(gradient_clipping)
    return report


def validation_scope(
    *,
    epoch: int,
    validation_records: int,
    panel_records: int = 48,
    all_records_every: int = 5,
) -> tuple[str, int]:
    current = int(epoch)
    records = int(validation_records)
    panel = int(panel_records)
    cadence = int(all_records_every)
    if current <= 0 or records <= 0 or panel <= 0 or panel > records or cadence <= 0:
        raise ValueError("validation scope configuration is invalid")
    return ("all_records", records) if current % cadence == 0 else ("panel", panel)


def validation_panel_indices(
    *, validation_records: int, panel_records: int, epoch: int, seed: int
) -> tuple[int, ...]:
    records = int(validation_records)
    panel = int(panel_records)
    current = int(epoch)
    if records <= 0 or panel <= 0 or panel > records or current <= 0:
        raise ValueError("validation panel configuration is invalid")
    order = np.random.default_rng(int(seed)).permutation(records)
    start = ((current - 1) * panel) % records
    return tuple(int(order[(start + offset) % records]) for offset in range(panel))


def validation_plan(
    config: Mapping[str, object],
    *,
    epoch: int,
    validation_records: int,
    stored_time_count: int,
) -> tuple[str, tuple[int, ...], str, int]:
    """Return a comparable fixed-record validation plan for one epoch."""

    validation = dict(config["validation"])
    panel_records = int(validation["panel_records"])
    full_panel_records = int(validation.get("full_panel_records", panel_records))
    cadence = int(validation["all_records_every"])
    current = int(epoch)
    if full_panel_records != panel_records:
        raise ValueError("fixed full-time validation must use the same record panel")
    indices = validation_panel_indices(
        validation_records=int(validation_records),
        panel_records=panel_records,
        epoch=1,
        seed=int(config["seed"]),
    )
    if current % cadence == 0:
        frames = int(validation["final_frames_per_record"])
        if frames != int(stored_time_count):
            raise ValueError("full-time validation must cover every stored frame")
        return "fixed_full_time_panel", indices, "all_saved", frames
    frames = int(validation["frames_per_record"])
    return "fixed_panel", indices, "validation_fixed", frames


def epoch_gate_validation_plan(
    config: Mapping[str, object],
    *,
    validation_records: int,
) -> tuple[str, tuple[int, ...], str, int]:
    """Return the identical record/time protocol used by every epoch gate."""

    validation = dict(config["validation"])
    panel_records = int(validation["panel_records"])
    frames = int(validation["frames_per_record"])
    if panel_records <= 0 or panel_records > int(validation_records) or frames <= 0:
        raise ValueError("epoch validation gate protocol is invalid")
    indices = validation_panel_indices(
        validation_records=int(validation_records),
        panel_records=panel_records,
        epoch=1,
        seed=int(config["seed"]),
    )
    return "fixed_epoch_gate", indices, "validation_fixed", frames


def resolve_epoch_validation_control(
    config: Mapping[str, object], *, pilot_or_smoke: bool
) -> dict[str, object]:
    """Validate the fail-closed full-run epoch controller configuration."""

    raw = config.get("epoch_validation_control") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("epoch_validation_control must be a mapping")
    enabled = bool(raw.get("enabled", False)) and not bool(pilot_or_smoke)
    evaluation_split = str(raw.get("evaluation_split", "validation"))
    if evaluation_split not in {"train", "validation"}:
        raise ValueError(
            "epoch validation control evaluation_split must be train or validation"
        )
    metric = str(raw.get("metric", "aggregate_relative_l2"))
    if metric != "aggregate_relative_l2":
        raise ValueError("epoch validation control metric must be aggregate_relative_l2")
    minimum_improvement = float(raw.get("minimum_absolute_improvement", 0.0))
    backoff = float(raw.get("learning_rate_backoff", 0.5))
    minimum_multiplier = float(raw.get("minimum_learning_rate_multiplier", 1.0 / 256.0))
    maximum_attempts = int(raw.get("maximum_attempts_per_epoch", 8))
    time_selector_seed_offset = raw.get("time_selector_seed_offset", 7919)
    if isinstance(time_selector_seed_offset, bool) or not isinstance(
        time_selector_seed_offset, int
    ):
        raise ValueError("epoch validation time selector seed offset must be an integer")
    time_selector_seed_offset = int(time_selector_seed_offset)
    if time_selector_seed_offset < 0:
        raise ValueError(
            "epoch validation time selector seed offset must be nonnegative"
        )
    if minimum_improvement < 0.0 or not math.isfinite(minimum_improvement):
        raise ValueError("minimum epoch validation improvement must be finite and nonnegative")
    if not 0.0 < backoff < 1.0:
        raise ValueError("epoch validation learning-rate backoff must be between zero and one")
    if not 0.0 < minimum_multiplier <= 1.0:
        raise ValueError("minimum epoch validation learning-rate multiplier is invalid")
    if maximum_attempts <= 0:
        raise ValueError("maximum epoch validation attempts must be positive")
    if (
        enabled
        and evaluation_split == "train"
        and str(config.get("time_policy", "")) == "fixed_train_gate"
        and time_selector_seed_offset != 0
    ):
        raise ValueError(
            "fixed_train_gate requires epoch_validation_control."
            "time_selector_seed_offset=0 so training and gate frames align"
        )
    return {
        "enabled": enabled,
        "evaluation_split": evaluation_split,
        "metric": metric,
        "minimum_absolute_improvement": minimum_improvement,
        "learning_rate_backoff": backoff,
        "minimum_learning_rate_multiplier": minimum_multiplier,
        "maximum_attempts_per_epoch": maximum_attempts,
        "time_selector_seed_offset": time_selector_seed_offset,
    }


def evaluation_time_selector_seed(config: Mapping[str, object]) -> int:
    """Return the content-bound seed used for exact-time metric selection."""

    raw = config.get("epoch_validation_control") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("epoch_validation_control must be a mapping")
    offset = raw.get("time_selector_seed_offset", 7919)
    if isinstance(offset, bool) or not isinstance(offset, int) or int(offset) < 0:
        raise ValueError(
            "epoch validation time selector seed offset must be a nonnegative integer"
        )
    return int(config["seed"]) + int(offset)


def epoch_metric_evaluation_split(
    config: Mapping[str, object],
    epoch_control: Mapping[str, object],
    *,
    smoke: bool,
) -> str:
    """Keep an explicitly train-bound controller train-only during smoke tests."""

    if bool(epoch_control.get("enabled", False)):
        return str(epoch_control["evaluation_split"])
    raw = config.get("epoch_validation_control") or {}
    if (
        bool(smoke)
        and isinstance(raw, Mapping)
        and bool(raw.get("enabled", False))
    ):
        split = str(raw.get("evaluation_split", "validation"))
        if split not in {"train", "validation"}:
            raise ValueError("smoke evaluation split must be train or validation")
        return split
    return "validation"


def validation_score_improved(
    previous: float, current: float, *, minimum_absolute_improvement: float = 0.0
) -> bool:
    """Require a finite, strict same-protocol aggregate-error decrease."""

    old = float(previous)
    new = float(current)
    minimum = float(minimum_absolute_improvement)
    return (
        math.isfinite(old)
        and math.isfinite(new)
        and math.isfinite(minimum)
        and minimum >= 0.0
        and new < old - minimum
    )


def backed_off_learning_rate_multiplier(
    current: float, *, backoff: float, minimum: float
) -> float:
    """Return the next bounded multiplier after a rejected epoch."""

    value = float(current)
    factor = float(backoff)
    floor = float(minimum)
    if not 0.0 < value <= 1.0 or not 0.0 < factor < 1.0 or not 0.0 < floor <= 1.0:
        raise ValueError("epoch validation learning-rate backoff is invalid")
    return max(floor, value * factor)


def epoch_retry_exhausted(
    *, attempt: int, maximum_attempts: int, current_multiplier: float,
    next_multiplier: float
) -> bool:
    """Stop when the attempt budget or the distinct learning-rate ladder ends."""

    current_attempt = int(attempt)
    maximum = int(maximum_attempts)
    current = float(current_multiplier)
    following = float(next_multiplier)
    if current_attempt <= 0 or maximum <= 0:
        raise ValueError("epoch retry attempts must be positive")
    if not 0.0 < following <= current <= 1.0:
        raise ValueError("epoch retry learning-rate multipliers are invalid")
    return current_attempt >= maximum or following >= current


def recovery_stage_epoch(config: Mapping[str, object], *, epoch: int) -> int:
    """Map a local branch epoch onto its inherited recovery-stage epoch."""

    current = int(epoch)
    if current <= 0:
        raise ValueError("recovery stage epoch must be positive")
    recovery = config.get("residual_recovery")
    if not recovery:
        return current
    offset = int(recovery.get("stage_epoch_offset", 0))
    if offset < 0:
        raise ValueError("recovery stage epoch offset must be nonnegative")
    return current + offset


def microbatch_records_for_epoch(config: Mapping[str, object], *, epoch: int) -> int:
    """Reduce physical batches as geometry and medium backpropagation unfreeze."""

    expert_config = config.get("family_experts")
    current = (
        family_expert_stage_epoch(config, epoch=epoch)
        if expert_config
        else recovery_stage_epoch(config, epoch=epoch)
    )
    maximum = int(config["microbatch_records"])
    effective = int(config["macro_records"]) * int(config["macros_per_update"])
    if current <= 0 or maximum <= 0 or effective <= 0:
        raise ValueError("stage microbatch configuration must be positive")
    if expert_config:
        dense_epoch = expert_config.get("dense_unfreeze_epoch")
        shared_epoch = expert_config.get("shared_unfreeze_epoch")
        geometry_epoch = expert_config.get("geometry_unfreeze_epoch")
        dense_boundary = dense_epoch if dense_epoch is not None else shared_epoch
        if dense_boundary is None or current < int(dense_boundary):
            target = maximum
        elif geometry_epoch is None or current < int(geometry_epoch):
            # Rank-96 dense/source/fusion backpropagation reached the 24 GiB
            # boundary at four records in the registered recovery probes.
            target = min(maximum, 3)
        else:
            # Geometry and velocity-encoder stages retain the proven
            # two-record physical batch while accumulation preserves the
            # registered effective batch.
            target = min(maximum, 2)
    elif config.get("residual_recovery"):
        decoder_epochs = int(config["residual_recovery"].get("decoder_only_epochs", 2))
        if current <= decoder_epochs:
            target = maximum
        elif current <= 3:
            # Rank-96 temporal decoding consumed 23.41 GiB at the first
            # source/fusion unfreeze and failed the next 140 MiB allocation.
            target = min(maximum, 3)
        elif current == 4:
            # Four records exhausted the production 24 GiB card at this
            # inherited unfreeze boundary.  Three records retain 100% GPU
            # utilization while leaving safe allocator headroom.
            target = min(maximum, 3)
        elif current <= 7:
            # Production evidence supersedes the earlier estimate: the first
            # geometry-unfrozen update at three records reached 23.43 GiB and
            # failed its next 104 MiB FFT allocation. Two records preserve the
            # effective batch through accumulation and leave allocator headroom.
            target = min(maximum, 2)
        else:
            # Stage eight additionally backpropagates through the velocity
            # encoder, so retain the independently validated conservative size.
            target = min(maximum, 2)
    else:
        target = maximum if current <= 2 else min(maximum, 8 if current <= 5 else 4)
    return target


def training_dense_time_block(config: Mapping[str, object]) -> int:
    """Return the number of exact saved-time frames rendered per training block."""

    optimizer = config.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise ValueError("optimizer must be a mapping")
    block = int(optimizer.get("training_time_block", 1))
    if block <= 0:
        raise ValueError("optimizer.training_time_block must be positive")
    return block


def velocity_fields_per_record(
    velocity_mps: torch.Tensor, record_to_medium: torch.Tensor
) -> torch.Tensor:
    """Expand deduplicated medium fields to the source-record axis."""

    velocity = torch.as_tensor(velocity_mps)
    mapping = torch.as_tensor(
        record_to_medium, dtype=torch.long, device=velocity.device
    ).flatten()
    if velocity.ndim == 4 and velocity.shape[1] == 1:
        velocity = velocity[:, 0]
    if velocity.ndim != 3 or mapping.ndim != 1 or mapping.numel() == 0:
        raise ValueError(
            "velocity/record mapping must be [medium,1,z,x] or [medium,z,x] and [record]"
        )
    if bool((mapping < 0).any()) or bool((mapping >= velocity.shape[0]).any()):
        raise ValueError("record-to-medium mapping is outside the velocity axis")
    return velocity.index_select(0, mapping)


def training_loss_time_slices(
    config: Mapping[str, object], *, frame_count: int
) -> tuple[tuple[int, int, float], ...]:
    """Partition exact frames into bounded-graph loss/backward blocks.

    The returned weights sum to one, so every selected exact frame keeps the
    same nominal contribution even when the final block is shorter.  Omitting
    the option preserves the legacy single-backward behavior.
    """

    frames = int(frame_count)
    if frames <= 0:
        raise ValueError("training loss frame count must be positive")
    optimizer = config.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise ValueError("optimizer must be a mapping")
    raw_block = optimizer.get("training_loss_time_block", frames)
    if isinstance(raw_block, bool):
        raise ValueError("optimizer.training_loss_time_block must be positive")
    block = int(raw_block)
    if block <= 0:
        raise ValueError("optimizer.training_loss_time_block must be positive")
    return tuple(
        (start, min(start + block, frames), (min(start + block, frames) - start) / frames)
        for start in range(0, frames, block)
    )


def training_frames_per_record(config: Mapping[str, object]) -> int | None:
    """Return an optional exact stored-frame count for each training record."""

    value = config.get("training_frames_per_record")
    if value is None:
        return None
    frames = int(value)
    if frames <= 0:
        raise ValueError("training_frames_per_record must be positive")
    return frames


def numerical_teacher_time_pool(
    config: Mapping[str, object],
) -> tuple[int, ...] | None:
    """Return the registered exact-time pool for solver-guided pretraining."""

    raw = config.get("numerical_teacher")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("numerical_teacher must be a mapping")
    values = tuple(int(value) for value in raw.get("time_indices", ()))
    if (
        len(values) < 2
        or len(set(values)) != len(values)
        or values[0] != 0
        or any(left >= right for left, right in zip(values, values[1:]))
    ):
        raise ValueError("numerical teacher time indices must be a strict pool from zero")
    weights = tuple(
        float(raw.get(name, default))
        for name, default in (
            ("low_fidelity_weight", 0.5),
            ("residual_weight", 0.5),
            ("residual_energy_floor_fraction", 0.1),
        )
    )
    if (
        any(not math.isfinite(value) or value < 0.0 for value in weights[:2])
        or not 0.0 < weights[2] <= 1.0
        or not str(raw.get("cache_h5", ""))
    ):
        raise ValueError("numerical teacher cache or loss weights are invalid")
    return values


def coverage_frames_per_appearance(config: Mapping[str, object]) -> int:
    """Return the exact frame count materialized by an appearance policy."""

    policy = str(config.get("time_policy", "appearance4"))
    registered = training_frames_per_record(config)
    if policy == "numerical_teacher_pool":
        if numerical_teacher_time_pool(config) is None:
            raise ValueError("numerical teacher policy requires a registered cache")
        return 16 if registered is None else registered
    if policy == "appearance16":
        return 16 if registered is None else registered
    if policy == "appearance4":
        return 4 if registered is None else registered
    if policy == "fixed_train_gate":
        return 16 if registered is None else registered
    # Preserve the legacy four-frame coverage behavior for non-appearance
    # policies; those policies do not use an appearance counter in the loader.
    return 4


def pilot_gate(
    reports: Sequence[Mapping[str, object]], *, family_tolerance: float
) -> dict[str, object]:
    values = tuple(reports)
    tolerance = float(family_tolerance)
    if len(values) < 2 or tolerance < 0.0:
        raise ValueError("pilot gate requires at least two reports and nonnegative tolerance")
    scores = tuple(float(report["score"]) for report in values)
    first_family = {str(k): float(v) for k, v in dict(values[0]["family"]).items()}
    final_family = {str(k): float(v) for k, v in dict(values[-1]["family"]).items()}
    if set(first_family) != set(final_family):
        raise ValueError("pilot family metric keys changed")
    family_safe = all(
        final_family[name] <= first_family[name] * (1.0 + tolerance)
        for name in first_family
    )
    improved = min(scores) <= scores[0] and scores[-1] <= scores[0]
    recovery_report = "improvement_vs_coarse" in values[-1]
    better_than_coarse = (
        float(values[-1]["improvement_vs_coarse"]) > 0.0 if recovery_report else True
    )
    correction_bounded = (
        0.01 <= float(values[-1]["correction_ratio"]) <= 0.30
        if recovery_report
        else True
    )
    return {
        "passed": bool(
            improved and family_safe and better_than_coarse and correction_bounded
        ),
        "best_nonincreasing": bool(improved),
        "family_safe": bool(family_safe),
        "better_than_coarse": bool(better_than_coarse),
        "correction_bounded": bool(correction_bounded),
        "initial_score": scores[0],
        "final_score": scores[-1],
        "best_score": min(scores),
    }


def pilot_gate_on_main(
    reports: Sequence[Mapping[str, object]],
    *,
    is_main: bool,
    family_tolerance: float,
) -> dict[str, object] | None:
    """Evaluate validation reports only on the rank that collected them."""

    if not bool(is_main):
        return None
    return pilot_gate(reports, family_tolerance=family_tolerance)


def pilot_terminal_exit_code(
    status: str, *, external_evidence_gate: bool
) -> int:
    """Let a registered external gate inspect complete pilot metrics."""

    value = str(status)
    if value == "complete":
        return 0
    if value == "pilot_gate_failed" and bool(external_evidence_gate):
        return 0
    return 2


def _load_context(config: Mapping[str, object]):
    parent_identity = json.loads(Path(str(config["parent_identity"])).read_text())
    base = V3Config.from_yaml(parent_identity["config"]["base_config"])
    manifest = build_manifest(base.data.source_h5)
    validate_expected_counts(
        manifest,
        {
            "train": base.data.expected_train_records,
            "validation": base.data.expected_validation_records,
        },
    )
    parent_manifest_transfer_metadata(
        config,
        parent_identity=parent_identity,
        active_manifest_digest=manifest.digest,
    )
    return base, manifest, parent_identity


def _checkpoint_transfer_config(config: Mapping[str, object]) -> Mapping[str, object]:
    transfer = config.get("checkpoint_transfer", {})
    return transfer if isinstance(transfer, Mapping) else {}


def probe_variant_for_config(
    config: Mapping[str, object], parent_identity: Mapping[str, object]
) -> ProbeVariant:
    """Apply only checkpoint-schema-compatible architecture overrides."""

    values = dict(parent_identity["variant_config"])
    overrides = config.get("variant_overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("variant_overrides must be a mapping")
    allowed = {
        "coupled_axes",
        "local_differential_residual",
        "modes",
        "coupled_2d_rank",
        "temporal_basis_rank",
        "family_expert_rank",
        "band_adapter_rank",
        "band_adapter_architecture",
        "band_adapter_spectral_rank",
        "band_adapter_modes",
        "band_adapter_full_depth",
        "band_adapter_coarse_depth",
        "band_adapter_activation_checkpointing",
        "band_adapter_dropout",
        "band_adapter_preserve_high_band",
        "local_field",
        "local_field_residual",
        "local_field_channel_multipliers",
        "local_field_causal_width_s",
        "local_field_extended_late_features",
        "local_field_temporal_operator_rank",
        "local_field_temporal_operator_spatial_kernel",
        "local_field_warp",
        "local_field_warp_max_shift_cells",
        "local_field_warp_shift_dilation",
        "local_field_green_kernel",
        "local_field_green_kernel_size",
        "local_field_green_dilations",
        "local_field_temporal_latent_basis",
        "local_field_temporal_latent_rank",
        "local_field_temporal_latent_harmonics",
        "local_field_multi_arrival",
        "local_field_multi_arrival_paths",
        "local_field_multi_arrival_max_shift_cells",
        "local_field_multi_arrival_max_delay_frac",
        "local_field_dispersive_modal",
        "local_field_dispersive_modal_modes",
        "local_field_dispersive_modal_max_frequency",
        "local_field_windowed_propagation",
        "local_field_windowed_propagation_window",
        "local_field_windowed_propagation_stride",
        "local_field_windowed_propagation_rank",
        "local_field_windowed_propagation_max_advect_cells",
        "local_field_adapter_gate_init",
        "high_frequency_residual",
        "high_frequency_hidden",
        "high_frequency_depth",
    }
    unknown = set(str(name) for name in overrides) - allowed
    if unknown:
        raise ValueError(f"variant override is not checkpoint compatible: {sorted(unknown)}")
    for name, value in overrides.items():
        if str(name) in {"coupled_axes", "local_differential_residual"} and not isinstance(value, bool):
            if str(name) != "modes":
                raise ValueError(f"variant override {name} must be boolean")
        if str(name) == "modes":
            parent_modes = int(values["modes"])
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not parent_modes <= int(value) <= 101
            ):
                raise ValueError(
                    f"variant override modes must be an integer in [{parent_modes}, 101]"
                )
        if str(name) == "coupled_2d_rank":
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= int(value) <= 32
            ):
                raise ValueError(
                    "variant override coupled_2d_rank must be an integer in [0, 32]"
                )
        if str(name) == "temporal_basis_rank":
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= int(value) <= 128
            ):
                raise ValueError(
                    "variant override temporal_basis_rank must be an integer in [0, 128]"
                )
        if str(name) == "family_expert_rank":
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= int(value) <= 64
            ):
                raise ValueError(
                    "variant override family_expert_rank must be an integer in [0, 64]"
                )
        if str(name) == "band_adapter_rank":
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= int(value) <= 64
            ):
                raise ValueError(
                    "variant override band_adapter_rank must be an integer in [0, 64]"
                )
        if str(name) == "band_adapter_architecture" and str(value) not in {
            "low_rank",
            "multiscale_spectral",
            "dynamic_multiscale_spectral",
            "shared_dynamic_multiscale_spectral",
        }:
            raise ValueError(
                "band adapter architecture must be low_rank, multiscale_spectral, "
                "dynamic_multiscale_spectral, or shared_dynamic_multiscale_spectral"
            )
        if str(name) in {
            "band_adapter_spectral_rank",
            "band_adapter_full_depth",
            "band_adapter_coarse_depth",
        }:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= int(value) <= 128
            ):
                raise ValueError(
                    f"band adapter {str(name).removeprefix('band_adapter_')} "
                    "must be an integer in [1, 128]"
                )
        if str(name) == "band_adapter_modes":
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= int(value) <= 101
            ):
                raise ValueError(
                    "band adapter modes must be an integer in [1, 101]"
                )
        if str(name) == "band_adapter_activation_checkpointing" and not isinstance(
            value, bool
        ):
            raise ValueError(
                "band adapter activation checkpointing must be boolean"
            )
        if str(name) == "band_adapter_dropout":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) < 1.0
            ):
                raise ValueError("band adapter dropout must be a number in [0, 1)")
        if str(name) == "band_adapter_preserve_high_band" and not isinstance(
            value, bool
        ):
            raise ValueError(
                "band adapter high-band preservation must be boolean"
            )
        if str(name) in {"local_field", "local_field_residual"} and not isinstance(
            value, bool
        ):
            raise ValueError(f"variant override {name} must be boolean")
        if str(name) == "local_field_channel_multipliers":
            if (
                not isinstance(value, (list, tuple))
                or len(value) < 2
                or any(isinstance(m, bool) or not isinstance(m, int) or m <= 0 for m in value)
                or int(value[0]) != 1
            ):
                raise ValueError(
                    "local_field_channel_multipliers must be a list of >=2 positive "
                    "integers with a leading multiplier of 1"
                )
        if str(name) == "local_field_causal_width_s":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0.0:
                raise ValueError("local_field_causal_width_s must be a positive number")
        if str(name) == "local_field_warp" and not isinstance(value, bool):
            raise ValueError("variant override local_field_warp must be boolean")
        if str(name) == "local_field_warp_max_shift_cells":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0.0:
                raise ValueError(
                    "local_field_warp_max_shift_cells must be a positive number"
                )
        if str(name) == "local_field_warp_shift_dilation":
            if isinstance(value, bool) or not isinstance(value, int) or int(value) < 1:
                raise ValueError(
                    "local_field_warp_shift_dilation must be a positive integer"
                )
        if str(name) == "local_field_green_kernel" and not isinstance(value, bool):
            raise ValueError("variant override local_field_green_kernel must be boolean")
        if str(name) == "local_field_green_kernel_size":
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or int(value) <= 0
                or int(value) % 2 == 0
            ):
                raise ValueError(
                    "local_field_green_kernel_size must be a positive odd integer"
                )
        if str(name) == "local_field_green_dilations":
            if (
                not isinstance(value, (list, tuple))
                or len(value) == 0
                or any(isinstance(d, bool) or not isinstance(d, int) or int(d) <= 0 for d in value)
            ):
                raise ValueError(
                    "local_field_green_dilations must be a non-empty list of positive integers"
                )
        if str(name) == "local_field_temporal_latent_basis" and not isinstance(value, bool):
            raise ValueError("variant override local_field_temporal_latent_basis must be boolean")
        if str(name) in ("local_field_temporal_latent_rank", "local_field_temporal_latent_harmonics"):
            if isinstance(value, bool) or not isinstance(value, int) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if str(name) == "local_field_multi_arrival" and not isinstance(value, bool):
            raise ValueError("variant override local_field_multi_arrival must be boolean")
        if str(name) == "local_field_dispersive_modal" and not isinstance(value, bool):
            raise ValueError("variant override local_field_dispersive_modal must be boolean")
        if str(name) == "local_field_dispersive_modal_modes":
            if isinstance(value, bool) or not isinstance(value, int) or int(value) < 1:
                raise ValueError("local_field_dispersive_modal_modes must be an integer >= 1")
        if str(name) == "local_field_dispersive_modal_max_frequency":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0.0:
                raise ValueError(
                    "local_field_dispersive_modal_max_frequency must be a positive number"
                )
        if str(name) == "local_field_multi_arrival_paths":
            if isinstance(value, bool) or not isinstance(value, int) or int(value) < 1:
                raise ValueError("local_field_multi_arrival_paths must be an integer >= 1")
        if str(name) == "local_field_multi_arrival_max_shift_cells":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0.0:
                raise ValueError(
                    "local_field_multi_arrival_max_shift_cells must be a positive number"
                )
        if str(name) == "local_field_multi_arrival_max_delay_frac":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 < float(value) <= 1.0
            ):
                raise ValueError(
                    "local_field_multi_arrival_max_delay_frac must be in (0, 1]"
                )
        if str(name) == "local_field_adapter_gate_init":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0.0:
                raise ValueError(
                    "local_field_adapter_gate_init must be a non-negative number"
                )
        if str(name) == "local_field_windowed_propagation" and not isinstance(value, bool):
            raise ValueError("variant override local_field_windowed_propagation must be boolean")
        if str(name) in (
            "local_field_windowed_propagation_window",
            "local_field_windowed_propagation_stride",
            "local_field_windowed_propagation_rank",
        ):
            if isinstance(value, bool) or not isinstance(value, int) or int(value) < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        if str(name) == "local_field_windowed_propagation_max_advect_cells":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0.0:
                raise ValueError(
                    "local_field_windowed_propagation_max_advect_cells must be a positive number"
                )
    if bool(overrides.get("local_field_warp", False)) and not bool(
        overrides.get("local_field", values.get("local_field", False))
    ):
        raise ValueError("local_field_warp requires local_field")
    if bool(overrides.get("local_field_green_kernel", False)) and not bool(
        overrides.get("local_field", values.get("local_field", False))
    ):
        raise ValueError("local_field_green_kernel requires local_field")
    if bool(overrides.get("local_field_temporal_latent_basis", False)) and not bool(
        overrides.get("local_field", values.get("local_field", False))
    ):
        raise ValueError("local_field_temporal_latent_basis requires local_field")
    if bool(overrides.get("local_field_multi_arrival", False)) and not bool(
        overrides.get("local_field", values.get("local_field", False))
    ):
        raise ValueError("local_field_multi_arrival requires local_field")
    if bool(overrides.get("local_field_windowed_propagation", False)) and not bool(
        overrides.get("local_field", values.get("local_field", False))
    ):
        raise ValueError("local_field_windowed_propagation requires local_field")
    if bool(overrides.get("local_field_residual", False)) and not bool(
        overrides.get("local_field", values.get("local_field", False))
    ):
        raise ValueError("local_field_residual requires local_field")
    values.update({str(name): value for name, value in overrides.items()})
    # ProbeVariant wants an immutable channel-multiplier tuple.
    if "local_field_channel_multipliers" in values:
        values["local_field_channel_multipliers"] = tuple(
            int(m) for m in values["local_field_channel_multipliers"]
        )
    if "local_field_green_dilations" in values:
        values["local_field_green_dilations"] = tuple(
            int(d) for d in values["local_field_green_dilations"]
        )
    variant = ProbeVariant(**values)
    if (
        variant.band_adapter_architecture in {
            "multiscale_spectral",
            "dynamic_multiscale_spectral",
            "shared_dynamic_multiscale_spectral",
        }
        and int(variant.band_adapter_rank) <= 0
    ):
        raise ValueError("multiscale band adapter rank must be positive")
    return variant


def resolve_spectral_mode_transfer(
    config: Mapping[str, object],
    *,
    parent_identity: Mapping[str, object],
    checkpoint_identity: Mapping[str, object],
) -> tuple[ProbeVariant, ProbeVariant, bool]:
    """Resolve the actual checkpoint and candidate mode widths safely."""

    checkpoint_config = checkpoint_identity.get(
        "model_config", checkpoint_identity.get("config", {})
    )
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("parent checkpoint identity config must be a mapping")
    parent_variant = probe_variant_for_config(checkpoint_config, parent_identity)
    candidate_variant = probe_variant_for_config(config, parent_identity)
    if candidate_variant.modes < parent_variant.modes:
        raise ValueError("spectral mode transfer cannot shrink a trained checkpoint")
    expands = candidate_variant.modes > parent_variant.modes
    if expands:
        transfer = _checkpoint_transfer_config(config)
        if not bool(transfer.get("allow_spectral_mode_expansion", False)):
            raise ValueError(
                "spectral expansion requires allow_spectral_mode_expansion"
            )
        if bool(transfer.get("parent_optimizer_state", False)):
            raise ValueError(
                "spectral mode expansion cannot restore a shape-incompatible optimizer"
            )
    return parent_variant, candidate_variant, bool(expands)


def coupled_2d_missing_prefixes(
    config: Mapping[str, object],
    *,
    parent_variant: ProbeVariant,
    candidate_variant: ProbeVariant,
    block_count: int,
) -> tuple[str, ...]:
    """Authorize exactly one zero-gated 2-D branch expansion."""

    source_rank = int(parent_variant.coupled_2d_rank)
    target_rank = int(candidate_variant.coupled_2d_rank)
    transfer = _checkpoint_transfer_config(config)
    allowed = bool(transfer.get("allow_new_coupled_2d_parameters", False))
    if source_rank == target_rank:
        if allowed:
            raise ValueError("coupled 2-D transfer permission has no rank expansion")
        return ()
    if source_rank != 0 or target_rank <= 0:
        raise ValueError("coupled 2-D checkpoint transfer only supports rank 0 expansion")
    if not allowed:
        raise ValueError("coupled 2-D expansion requires allow_new_coupled_2d_parameters")
    if bool(transfer.get("parent_optimizer_state", False)):
        raise ValueError("coupled 2-D expansion cannot restore parent optimizer state")
    if int(parent_variant.modes) != int(candidate_variant.modes):
        raise ValueError("spectral mode and coupled 2-D expansions must be staged")
    count = int(block_count)
    if count <= 0:
        raise ValueError("coupled 2-D checkpoint expansion requires residual blocks")
    return tuple(
        f"dense_decoder.stack.blocks.{index}.coupled_2d."
        for index in range(count)
    )


def temporal_basis_missing_prefixes(
    config: Mapping[str, object],
    *,
    parent_variant: ProbeVariant,
    candidate_variant: ProbeVariant,
) -> tuple[str, ...]:
    """Authorize exactly one zero-gated temporal-basis branch expansion."""

    source_rank = int(parent_variant.temporal_basis_rank)
    target_rank = int(candidate_variant.temporal_basis_rank)
    transfer = _checkpoint_transfer_config(config)
    allowed = bool(transfer.get("allow_new_temporal_basis_parameters", False))
    if source_rank == target_rank:
        if allowed:
            raise ValueError("temporal-basis transfer permission has no rank expansion")
        return ()
    if source_rank != 0 or target_rank <= 0:
        raise ValueError("temporal-basis expansion only supports rank 0 to positive")
    if not allowed:
        raise ValueError("temporal-basis expansion requires explicit transfer permission")
    if bool(transfer.get("parent_optimizer_state", False)):
        raise ValueError("temporal-basis expansion cannot restore parent optimizer state")
    if (
        int(parent_variant.modes) != int(candidate_variant.modes)
        or int(parent_variant.coupled_2d_rank) != int(candidate_variant.coupled_2d_rank)
        or bool(parent_variant.local_differential_residual)
        != bool(candidate_variant.local_differential_residual)
    ):
        raise ValueError("temporal-basis and other architecture expansions must be staged")
    return ("dense_decoder.temporal_basis.",)


def family_expert_missing_prefixes(
    config: Mapping[str, object],
    *,
    parent_variant: ProbeVariant,
    candidate_variant: ProbeVariant,
) -> tuple[str, ...]:
    """Authorize only a new zero-output velocity-routed expert branch."""

    source_rank = int(parent_variant.family_expert_rank)
    target_rank = int(candidate_variant.family_expert_rank)
    transfer = _checkpoint_transfer_config(config)
    allowed = bool(transfer.get("allow_new_family_expert_parameters", False))
    if source_rank == target_rank:
        if allowed:
            raise ValueError("family expert transfer permission has no rank expansion")
        return ()
    if source_rank != 0 or target_rank <= 0:
        raise ValueError(
            "family expert checkpoint parent must not already contain experts"
        )
    if not allowed:
        raise ValueError("family expert expansion requires explicit transfer permission")
    if bool(transfer.get("parent_optimizer_state", False)):
        raise ValueError("family expert expansion cannot restore parent optimizer state")
    parent_signature = (
        int(parent_variant.modes),
        int(parent_variant.coupled_2d_rank),
        int(parent_variant.temporal_basis_rank),
        bool(parent_variant.local_differential_residual),
    )
    candidate_signature = (
        int(candidate_variant.modes),
        int(candidate_variant.coupled_2d_rank),
        int(candidate_variant.temporal_basis_rank),
        bool(candidate_variant.local_differential_residual),
    )
    if parent_signature != candidate_signature:
        raise ValueError("family experts and other architecture expansions must be staged")
    return ("dense_decoder.family_experts.",)


def family_expert_identity_report(
    model, *, expected_rank: int
) -> dict[str, int | bool]:
    """Prove a freshly expanded expert path contributes exactly zero."""

    path = getattr(getattr(model, "dense_decoder", None), "family_experts", None)
    rank = int(expected_rank)
    if path is None or rank <= 0 or len(path.experts) != 3:
        raise ValueError("family expert expansion does not match the registered rank")
    observed_ranks = {int(expert.down.out_channels) for expert in path.experts}
    zero_outputs = all(
        torch.count_nonzero(expert.output.weight).item() == 0
        and torch.count_nonzero(expert.output.bias).item() == 0
        for expert in path.experts
    )
    if observed_ranks != {rank} or not zero_outputs:
        raise ValueError("family expert expansion does not preserve exact parent identity")
    return {
        "family_expert_rank": rank,
        "expert_count": 3,
        "exact_parent_identity": True,
    }


def band_adapter_missing_prefixes(
    config: Mapping[str, object],
    *,
    parent_variant: ProbeVariant,
    candidate_variant: ProbeVariant,
) -> tuple[str, ...]:
    """Authorize only a zero-output, high-band-safe adapter expansion."""

    source_rank = int(parent_variant.band_adapter_rank)
    target_rank = int(candidate_variant.band_adapter_rank)
    transfer = _checkpoint_transfer_config(config)
    allowed = bool(transfer.get("allow_new_band_adapter_parameters", False))
    if source_rank == target_rank:
        if allowed:
            raise ValueError("band adapter transfer permission has no rank expansion")
        return ()
    if source_rank != 0 or target_rank <= 0:
        raise ValueError("band adapter checkpoint transfer only supports rank 0 expansion")
    if not allowed:
        raise ValueError("band adapter expansion requires explicit transfer permission")
    if bool(transfer.get("parent_optimizer_state", False)):
        raise ValueError("band adapter expansion cannot restore parent optimizer state")
    parent_signature = (
        int(parent_variant.modes),
        int(parent_variant.coupled_2d_rank),
        int(parent_variant.temporal_basis_rank),
        int(parent_variant.family_expert_rank),
        bool(parent_variant.local_differential_residual),
    )
    candidate_signature = (
        int(candidate_variant.modes),
        int(candidate_variant.coupled_2d_rank),
        int(candidate_variant.temporal_basis_rank),
        int(candidate_variant.family_expert_rank),
        bool(candidate_variant.local_differential_residual),
    )
    if parent_signature != candidate_signature:
        raise ValueError("band adapter and other architecture expansions must be staged")
    return ("dense_decoder.band_limited_adapter.",)


def local_field_missing_prefixes(
    config: Mapping[str, object],
    *,
    parent_variant: ProbeVariant,
    candidate_variant: ProbeVariant,
) -> tuple[str, ...]:
    """Authorize a zero-init local-propagation residual generator added on top of a
    parent checkpoint that predates it (ControlNet-style warm start).

    Only the residual variant is a safe on-parent expansion: its output conv is
    zero-initialised, so update-0 reproduces the parent exactly and the MIONet
    coarse path the parent trained stays live. A replace-mode expansion would
    orphan that trained coarse path, so it is rejected here.
    """

    parent_on = bool(parent_variant.local_field)
    candidate_on = bool(candidate_variant.local_field)
    transfer = _checkpoint_transfer_config(config)
    allowed = bool(transfer.get("allow_new_local_field_parameters", False))
    if parent_on == candidate_on:
        if allowed and not candidate_on:
            raise ValueError("local field transfer permission has no generator to add")
        return ()
    if parent_on and not candidate_on:
        raise ValueError("local field checkpoint transfer cannot drop a trained generator")
    if not allowed:
        raise ValueError(
            "adding a local field generator requires allow_new_local_field_parameters"
        )
    if not bool(candidate_variant.local_field_residual):
        raise ValueError(
            "local field expansion on a parent requires local_field_residual "
            "(a zero-init residual on top of the parent coarse)"
        )
    if bool(transfer.get("parent_optimizer_state", False)):
        raise ValueError("local field expansion cannot restore parent optimizer state")
    return ("local_field.",)


def band_adapter_identity_report(
    model,
    *,
    expected_rank: int,
    expected_architecture: str = "low_rank",
) -> dict[str, int | bool | str]:
    """Prove a freshly expanded band adapter contributes exactly zero."""

    path = getattr(getattr(model, "dense_decoder", None), "band_limited_adapter", None)
    rank = int(expected_rank)
    expected_experts = (
        1
        if str(expected_architecture) == "shared_dynamic_multiscale_spectral"
        else 3
    )
    if path is None or rank <= 0 or len(path.experts) != expected_experts:
        raise ValueError("band adapter expansion does not match the registered rank")
    architecture = str(expected_architecture)
    observed_architecture = str(getattr(path, "architecture", "low_rank"))
    if architecture not in {
        "low_rank",
        "multiscale_spectral",
        "dynamic_multiscale_spectral",
        "shared_dynamic_multiscale_spectral",
    }:
        raise ValueError("band adapter identity architecture is invalid")
    observed_ranks = {
        int(
            expert.down.out_channels
            if hasattr(expert, "down")
            else expert.latent_width
        )
        for expert in path.experts
    }
    zero_outputs = all(
        torch.count_nonzero(expert.output.weight).item() == 0
        and torch.count_nonzero(expert.output.bias).item() == 0
        for expert in path.experts
    )
    if (
        observed_architecture != architecture
        or observed_ranks != {rank}
        or not zero_outputs
    ):
        raise ValueError("band adapter expansion does not preserve exact parent identity")
    return {
        "band_adapter_rank": rank,
        "band_adapter_architecture": architecture,
        "expert_count": len(path.experts),
        "trainable_parameters": sum(
            parameter.numel() for parameter in path.parameters()
        ),
        "exact_parent_identity": True,
    }


def activate_band_adapter_output_if_requested(
    model, config: Mapping[str, object]
) -> dict[str, object] | None:
    """Apply an explicit dynamic readout wakeup after exact transfer audit."""

    adapter_config = config.get("band_limited_adapter")
    if not isinstance(adapter_config, Mapping):
        return None
    raw_std = adapter_config.get("output_initialization_std")
    if raw_std is None:
        return None
    adapter = getattr(
        getattr(model, "dense_decoder", None), "band_limited_adapter", None
    )
    if adapter is None:
        raise ValueError("band adapter output initialization requires an enabled adapter")
    return activate_band_adapter_output(
        adapter,
        std=float(raw_std),
        seed=int(config["seed"]),
    )


def absorb_temporal_basis_gate_if_requested(
    model, config: Mapping[str, object]
) -> dict[str, float] | None:
    """Apply an exact function-preserving gate reparameterization on request."""

    recovery = config.get("residual_recovery", {})
    if not isinstance(recovery, Mapping) or not bool(
        recovery.get("absorb_temporal_basis_gate", False)
    ):
        return None
    if bool(
        _checkpoint_transfer_config(config).get("parent_optimizer_state", False)
    ):
        raise ValueError(
            "temporal basis gate absorption is incompatible with optimizer restore"
        )
    path = getattr(model.dense_decoder, "temporal_basis", None)
    if path is None:
        raise ValueError("temporal gate absorption requires an enabled temporal basis")
    return path.absorb_gate_into_coefficient()


def parent_manifest_transfer_metadata(
    config: Mapping[str, object],
    *,
    parent_identity: Mapping[str, object],
    active_manifest_digest: str,
) -> dict[str, object]:
    parent_digest = str(parent_identity["manifest_digest"])
    active_digest = str(active_manifest_digest)
    allowed = bool(
        _checkpoint_transfer_config(config).get("allow_parent_manifest_mismatch", False)
    )
    if parent_digest == active_digest:
        return {
            "allowed": False,
            "parent_manifest_digest": parent_digest,
            "active_manifest_digest": active_digest,
            "reason": "parent_manifest_matches_active_dataset",
        }
    if not allowed:
        raise ValueError("full-support parent manifest identity mismatch")
    return {
        "allowed": True,
        "parent_manifest_digest": parent_digest,
        "active_manifest_digest": active_digest,
        "reason": "explicit_checkpoint_weight_transfer_after_dataset_repair",
    }


def parent_checkpoint_expected_manifest_digest(
    config: Mapping[str, object],
    *,
    active_manifest_digest: str,
    checkpoint_identity: Mapping[str, object],
) -> str:
    if (
        str(checkpoint_identity["manifest_digest"]) != str(active_manifest_digest)
        and bool(
            _checkpoint_transfer_config(config).get(
                "allow_parent_manifest_mismatch", False
            )
        )
    ):
        return str(checkpoint_identity["manifest_digest"])
    return str(active_manifest_digest)


def adaptive_record_schedule_parameters(
    adaptive: Mapping[str, object],
    record_families: Sequence[str],
) -> tuple[np.ndarray | None, float]:
    """Resolve either RAD or uniform random record oversampling.

    Uniform mode preserves the same epoch length and coverage contract as RAD,
    isolating the sampling distribution from the number of optimizer updates.
    """

    if not bool(adaptive.get("enabled", False)) or not bool(
        adaptive.get("record_axis", False)
    ):
        return None, 1.0
    oversample = float(adaptive.get("record_oversample", 1.5))
    if not math.isfinite(oversample) or oversample < 1.0:
        raise ValueError("adaptive_sampling.record_oversample must be >= 1.0")
    families = tuple(str(value) for value in record_families)
    if not families:
        raise ValueError("adaptive record sampling requires training records")
    strategy = str(adaptive.get("record_axis_strategy", "rad"))
    if strategy == "uniform":
        return np.full(len(families), 1.0 / len(families), dtype=np.float64), oversample
    if strategy != "rad":
        raise ValueError(
            "adaptive_sampling.record_axis_strategy must be rad or uniform"
        )
    family_errors = adaptive.get("family_errors")
    if not isinstance(family_errors, Mapping):
        raise ValueError(
            "adaptive_sampling.record_axis requires family_errors "
            "{uniform:.., layered:.., marmousi:..} in rad mode"
        )
    try:
        per_record = np.array(
            [float(family_errors[family]) for family in families],
            dtype=np.float64,
        )
    except KeyError as error:
        raise ValueError(
            f"adaptive_sampling.family_errors is missing family {error.args[0]}"
        ) from error
    return (
        rad_record_weights(
            per_record,
            k=float(adaptive.get("k", 1.0)),
            c=float(adaptive.get("c", 1.0)),
            floor=float(adaptive.get("floor", 0.0)),
        ),
        oversample,
    )


def restore_parent_optimizer_state(
    config: Mapping[str, object],
    *,
    model,
    optimizer,
    active_manifest_digest: str,
    parent_identity: Mapping[str, object],
    device: torch.device,
) -> bool:
    """Restore Adam moments only for an explicitly registered continuation."""

    if not bool(
        _checkpoint_transfer_config(config).get("parent_optimizer_state", False)
    ):
        return False
    configured_learning_rates = {
        str(group["group_name"]): (
            float(group["lr"]),
            float(group.get("initial_lr", group["lr"])),
        )
        for group in optimizer.param_groups
    }
    checkpoint_identity = parent_identity
    if config.get("parent_checkpoint_identity"):
        checkpoint_identity = json.loads(
            Path(str(config["parent_checkpoint_identity"])).read_text()
        )
    load_checkpoint(
        str(config["parent_checkpoint"]),
        model=model,
        optimizer=optimizer,
        expected_manifest_digest=parent_checkpoint_expected_manifest_digest(
            config,
            active_manifest_digest=active_manifest_digest,
            checkpoint_identity=checkpoint_identity,
        ),
        expected_config_digest=str(checkpoint_identity["run_digest"]),
        restore_rng=False,
        map_location=device,
    )
    loaded_names = {str(group["group_name"]) for group in optimizer.param_groups}
    if loaded_names != set(configured_learning_rates):
        raise ValueError("parent optimizer parameter groups do not match the active config")
    for group in optimizer.param_groups:
        current_lr, initial_lr = configured_learning_rates[str(group["group_name"])]
        group["lr"] = current_lr
        group["initial_lr"] = initial_lr
    return True


def _load_parent_model(config, base, manifest, parent_identity, device):
    checkpoint_identity = parent_identity
    if config.get("parent_checkpoint_identity"):
        checkpoint_identity = json.loads(
            Path(str(config["parent_checkpoint_identity"])).read_text()
        )
    parent_variant, variant, expand_modes = resolve_spectral_mode_transfer(
        config,
        parent_identity=parent_identity,
        checkpoint_identity=checkpoint_identity,
    )
    model = _model(base, manifest, variant).to(device)
    transfer = _checkpoint_transfer_config(config)
    allow_local_differential = bool(
        transfer.get("allow_new_local_differential_parameters", False)
    )
    if allow_local_differential and not variant.local_differential_residual:
        raise ValueError(
            "local differential checkpoint expansion requires the architecture override"
        )
    allowed_missing_prefixes = (
        tuple(
            f"dense_decoder.stack.blocks.{index}.local_differential."
            for index in range(len(model.dense_decoder.stack.blocks))
        )
        if allow_local_differential
        else ()
    )
    allowed_missing_prefixes += coupled_2d_missing_prefixes(
        config,
        parent_variant=parent_variant,
        candidate_variant=variant,
        block_count=len(model.dense_decoder.stack.blocks),
    )
    allowed_missing_prefixes += temporal_basis_missing_prefixes(
        config,
        parent_variant=parent_variant,
        candidate_variant=variant,
    )
    expert_missing_prefixes = family_expert_missing_prefixes(
        config,
        parent_variant=parent_variant,
        candidate_variant=variant,
    )
    allowed_missing_prefixes += expert_missing_prefixes
    adapter_missing_prefixes = band_adapter_missing_prefixes(
        config,
        parent_variant=parent_variant,
        candidate_variant=variant,
    )
    allowed_missing_prefixes += adapter_missing_prefixes
    local_field_prefixes = local_field_missing_prefixes(
        config,
        parent_variant=parent_variant,
        candidate_variant=variant,
    )
    allowed_missing_prefixes += local_field_prefixes
    # A newly enabled high-frequency residual head is zero-initialized (its output
    # conv is zeros), so warm-starting a parent that predates it reproduces the
    # parent exactly at load; allow its tensors to be missing from the checkpoint.
    if bool(getattr(variant, "high_frequency_residual", False)) and not bool(
        getattr(parent_variant, "high_frequency_residual", False)
    ):
        allowed_missing_prefixes += ("dense_decoder.high_frequency_head.",)
    # A newly enabled query-invariant temporal propagation operator (zero-init gate)
    # is likewise an exact no-op at load; allow its tensors to be missing when the
    # parent predates it.
    if int(getattr(variant, "local_field_temporal_operator_rank", 0)) > 0 and int(
        getattr(parent_variant, "local_field_temporal_operator_rank", 0)
    ) <= 0:
        allowed_missing_prefixes += ("local_field.temporal_operator.",)
    # The arrival-aligned warp (r4) is zero-init (identity at load), so it likewise
    # reproduces the parent exactly; allow its tensors to be missing when the parent
    # predates it.
    if bool(getattr(variant, "local_field_warp", False)) and not bool(
        getattr(parent_variant, "local_field_warp", False)
    ):
        allowed_missing_prefixes += ("local_field.warp.",)
    # The dynamic Green/scattering kernel (r5) is zero-init gated (exact no-op at
    # load), so warm-starting a parent that predates it is byte-reproducible; allow
    # its tensors to be missing when the parent lacks it.
    if bool(getattr(variant, "local_field_green_kernel", False)) and not bool(
        getattr(parent_variant, "local_field_green_kernel", False)
    ):
        allowed_missing_prefixes += ("local_field.green_kernel.",)
    # The continuous temporal latent basis (A3, class-A) is zero-init gated (exact
    # no-op at load), so warm-starting a parent that predates it is byte-reproducible;
    # allow its tensors to be missing when the parent lacks it.
    if bool(getattr(variant, "local_field_temporal_latent_basis", False)) and not bool(
        getattr(parent_variant, "local_field_temporal_latent_basis", False)
    ):
        allowed_missing_prefixes += ("local_field.temporal_latent.",)
    # The multi-arrival extra-path warp (A4, class-A) is zero-gated (exact no-op at
    # load: extra paths contribute nothing until their path_gate learns off zero), so
    # warm-starting a parent that predates it is byte-reproducible -- and because A4 is
    # additive ON TOP of the parent-trained local_field.warp (Option B, path 0 loads
    # unchanged) the model starts EXACTLY at the loaded warp ceiling; allow its tensors
    # to be missing when the parent lacks it.
    if bool(getattr(variant, "local_field_multi_arrival", False)) and not bool(
        getattr(parent_variant, "local_field_multi_arrival", False)
    ):
        allowed_missing_prefixes += ("local_field.multi_arrival.",)
    # The dispersive modal field (A5, class-A escalation) is zero-gated (exact no-op at
    # load until its gate learns off zero) and additive on the coarse field, so warm-starting
    # a parent that predates it is byte-reproducible; allow its tensors to be missing.
    if bool(getattr(variant, "local_field_dispersive_modal", False)) and not bool(
        getattr(parent_variant, "local_field_dispersive_modal", False)
    ):
        allowed_missing_prefixes += ("local_field.dispersive_modal.",)
    if bool(getattr(variant, "local_field_windowed_propagation", False)) and not bool(
        getattr(parent_variant, "local_field_windowed_propagation", False)
    ):
        allowed_missing_prefixes += ("local_field.windowed_propagation.",)
    expected_manifest = parent_checkpoint_expected_manifest_digest(
        config,
        active_manifest_digest=manifest.digest,
        checkpoint_identity=checkpoint_identity,
    )
    if expand_modes:
        if allow_local_differential:
            raise ValueError(
                "spectral mode and local-differential checkpoint expansions must be staged"
            )
        parent_model = _model(base, manifest, parent_variant).cpu()
        load_checkpoint(
            str(config["parent_checkpoint"]),
            model=parent_model,
            expected_manifest_digest=expected_manifest,
            expected_config_digest=checkpoint_identity["run_digest"],
            map_location="cpu",
        )
        model.spectral_mode_expansion_report = transfer_expanded_spectral_modes(
            parent_model, model
        )
        del parent_model
    else:
        load_checkpoint(
            str(config["parent_checkpoint"]),
            model=model,
            expected_manifest_digest=expected_manifest,
            expected_config_digest=checkpoint_identity["run_digest"],
            map_location=device,
            allowed_missing_prefixes=allowed_missing_prefixes,
        )
    if expert_missing_prefixes:
        model.family_expert_transfer_report = family_expert_identity_report(
            model,
            expected_rank=int(variant.family_expert_rank),
        )
    if adapter_missing_prefixes:
        model.band_adapter_transfer_report = band_adapter_identity_report(
            model,
            expected_rank=int(variant.band_adapter_rank),
            expected_architecture=str(variant.band_adapter_architecture),
        )
    wakeup = activate_band_adapter_output_if_requested(model, config)
    if wakeup is not None:
        model.band_adapter_wakeup_report = wakeup
    recovery = config.get("residual_recovery")
    if recovery:
        activation = residual_activation_config(recovery, transfer)
        model.residual_activation_report = activate_residual_head(
            model.dense_decoder,
            activation,
            seed=int(config["seed"]),
        )
        absorption = absorb_temporal_basis_gate_if_requested(model, config)
        if absorption is not None:
            model.residual_activation_report[
                "temporal_basis_gate_absorption"
            ] = absorption
    return model


def _training_source_metadata(base, manifest):
    records = tuple(record for record in manifest.records if record.split == "train")
    source_indices = np.asarray([record.source_index for record in records], dtype=np.int64)
    with h5py.File(base.data.source_h5, "r", swmr=True) as handle:
        onset = np.asarray(handle["source_t0_s"][source_indices], dtype=np.float64)
        frequency = np.asarray(handle["source_f0_hz"][source_indices], dtype=np.float64)
    return onset, frequency, tuple(record.sample_id for record in records)


def _coverage_update(
    coverage,
    specs,
    *,
    time_s,
    source_t0_s,
    source_f0_hz,
    sample_ids,
    seed,
    frames_per_appearance=4,
    time_policy="appearance16",
    time_index_pool=None,
):
    for spec in specs:
        for record_index, appearance in zip(
            spec.record_indices, spec.appearance_indices, strict=True
        ):
            if str(time_policy) == "numerical_teacher_pool":
                if time_index_pool is None:
                    raise ValueError("numerical teacher coverage requires its time pool")
                indices = numerical_teacher_pool_indices(
                    time_index_pool,
                    sample_id=sample_ids[record_index],
                    appearance=int(appearance),
                    seed=int(seed),
                    count=int(frames_per_appearance),
                )
            elif str(time_policy) == "fixed_train_gate":
                indices = validation_time_indices(
                    time_s,
                    source_t0_s=float(source_t0_s[record_index]),
                    source_f0_hz=float(source_f0_hz[record_index]),
                    sample_id=sample_ids[record_index],
                    panel_offset=0,
                    seed=int(seed),
                    count=int(frames_per_appearance),
                )
            else:
                indices = appearance_time_indices(
                    time_s,
                    source_t0_s=float(source_t0_s[record_index]),
                    source_f0_hz=(
                        float(source_f0_hz[record_index])
                        if int(frames_per_appearance) == 16
                        else None
                    ),
                    sample_id=sample_ids[record_index],
                    appearance=int(appearance),
                    seed=int(seed),
                    count=int(frames_per_appearance),
                )
            coverage[record_index, indices] = True


def _coverage_report(coverage: np.ndarray) -> dict[str, int | float]:
    counts = np.asarray(coverage, dtype=np.uint8).sum(axis=1, dtype=np.int64)
    return {
        "record_count": int(len(counts)),
        "minimum_unique_indices": int(counts.min()),
        "median_unique_indices": float(np.median(counts)),
        "maximum_unique_indices": int(counts.max()),
    }


def _save_coverage(path: Path, coverage: np.ndarray, *, epoch: int, digest: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        with partial.open("xb") as handle:
            np.savez_compressed(
                handle,
                packed=np.packbits(coverage, axis=1, bitorder="little"),
                epoch=np.asarray(int(epoch), dtype=np.int64),
                schedule_digest=np.asarray(str(digest)),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _load_coverage(path: Path, *, records: int, times: int, epoch: int, digest: str):
    with np.load(path, allow_pickle=False) as payload:
        if int(payload["epoch"]) != int(epoch) or str(payload["schedule_digest"]) != digest:
            raise ValueError("coverage ledger identity mismatch")
        unpacked = np.unpackbits(payload["packed"], axis=1, bitorder="little")[:, :times]
    if unpacked.shape != (records, times):
        raise ValueError("coverage ledger shape mismatch")
    return unpacked.astype(bool, copy=False)


def _gradient_report(model, active_prefixes: Sequence[str]) -> dict[str, float]:
    report: dict[str, float] = {}
    for prefix in active_prefixes:
        values = [
            parameter.grad.detach().float().norm()
            for name, parameter in model.named_parameters()
            if (
                (
                    name.startswith(f"{prefix}.")
                    if "." in prefix
                    else name.split(".", 1)[0] == prefix
                )
                and parameter.grad is not None
            )
        ]
        if not values:
            raise RuntimeError(f"trainable parameter group has no gradient: {prefix}")
        norm = float(torch.linalg.vector_norm(torch.stack(values)))
        if not math.isfinite(norm) or norm <= 0.0:
            raise FloatingPointError(f"non-finite or zero gradient in parameter group: {prefix}")
        report[prefix] = norm
    return report


def local_differential_gradient_norms(model) -> dict[str, float]:
    """Report the gate and feature gradients for an enabled V15 residual path."""

    gates: list[torch.nn.Parameter] = []
    features: list[torch.nn.Parameter] = []
    for block in model.dense_decoder.stack.blocks:
        path = block.local_differential
        if path is None:
            continue
        gates.append(path.scale)
        features.extend(
            parameter
            for name, parameter in path.named_parameters()
            if name != "scale"
        )
    if not gates:
        return {}

    def norm(parameters: Sequence[torch.nn.Parameter]) -> float:
        squared = 0.0
        for parameter in parameters:
            if parameter.grad is None:
                continue
            squared += float(parameter.grad.detach().float().square().sum().item())
        return math.sqrt(squared)

    return {
        "local_differential_gate": norm(gates),
        "local_differential_features": norm(features),
    }


def coupled_2d_gradient_norms(model) -> dict[str, float]:
    """Report staged gate and feature gradients for joint-frequency branches."""

    gates: list[torch.nn.Parameter] = []
    features: list[torch.nn.Parameter] = []
    for block in model.dense_decoder.stack.blocks:
        path = block.coupled_2d
        if path is None:
            continue
        gates.append(path.scale)
        features.extend(
            parameter
            for name, parameter in path.named_parameters()
            if name != "scale"
        )
    if not gates:
        return {}

    def norm(parameters: Sequence[torch.nn.Parameter]) -> float:
        squared = 0.0
        for parameter in parameters:
            if parameter.grad is None:
                continue
            squared += float(parameter.grad.detach().float().square().sum().item())
        return math.sqrt(squared)

    return {
        "coupled_2d_gate": norm(gates),
        "coupled_2d_features": norm(features),
    }


def temporal_basis_gradient_norms(model) -> dict[str, float]:
    """Report staged gate and feature gradients for the temporal basis branch."""

    path = getattr(model.dense_decoder, "temporal_basis", None)
    if path is None:
        return {}
    features = tuple(path.feature_parameters())
    if not any(parameter.requires_grad for parameter in (path.gate, *features)):
        return {}

    def norm(parameters: Sequence[torch.nn.Parameter]) -> float:
        squared = 0.0
        for parameter in parameters:
            if parameter.grad is None:
                continue
            squared += float(parameter.grad.detach().float().square().sum().item())
        return math.sqrt(squared)

    return {
        "temporal_basis_gate": norm((path.gate,)),
        "temporal_basis_features": norm(features),
    }


def clip_trainable_gradients(
    model: torch.nn.Module,
    *,
    maximum_norm: float,
    mode: str = "global",
    prefix_limits: Mapping[str, float] | None = None,
    return_report: bool = False,
) -> float | tuple[float, dict[str, object]]:
    """Clip active gradients globally or independently by top-level module."""

    limit = float(maximum_norm)
    strategy = str(mode)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("gradient clip maximum norm must be positive and finite")
    if strategy not in {"global", "prefix", "prefix_limits"}:
        raise ValueError(
            "gradient clip mode must be 'global', 'prefix', or 'prefix_limits'"
        )
    if strategy == "prefix_limits":
        if not isinstance(prefix_limits, Mapping):
            raise ValueError(
                "prefix_limits mode requires gradient_clip_prefix_limits"
            )
        bounded = clip_gradients_by_prefix(model, prefix_limits)
        if return_report:
            telemetry = bounded.as_dict()
            telemetry["mode"] = strategy
            return bounded.total_before, telemetry
        return bounded.total_before
    named = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    if not named:
        empty = {
            "mode": strategy,
            "limit": limit,
            "total_before": 0.0,
            "total_after": 0.0,
            "prefixes": {},
        }
        return (0.0, empty) if return_report else 0.0
    norms = torch.stack(
        [parameter.grad.detach().float().norm(2) for _, parameter in named]
    )
    total_before = float(norms.norm(2))
    groups: dict[str, list[torch.nn.Parameter]] = {}
    for name, parameter in named:
        groups.setdefault(name.split(".", 1)[0], []).append(parameter)
    before_by_prefix = {
        name: float(
            torch.stack(
                [parameter.grad.detach().float().norm(2) for parameter in parameters]
            ).norm(2)
        )
        for name, parameters in groups.items()
    }
    if strategy == "global":
        torch.nn.utils.clip_grad_norm_(
            tuple(parameter for _, parameter in named), limit
        )
    else:
        for parameters in groups.values():
            torch.nn.utils.clip_grad_norm_(tuple(parameters), limit)
    after_by_prefix = {
        name: float(
            torch.stack(
                [parameter.grad.detach().float().norm(2) for parameter in parameters]
            ).norm(2)
        )
        for name, parameters in groups.items()
    }
    telemetry = {
        "mode": strategy,
        "limit": limit,
        "total_before": total_before,
        "total_after": math.sqrt(
            sum(value * value for value in after_by_prefix.values())
        ),
        "prefixes": {
            name: {
                "before": before_by_prefix[name],
                "after": after,
                "limit": limit,
                "scale": (
                    1.0
                    if before_by_prefix[name] == 0.0
                    else after / before_by_prefix[name]
                ),
            }
            for name, after in after_by_prefix.items()
        },
    }
    return (total_before, telemetry) if return_report else total_before


def _load_pilot_reports(path: Path, *, completed_epochs: int) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    if not path.exists():
        return reports
    for line in path.read_text().splitlines():
        payload = json.loads(line)
        if payload.get("event") != "epoch" or int(payload["epoch"]) > completed_epochs:
            continue
        metrics = payload["metrics"]
        reports.append(
            {
                "score": float(metrics["aggregate_relative_l2"]),
                "family": metrics["family_relative_l2"],
            }
        )
    if len(reports) != completed_epochs:
        raise ValueError("pilot metric history does not match the resume epoch")
    return reports


def recovery_time_indices(batch, device: torch.device) -> torch.Tensor:
    """Move exact-time metadata omitted by the legacy model-input transfer helper."""

    return batch.left_index.to(device, non_blocking=device.type == "cuda")


def _microbatch_record_weights(pieces) -> tuple[float, ...]:
    """Weight ragged microbatches so one macro remains a per-record mean."""

    counts = tuple(len(piece.sample_id) for piece in pieces)
    total = sum(counts)
    if not counts or any(count <= 0 for count in counts) or total <= 0:
        raise ValueError("microbatch pieces must contain records")
    return tuple(count / total for count in counts)


def backward_with_isolated_family_weighting(
    field_loss: torch.Tensor,
    *,
    router_loss: torch.Tensor | None,
    expert_parameters,
    piece_weight: float,
    family_scale: float,
    router_weight: float,
    retain_graph: bool = False,
) -> None:
    """Scale shared gradients by family while leaving expert gradients unscaled.

    ``retain_graph`` keeps the (shared front-end) graph alive after this slice's
    backward so later full-model streaming slices can accumulate into it.
    """

    piece = float(piece_weight)
    scale = float(family_scale)
    route_scale = float(router_weight)
    if (
        not math.isfinite(piece)
        or piece <= 0.0
        or not math.isfinite(scale)
        or scale <= 0.0
        or not math.isfinite(route_scale)
        or route_scale < 0.0
    ):
        raise ValueError("family-isolated backward weights are invalid")
    parameters = tuple(
        parameter for parameter in expert_parameters if parameter.requires_grad
    )
    hooks = [
        parameter.register_hook(lambda gradient, divisor=scale: gradient / divisor)
        for parameter in parameters
    ]
    route_active = router_loss is not None and route_scale > 0.0
    try:
        (field_loss * piece * scale).backward(
            retain_graph=bool(route_active or retain_graph)
        )
    finally:
        for hook in hooks:
            hook.remove()
    if route_active:
        (router_loss * piece * route_scale).backward(retain_graph=bool(retain_graph))


def _train_update(
    model,
    optimizer,
    batch,
    normalizer,
    device,
    config,
    *,
    microbatch_records: int,
    numerical_teacher_cache: NumericalTeacherCache | None = None,
    background_provider=None,
    record_residual_sink: dict[str, tuple[float, float]] | None = None,
    frame_time_weights_override=None,
    frame_residual_sink: dict[int, tuple[float, float]] | None = None,
    time_index_residual_sink: dict[int, tuple[float, float]] | None = None,
):
    """Run one optimizer update.

    ``record_residual_sink`` (optional, opt-in) accumulates a per-record relative-L2
    residual signal for residual-adaptive sampling (vRBA RAD).  For every training
    record it adds ``(sum ||pred - target||^2, sum ||target||^2)`` -- summed over the
    normalized wavefield and across time-slices/microbatches -- keyed by
    ``sample_id``.  The caller turns the accumulated ratio into a per-record score.

    ``frame_time_weights_override`` (optional, opt-in) is a 1-D tensor over the full
    selected-frame axis that REPLACES the internally-derived ``late_frame_gain`` ramp,
    letting the caller drive the per-frame loss weight from a vRBA frame-RBA bounded
    EMA.  Like the ramp it only applies to the per-frame-normalized frame term, so it
    requires ``loss.per_frame_frame``.

    ``frame_residual_sink`` (optional, opt-in) accumulates the per-frame-slot
    ``(sum ||pred - target||^2, sum ||target||^2)`` -- summed over records and
    time-slices/microbatches -- keyed by absolute frame-slot index, so the caller can
    update the frame-RBA EMA from the same relative-L2 the loss sees.

    ``time_index_residual_sink`` accumulates the same signal by the actual exact saved
    time index (which can differ across records in one batch). The caller can use that
    residual map to increase the next segment's probability of sampling hard times.

    When all three are ``None`` (the default for every existing caller) this function
    is bit-for-bit identical to before: no extra tensor is materialized.
    """
    pieces = split_pilot_batch(batch, microbatch_records=int(microbatch_records))
    piece_weights = _microbatch_record_weights(pieces)
    delta_global_target_square = None
    if str(config.get("loss", {}).get("delta_reduction", "per_record")) == "global_squared":
        if str(config["loss"].get("delta_reference", "model_coarse")) != "target":
            raise ValueError("global-squared delta reduction requires target reference")
        with torch.no_grad():
            target_square = 0.0
            for reference_piece in pieces:
                reference_source = torch.as_tensor(reference_piece.source_parameters)
                reference_target = normalizer.encode_pressure(
                    torch.as_tensor(reference_piece.dense_target_physical),
                    reference_source[:, 4],
                )
                target_square += float(reference_target.double().square().sum())
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            distributed_square = torch.tensor(
                target_square, dtype=torch.float64, device=device
            )
            torch.distributed.all_reduce(
                distributed_square, op=torch.distributed.ReduceOp.SUM
            )
            # Gradients are averaged across ranks later.  Dividing the global
            # denominator by world size makes that averaged gradient exactly the
            # derivative of sum(error^2) / sum(target^2) over all ranks.
            target_square = float(distributed_square) / float(
                torch.distributed.get_world_size()
            )
        delta_global_target_square = target_square
    optimizer.zero_grad(set_to_none=True)
    component_values: defaultdict[str, float] = defaultdict(float)
    recovery = "delta" in config["loss"]
    numerical_teacher_config = config.get("numerical_teacher")
    numerical_teacher_active = numerical_teacher_config is not None
    background_conditioning_active = (
        getattr(
            getattr(model, "local_field", None),
            "helmholtz_background_conditioner",
            None,
        )
        is not None
    )
    if background_conditioning_active and background_provider is None:
        raise ValueError(
            "Helmholtz background conditioning requires a background field provider"
        )
    if numerical_teacher_active != (numerical_teacher_cache is not None):
        raise ValueError("numerical teacher config and cache must be enabled together")
    if numerical_teacher_active and not isinstance(numerical_teacher_config, Mapping):
        raise ValueError("numerical_teacher must be a mapping")
    expert_config = config.get("family_experts")
    time_block = training_dense_time_block(config)
    loss_time_slices = training_loss_time_slices(
        config, frame_count=int(batch.requested_time_s.shape[1])
    )
    stream_time_backward = len(loss_time_slices) > 1
    # Optional late-frame reweighting of the streamed relative-energy frame loss:
    # w(t) = 1 + late_gain * clamp((i - start) / (T - 1 - start), 0, 1) over the
    # full frame axis.  late_gain=0 (default) keeps the legacy record-normalized
    # objective bit-for-bit; >0 makes low-energy late frames visible to the gradient.
    _loss_cfg = config.get("loss", {})
    _late_gain = float(_loss_cfg.get("late_frame_gain", 0.0))
    frame_time_weights = None
    if frame_time_weights_override is not None:
        # vRBA frame-RBA path: the caller supplies a 1-D per-frame weight (mean-1 tilt)
        # that replaces the fixed late_frame_gain ramp. Same contract: it only tilts
        # the per-frame-normalized frame term, so per_frame_frame must be enabled.
        if not bool(_loss_cfg.get("per_frame_frame", False)):
            raise ValueError(
                "frame_time_weights_override requires loss.per_frame_frame=true"
            )
        _frames = int(batch.requested_time_s.shape[1])
        frame_time_weights = torch.as_tensor(
            frame_time_weights_override, dtype=torch.float32
        )
        if frame_time_weights.ndim != 1 or frame_time_weights.numel() != _frames:
            raise ValueError(
                "frame_time_weights_override must be 1-D over all selected frames"
            )
    elif _late_gain > 0.0:
        if not bool(_loss_cfg.get("per_frame_frame", False)):
            raise ValueError(
                "loss.late_frame_gain>0 requires loss.per_frame_frame=true "
                "(time weights only apply to the per-frame-normalized frame term)"
            )
        _frames = int(batch.requested_time_s.shape[1])
        _start_frac = float(_loss_cfg.get("late_frame_start_fraction", 0.5))
        _start = min(max(int(_start_frac * _frames), 0), max(_frames - 1, 0))
        _ramp = torch.zeros(_frames, dtype=torch.float32)
        if _frames - 1 > _start:
            idx = torch.arange(_start, _frames, dtype=torch.float32)
            _ramp[_start:] = (idx - _start) / float(_frames - 1 - _start)
        frame_time_weights = 1.0 + _late_gain * _ramp
    loss_objective = str(
        config["optimizer"].get("training_loss_objective", "relative_l2")
    )
    initial_condition_weight = float(_loss_cfg.get("initial_condition", 0.0))
    initial_first_step_weight = float(
        _loss_cfg.get("initial_first_step_weight", 1.0)
    )
    initial_dt_s = float(_loss_cfg.get("initial_dt_s", 0.0025))
    if (
        not math.isfinite(initial_condition_weight)
        or initial_condition_weight < 0.0
        or not math.isfinite(initial_first_step_weight)
        or initial_first_step_weight < 0.0
        or not math.isfinite(initial_dt_s)
        or initial_dt_s <= 0.0
    ):
        raise ValueError("initial-condition loss settings are invalid")
    if expert_config is not None and not isinstance(expert_config, Mapping):
        raise ValueError("family_experts must be a mapping")
    stream_full_model = bool(config["optimizer"].get("stream_full_model", False))
    if stream_time_backward:
        if numerical_teacher_active:
            raise ValueError(
                "numerical teacher pilot currently requires one complete loss block"
            )
        if not stream_full_model:
            if config.get("band_limited_adapter") is None:
                raise ValueError(
                    "training loss time blocking currently requires adapter-only training"
                )
            if bool(config["optimizer"].get("full_forward_checkpointing", False)):
                raise ValueError(
                    "training loss time blocking replaces full-forward checkpointing"
                )
        if loss_objective != "squared_relative_energy":
            raise ValueError(
                "training loss time blocking requires squared_relative_energy"
            )
        streamed_zero_terms = (
            "delta",
            "temporal_difference",
            "spatial_gradient",
        )
        if any(float(config["loss"].get(name, 0.0)) != 0.0 for name in streamed_zero_terms):
            raise ValueError(
                "squared-relative time blocking requires zero delta, temporal, and gradient weights"
            )
        if not stream_full_model:
            unexpected = tuple(
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
                and not name.startswith("dense_decoder.band_limited_adapter.")
            )
            if unexpected:
                raise ValueError(
                    "training loss time blocking found trainable non-adapter parameters: "
                    + ", ".join(unexpected[:3])
                )
    for micro, piece_weight in zip(pieces, piece_weights, strict=True):
        family_weights = config.get("family_gradient_weights")
        family_scale = (
            1.0
            if family_weights is None
            else homogeneous_family_scale(micro.medium_type, family_weights)
        )
        weighted_piece = piece_weight * family_scale
        tensors = _to_device(micro, device)
        source = tensors["source_parameters"]
        background_normalized = torch.empty(0, dtype=torch.float32, device=device)
        if background_conditioning_active:
            background_physical = background_provider.full_physical(
                micro.sample_id,
                device=device,
                dtype=tensors["dense_target_physical"].dtype,
            )
            background_normalized = normalizer.encode_pressure(
                background_physical, source[:, 4]
            )
        numerical_target_full = None
        if numerical_teacher_cache is not None:
            numerical_target_full = normalizer.encode_pressure(
                numerical_teacher_cache.read(micro.sample_id, micro.left_index).to(
                    device, non_blocking=device.type == "cuda"
                ),
                source[:, 4],
            )
        route_override = (
            family_route_targets(
                micro.medium_type,
                medium_count=tensors["velocity_mps"].shape[0],
                record_to_medium=tensors["record_to_medium"],
                device=device,
            )
            if expert_config is not None
            and bool(expert_config.get("teacher_forced_routing", False))
            else torch.empty(0, dtype=torch.long, device=device)
        )

        dense_travel_time_s = (
            torch.empty(0, dtype=torch.float32, device=device)
            if micro.dense_travel_time_s is None
            else micro.dense_travel_time_s.to(
                device, non_blocking=device.type == "cuda"
            )
        )

        def forward(
            velocity,
            source_parameters,
            source_map,
            record_to_medium,
            times,
            x_m,
            z_m,
            dense_travel_time_s,
            family_route_override,
            complete_background_normalized,
        ):
            prepared = model.prepare_sources(
                model.encode_medium(velocity, normalizer),
                source_parameters,
                source_map,
                normalizer,
                record_to_medium=record_to_medium,
            )
            dense_grid = model.prepare_dense_grid(
                prepared,
                x_m=x_m,
                z_m=z_m,
                travel_time_s=(
                    None if dense_travel_time_s.numel() == 0 else dense_travel_time_s
                ),
            )
            if expert_config is not None:
                return model.dense_normalized_with_coarse_and_routing(
                    prepared,
                    times,
                    dense_grid=dense_grid,
                    time_block=time_block,
                    route_override=(
                        None
                        if family_route_override.numel() == 0
                        else family_route_override
                    ),
                    background_normalized=(
                        None
                        if complete_background_normalized.numel() == 0
                        else complete_background_normalized
                    ),
                )
            if recovery:
                prediction, coarse = model.dense_normalized_with_coarse(
                    prepared,
                    times,
                    dense_grid=dense_grid,
                    time_block=time_block,
                    background_normalized=(
                        None
                        if complete_background_normalized.numel() == 0
                        else complete_background_normalized
                    ),
                )
                return prediction, coarse, prediction.new_empty((0, 3))
            prediction = model.dense_normalized(
                prepared,
                times,
                dense_grid=dense_grid,
                time_block=time_block,
                background_normalized=(
                    None
                    if complete_background_normalized.numel() == 0
                    else complete_background_normalized
                ),
            )
            return prediction, prediction.detach(), prediction.new_empty((0, 3))

        prepared_static = None
        dense_grid_static = None
        target_full = None
        energy_reference = None
        if stream_time_backward:
            # Adapter-only streaming freezes the front end and reuses it across
            # time-loss blocks.  Full-model streaming (stream_full_model) instead
            # computes the front end WITH gradients so it is trained too; its
            # activations stay alive across slices (small vs the per-slice decoder
            # graph, which is still freed after each slice's backward), and every
            # slice's backward accumulates gradient into the shared front end.
            _front_end_ctx = (
                torch.enable_grad() if stream_full_model else torch.no_grad()
            )
            with _front_end_ctx:
                prepared_static = model.prepare_sources(
                    model.encode_medium(tensors["velocity_mps"], normalizer),
                    source,
                    tensors["source_map"],
                    normalizer,
                    record_to_medium=tensors["record_to_medium"],
                )
                dense_grid_static = model.prepare_dense_grid(
                    prepared_static,
                    x_m=tensors["x_m"],
                    z_m=tensors["z_m"],
                    travel_time_s=(
                        None
                        if dense_travel_time_s.numel() == 0
                        else dense_travel_time_s
                    ),
                )
            target_full = normalizer.encode_pressure(
                tensors["dense_target_physical"], source[:, 4]
            )
            energy_reference = relative_energy_squared_reference(
                target_full,
                energy_floor_fraction=float(
                    config["loss"].get("spectrum_energy_floor_fraction", 0.05)
                ),
            )

        for _slice_index, (time_start, time_stop, time_weight) in enumerate(loss_time_slices):
            _is_last_slice = _slice_index == len(loss_time_slices) - 1
            requested_time_s = tensors["requested_time_s"][:, time_start:time_stop]
            if stream_time_backward:
                assert prepared_static is not None and dense_grid_static is not None
                if expert_config is not None:
                    prediction, coarse, router_logits = (
                        model.dense_normalized_with_coarse_and_routing(
                            prepared_static,
                            requested_time_s,
                            dense_grid=dense_grid_static,
                            time_block=time_block,
                            route_override=(
                                None if route_override.numel() == 0 else route_override
                            ),
                            background_normalized=(
                                None
                                if background_normalized.numel() == 0
                                else background_normalized
                            ),
                        )
                    )
                elif recovery:
                    prediction, coarse = model.dense_normalized_with_coarse(
                        prepared_static,
                        requested_time_s,
                        dense_grid=dense_grid_static,
                        time_block=time_block,
                        background_normalized=(
                            None
                            if background_normalized.numel() == 0
                            else background_normalized
                        ),
                    )
                    router_logits = prediction.new_empty((0, 3))
                else:
                    prediction = model.dense_normalized(
                        prepared_static,
                        requested_time_s,
                        dense_grid=dense_grid_static,
                        time_block=time_block,
                        background_normalized=(
                            None
                            if background_normalized.numel() == 0
                            else background_normalized
                        ),
                    )
                    coarse = prediction.detach()
                    router_logits = prediction.new_empty((0, 3))
            else:
                inputs = (
                    tensors["velocity_mps"],
                    source,
                    tensors["source_map"],
                    tensors["record_to_medium"],
                    requested_time_s,
                    tensors["x_m"],
                    tensors["z_m"],
                    dense_travel_time_s,
                    route_override,
                    background_normalized,
                )
                prediction, coarse, router_logits = (
                    checkpoint(forward, *inputs, use_reentrant=False)
                    if bool(config["optimizer"]["full_forward_checkpointing"])
                    else forward(*inputs)
                )
            target = (
                target_full[:, time_start:time_stop]
                if target_full is not None
                else normalizer.encode_pressure(
                    tensors["dense_target_physical"][:, time_start:time_stop],
                    source[:, 4],
                )
            )
            numerical_target = (
                None
                if numerical_target_full is None
                else numerical_target_full[:, time_start:time_stop]
            )
            if bool(config["loss"].get("hard_causality", False)):
                causal_onset_s = source_causality_onset_s(
                    source,
                    lead_cycles=float(
                        config["loss"].get("hard_causality_lead_cycles", 0.0)
                    ),
                )
                prediction = apply_hard_causality(
                    prediction, requested_time_s, causal_onset_s
                )
                coarse = apply_hard_causality(
                    coarse, requested_time_s, causal_onset_s
                )
                if numerical_target is not None:
                    target = apply_hard_causality(
                        target, requested_time_s, causal_onset_s
                    )
                    numerical_target = apply_hard_causality(
                        numerical_target, requested_time_s, causal_onset_s
                    )
            if record_residual_sink is not None:
                # Per-record relative-L2 residual signal for vRBA RAD. Captured AFTER
                # hard-causality masking so it matches the residual the loss sees.
                # Accumulate squared numerator/denominator over every time-slice and
                # microbatch -> the per-record ratio is its true relative-L2 over all
                # frames it appeared in this update. Detached, off the autograd path.
                with torch.no_grad():
                    _flat = prediction.detach().reshape(prediction.shape[0], -1).double()
                    _tflat = target.detach().reshape(target.shape[0], -1).double()
                    _num = ((_flat - _tflat) ** 2).sum(dim=1)
                    _den = (_tflat ** 2).sum(dim=1)
                for _row, _sid in enumerate(micro.sample_id):
                    _prev_num, _prev_den = record_residual_sink.get(_sid, (0.0, 0.0))
                    record_residual_sink[_sid] = (
                        _prev_num + float(_num[_row]),
                        _prev_den + float(_den[_row]),
                    )
            if frame_residual_sink is not None:
                # Per-frame-slot relative-L2 signal for vRBA frame RBA. Sum squared
                # numerator/denominator over records (accumulated across time-slices and
                # microbatches), keyed by ABSOLUTE frame-slot index within the full
                # selected axis (time_start + local). appearance16 returns time-sorted
                # frames, so a slot index is a stable early->late position across
                # updates. Captured post-masking; detached, off the autograd path.
                with torch.no_grad():
                    _fdiff = (prediction.detach() - target.detach()).double()
                    _fnum = _fdiff.reshape(_fdiff.shape[0], _fdiff.shape[1], -1).square().sum(dim=(0, 2))
                    _ftgt = target.detach().double()
                    _fden = _ftgt.reshape(_ftgt.shape[0], _ftgt.shape[1], -1).square().sum(dim=(0, 2))
                for _local in range(_fnum.shape[0]):
                    _slot = int(time_start) + _local
                    _pn, _pd = frame_residual_sink.get(_slot, (0.0, 0.0))
                    frame_residual_sink[_slot] = (
                        _pn + float(_fnum[_local]),
                        _pd + float(_fden[_local]),
                    )
            if time_index_residual_sink is not None:
                # Actual-time RAD signal. Unlike frame_residual_sink, this retains the
                # saved-time identity for each record rather than pooling by sorted slot.
                with torch.no_grad():
                    _tdiff = (prediction.detach() - target.detach()).double()
                    _tnum = _tdiff.reshape(
                        _tdiff.shape[0], _tdiff.shape[1], -1
                    ).square().sum(dim=2)
                    _ttgt = target.detach().double()
                    _tden = _ttgt.reshape(
                        _ttgt.shape[0], _ttgt.shape[1], -1
                    ).square().sum(dim=2)
                _time_indices = micro.left_index[:, time_start:time_stop]
                for _row in range(_tnum.shape[0]):
                    for _local in range(_tnum.shape[1]):
                        _time_index = int(_time_indices[_row, _local])
                        _pn, _pd = time_index_residual_sink.get(
                            _time_index, (0.0, 0.0)
                        )
                        time_index_residual_sink[_time_index] = (
                            _pn + float(_tnum[_row, _local]),
                            _pd + float(_tden[_row, _local]),
                        )
            if numerical_target is not None:
                assert isinstance(numerical_teacher_config, Mapping)
                multifidelity_reference = multifidelity_energy_reference(
                    target,
                    numerical_target,
                    residual_energy_floor_fraction=float(
                        numerical_teacher_config.get(
                            "residual_energy_floor_fraction", 0.1
                        )
                    ),
                    spectrum_energy_floor_fraction=float(
                        config["loss"].get(
                            "spectrum_energy_floor_fraction", 0.05
                        )
                    ),
                )
                parts = multifidelity_distillation_block_loss(
                    prediction,
                    coarse,
                    target,
                    numerical_target,
                    reference=multifidelity_reference,
                    low_fidelity_weight=float(
                        numerical_teacher_config.get("low_fidelity_weight", 0.5)
                    ),
                    residual_weight=float(
                        numerical_teacher_config.get("residual_weight", 0.5)
                    ),
                    spectrum_weight=float(config["loss"]["spectrum"]),
                )
                loss = parts.total
                for name in ("total", "high", "low", "residual", "spectrum"):
                    component_values[name] += (
                        weighted_piece
                        * time_weight
                        * float(getattr(parts, name).detach())
                    )
            elif stream_time_backward:
                assert energy_reference is not None
                streamed_parts = relative_energy_squared_block_loss(
                    prediction,
                    target,
                    reference=energy_reference,
                    spectrum_weight=float(config["loss"]["spectrum"]),
                )
                loss = streamed_parts.total
                for name in ("total", "frame", "spectrum"):
                    component_values[name] += weighted_piece * float(
                        getattr(streamed_parts, name).detach()
                    )
            elif recovery:
                # Optional PINN-style wave-equation residual (P1). Default weight
                # 0.0 keeps this path bit-identical to the pre-existing behaviour;
                # a positive weight adds a relative, source-free acoustic residual
                # computed on the dense predicted field via finite differences.
                pde_weight = float(config["loss"].get("pde_residual", 0.0))
                pml_weight = float(config["loss"].get("pml_interface", 0.0))
                pde_kwargs: dict[str, object] = {}
                if pde_weight > 0.0 or pml_weight > 0.0:
                    records_b, frames_b, height_b, width_b = prediction.shape
                    dz_m = float(tensors["z_m"][1] - tensors["z_m"][0])
                    dx_m = float(tensors["x_m"][1] - tensors["x_m"][0])
                    physics_velocity = velocity_fields_per_record(
                        tensors["velocity_mps"], tensors["record_to_medium"]
                    )
                    # Spatial source-free mask [R,Z,X]: exclude cells whose eikonal
                    # arrival time is below a small onset (the injection region where
                    # the forcing f != 0 and the source-free residual does not hold).
                    spatial_free = None
                    if pde_weight > 0.0 and (
                        dense_travel_time_s.numel()
                        == records_b * height_b * width_b
                    ):
                        spatial_free = dense_travel_time_s.reshape(
                            records_b, height_b, width_b
                        ) > float(
                            config["loss"].get("pde_source_free_onset_s", 0.0)
                        )
                    # When hard causality masks pre-onset frames to zero, the 3-frame
                    # 2nd time-derivative straddling the onset edge is a spurious
                    # discontinuity, not a physical acceleration. Restrict the residual
                    # to interior frame-triples that are fully post-onset [R,frames-2].
                    source_free_mask = spatial_free
                    if pde_weight > 0.0 and frames_b >= 3 and bool(
                        config["loss"].get("hard_causality", False)
                    ):
                        onset_s = source_causality_onset_s(
                            source,
                            lead_cycles=float(
                                config["loss"].get("hard_causality_lead_cycles", 0.0)
                            ),
                        )
                        post = requested_time_s >= onset_s[:, None]
                        triple_post = post[:, :-2] & post[:, 1:-1] & post[:, 2:]
                        time_space = triple_post[:, :, None, None]
                        if spatial_free is not None:
                            time_space = time_space & spatial_free[:, None, :, :]
                        source_free_mask = time_space.expand(
                            records_b, frames_b - 2, height_b, width_b
                        )
                    pde_kwargs = dict(
                        pde_weight=pde_weight,
                        velocity_mps=physics_velocity,
                        time_values_s=requested_time_s,
                        pde_dz_m=dz_m,
                        pde_dx_m=dx_m,
                        pde_source_free_mask=source_free_mask,
                        pml_weight=pml_weight,
                        pml_boundary_band_cells=int(
                            config["loss"].get("pml_boundary_band_cells", 4)
                        ),
                    )
                parts = residual_recovery_loss(
                    prediction,
                    coarse,
                    target,
                    time_indices=micro.left_index[:, time_start:time_stop].to(
                        device, non_blocking=device.type == "cuda"
                    ),
                    frame_weight=float(
                        config["loss"].get("full_field_frame", 1.0)
                    ),
                    delta_weight=float(config["loss"]["delta"]),
                    delta_reference=str(
                        config["loss"].get("delta_reference", "model_coarse")
                    ),
                    delta_reduction=str(
                        config["loss"].get("delta_reduction", "per_record")
                    ),
                    delta_global_target_square=delta_global_target_square,
                    delta_piece_weight=piece_weight,
                    temporal_weight=float(config["loss"]["temporal_difference"]),
                    gradient_weight=float(config["loss"]["spatial_gradient"]),
                    spectrum_weight=float(config["loss"]["spectrum"]),
                    delta_energy_floor_fraction=float(
                        config["loss"].get("delta_energy_floor_fraction", 0.0)
                    ),
                    per_frame_frame=bool(
                        config["loss"].get("per_frame_frame", False)
                    ),
                    frame_energy_floor_fraction=float(
                        config["loss"].get("frame_energy_floor_fraction", 0.0)
                    ),
                    frame_time_weights=(
                        None
                        if frame_time_weights is None
                        else frame_time_weights[time_start:time_stop].to(prediction.device)
                    ),
                    **pde_kwargs,
                )
                loss = parts.total
                for name in (
                    "total", "frame", "delta", "temporal", "gradient", "spectrum"
                ):
                    component_values[name] += (
                        weighted_piece
                        * time_weight
                        * float(getattr(parts, name).detach())
                    )
                if parts.pde is not None:
                    component_values["pde"] += (
                        weighted_piece
                        * time_weight
                        * float(parts.pde.detach())
                    )
                if parts.pml is not None:
                    component_values["pml"] += (
                        weighted_piece
                        * time_weight
                        * float(parts.pml.detach())
                    )
            else:
                loss, frame, gradient, spectrum = _loss(
                    prediction,
                    target,
                    gradient_weight=float(config["loss"]["spatial_gradient"]),
                    spectrum_weight=float(config["loss"]["spectrum"]),
                    per_frame=bool(config["loss"].get("per_frame_frame", False)),
                    frame_energy_floor_fraction=float(
                        config["loss"].get("frame_energy_floor_fraction", 0.0)
                    ),
                    frame_time_weights=(
                        None
                        if frame_time_weights is None
                        else frame_time_weights[time_start:time_stop].to(prediction.device)
                    ),
                )
                for name, value in (
                    ("total", loss),
                    ("frame", frame),
                    ("gradient", gradient),
                    ("spectrum", spectrum),
                ):
                    component_values[name] += (
                        weighted_piece * time_weight * float(value.detach())
                    )
            router_loss = None
            if expert_config is not None:
                router_loss, router_report = family_router_loss(
                    router_logits,
                    micro.medium_type,
                    record_to_medium=tensors["record_to_medium"],
                )
                component_values["router"] += (
                    piece_weight * time_weight * float(router_loss.detach())
                )
                component_values["router_accuracy"] += (
                    piece_weight * time_weight * float(router_report["accuracy"])
                )
                component_values["router_entropy"] += (
                    piece_weight * time_weight * float(router_report["entropy"])
                )
                for family in ("uniform", "layered", "marmousi"):
                    key = f"route_probability_{family}"
                    component_values[key] += (
                        piece_weight * time_weight * float(router_report[key])
                    )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite full-support training loss")
            field_piece_weight = piece_weight * (
                1.0 if stream_time_backward else time_weight
            )
            # Full-model streaming shares one front-end graph across all slices;
            # retain it until the final slice so every slice's backward accumulates
            # gradient into the front end (released on the last slice).
            _retain = stream_full_model and not _is_last_slice
            if expert_config is None:
                (loss * family_scale * field_piece_weight).backward(retain_graph=_retain)
            else:
                backward_with_isolated_family_weighting(
                    loss,
                    router_loss=router_loss,
                    expert_parameters=model.dense_decoder.family_experts.parameters(),
                    piece_weight=field_piece_weight,
                    family_scale=family_scale,
                    router_weight=float(expert_config["router_loss_weight"]),
                    retain_graph=_retain,
                )
        if initial_condition_weight > 0.0:
            if stream_time_backward or expert_config is not None:
                raise ValueError(
                    "initial-condition training currently requires non-streamed, non-expert recovery"
                )
            initial_times = torch.tensor(
                (0.0, initial_dt_s),
                dtype=tensors["requested_time_s"].dtype,
                device=device,
            )[None, :].expand(tensors["velocity_mps"].shape[0], 2)
            initial_inputs = (
                tensors["velocity_mps"],
                source,
                tensors["source_map"],
                tensors["record_to_medium"],
                initial_times,
                tensors["x_m"],
                tensors["z_m"],
                dense_travel_time_s,
                route_override,
                background_normalized,
            )
            initial_prediction, _, _ = (
                checkpoint(forward, *initial_inputs, use_reentrant=False)
                if bool(config["optimizer"]["full_forward_checkpointing"])
                else forward(*initial_inputs)
            )
            with torch.no_grad():
                reference = normalizer.encode_pressure(
                    tensors["dense_target_physical"], source[:, 4]
                )
                reference_rms = reference.float().square().flatten(1).mean(dim=1).sqrt()
            initial = zero_initial_condition_loss(
                initial_prediction,
                reference_rms=reference_rms,
                first_step_weight=initial_first_step_weight,
            )
            initial_objective = initial_condition_weight * initial
            if not bool(torch.isfinite(initial_objective)):
                raise FloatingPointError("non-finite initial-condition loss")
            (initial_objective * family_scale * piece_weight).backward()
            component_values["initial"] += weighted_piece * float(initial.detach())
            component_values["total"] += (
                weighted_piece * initial_condition_weight * float(initial.detach())
            )
    if stream_time_backward:
        for name in ("delta", "temporal", "gradient"):
            component_values.setdefault(name, 0.0)
    return dict(component_values)


def _validation_schedule(indices: Sequence[int], *, epoch_offset: int) -> tuple[FullSupportStepSpec, ...]:
    selected = tuple(int(value) for value in indices)
    if not selected or len(selected) % 12:
        raise ValueError("validation records must divide into 12-record macros")
    return tuple(
        FullSupportStepSpec(
            step=100_000 + epoch_offset * 1000 + start // 12,
            epoch=int(epoch_offset),
            record_indices=selected[start : start + 12],
            appearance_indices=(int(epoch_offset),) * 12,
        )
        for start in range(0, len(selected), 12)
    )


@torch.inference_mode()
def _evaluate(
    model,
    base,
    manifest,
    normalizer,
    device,
    config,
    indices,
    *,
    epoch_offset,
    split="validation",
    time_policy="validation16",
    frames_per_record=16,
):
    evaluation_split = str(split)
    if evaluation_split not in {"train", "validation"}:
        raise ValueError("evaluation split must be train or validation")
    dataset = ExactStoredTimeBatchDataset(
        base.data.source_h5,
        manifest,
        split=evaluation_split,
        schedule=_validation_schedule(indices, epoch_offset=epoch_offset),
        query_points=1,
        seed=evaluation_time_selector_seed(config),
        time_policy=str(time_policy),
        frames_per_record=int(frames_per_record),
        travel_time_h5=config.get("travel_time_h5"),
    )
    accumulator = ExactWavefieldMetricAccumulator(
        energy_floor_fraction=float(config.get("energy_floor_fraction", 0.01)),
        require_unique=True,
        stored_time_count=len(manifest.time_s),
    )
    coarse_accumulator = ExactWavefieldMetricAccumulator(
        energy_floor_fraction=float(config.get("energy_floor_fraction", 0.01)),
        require_unique=True,
        stored_time_count=len(manifest.time_s),
    )
    correction_square = 0.0
    coarse_square = 0.0
    router_totals: defaultdict[str, float] = defaultdict(float)
    router_medium_count = 0
    model.eval()
    for batch_index in range(len(dataset)):
        batch = dataset[batch_index]
        for micro in split_pilot_batch(
            batch,
            microbatch_records=int(config["validation"]["microbatch_records"]),
        ):
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
            if config.get("family_experts") is not None:
                prediction, coarse, router_logits = (
                    model.dense_normalized_with_coarse_and_routing(
                        prepared,
                        tensors["requested_time_s"],
                        dense_grid=dense_grid,
                        time_block=1,
                    )
                )
                _, router_report = family_router_loss(
                    router_logits,
                    micro.medium_type,
                    record_to_medium=tensors["record_to_medium"],
                )
                medium_count = int(router_logits.shape[0])
                router_medium_count += medium_count
                for name, value in router_report.items():
                    router_totals[name] += medium_count * float(value)
            else:
                prediction, coarse = model.dense_normalized_with_coarse(
                    prepared,
                    tensors["requested_time_s"],
                    dense_grid=dense_grid,
                    time_block=1,
                )
            target = normalizer.encode_pressure(
                tensors["dense_target_physical"], source[:, 4]
            )
            if bool(config["loss"].get("hard_causality", False)):
                causal_onset_s = source_causality_onset_s(
                    source,
                    lead_cycles=float(
                        config["loss"].get("hard_causality_lead_cycles", 0.0)
                    ),
                )
                prediction = apply_hard_causality(
                    prediction, tensors["requested_time_s"], causal_onset_s
                )
                coarse = apply_hard_causality(
                    coarse, tensors["requested_time_s"], causal_onset_s
                )
            metric_onset_s = source[:, 3]
            if config.get("residual_recovery"):
                metric_onset_s = torch.clamp(
                    source[:, 3] - source[:, 2].reciprocal(),
                    min=float(manifest.time_s[0]),
                )
            onset_indices = torch.searchsorted(
                torch.as_tensor(manifest.time_s, device=device),
                metric_onset_s.contiguous(),
            ).cpu().tolist()
            accumulator.update(
                prediction,
                target,
                families=micro.medium_type,
                group_ids=micro.group_id,
                sample_ids=micro.sample_id,
                time_indices=micro.left_index,
                source_onset_indices=onset_indices,
            )
            coarse_accumulator.update(
                coarse,
                target,
                families=micro.medium_type,
                group_ids=micro.group_id,
                sample_ids=micro.sample_id,
                time_indices=micro.left_index,
                source_onset_indices=onset_indices,
            )
            correction_square += float((prediction.float() - coarse.float()).square().sum())
            coarse_square += float(coarse.float().square().sum())
    metrics = accumulator.finalize()
    coarse_metrics = coarse_accumulator.finalize()
    metrics["coarse_metrics"] = coarse_metrics
    metrics["correction_to_coarse_l2_ratio"] = math.sqrt(correction_square) / math.sqrt(
        max(coarse_square, 1.0e-16)
    )
    coarse_score = float(coarse_metrics["aggregate_relative_l2"])
    metrics["relative_improvement_vs_coarse"] = (
        coarse_score - float(metrics["aggregate_relative_l2"])
    ) / max(coarse_score, 1.0e-16)
    if router_medium_count:
        metrics["router_accuracy"] = (
            router_totals["accuracy"] / router_medium_count
        )
        metrics["router_entropy"] = (
            router_totals["entropy"] / router_medium_count
        )
        metrics["router_route_probabilities"] = {
            family: router_totals[f"route_probability_{family}"]
            / router_medium_count
            for family in ("uniform", "layered", "marmousi")
        }
    return metrics


def _set_epoch_learning_rates(
    optimizer,
    config,
    *,
    epoch_index: int,
    total_epochs: int,
    control_multiplier: float = 1.0,
):
    optimizer_config = config["optimizer"]
    schedule_offset = int(optimizer_config.get("schedule_epoch_offset", 0))
    schedule_total = int(
        optimizer_config.get("schedule_total_epochs", total_epochs)
    )
    if schedule_offset < 0:
        raise ValueError("optimizer schedule epoch offset must be nonnegative")
    schedule_index = min(schedule_offset + int(epoch_index), schedule_total - 1)
    factor = warmup_cosine_factor(
        schedule_index,
        total_epochs=schedule_total,
        warmup_epochs=int(optimizer_config["warmup_epochs"]),
        minimum_factor=float(optimizer_config["minimum_factor"]),
    )
    multiplier = float(control_multiplier)
    if not 0.0 < multiplier <= 1.0:
        raise ValueError("epoch validation learning-rate multiplier is invalid")
    effective_factor = factor * multiplier
    for group in optimizer.param_groups:
        group["lr"] = float(group["initial_lr"]) * effective_factor
    return effective_factor


def _run(config, args) -> int:
    ddp = distributed_context()
    base, manifest, parent_identity = _load_context(config)
    registered_epochs = int(config["epochs"])
    run_epochs = run_epochs_for_mode(
        registered_epochs=registered_epochs,
        pilot_epochs=int(config["gate"]["pilot_epochs"]),
        pilot=bool(args.pilot),
        smoke_updates=int(args.smoke_updates),
    )
    train_records = tuple(record for record in manifest.records if record.split == "train")
    if len(train_records) != int(base.data.expected_train_records):
        raise ValueError("full-support train manifest count changed")

    # --- Residual-adaptive sampling (RAD) for pretraining, config-driven -------
    # Reallocates the frame budget across time bins (early/mid/late) and, with
    # record_oversample>1, the record-appearance frequency across media, both in
    # proportion to a measured error profile.  Promotion-eligible configs must
    # bind that profile to train-only evidence.  Disabled by default.
    _adaptive = config.get("adaptive_sampling") or {}
    _adaptive_frame_budget = None
    _adaptive_record_weights, _adaptive_record_oversample = (
        adaptive_record_schedule_parameters(
            _adaptive,
            tuple(record.medium_type for record in train_records),
        )
    )
    if bool(_adaptive.get("enabled", False)):
        _ak = float(_adaptive.get("k", 1.0))
        _ac = float(_adaptive.get("c", 1.0))
        if bool(_adaptive.get("frame_axis", True)):
            bin_err = _adaptive.get("time_bin_errors")
            if bin_err is None:
                raise ValueError(
                    "adaptive_sampling.frame_axis requires time_bin_errors "
                    "[early, middle, late] (measured error profile)"
                )
            total_active = int(training_frames_per_record(config) or 16) - 3  # onset pair + pre-onset fixed
            _adaptive_frame_budget = rad_bin_budget(
                [float(v) for v in bin_err], total_active,
                k=_ak, c=_ac, floor_per_bin=int(_adaptive.get("floor_per_bin", 1)),
            )
    family_curriculum = config.get("family_curriculum")
    if family_curriculum:
        complete_schedule = build_family_curriculum_schedule(
            tuple(record.medium_type for record in train_records),
            stages=tuple(family_curriculum["stages"]),
            epochs=run_epochs,
            macro_records=int(config["macro_records"]),
            macros_per_update=int(config["macros_per_update"]),
            seed=int(config["seed"]),
            appearance_offset=int(config.get("time_appearance_offset", 0)),
        )
    else:
        complete_schedule = build_full_support_schedule(
            int(base.data.expected_train_records),
            epochs=run_epochs,
            macro_records=int(config["macro_records"]),
            macros_per_update=int(config["macros_per_update"]),
            seed=int(config["seed"]),
            appearance_offset=int(config.get("time_appearance_offset", 0)),
            epoch_offset=int(config.get("schedule_epoch_offset", 0)),
            record_weights=_adaptive_record_weights,
            record_oversample=_adaptive_record_oversample,
        )
    schedule_sha = schedule_digest(complete_schedule)
    if family_curriculum:
        audits = [
            audit_family_curriculum_epoch_schedule(
                complete_schedule,
                epoch=epoch,
                record_families=tuple(record.medium_type for record in train_records),
                macros_per_update=int(config["macros_per_update"]),
            )
            for epoch in range(run_epochs)
        ]
    else:
        audits = [
            audit_epoch_schedule(
                complete_schedule,
                epoch=epoch,
                record_count=int(base.data.expected_train_records),
                macros_per_update=int(config["macros_per_update"]),
                allow_oversample=_adaptive_record_oversample > 1.0,
            )
            for epoch in range(run_epochs)
        ]
    audit_payload = {
        "schedule_digest": schedule_sha,
        "epochs": run_epochs,
        "audits": [audit.__dict__ for audit in audits],
    }
    if args.audit_only:
        if bool(ddp["is_main"]):
            print(json.dumps(audit_payload, sort_keys=True))
        distributed_cleanup(ddp)
        return 0

    if int(config["macros_per_update"]) % int(ddp["world_size"]):
        distributed_cleanup(ddp)
        raise ValueError("macros_per_update must divide evenly across DDP ranks")

    mode = "smoke" if args.smoke_updates else "pilot" if args.pilot else "run"
    configured_workers = 0 if args.smoke_updates else int(config["workers"])
    effective_workers = (
        configured_workers
        if args.workers_override is None
        else int(args.workers_override)
    )
    if effective_workers < 0:
        raise ValueError("workers override must be nonnegative")
    epoch_control = resolve_epoch_validation_control(
        config, pilot_or_smoke=bool(args.pilot or args.smoke_updates)
    )
    root = Path(config["artifact_dir"]) / mode
    configured_checkpoint_keep_last = config.get("checkpoint_keep_last")
    effective_checkpoint_keep_last = (
        None if args.keep_all_checkpoints else configured_checkpoint_keep_last
    )
    if bool(ddp["is_main"]):
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json(audit_payload, root / "schedule_audit.json")
        _atomic_json(
            {
                "schema": "saved_time_checkpoint_retention_v1",
                "keep_all_epoch_checkpoints": bool(args.keep_all_checkpoints),
                "configured_keep_last": configured_checkpoint_keep_last,
                "effective_keep_last": effective_checkpoint_keep_last,
            },
            root / "checkpoint_retention.json",
        )
        _atomic_json(
            {
                "schema": "saved_time_runtime_resource_control_v1",
                "configured_workers_per_rank": configured_workers,
                "effective_workers_per_rank": effective_workers,
                "sampling_order_changed": False,
                "training_identity_changed": False,
            },
            root / "runtime_resource_control.json",
        )
    identity = {
        "schema": "saved_time_v5_training_contract_recovery_v1",
        "mode": mode,
        "config": config,
        "manifest_digest": manifest.digest,
        "parent_run_digest": parent_identity["run_digest"],
        "parent_manifest_transfer": parent_manifest_transfer_metadata(
            config,
            parent_identity=parent_identity,
            active_manifest_digest=manifest.digest,
        ),
        "schedule_digest": schedule_sha,
        "time_axis_sha256": time_axis_sha256(manifest.time_s),
    }
    identity["run_digest"] = _digest(identity)
    identity_path = root / "run_identity.json"
    if bool(ddp["is_main"]):
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("full-support run identity mismatch")
        if not identity_path.exists():
            _atomic_json(identity, identity_path)
    distributed_barrier(ddp)
    terminal_path = root / "terminal.json"
    if terminal_path.exists():
        terminal = json.loads(terminal_path.read_text())
        if terminal.get("status") in {"complete", "pilot_gate_failed"}:
            if bool(ddp["is_main"]):
                print(json.dumps(terminal, sort_keys=True))
            distributed_cleanup(ddp)
            return pilot_terminal_exit_code(
                str(terminal["status"]),
                external_evidence_gate=bool(
                    config.get("gate", {}).get("external_evidence_gate", False)
                ),
            )

    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    device = torch.device(f"cuda:{int(ddp['local_rank'])}")
    model = _load_parent_model(config, base, manifest, parent_identity, device)
    band_adapter_rates = registered_band_adapter_learning_rates(config)
    optimizer_name = str(config["optimizer"].get("name", "adamw")).lower()
    if optimizer_name not in {"adamw", "muon_adamw"}:
        raise ValueError("optimizer.name must be adamw or muon_adamw")
    if band_adapter_rates is not None and optimizer_name != "adamw":
        raise ValueError("band-limited adapter training currently requires AdamW")
    if band_adapter_rates is None:
        optimizer_builder = (
            build_staged_muon_adamw
            if optimizer_name == "muon_adamw"
            else build_staged_adamw
        )
        optimizer_kwargs = dict(
            model=model,
            dense_lr=float(config["optimizer"]["dense_learning_rate"]),
            geometry_lr=float(config["optimizer"]["geometry_learning_rate"]),
            backbone_lr=float(config["optimizer"]["backbone_learning_rate"]),
            weight_decay=float(config["optimizer"]["weight_decay"]),
            temporal_basis_lr=(
                None
                if config["optimizer"].get("temporal_basis_learning_rate") is None
                else float(config["optimizer"]["temporal_basis_learning_rate"])
            ),
            expert_lr=(
                None
                if config["optimizer"].get("family_expert_learning_rate") is None
                else float(config["optimizer"]["family_expert_learning_rate"])
            ),
            temporal_operator_lr=(
                None
                if config["optimizer"].get("temporal_operator_learning_rate") is None
                else float(config["optimizer"]["temporal_operator_learning_rate"])
            ),
            warp_lr=(
                None
                if config["optimizer"].get("warp_learning_rate") is None
                else float(config["optimizer"]["warp_learning_rate"])
            ),
            green_kernel_lr=(
                None
                if config["optimizer"].get("green_kernel_learning_rate") is None
                else float(config["optimizer"]["green_kernel_learning_rate"])
            ),
            temporal_latent_lr=(
                None
                if config["optimizer"].get("temporal_latent_learning_rate") is None
                else float(config["optimizer"]["temporal_latent_learning_rate"])
            ),
            multi_arrival_lr=(
                None
                if config["optimizer"].get("multi_arrival_learning_rate") is None
                else float(config["optimizer"]["multi_arrival_learning_rate"])
            ),
            dispersive_modal_lr=(
                None
                if config["optimizer"].get("dispersive_modal_learning_rate") is None
                else float(config["optimizer"]["dispersive_modal_learning_rate"])
            ),
            windowed_propagation_lr=(
                None
                if config["optimizer"].get("windowed_propagation_learning_rate") is None
                else float(config["optimizer"]["windowed_propagation_learning_rate"])
            ),
            local_field_lr=(
                None
                if config["optimizer"].get("local_field_learning_rate") is None
                else float(config["optimizer"]["local_field_learning_rate"])
            ),
        )
        if optimizer_name == "muon_adamw":
            optimizer_kwargs.update(
                muon_lr_scale=float(config["optimizer"].get("muon_lr_scale", 40.0)),
                momentum=float(config["optimizer"].get("muon_momentum", 0.95)),
                nesterov=bool(config["optimizer"].get("muon_nesterov", True)),
                ns_steps=int(config["optimizer"].get("muon_ns_steps", 5)),
                adamw_implementation=str(
                    config["optimizer"].get("adamw_implementation", "single_tensor")
                ),
                adamw_betas=(
                    float(config["optimizer"].get("adamw_beta1", 0.9)),
                    float(config["optimizer"].get("adamw_beta2", 0.999)),
                ),
                adamw_eps=float(config["optimizer"].get("adamw_eps", 1.0e-8)),
            )
        else:
            optimizer_kwargs.update(
                implementation=str(
                    config["optimizer"].get("adamw_implementation", "single_tensor")
                ),
                betas=(
                    float(config["optimizer"].get("adamw_beta1", 0.9)),
                    float(config["optimizer"].get("adamw_beta2", 0.999)),
                ),
                eps=float(config["optimizer"].get("adamw_eps", 1.0e-8)),
            )
        optimizer = optimizer_builder(**optimizer_kwargs)
    else:
        optimizer = build_band_adapter_adamw(
            model,
            feature_lr=band_adapter_rates[0],
            output_lr=band_adapter_rates[1],
            weight_decay=float(config["optimizer"]["weight_decay"]),
            implementation=str(
                config["optimizer"].get("adamw_implementation", "single_tensor")
            ),
        )
    restore_parent_optimizer_state(
        config,
        model=model,
        optimizer=optimizer,
        active_manifest_digest=manifest.digest,
        parent_identity=parent_identity,
        device=device,
    )
    normalizer = load_normalizer(base, manifest.digest)
    start_epoch = 0
    global_step = 0
    latest = root / "latest.pt"
    if latest.exists():
        metadata = load_checkpoint(
            latest,
            model=model,
            optimizer=optimizer,
            expected_manifest_digest=manifest.digest,
            expected_config_digest=identity["run_digest"],
            restore_rng=True,
            map_location=device,
        )
        start_epoch = metadata.epoch
        global_step = metadata.global_step
    train_model = model

    source_t0_s, source_f0_hz, train_sample_ids = _training_source_metadata(base, manifest)
    teacher_time_pool = numerical_teacher_time_pool(config)
    numerical_teacher_cache = None
    if teacher_time_pool is not None:
        teacher_config = config["numerical_teacher"]
        assert isinstance(teacher_config, Mapping)
        with h5py.File(base.data.source_h5, "r", swmr=True) as source_handle:
            source_manifest_sha256 = str(
                source_handle.attrs.get("manifest_sha256", "")
            )
        numerical_teacher_cache = NumericalTeacherCache(
            str(teacher_config["cache_h5"]),
            expected_source_manifest_sha256=source_manifest_sha256,
            expected_sample_ids=train_sample_ids,
        )
        if numerical_teacher_cache.time_indices != teacher_time_pool:
            raise ValueError("numerical teacher config and cache time pools differ")
    coverage = np.zeros(
        (base.data.expected_train_records, len(manifest.time_s)), dtype=bool
    )
    if start_epoch:
        coverage = _load_coverage(
            root / "coverage" / f"epoch_{start_epoch:04d}.npz",
            records=base.data.expected_train_records,
            times=len(manifest.time_s),
            epoch=start_epoch,
            digest=schedule_sha,
        )

    macros_per_epoch = audits[0].macro_count
    ranges = epoch_step_ranges(
        total_macros=len(complete_schedule), macros_per_epoch=macros_per_epoch
    )
    pilot_reports = (
        _load_pilot_reports(root / "metrics.jsonl", completed_epochs=start_epoch)
        if args.pilot
        else []
    )
    best: dict[str, object] | None = None
    if bool(ddp["is_main"]) and (root / "best.json").exists():
        best = json.loads((root / "best.json").read_text())
    control_state_path = root / "epoch_validation_control.json"
    control_history_path = root / "epoch_validation_control.jsonl"
    accepted_score = float("nan")
    control_lr_multiplier = 1.0
    epoch_attempt = 1
    if bool(epoch_control["enabled"]):
        control_evaluation_split = str(epoch_control["evaluation_split"])
        control_record_count = (
            base.data.expected_train_records
            if control_evaluation_split == "train"
            else base.data.expected_validation_records
        )
        if control_state_path.exists():
            control_state = json.loads(control_state_path.read_text())
            if int(control_state.get("accepted_epoch", -1)) != int(start_epoch):
                raise ValueError(
                    "epoch validation control state does not match the latest checkpoint"
                )
            if str(control_state.get("evaluation_split", "validation")) != control_evaluation_split:
                raise ValueError(
                    "epoch validation control state evaluation split changed"
                )
            accepted_score = float(control_state["accepted_score"])
            control_lr_multiplier = float(
                control_state.get("learning_rate_multiplier", 1.0)
            )
            epoch_attempt = int(control_state.get("next_attempt", 1))
            if not math.isfinite(accepted_score) or epoch_attempt <= 0:
                raise ValueError("epoch validation control state is invalid")
        elif start_epoch:
            raise ValueError("accepted checkpoint lacks epoch validation control state")
        else:
            (
                baseline_scope,
                baseline_indices,
                baseline_time_policy,
                baseline_frames,
            ) = epoch_gate_validation_plan(
                config,
                validation_records=control_record_count,
            )
            if bool(ddp["is_main"]):
                baseline_metrics = _evaluate(
                    model,
                    base,
                    manifest,
                    normalizer,
                    device,
                    config,
                    baseline_indices,
                    epoch_offset=0,
                    split=control_evaluation_split,
                    time_policy=baseline_time_policy,
                    frames_per_record=baseline_frames,
                )
                accepted_score = float(
                    baseline_metrics[str(epoch_control["metric"])]
                )
            else:
                baseline_metrics = {}
            accepted_score = distributed_broadcast_float(
                accepted_score, device=device, ctx=ddp
            )
            if not math.isfinite(accepted_score):
                raise FloatingPointError("initial epoch validation baseline is non-finite")
            baseline_checkpoint = root / "checkpoints" / "epoch_0000.pt"
            if bool(ddp["is_main"]):
                save_checkpoint_atomic(
                    baseline_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    epoch=0,
                    global_step=0,
                    manifest_digest=manifest.digest,
                    config_digest=identity["run_digest"],
                    metrics={control_evaluation_split: accepted_score},
                )
                _atomic_hardlink(baseline_checkpoint, latest)
                control_state = {
                    "schema": "saved_time_epoch_validation_control_v1",
                    "accepted_epoch": 0,
                    "accepted_score": accepted_score,
                    "learning_rate_multiplier": 1.0,
                    "next_attempt": 1,
                    "metric": str(epoch_control["metric"]),
                    "evaluation_split": control_evaluation_split,
                    "validation_scope": baseline_scope,
                    "validation_records": len(baseline_indices),
                    "validation_frames_per_record": baseline_frames,
                    "checkpoint": str(baseline_checkpoint),
                }
                _atomic_json(control_state, control_state_path)
                _append_jsonl(
                    control_history_path,
                    {
                        "event": "validation_baseline",
                        "epoch": 0,
                        "score": accepted_score,
                        "metrics": baseline_metrics,
                        "evaluation_split": control_evaluation_split,
                        "validation_scope": baseline_scope,
                        "checkpoint": str(baseline_checkpoint),
                    },
                )
                print(
                    json.dumps(
                        {
                            "event": "validation_baseline",
                            "epoch": 0,
                            "score": accepted_score,
                            "validation_scope": baseline_scope,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            distributed_barrier(ddp)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    epoch_index = start_epoch
    while epoch_index < run_epochs:
        epoch_number = epoch_index + 1
        coverage_before_epoch = coverage.copy()
        if config.get("band_limited_adapter"):
            stage = configure_band_adapter_stage(model, epoch=epoch_number)
        elif config.get("pin_trainable_prefixes"):
            # Staged-adapter warm-start: hold a fixed trainable set for every epoch so
            # the near-frozen parent is not progressively unfrozen (which both breaks
            # the "develop the zero-init operator on a stable optimum" premise and, with
            # the temporal operator's recompute added, OOMs the 24 GiB card once the
            # geometry path retains activations for backward).
            stage = configure_pinned_stage(
                model, prefixes=tuple(config["pin_trainable_prefixes"])
            )
        elif config.get("family_experts"):
            stage = configure_family_expert_stage(
                model,
                epoch=family_expert_stage_epoch(config, epoch=epoch_number),
                head_only_epochs=int(config["family_experts"]["head_only_epochs"]),
                dense_unfreeze_epoch=(
                    None
                    if config["family_experts"].get("dense_unfreeze_epoch") is None
                    else int(config["family_experts"]["dense_unfreeze_epoch"])
                ),
                shared_unfreeze_epoch=(
                    None
                    if config["family_experts"].get("shared_unfreeze_epoch") is None
                    else int(config["family_experts"]["shared_unfreeze_epoch"])
                ),
                geometry_unfreeze_epoch=(
                    None
                    if config["family_experts"].get("geometry_unfreeze_epoch") is None
                    else int(config["family_experts"]["geometry_unfreeze_epoch"])
                ),
                backbone_unfreeze_epoch=(
                    None
                    if config["family_experts"].get("backbone_unfreeze_epoch") is None
                    else int(config["family_experts"]["backbone_unfreeze_epoch"])
                ),
            )
        elif config.get("residual_recovery"):
            inherited_stage_epoch = recovery_stage_epoch(config, epoch=epoch_number)
            stage = configure_recovery_stage(
                model,
                epoch=inherited_stage_epoch,
                decoder_only_epochs=int(
                    config["residual_recovery"].get("decoder_only_epochs", 2)
                ),
            )
        else:
            stage = configure_trainable_stage(model, epoch=epoch_number)
        physical_microbatch = microbatch_records_for_epoch(config, epoch=epoch_number)
        lr_factor = _set_epoch_learning_rates(
            optimizer,
            config,
            epoch_index=epoch_index,
            total_epochs=registered_epochs,
            control_multiplier=control_lr_multiplier,
        )
        start, stop = ranges[epoch_index]
        epoch_specs = complete_schedule[start:stop]
        local_epoch_specs = (
            ddp_update_specs(
                epoch_specs,
                macros_per_update=int(config["macros_per_update"]),
                rank=int(ddp["rank"]),
                world_size=int(ddp["world_size"]),
            )
            if bool(ddp["enabled"])
            else epoch_specs
        )
        dataset = ExactStoredTimeBatchDataset(
            base.data.source_h5,
            manifest,
            split="train",
            schedule=local_epoch_specs,
            query_points=int(config["query_points"]),
            seed=int(config["seed"]),
            time_policy=str(config.get("time_policy", "appearance4")),
            frames_per_record=training_frames_per_record(config),
            travel_time_h5=config.get("travel_time_h5"),
            time_index_pool=teacher_time_pool,
            bin_frame_budget=_adaptive_frame_budget,
        )
        loader = make_pilot_loader(
            dataset,
            workers=effective_workers,
            prefetch_factor=int(config["prefetch_factor"]),
            pin_memory=True,
        )
        iterator = iter(loader)
        model.train()
        train_model.train()
        torch.cuda.reset_peak_memory_stats()
        epoch_losses: list[float] = []
        last_loss_components: dict[str, float] = {}
        last_gradients: dict[str, float] = {}
        training_gpu: dict[str, float | None] | None = None
        updates = len(epoch_specs) // int(config["macros_per_update"])
        local_macros_per_update = int(config["macros_per_update"]) // int(ddp["world_size"])
        if args.smoke_updates:
            updates = min(updates, int(args.smoke_updates))
        for update_index in range(updates):
            macros = tuple(next(iterator) for _ in range(local_macros_per_update))
            batch = merge_pilot_batches(macros)
            last_loss_components = _train_update(
                train_model,
                optimizer,
                batch,
                normalizer,
                device,
                config,
                microbatch_records=physical_microbatch,
                numerical_teacher_cache=numerical_teacher_cache,
            )
            epoch_losses.append(float(last_loss_components["total"]))
            distributed_average_gradients(model, ddp)
            last_gradients = _gradient_report(model, stage.trainable_prefixes)
            differential_gradients = local_differential_gradient_norms(model)
            if differential_gradients:
                gate_gradient = float(
                    differential_gradients["local_differential_gate"]
                )
                feature_gradient = float(
                    differential_gradients["local_differential_features"]
                )
                if not math.isfinite(gate_gradient) or gate_gradient <= 0.0:
                    raise FloatingPointError(
                        "local differential gate gradient is non-finite or zero"
                    )
                if update_index >= 1 and (
                    not math.isfinite(feature_gradient) or feature_gradient <= 0.0
                ):
                    raise FloatingPointError(
                        "local differential feature gradient did not activate"
                    )
                last_gradients.update(differential_gradients)
            coupled_gradients = coupled_2d_gradient_norms(model)
            if coupled_gradients:
                gate_gradient = float(coupled_gradients["coupled_2d_gate"])
                feature_gradient = float(coupled_gradients["coupled_2d_features"])
                if not math.isfinite(gate_gradient) or gate_gradient <= 0.0:
                    raise FloatingPointError(
                        "coupled 2-D gate gradient is non-finite or zero"
                    )
                if update_index >= 1 and (
                    not math.isfinite(feature_gradient) or feature_gradient <= 0.0
                ):
                    raise FloatingPointError(
                        "coupled 2-D feature gradient did not activate"
                    )
                last_gradients.update(coupled_gradients)
            temporal_gradients = temporal_basis_gradient_norms(model)
            if temporal_gradients:
                gate_gradient = float(temporal_gradients["temporal_basis_gate"])
                feature_gradient = float(
                    temporal_gradients["temporal_basis_features"]
                )
                if not math.isfinite(gate_gradient) or gate_gradient <= 0.0:
                    raise FloatingPointError(
                        "temporal-basis gate gradient is non-finite or zero"
                    )
                if update_index >= 1 and (
                    not math.isfinite(feature_gradient) or feature_gradient <= 0.0
                ):
                    raise FloatingPointError(
                        "temporal-basis feature gradient did not activate"
                    )
                last_gradients.update(temporal_gradients)
            gradient_norm, gradient_clipping = clip_trainable_gradients(
                model,
                maximum_norm=float(config["optimizer"]["gradient_clip"]),
                mode=str(config["optimizer"].get("gradient_clip_mode", "global")),
                prefix_limits=config["optimizer"].get(
                    "gradient_clip_prefix_limits"
                ),
                return_report=True,
            )
            if not math.isfinite(gradient_norm):
                raise FloatingPointError("non-finite clipped full-support gradient")
            optimizer.step()
            global_step += 1
            training_gpu = _gpu_snapshot()
            if bool(ddp["is_main"]):
                update_report = build_update_report(
                    epoch=epoch_number,
                    update_index=update_index + 1,
                    updates_per_epoch=updates,
                    global_step=global_step,
                    loss_components=last_loss_components,
                    gradient_norm=gradient_norm,
                    gradient_norms=last_gradients,
                    gradient_clipping=gradient_clipping,
                    learning_rates={
                        str(group["group_name"]): float(group["lr"])
                        for group in optimizer.param_groups
                    },
                    physical_microbatch_records=physical_microbatch,
                    gpu=training_gpu,
                    elapsed_seconds=time.monotonic() - started,
                    attempt=epoch_attempt,
                )
                _append_jsonl(root / "updates.jsonl", update_report)
                print(json.dumps(update_report, sort_keys=True), flush=True)
        del iterator, loader, dataset
        distributed_barrier(ddp)
        if args.smoke_updates:
            used_specs = epoch_specs[: updates * int(config["macros_per_update"])]
        else:
            used_specs = epoch_specs
        _coverage_update(
            coverage,
            used_specs,
            time_s=manifest.time_s,
            source_t0_s=source_t0_s,
            source_f0_hz=source_f0_hz,
            sample_ids=train_sample_ids,
            seed=int(config["seed"]),
            frames_per_appearance=coverage_frames_per_appearance(config),
            time_policy=str(config.get("time_policy", "appearance4")),
            time_index_pool=teacher_time_pool,
        )
        coverage_report = _coverage_report(coverage)
        if not args.smoke_updates and epoch_number >= 30 and coverage_report["minimum_unique_indices"] < 120:
            raise RuntimeError("epoch-30 exact-time coverage gate failed")
        if not args.smoke_updates and epoch_number >= 50 and coverage_report["median_unique_indices"] <= 180:
            raise RuntimeError("epoch-50 exact-time median coverage gate failed")

        metric_evaluation_split = epoch_metric_evaluation_split(
            config,
            epoch_control,
            smoke=bool(args.smoke_updates),
        )
        if args.smoke_updates:
            smoke_record_count = (
                base.data.expected_train_records
                if metric_evaluation_split == "train"
                else base.data.expected_validation_records
            )
            selected_indices = validation_panel_indices(
                validation_records=smoke_record_count,
                panel_records=12,
                epoch=1,
                seed=int(config["seed"]),
            )
            scope = "smoke"
            validation_epoch_offset = 0
        elif args.pilot:
            selected_indices = validation_panel_indices(
                validation_records=base.data.expected_validation_records,
                panel_records=int(config["validation"]["panel_records"]),
                epoch=1,
                seed=int(config["seed"]),
            )
            scope = "pilot_fixed_panel"
            validation_epoch_offset = 0
        elif bool(epoch_control["enabled"]):
            control_evaluation_split = str(epoch_control["evaluation_split"])
            control_record_count = (
                base.data.expected_train_records
                if control_evaluation_split == "train"
                else base.data.expected_validation_records
            )
            (
                scope,
                selected_indices,
                validation_time_policy,
                validation_frames,
            ) = epoch_gate_validation_plan(
                config,
                validation_records=control_record_count,
            )
            validation_epoch_offset = 0
        else:
            scope, selected_indices, validation_time_policy, validation_frames = validation_plan(
                config,
                epoch=epoch_number,
                validation_records=base.data.expected_validation_records,
                stored_time_count=len(manifest.time_s),
            )
            validation_epoch_offset = 0
        if args.smoke_updates or args.pilot:
            validation_time_policy = "validation_fixed"
            validation_frames = int(config["validation"]["frames_per_record"])
        if bool(ddp["is_main"]):
            metrics = _evaluate(
                model,
                base,
                manifest,
                normalizer,
                device,
                config,
                selected_indices,
                epoch_offset=validation_epoch_offset,
                split=metric_evaluation_split,
                time_policy=validation_time_policy,
                frames_per_record=validation_frames,
            )
            score = float(metrics[str(epoch_control["metric"])])
        else:
            metrics = {}
            score = float("nan")
        full_time_audit_metrics = None
        if bool(epoch_control["enabled"]):
            score = distributed_broadcast_float(score, device=device, ctx=ddp)
            improved = validation_score_improved(
                accepted_score,
                score,
                minimum_absolute_improvement=float(
                    epoch_control["minimum_absolute_improvement"]
                ),
            )
            if not improved:
                next_multiplier = backed_off_learning_rate_multiplier(
                    control_lr_multiplier,
                    backoff=float(epoch_control["learning_rate_backoff"]),
                    minimum=float(
                        epoch_control["minimum_learning_rate_multiplier"]
                    ),
                )
                exhausted = epoch_retry_exhausted(
                    attempt=epoch_attempt,
                    maximum_attempts=int(
                        epoch_control["maximum_attempts_per_epoch"]
                    ),
                    current_multiplier=control_lr_multiplier,
                    next_multiplier=next_multiplier,
                )
                rejection = {
                    "event": "epoch_rejected",
                    "epoch": epoch_number,
                    "attempt": epoch_attempt,
                    "accepted_score": accepted_score,
                    "candidate_score": score,
                    "delta": score - accepted_score,
                    "learning_rate_multiplier": control_lr_multiplier,
                    "next_learning_rate_multiplier": next_multiplier,
                    "maximum_attempts_exhausted": exhausted,
                    "evaluation_split": str(epoch_control["evaluation_split"]),
                    "metrics": metrics,
                }
                if bool(ddp["is_main"]):
                    _append_jsonl(control_history_path, rejection)
                    _atomic_json(
                        {
                            "schema": "saved_time_epoch_validation_control_v1",
                            "accepted_epoch": epoch_index,
                            "accepted_score": accepted_score,
                            "learning_rate_multiplier": next_multiplier,
                            "next_attempt": epoch_attempt + 1,
                            "metric": str(epoch_control["metric"]),
                            "evaluation_split": str(
                                epoch_control["evaluation_split"]
                            ),
                            "validation_scope": scope,
                            "validation_records": len(selected_indices),
                            "validation_frames_per_record": validation_frames,
                            "checkpoint": str(latest),
                            "last_rejection": rejection,
                        },
                        control_state_path,
                    )
                    print(json.dumps(rejection, sort_keys=True), flush=True)
                distributed_barrier(ddp)
                if exhausted:
                    raise RuntimeError(
                        "epoch validation did not improve after the configured parameter retries"
                    )
                metadata = load_checkpoint(
                    latest,
                    model=model,
                    optimizer=optimizer,
                    expected_manifest_digest=manifest.digest,
                    expected_config_digest=identity["run_digest"],
                    restore_rng=True,
                    map_location=device,
                )
                global_step = metadata.global_step
                coverage = coverage_before_epoch
                control_lr_multiplier = next_multiplier
                epoch_attempt += 1
                distributed_barrier(ddp)
                continue

            cadence = int(config["validation"]["all_records_every"])
            if cadence <= 0:
                raise ValueError("full-time validation audit cadence must be positive")
            if epoch_number % cadence == 0 and bool(ddp["is_main"]):
                full_time_audit_metrics = _evaluate(
                    model,
                    base,
                    manifest,
                    normalizer,
                    device,
                    config,
                    selected_indices,
                    epoch_offset=0,
                    split=str(epoch_control["evaluation_split"]),
                    time_policy="all_saved",
                    frames_per_record=int(
                        config["validation"]["final_frames_per_record"]
                    ),
                )
        peak_cuda_bytes = distributed_max_int(
            int(torch.cuda.max_memory_allocated()), device=device, ctx=ddp
        )
        maximum_peak = float(config["gate"]["maximum_peak_cuda_gib"]) * 1024**3
        if peak_cuda_bytes > maximum_peak:
            raise RuntimeError("full-support peak CUDA memory exceeded the registered limit")
        checkpoint = root / "checkpoints" / f"epoch_{epoch_number:04d}.pt"
        if bool(ddp["is_main"]):
            save_checkpoint_atomic(
                checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch_number,
                global_step=global_step,
                manifest_digest=manifest.digest,
                config_digest=identity["run_digest"],
                metrics={"validation": score},
            )
            _atomic_hardlink(checkpoint, latest)
            _save_coverage(
                root / "coverage" / f"epoch_{epoch_number:04d}.npz",
                coverage,
                epoch=epoch_number,
                digest=schedule_sha,
            )
            report = {
                "event": "epoch",
                "epoch": epoch_number,
                "attempt": epoch_attempt,
                "global_step": global_step,
                "train_loss": float(np.mean(epoch_losses)),
                "last_update_loss_components": last_loss_components,
                "validation_scope": scope,
                "metrics": metrics,
                "validation_previous_score": (
                    accepted_score if bool(epoch_control["enabled"]) else None
                ),
                "validation_delta": (
                    score - accepted_score if bool(epoch_control["enabled"]) else None
                ),
                "epoch_validation_gate_passed": (
                    True if bool(epoch_control["enabled"]) else None
                ),
                "full_time_audit_metrics": full_time_audit_metrics,
                "coverage": coverage_report,
                "trainable_stage": stage.__dict__,
                "residual_activation": getattr(
                    model, "residual_activation_report", None
                ),
                "physical_microbatch_records": physical_microbatch,
                "gradient_norms": last_gradients,
                "learning_rate_factor": lr_factor,
                "learning_rate_control_multiplier": control_lr_multiplier,
                "learning_rates": {
                    str(group["group_name"]): float(group["lr"])
                    for group in optimizer.param_groups
                },
                "training_gpu": training_gpu,
                "post_validation_gpu": _gpu_snapshot(),
                "peak_cuda_bytes": peak_cuda_bytes,
                "ddp": {
                    "world_size": int(ddp["world_size"]),
                    "local_macros_per_update": local_macros_per_update,
                    "global_macros_per_update": int(config["macros_per_update"]),
                },
                "elapsed_seconds": time.monotonic() - started,
                "checkpoint": str(checkpoint),
            }
            _append_jsonl(root / "metrics.jsonl", report)
            eligible = checkpoint_is_eligible(
                config,
                metrics=metrics,
                validation_scope=scope,
                pilot_or_smoke=bool(args.pilot or args.smoke_updates),
            ) or bool(epoch_control["enabled"])
            if eligible and (best is None or score < float(best["score"])):
                best = {
                    "epoch": epoch_number,
                    "score": score,
                    "metrics": metrics,
                    "checkpoint": str(checkpoint),
                }
                _atomic_hardlink(checkpoint, root / "best.pt")
                _atomic_json(best, root / "best.json")
            if bool(epoch_control["enabled"]):
                accepted_state = {
                    "schema": "saved_time_epoch_validation_control_v1",
                    "accepted_epoch": epoch_number,
                    "accepted_score": score,
                    "previous_score": accepted_score,
                    "improvement": accepted_score - score,
                    "learning_rate_multiplier": control_lr_multiplier,
                    "next_attempt": 1,
                    "metric": str(epoch_control["metric"]),
                    "evaluation_split": str(epoch_control["evaluation_split"]),
                    "validation_scope": scope,
                    "validation_records": len(selected_indices),
                    "validation_frames_per_record": validation_frames,
                    "checkpoint": str(checkpoint),
                }
                _atomic_json(accepted_state, control_state_path)
                _append_jsonl(
                    control_history_path,
                    {
                        "event": "epoch_accepted",
                        "epoch": epoch_number,
                        "attempt": epoch_attempt,
                        "previous_score": accepted_score,
                        "score": score,
                        "improvement": accepted_score - score,
                        "evaluation_split": str(epoch_control["evaluation_split"]),
                        "learning_rate_multiplier": control_lr_multiplier,
                        "checkpoint": str(checkpoint),
                    },
                )
            pruned_checkpoints = prune_epoch_checkpoints(
                root / "checkpoints",
                keep_last=effective_checkpoint_keep_last,
                best_epoch=(int(best["epoch"]) if best is not None else None),
            )
            if pruned_checkpoints:
                print(json.dumps({
                    "event": "checkpoint_prune",
                    "epoch": epoch_number,
                    "keep_last": int(config["checkpoint_keep_last"]),
                    "best_epoch": (int(best["epoch"]) if best is not None else None),
                    "removed_epochs": pruned_checkpoints,
                }, sort_keys=True), flush=True)
            if args.pilot:
                pilot_reports.append(
                    {
                        "score": score,
                        "family": metrics["family_relative_l2"],
                        "improvement_vs_coarse": metrics["relative_improvement_vs_coarse"],
                        "correction_ratio": metrics["correction_to_coarse_l2_ratio"],
                    }
                )
            print(json.dumps(report, sort_keys=True), flush=True)
        if bool(epoch_control["enabled"]):
            accepted_score = score
            epoch_attempt = 1
        distributed_barrier(ddp)
        if args.smoke_updates and epoch_number >= int(args.smoke_epochs):
            if bool(ddp["is_main"]):
                terminal = {
                    "status": "complete",
                    "mode": mode,
                    "global_step": global_step,
                    "best": best,
                    "run_digest": identity["run_digest"],
                }
                _atomic_json(terminal, terminal_path)
            distributed_cleanup(ddp)
            return 0

        epoch_index += 1

    gate = None
    status = "complete"
    if args.pilot:
        gate = pilot_gate_on_main(
            pilot_reports,
            is_main=bool(ddp["is_main"]),
            family_tolerance=float(config["gate"]["family_regression_tolerance"]),
        )
        if gate is not None and not gate["passed"]:
            status = "pilot_gate_failed"
        if bool(ddp["enabled"]):
            status_code = torch.tensor(
                [1 if status == "pilot_gate_failed" else 0],
                dtype=torch.int64,
                device=device,
            )
            dist.broadcast(status_code, src=0)
            status = "pilot_gate_failed" if int(status_code.item()) else "complete"
    terminal = {
        "status": status,
        "mode": mode,
        "global_step": global_step,
        "best": best,
        "pilot_gate": gate,
        "run_digest": identity["run_digest"],
    }
    if bool(ddp["is_main"]):
        _atomic_json(terminal, terminal_path)
        print(json.dumps(terminal, sort_keys=True), flush=True)
    distributed_cleanup(ddp)
    return pilot_terminal_exit_code(
        status,
        external_evidence_gate=bool(
            config.get("gate", {}).get("external_evidence_gate", False)
        ),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-updates", type=int, default=0)
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--keep-all-checkpoints",
        action="store_true",
        help="retain every epoch_NNNN.pt snapshot without changing training identity",
    )
    parser.add_argument(
        "--workers-override",
        type=int,
        default=None,
        help=(
            "runtime-only DataLoader workers per rank; does not alter the "
            "registered sampling schedule or checkpoint identity"
        ),
    )
    args = parser.parse_args(argv)
    if args.pilot and args.smoke_updates:
        raise ValueError("pilot and smoke modes are mutually exclusive")
    config = yaml.safe_load(Path(args.config).read_text())
    try:
        return _run(config, args)
    except Exception as error:
        artifact = config.get("artifact_dir") if isinstance(config, Mapping) else None
        if artifact and not args.audit_only:
            mode = "smoke" if args.smoke_updates else "pilot" if args.pilot else "run"
            _atomic_json(
                {"status": "failed", "error_type": type(error).__name__, "error": str(error)},
                Path(str(artifact)) / mode / "terminal.json",
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "activate_band_adapter_output_if_requested",
    "band_adapter_identity_report",
    "band_adapter_missing_prefixes",
    "backed_off_learning_rate_multiplier",
    "backward_with_isolated_family_weighting",
    "build_update_report",
    "checkpoint_is_eligible",
    "coverage_frames_per_appearance",
    "epoch_gate_validation_plan",
    "evaluation_time_selector_seed",
    "epoch_metric_evaluation_split",
    "epoch_retry_exhausted",
    "epoch_step_ranges",
    "family_expert_missing_prefixes",
    "family_expert_stage_epoch",
    "family_expert_identity_report",
    "microbatch_records_for_epoch",
    "training_dense_time_block",
    "training_frames_per_record",
    "training_loss_time_slices",
    "velocity_fields_per_record",
    "parent_checkpoint_expected_manifest_digest",
    "parent_manifest_transfer_metadata",
    "pilot_gate",
    "recovery_time_indices",
    "resolve_epoch_validation_control",
    "run_epochs_for_mode",
    "restore_parent_optimizer_state",
    "validation_panel_indices",
    "validation_plan",
    "validation_score_improved",
    "validation_scope",
]
