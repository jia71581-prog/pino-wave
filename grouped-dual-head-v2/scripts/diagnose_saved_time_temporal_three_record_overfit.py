#!/usr/bin/env python3
"""Test whether the temporal operator can memorize one record per medium family."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.training.checkpoint import (
    load_checkpoint,
    save_checkpoint_atomic,
)
from saved_time_phase_operator_v4.data import (
    ExactStoredTimeBatchDataset,
    split_pilot_batch,
)
from saved_time_phase_operator_v4.band_adapter import (
    build_band_adapter_adamw,
    configure_band_adapter_stage,
    registered_band_adapter_learning_rates,
    registered_high_band_mask,
)
from saved_time_phase_operator_v4.full_support import (
    FullSupportStepSpec,
    build_staged_adamw,
    configure_family_expert_stage,
    configure_recovery_stage,
)
from saved_time_phase_operator_v4.family_gradients import (
    homogeneous_family_scale,
    inverse_norm_family_weights,
)
from saved_time_phase_operator_v4.losses import (
    apply_hard_causality,
    source_causality_onset_s,
)
from saved_time_phase_operator_v4.streaming_metrics import (
    ExactWavefieldMetricAccumulator,
)
from saved_time_phase_operator_v4.evaluation import sha256_file
from scripts.refine_saved_time_v4_lbfgs import _append_jsonl, _gpu_snapshot
from scripts.train_grouped_v3_pilot import _to_device, load_normalizer
from scripts.train_saved_time_v4_full_support import (
    _atomic_hardlink,
    _atomic_json,
    _digest,
    _gradient_report,
    _load_context,
    _load_parent_model,
    _train_update,
    clip_trainable_gradients,
    temporal_basis_gradient_norms,
)


FAMILIES = ("uniform", "layered", "marmousi")


def _retarget_batch_to_scattering(batch, provider):
    """A+1: subtract the fixed physical background P_bg from a batch's dense targets so
    the shared trainer learns the scattering residual wf - P_bg. encode_pressure is
    linear, so training on encode(wf - P_bg) == training the model toward encode(scat)
    with the coarse field carrying encode(P_bg). Frame-aligned by sample_id + left_index
    (exact saved-time index; left==right for the exact dataset)."""
    if provider is None:
        return batch
    pbg = provider.physical(
        batch.sample_id, batch.left_index,
        device=batch.dense_target_physical.device,
        dtype=batch.dense_target_physical.dtype,
    )
    if pbg.shape != batch.dense_target_physical.shape:
        raise ValueError(
            f"background shape {tuple(pbg.shape)} != target "
            f"{tuple(batch.dense_target_physical.shape)}"
        )
    import dataclasses
    return dataclasses.replace(
        batch, dense_target_physical=batch.dense_target_physical - pbg
    )
REMOTE_HOME_PREFIX = "/root/autodl-tmp/home/jiayh"
LOCAL_HOME_PREFIX = "/home/jiayh"


def localize_remote_paths(value):
    """Map recursively synchronized remote artifact paths onto this host."""

    if isinstance(value, str):
        return value.replace(REMOTE_HOME_PREFIX, LOCAL_HOME_PREFIX)
    if isinstance(value, list):
        return [localize_remote_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(localize_remote_paths(item) for item in value)
    if isinstance(value, dict):
        return {key: localize_remote_paths(item) for key, item in value.items()}
    return value


def select_one_index_per_family(records: Sequence[object], *, split: str) -> tuple[int, ...]:
    """Return deterministic split-relative indices in uniform/layered/marmousi order."""

    selected_records = tuple(record for record in records if str(record.split) == str(split))
    result: list[int] = []
    for family in FAMILIES:
        match = next(
            (
                index
                for index, record in enumerate(selected_records)
                if str(record.medium_type) == family
            ),
            None,
        )
        if match is None:
            raise ValueError(f"split {split!r} has no {family} record")
        result.append(int(match))
    return tuple(result)


def build_repeat_schedule(
    record_indices: Sequence[int], *, updates: int
) -> tuple[FullSupportStepSpec, ...]:
    """Repeat a fixed record triplet while advancing exact-time appearances."""

    selected = tuple(int(value) for value in record_indices)
    count = int(updates)
    if len(selected) != len(FAMILIES) or count <= 0:
        raise ValueError("overfit schedule requires three records and positive updates")
    return tuple(
        FullSupportStepSpec(
            step=900_000 + update,
            epoch=0,
            record_indices=selected,
            appearance_indices=(update,) * len(selected),
        )
        for update in range(count)
    )


def resolve_training_sampling(
    policy: str, *, training_frames: int, validation_frames: int
) -> tuple[str, int]:
    """Resolve the diagnostic's one-variable time-sampling intervention."""

    normalized = str(policy)
    if normalized not in {"appearance16", "validation_fixed"}:
        raise ValueError(f"unsupported training time policy: {normalized!r}")
    if int(training_frames) <= 0 or int(validation_frames) <= 0:
        raise ValueError("training and validation frame counts must be positive")
    if normalized == "validation_fixed":
        return normalized, int(validation_frames)
    return normalized, int(training_frames)


def resolve_microbatch_records(value: int) -> int:
    """Validate the physical record count used by the diagnostic."""

    result = int(value)
    if result <= 0:
        raise ValueError("microbatch record count must be positive")
    return result


def resolve_warmstart_paths(
    checkpoint: str | Path | None,
    identity: str | Path | None,
) -> tuple[Path | None, Path | None]:
    """Require a strict checkpoint/run-identity pair for adapter continuation."""

    if checkpoint is None and identity is None:
        return None, None
    if checkpoint is None or identity is None:
        raise ValueError("warmstart checkpoint and identity must be provided together")
    checkpoint_path = Path(checkpoint).resolve()
    identity_path = Path(identity).resolve()
    if not checkpoint_path.is_file() or not identity_path.is_file():
        raise FileNotFoundError("warmstart checkpoint or identity is unavailable")
    payload = json.loads(identity_path.read_text())
    if not isinstance(payload, Mapping) or not payload.get("run_digest"):
        raise ValueError("warmstart identity lacks a run digest")
    return checkpoint_path, identity_path


def resolve_family_expert_overfit_stage(
    family_experts: Mapping[str, object],
) -> dict[str, int | None]:
    """Select the most expressive registered family-expert stage for memorization."""

    head_only_epochs = int(family_experts["head_only_epochs"])
    dense_value = family_experts.get("dense_unfreeze_epoch")
    dense_unfreeze_epoch = None if dense_value is None else int(dense_value)
    result: dict[str, int | None] = {
        "epoch": head_only_epochs + 1,
        "head_only_epochs": head_only_epochs,
        "dense_unfreeze_epoch": dense_unfreeze_epoch,
    }
    registered_epochs = [
        value for value in (dense_unfreeze_epoch,) if value is not None
    ]
    for name in (
        "shared_unfreeze_epoch",
        "geometry_unfreeze_epoch",
        "backbone_unfreeze_epoch",
    ):
        value = family_experts.get(name)
        if value is not None:
            resolved = int(value)
            result[name] = resolved
            registered_epochs.append(resolved)
    result["epoch"] = max((head_only_epochs + 1, *registered_epochs))
    return result


def configure_overfit_trainable_stage(model, config: Mapping[str, object]):
    """Select the most expressive registered stage, with adapter isolation first."""

    if config.get("band_limited_adapter") is not None:
        stage = configure_band_adapter_stage(model, epoch=1)
        if stage.trainable_prefixes != ("dense_decoder.band_limited_adapter",):
            raise ValueError("band-adapter overfit stage is invalid")
        return stage
    if config.get("family_experts") is not None:
        family_experts = config["family_experts"]
        if not isinstance(family_experts, Mapping):
            raise ValueError("family-expert overfit configuration is invalid")
        stage_kwargs = resolve_family_expert_overfit_stage(family_experts)
        stage = configure_family_expert_stage(model, **stage_kwargs)
        if stage_kwargs.get("backbone_unfreeze_epoch") is not None:
            expected_prefixes = (
                "dense_decoder",
                "source_encoder",
                "fusion",
                "coordinate_encoder",
                "travel_branch",
                "medium_encoder",
            )
        elif stage_kwargs.get("geometry_unfreeze_epoch") is not None:
            expected_prefixes = (
                "dense_decoder",
                "source_encoder",
                "fusion",
                "coordinate_encoder",
                "travel_branch",
            )
        elif stage_kwargs.get("shared_unfreeze_epoch") is not None:
            expected_prefixes = ("dense_decoder", "source_encoder", "fusion")
        elif stage_kwargs["dense_unfreeze_epoch"] is not None:
            expected_prefixes = ("dense_decoder",)
        else:
            expected_prefixes = (
                "dense_decoder.family_experts",
                "dense_decoder.temporal_basis",
            )
        if stage.trainable_prefixes != expected_prefixes:
            raise ValueError("family-expert overfit stage is invalid")
        return stage
    stage = configure_recovery_stage(model, epoch=1, decoder_only_epochs=10_000)
    if stage.trainable_prefixes != ("dense_decoder",):
        raise ValueError("overfit diagnostic must train only the dense decoder")
    return stage


def apply_diagnostic_overrides(
    config: Mapping[str, object],
    *,
    dense_learning_rate: float,
    temporal_basis_learning_rate: float,
    delta_weight: float,
    dense_gradient_clip_limit: float | None = None,
    auxiliary_loss_scale: float = 1.0,
    equal_family_weights: bool = False,
    band_adapter_feature_learning_rate: float | None = None,
    band_adapter_output_learning_rate: float | None = None,
) -> dict[str, object]:
    """Apply one-variable optimizer/loss interventions to any candidate source."""

    dense_lr = float(dense_learning_rate)
    temporal_lr = float(temporal_basis_learning_rate)
    delta = float(delta_weight)
    auxiliary_scale = float(auxiliary_loss_scale)
    if not math.isfinite(dense_lr) or dense_lr <= 0.0:
        raise ValueError("dense learning rate must be positive and finite")
    if not math.isfinite(temporal_lr) or temporal_lr <= 0.0:
        raise ValueError("temporal-basis learning rate must be positive and finite")
    if not math.isfinite(delta) or delta < 0.0:
        raise ValueError("delta loss weight must be finite and nonnegative")
    if not math.isfinite(auxiliary_scale) or auxiliary_scale < 0.0:
        raise ValueError("auxiliary loss scale must be finite and nonnegative")

    result = dict(config)
    loss = dict(result["loss"])
    loss["delta"] = delta
    loss["delta_energy_floor_fraction"] = 0.1
    for name in ("temporal_difference", "spatial_gradient", "spectrum"):
        if name in loss:
            loss[name] = float(loss[name]) * auxiliary_scale
    result["loss"] = loss

    if bool(equal_family_weights):
        result["family_gradient_weights"] = {
            family: 1.0 for family in FAMILIES
        }

    optimizer = dict(result["optimizer"])
    optimizer["dense_learning_rate"] = dense_lr
    optimizer["temporal_basis_learning_rate"] = temporal_lr
    optimizer["weight_decay"] = 0.0
    feature_lr = band_adapter_feature_learning_rate
    output_lr = band_adapter_output_learning_rate
    if (feature_lr is None) != (output_lr is None):
        raise ValueError("band adapter learning rates must be provided together")
    if feature_lr is not None and output_lr is not None:
        split_rates = (float(feature_lr), float(output_lr))
        if any(
            not math.isfinite(value) or value <= 0.0 for value in split_rates
        ):
            raise ValueError(
                "band adapter learning rates must be positive and finite"
            )
        optimizer["band_adapter_feature_learning_rate"] = split_rates[0]
        optimizer["band_adapter_output_learning_rate"] = split_rates[1]
    if dense_gradient_clip_limit is not None:
        clip = float(dense_gradient_clip_limit)
        if not math.isfinite(clip) or clip <= 0.0:
            raise ValueError("dense gradient clip limit must be positive and finite")
        if str(optimizer.get("gradient_clip_mode", "global")) != "prefix_limits":
            raise ValueError("dense gradient clip override requires prefix_limits mode")
        limits = dict(optimizer.get("gradient_clip_prefix_limits", {}))
        limits["dense_decoder"] = clip
        optimizer["gradient_clip_prefix_limits"] = limits
    result["optimizer"] = optimizer
    return result


def _candidate_config(
    checkpoint_identity: Path,
    checkpoint: Path,
    artifact_dir: Path,
    *,
    dense_learning_rate: float,
    temporal_basis_learning_rate: float,
    delta_weight: float = 0.5,
    family_gradient_weights: Mapping[str, float] | None = None,
    auxiliary_loss_scale: float = 1.0,
    equal_family_weights: bool = False,
) -> dict[str, object]:
    identity = json.loads(checkpoint_identity.read_text())
    config = localize_remote_paths(identity["config"])
    config["parent_checkpoint"] = str(checkpoint.resolve())
    config["parent_checkpoint_identity"] = str(checkpoint_identity.resolve())
    config["artifact_dir"] = str(artifact_dir.resolve())
    transfer = dict(config.get("checkpoint_transfer", {}))
    for permission in (
        "allow_spectral_mode_expansion",
        "allow_new_coupled_2d_parameters",
        "allow_new_local_differential_parameters",
        "allow_new_temporal_basis_parameters",
    ):
        transfer.pop(permission, None)
    transfer["parent_residual_already_active"] = True
    transfer["parent_optimizer_state"] = False
    config["checkpoint_transfer"] = transfer
    recovery = dict(config.get("residual_recovery", {}))
    recovery.update(
        {
            "activation_mode": "preserve",
            "stage_epoch_offset": 0,
            "decoder_only_epochs": 10_000,
            "absorb_temporal_basis_gate": True,
        }
    )
    config["residual_recovery"] = recovery
    config = apply_diagnostic_overrides(
        config,
        dense_learning_rate=dense_learning_rate,
        temporal_basis_learning_rate=temporal_basis_learning_rate,
        delta_weight=delta_weight,
        auxiliary_loss_scale=auxiliary_loss_scale,
        equal_family_weights=equal_family_weights,
    )
    if family_gradient_weights is not None and not bool(equal_family_weights):
        weights = {
            family: float(family_gradient_weights[family]) for family in FAMILIES
        }
        for family in FAMILIES:
            homogeneous_family_scale((family,), weights)
        config["family_gradient_weights"] = weights
    return config


def _dataset(
    config,
    base,
    manifest,
    indices,
    *,
    split: str,
    schedule,
    time_policy: str,
    frames_per_record: int,
):
    return ExactStoredTimeBatchDataset(
        base.data.source_h5,
        manifest,
        split=str(split),
        schedule=tuple(schedule),
        query_points=1,
        seed=int(config["seed"]) + 17_171,
        time_policy=str(time_policy),
        frames_per_record=int(frames_per_record),
        travel_time_h5=config.get("travel_time_h5"),
        allow_travel_source_path_mismatch=bool(
            config.get("allow_travel_source_path_mismatch", False)
        ),
    )


@torch.inference_mode()
def _evaluate_triplet(
    model,
    base,
    manifest,
    normalizer,
    device,
    config,
    indices,
    *,
    split: str,
    time_policy: str,
    frames_per_record: int,
    background_provider=None,
    target_provider=None,
    evaluation_macro_records: int | None = None,
    apply_correction: bool = True,
    evaluation_time_block: int = 1,
):
    selected = tuple(int(value) for value in indices)
    macro = len(selected) if evaluation_macro_records is None else int(
        evaluation_macro_records
    )
    if macro <= 0:
        raise ValueError("evaluation_macro_records must be positive")
    schedule = tuple(
        FullSupportStepSpec(
            step=990_000 + step,
            epoch=0,
            record_indices=selected[start : start + macro],
            appearance_indices=(0,) * len(selected[start : start + macro]),
        )
        for step, start in enumerate(range(0, len(selected), macro))
    )
    dataset = _dataset(
        config,
        base,
        manifest,
        indices,
        split=split,
        schedule=schedule,
        time_policy=time_policy,
        frames_per_record=frames_per_record,
    )
    predicted_metrics = ExactWavefieldMetricAccumulator(
        energy_floor_fraction=float(config.get("energy_floor_fraction", 0.01)),
        require_unique=True,
        stored_time_count=len(manifest.time_s),
    )
    reference_metrics = ExactWavefieldMetricAccumulator(
        energy_floor_fraction=float(config.get("energy_floor_fraction", 0.01)),
        require_unique=True,
        stored_time_count=len(manifest.time_s),
    )
    correction_square = 0.0
    reference_square = 0.0
    scattering_error_square = 0.0
    scattering_target_square = 0.0
    scattering_prediction_square = 0.0
    high_delta_square = 0.0
    high_anchor_square = 0.0
    adapter_enabled = (
        getattr(model.dense_decoder, "band_limited_adapter", None) is not None
    )
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
            "Helmholtz background conditioning evaluation requires P_bg"
        )
    model.eval()
    for batch_index in range(len(dataset)):
        batch = dataset[batch_index]
        for micro in split_pilot_batch(batch, microbatch_records=1):
            tensors = _to_device(micro, device)
            source = tensors["source_parameters"]
            background_normalized = None
            if background_conditioning_active:
                background_physical = background_provider.full_physical(
                    micro.sample_id,
                    device=device,
                    dtype=tensors["dense_target_physical"].dtype,
                )
                background_normalized = normalizer.encode_pressure(
                    background_physical, source[:, 4]
                )
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
            if not bool(apply_correction):
                prediction = model.dense_normalized(
                    prepared,
                    tensors["requested_time_s"],
                    dense_grid=dense_grid,
                    time_block=int(evaluation_time_block),
                    apply_correction=False,
                    background_normalized=background_normalized,
                )
                reference = prediction
            elif adapter_enabled:
                anchor, increment, _ = (
                    model.dense_normalized_with_anchor_increment_and_routing(
                        prepared,
                        tensors["requested_time_s"],
                        dense_grid=dense_grid,
                        time_block=int(evaluation_time_block),
                        background_normalized=background_normalized,
                    )
                )
                prediction = anchor + increment
                reference = anchor
            else:
                prediction, reference = model.dense_normalized_with_coarse(
                    prepared,
                    tensors["requested_time_s"],
                    dense_grid=dense_grid,
                    time_block=int(evaluation_time_block),
                    background_normalized=background_normalized,
                )
            target_physical = tensors["dense_target_physical"]
            if target_provider is not None:
                target_physical = target_provider.physical(
                    micro.sample_id,
                    micro.left_index,
                    device=device,
                    dtype=target_physical.dtype,
                )
            target = normalizer.encode_pressure(target_physical, source[:, 4])
            if background_provider is not None:
                # A+1: the model predicts the normalized scattering residual; the
                # full field = residual + encode(P_bg). Add the (fixed physical)
                # background back to BOTH prediction and reference for full-field
                # scoring. encode_pressure is linear so this is exact.
                pbg_phys = background_provider.physical(
                    micro.sample_id, micro.left_index, device=device,
                    dtype=tensors["dense_target_physical"].dtype,
                )
                pbg_norm = normalizer.encode_pressure(pbg_phys, source[:, 4])
                predicted_scattering = prediction
                true_scattering = target - pbg_norm
                prediction = prediction + pbg_norm
                reference = reference + pbg_norm
            if bool(config["loss"].get("hard_causality", False)):
                onset = source_causality_onset_s(
                    source,
                    lead_cycles=float(
                        config["loss"].get("hard_causality_lead_cycles", 0.0)
                    ),
                )
                prediction = apply_hard_causality(
                    prediction, tensors["requested_time_s"], onset
                )
                reference = apply_hard_causality(
                    reference, tensors["requested_time_s"], onset
                )
                if background_provider is not None:
                    predicted_scattering = apply_hard_causality(
                        predicted_scattering, tensors["requested_time_s"], onset
                    )
                    true_scattering = apply_hard_causality(
                        true_scattering, tensors["requested_time_s"], onset
                    )
            if background_provider is not None:
                scattering_error_square += float(
                    (predicted_scattering.float() - true_scattering.float())
                    .square()
                    .sum()
                )
                scattering_target_square += float(
                    true_scattering.float().square().sum()
                )
                scattering_prediction_square += float(
                    predicted_scattering.float().square().sum()
                )
            metric_onset = torch.clamp(
                source[:, 3] - source[:, 2].reciprocal(),
                min=float(manifest.time_s[0]),
            )
            onset_indices = torch.searchsorted(
                torch.as_tensor(manifest.time_s, device=device),
                metric_onset.contiguous(),
            ).cpu().tolist()
            common = {
                "families": micro.medium_type,
                "group_ids": micro.group_id,
                "sample_ids": micro.sample_id,
                "time_indices": micro.left_index,
                "source_onset_indices": onset_indices,
            }
            predicted_metrics.update(prediction, target, **common)
            reference_metrics.update(reference, target, **common)
            correction_square += float(
                (prediction.float() - reference.float()).square().sum()
            )
            reference_square += float(reference.float().square().sum())
            if adapter_enabled:
                height, width = prediction.shape[-2:]
                high = registered_high_band_mask(height, width, prediction.device)
                prediction_fft = torch.fft.rfft2(prediction.float(), norm="ortho")
                anchor_fft = torch.fft.rfft2(reference.float(), norm="ortho")
                high_delta_square += float(
                    (prediction_fft[..., high] - anchor_fft[..., high])
                    .abs()
                    .square()
                    .sum()
                )
                high_anchor_square += float(
                    anchor_fft[..., high].abs().square().sum()
                )
    metrics = predicted_metrics.finalize()
    reference = reference_metrics.finalize()
    metrics["coarse_metrics"] = reference
    metrics["reference_kind"] = "anchor" if adapter_enabled else "coarse"
    if adapter_enabled:
        metrics["anchor_metrics"] = reference
        metrics["candidate_high_band_relative_l2"] = float(
            metrics["spectrum_relative_l2"]["high"]
        )
        metrics["anchor_high_band_relative_l2"] = float(
            reference["spectrum_relative_l2"]["high"]
        )
        metrics["high_band_anchor_delta"] = math.sqrt(high_delta_square) / math.sqrt(
            max(high_anchor_square, 1.0e-16)
        )
    metrics["correction_to_coarse_l2_ratio"] = math.sqrt(
        correction_square
    ) / math.sqrt(max(reference_square, 1.0e-16))
    if background_provider is not None:
        scattering_denominator = math.sqrt(max(scattering_target_square, 1.0e-16))
        metrics["scattering_relative_l2"] = (
            math.sqrt(scattering_error_square) / scattering_denominator
        )
        metrics["predicted_to_true_scattering_l2_ratio"] = (
            math.sqrt(scattering_prediction_square) / scattering_denominator
        )
    reference_score = float(reference["aggregate_relative_l2"])
    metrics["relative_improvement_vs_coarse"] = (
        reference_score - float(metrics["aggregate_relative_l2"])
    ) / max(reference_score, 1.0e-16)
    return metrics


def _meets_target(metrics: Mapping[str, object]) -> bool:
    families = metrics.get("family_relative_l2", {})
    return float(metrics["aggregate_relative_l2"]) < 0.10 and all(
        float(families.get(family, math.inf)) < 0.12 for family in FAMILIES
    )


def save_overfit_checkpoint(
    root: str | Path,
    *,
    model,
    update: int,
    manifest_digest: str,
    config_digest: str,
    aggregate_relative_l2: float,
    extra_metrics: Mapping[str, float] | None = None,
) -> Path:
    """Persist one identity-bound diagnostic state and advance latest.pt."""

    checkpoint_path = Path(root) / "checkpoints" / f"update_{int(update):04d}.pt"
    checkpoint_metrics = {
        "triplet_aggregate_relative_l2": float(aggregate_relative_l2)
    }
    if extra_metrics is not None:
        checkpoint_metrics.update(
            {str(name): float(value) for name, value in extra_metrics.items()}
        )
    save_checkpoint_atomic(
        checkpoint_path,
        model=model,
        optimizer=None,
        epoch=0,
        global_step=int(update),
        manifest_digest=manifest_digest,
        config_digest=config_digest,
        metrics=checkpoint_metrics,
    )
    _atomic_hardlink(checkpoint_path, Path(root) / "latest.pt")
    return checkpoint_path


def restore_best_overfit_checkpoint(
    checkpoint: str | Path,
    *,
    model,
    manifest_digest: str,
    config_digest: str,
    map_location: str | torch.device,
):
    """Restore the selected diagnostic state before expensive all-time evaluation."""

    return load_checkpoint(
        checkpoint,
        model=model,
        optimizer=None,
        expected_manifest_digest=manifest_digest,
        expected_config_digest=config_digest,
        map_location=map_location,
    )


def build_overfit_terminal_report(
    *,
    updates_completed: int,
    baseline_relative_l2: float,
    best_fixed_relative_l2: float,
    best_checkpoint: str | Path,
    all_saved_metrics: Mapping[str, object],
) -> dict[str, object]:
    """Keep stage-local progress distinct from cumulative anchor progress."""

    baseline_score = float(baseline_relative_l2)
    best_score = float(best_fixed_relative_l2)
    can_memorize = _meets_target(all_saved_metrics)
    return {
        "status": "complete",
        "updates_completed": int(updates_completed),
        "best_fixed_aggregate_relative_l2": best_score,
        "initial_relative_l2": baseline_score,
        "final_relative_l2": best_score,
        "relative_reduction": (baseline_score - best_score)
        / max(baseline_score, 1.0e-16),
        "anchor_relative_reduction": float(
            all_saved_metrics["relative_improvement_vs_coarse"]
        ),
        "best_checkpoint": str(best_checkpoint),
        "all_saved_metrics": all_saved_metrics,
        "can_memorize_below_target": can_memorize,
        "interpretation": (
            "capacity_is_sufficient"
            if can_memorize
            else "capacity_or_optimization_is_insufficient"
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-identity")
    parser.add_argument("--candidate-config")
    parser.add_argument("--warmstart-checkpoint")
    parser.add_argument("--warmstart-identity")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--updates", "--overfit-updates", dest="updates", type=int, default=120)
    parser.add_argument("--evaluate-every", type=int, default=20)
    parser.add_argument("--validation-frames", type=int, default=32)
    parser.add_argument("--training-frames", type=int, default=16)
    parser.add_argument("--microbatch-records", type=int, default=3)
    parser.add_argument(
        "--training-time-policy",
        choices=("appearance16", "validation_fixed"),
        default="appearance16",
    )
    parser.add_argument("--dense-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--band-adapter-feature-learning-rate", type=float)
    parser.add_argument("--band-adapter-output-learning-rate", type=float)
    parser.add_argument("--delta-weight", type=float, default=0.5)
    parser.add_argument("--dense-gradient-clip-limit", type=float)
    parser.add_argument("--auxiliary-loss-scale", type=float, default=1.0)
    parser.add_argument("--equal-family-weights", action="store_true")
    parser.add_argument("--family-gradient-report")
    parser.add_argument(
        "--temporal-basis-learning-rate", type=float, default=1.0e-5
    )
    parser.add_argument(
        "--background-cache",
        help=(
            "Direction A+1: NumericalTeacherCache-format smoothed-velocity background "
            "P_bg (built by scripts/build_smoothed_background_cache.py). When set, the "
            "model is retargeted onto the scattering residual encode(scat)=encode(wf)-"
            "encode(P_bg); full-field metrics add encode(P_bg) back. encode_pressure is "
            "linear so this is exact."
        ),
    )
    args = parser.parse_args(argv)
    if args.updates <= 0 or args.evaluate_every <= 0:
        raise ValueError("updates, evaluation cadence, and frames must be positive")
    training_time_policy, training_frames = resolve_training_sampling(
        args.training_time_policy,
        training_frames=int(args.training_frames),
        validation_frames=int(args.validation_frames),
    )
    microbatch_records = resolve_microbatch_records(args.microbatch_records)
    background_provider = None
    if args.background_cache:
        from saved_time_phase_operator_v4.background_field import BackgroundFieldProvider
        background_provider = BackgroundFieldProvider(str(args.background_cache))
    warmstart_checkpoint, warmstart_identity = resolve_warmstart_paths(
        args.warmstart_checkpoint,
        args.warmstart_identity,
    )
    root = Path(args.artifact_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    terminal_path = root / "terminal.json"
    if terminal_path.exists():
        print(terminal_path.read_text().strip())
        return 0

    family_gradient_report = (
        None
        if args.family_gradient_report is None
        else Path(args.family_gradient_report).resolve()
    )
    family_gradient_weights = None
    if family_gradient_report is not None:
        evidence = json.loads(family_gradient_report.read_text())
        if evidence.get("schema") != "saved_time_family_gradient_conflict_v1":
            raise ValueError("family gradient report schema is invalid")
        family_gradient_weights = inverse_norm_family_weights(
            evidence.get("gradient_report", {}).get("gradient_norm", {})
        )
    if args.candidate_config is not None:
        if args.checkpoint is not None or args.checkpoint_identity is not None:
            raise ValueError("candidate config and explicit checkpoint inputs are exclusive")
        config = yaml.safe_load(Path(args.candidate_config).read_text())
        if not isinstance(config, dict) or config.get("family_experts") is None:
            raise ValueError("candidate config must enable family experts")
        config["artifact_dir"] = str(root)
        checkpoint = Path(config["parent_checkpoint"]).resolve()
        checkpoint_identity = Path(config["parent_checkpoint_identity"]).resolve()
        config = apply_diagnostic_overrides(
            config,
            dense_learning_rate=float(args.dense_learning_rate),
            temporal_basis_learning_rate=float(args.temporal_basis_learning_rate),
            delta_weight=float(args.delta_weight),
            dense_gradient_clip_limit=args.dense_gradient_clip_limit,
            auxiliary_loss_scale=float(args.auxiliary_loss_scale),
            equal_family_weights=bool(args.equal_family_weights),
            band_adapter_feature_learning_rate=(
                args.band_adapter_feature_learning_rate
            ),
            band_adapter_output_learning_rate=(
                args.band_adapter_output_learning_rate
            ),
        )
    else:
        if args.checkpoint is None or args.checkpoint_identity is None:
            raise ValueError("explicit checkpoint and identity are required")
        checkpoint = Path(args.checkpoint).resolve()
        checkpoint_identity = Path(args.checkpoint_identity).resolve()
        config = _candidate_config(
            checkpoint_identity,
            checkpoint,
            root,
            dense_learning_rate=float(args.dense_learning_rate),
            temporal_basis_learning_rate=float(args.temporal_basis_learning_rate),
            delta_weight=float(args.delta_weight),
            family_gradient_weights=family_gradient_weights,
            auxiliary_loss_scale=float(args.auxiliary_loss_scale),
            equal_family_weights=bool(args.equal_family_weights),
        )
    base, manifest, parent_identity = _load_context(config)
    indices = select_one_index_per_family(manifest.records, split=args.split)
    selected_records = tuple(
        record for record in manifest.records if record.split == args.split
    )
    sample_ids = tuple(selected_records[index].sample_id for index in indices)
    device = torch.device("cuda")
    seed = int(config["seed"]) + 40_401
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    model = _load_parent_model(config, base, manifest, parent_identity, device)
    warmstart_digest = None
    if warmstart_checkpoint is not None and warmstart_identity is not None:
        warmstart_evidence = json.loads(warmstart_identity.read_text())
        warmstart_digest = str(warmstart_evidence["run_digest"])
        load_checkpoint(
            warmstart_checkpoint,
            model=model,
            optimizer=None,
            expected_manifest_digest=manifest.digest,
            expected_config_digest=warmstart_digest,
            map_location=device,
        )
    stage = configure_overfit_trainable_stage(model, config)
    band_adapter_rates = registered_band_adapter_learning_rates(config)
    if band_adapter_rates is None:
        optimizer = build_staged_adamw(
            model,
            dense_lr=float(config["optimizer"]["dense_learning_rate"]),
            geometry_lr=float(config["optimizer"]["geometry_learning_rate"]),
            backbone_lr=float(config["optimizer"]["backbone_learning_rate"]),
            weight_decay=0.0,
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
        )
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
    normalizer = load_normalizer(base, manifest.digest)
    schedule = build_repeat_schedule(indices, updates=int(args.updates))
    train_data = _dataset(
        config,
        base,
        manifest,
        indices,
        split=args.split,
        schedule=schedule,
        time_policy=training_time_policy,
        frames_per_record=training_frames,
    )
    identity = {
        "schema": "saved_time_temporal_three_record_overfit_v1",
        "parent_checkpoint": str(checkpoint),
        "parent_checkpoint_identity": str(checkpoint_identity),
        "warmstart_checkpoint": (
            None if warmstart_checkpoint is None else str(warmstart_checkpoint)
        ),
        "warmstart_checkpoint_sha256": (
            None
            if warmstart_checkpoint is None
            else sha256_file(warmstart_checkpoint)
        ),
        "warmstart_identity": (
            None if warmstart_identity is None else str(warmstart_identity)
        ),
        "warmstart_identity_sha256": (
            None if warmstart_identity is None else sha256_file(warmstart_identity)
        ),
        "warmstart_run_digest": warmstart_digest,
        "manifest_digest": manifest.digest,
        "split": str(args.split),
        "record_indices": indices,
        "sample_ids": sample_ids,
        "families": FAMILIES,
        "updates": int(args.updates),
        "evaluate_every": int(args.evaluate_every),
        "training_time_policy": training_time_policy,
        "training_frames_per_record": training_frames,
        "microbatch_records": microbatch_records,
        "validation_frames_per_record": int(args.validation_frames),
        "dense_learning_rate": float(config["optimizer"]["dense_learning_rate"]),
        "band_adapter_feature_learning_rate": (
            None if band_adapter_rates is None else band_adapter_rates[0]
        ),
        "band_adapter_output_learning_rate": (
            None if band_adapter_rates is None else band_adapter_rates[1]
        ),
        "temporal_basis_learning_rate": float(
            config["optimizer"]["temporal_basis_learning_rate"]
        ),
        "delta_weight": float(config["loss"]["delta"]),
        "dense_gradient_clip_limit": float(
            config["optimizer"].get("gradient_clip_prefix_limits", {}).get(
                "dense_decoder", config["optimizer"]["gradient_clip"]
            )
        ),
        "auxiliary_loss_scale": float(args.auxiliary_loss_scale),
        "equal_family_weights": bool(args.equal_family_weights),
        "family_gradient_report": (
            None if family_gradient_report is None else str(family_gradient_report)
        ),
        "family_gradient_weights": config.get("family_gradient_weights"),
        "gate_absorbed": bool(
            config.get("residual_recovery", {}).get(
                "absorb_temporal_basis_gate", False
            )
        ),
        "one_source_per_record": True,
        "receiver_input": False,
        "wavefield_shape": [201, 201],
        "exact_saved_times_only": True,
        "stage": stage.__dict__,
    }
    identity["run_digest"] = _digest(identity)
    _atomic_json(identity, root / "run_identity.json")

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
    _append_jsonl(
        root / "metrics.jsonl",
        {"event": "baseline", "update": 0, "metrics": baseline},
    )
    baseline_checkpoint = save_overfit_checkpoint(
        root,
        model=model,
        update=0,
        manifest_digest=manifest.digest,
        config_digest=identity["run_digest"],
        aggregate_relative_l2=float(baseline["aggregate_relative_l2"]),
    )
    _atomic_hardlink(baseline_checkpoint, root / "best.pt")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    best_score = float(baseline["aggregate_relative_l2"])
    best_checkpoint = str(baseline_checkpoint)
    last_update = 0
    for update, batch in enumerate(train_data, start=1):
        model.train()
        batch = _retarget_batch_to_scattering(batch, background_provider)
        components = _train_update(
            model,
            optimizer,
            batch,
            normalizer,
            device,
            config,
            microbatch_records=microbatch_records,
        )
        gradients = _gradient_report(model, stage.trainable_prefixes)
        gradients.update(temporal_basis_gradient_norms(model))
        gradient_norm, clipping = clip_trainable_gradients(
            model,
            maximum_norm=float(config["optimizer"]["gradient_clip"]),
            mode=str(config["optimizer"].get("gradient_clip_mode", "global")),
            prefix_limits=config["optimizer"].get("gradient_clip_prefix_limits"),
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
                "gradient_norms": gradients,
                "gradient_clipping": clipping,
                "gpu": _gpu_snapshot(),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        if update % int(args.evaluate_every) and update != int(args.updates):
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
        if score <= best_score:
            best_score = score
            best_checkpoint = str(checkpoint_path)
            _atomic_hardlink(checkpoint_path, root / "best.pt")
        report = {
            "event": "evaluation",
            "update": update,
            "metrics": metrics,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "elapsed_seconds": time.monotonic() - started,
            "checkpoint": str(checkpoint_path),
        }
        _append_jsonl(root / "metrics.jsonl", report)
        print(json.dumps(report, sort_keys=True), flush=True)
        if _meets_target(metrics):
            break

    selected_metadata = restore_best_overfit_checkpoint(
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
        {
            "event": "all_saved_evaluation",
            "update": int(selected_metadata.global_step),
            "checkpoint": best_checkpoint,
            "metrics": full_metrics,
        },
    )
    terminal = build_overfit_terminal_report(
        updates_completed=last_update,
        baseline_relative_l2=float(baseline["aggregate_relative_l2"]),
        best_fixed_relative_l2=best_score,
        best_checkpoint=best_checkpoint,
        all_saved_metrics=full_metrics,
    )
    _atomic_json(terminal, terminal_path)
    print(json.dumps(terminal, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
