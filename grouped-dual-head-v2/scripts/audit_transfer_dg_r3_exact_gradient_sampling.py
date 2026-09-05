#!/usr/bin/env python3
"""Train-only exact-gradient audit of the frozen r3 block corrector.

This diagnostic never trains, calls ``optimizer.step``, or saves model weights.
It compares full-rollout BPTT gradients, the algebraically equivalent mean
Horvitz--Thompson (HT) gradient on the same graph, and the existing detached
training-window gradient.  Validation and test_id future truth stay sealed.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from saved_time_phase_operator_v4.parent_anchored_block import (  # noqa: E402
    ParentAnchoredBlockCorrector,
    parameter_count,
    render_retained_rfft_frames,
)
from saved_time_phase_operator_v4.parent_anchored_relative_loss import (  # noqa: E402
    record_energy_squared_loss,
    unbiased_window_weights,
)
from scripts.train_transfer_dg_parent_anchored_block32 import (  # noqa: E402
    ParentBlockCache,
    relative_rows,
    training_window,
)


SCHEMA = "transfer_dg_r3_exact_gradient_sampling_audit_v1"
FAMILIES = ("uniform", "layered", "anomaly", "marmousi")
STARTS = (2, 34, 66, 98, 130, 162, 194, 226, 258, 290, 322, 354, 386)
FUTURE_START = 2
TIME_COUNT = 401
ROLLOUT_BLOCKS = 2
EXPECTED_PARAMETER_TENSORS = 62
EXPECTED_PARAMETER_SCALARS = 91396
CLIP_THRESHOLD = 1.0
RHO_GRID = (0.0, 1.0e-6, 3.0e-6, 1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3)
COMPONENT_WEIGHTS = OrderedDict(
    (
        ("main", 1.0),
        ("derivative", 0.01),
        ("spectral", 0.02),
        ("nonworse", 0.30),
        ("correction", 0.01),
    )
)
SPECTRAL_FLOOR_FRACTION = 0.005
AUTHORITY_RECORDS_SHA256 = "e22704d9c8627b03572945a6aeee7ddfb3d0ebd6de60ee7a923f75ba85ce5464"
COMPATIBILITY_RECORDS_SHA256 = "b365de4d94f3526570c312e143f791887404834d5b58d545284c37a3622da2f7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"stale atomic-JSON staging path: {staging}")
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with staging.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left64 = left.detach().cpu().double().flatten()
    right64 = right.detach().cpu().double().flatten()
    denominator = float(left64.norm() * right64.norm())
    if denominator <= 1.0e-24:
        return 1.0 if torch.equal(left64, right64) else 0.0
    return float(torch.dot(left64, right64) / denominator)


def vector_comparison(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate64 = candidate.detach().cpu().double().flatten()
    reference64 = reference.detach().cpu().double().flatten()
    candidate_norm = float(candidate64.norm())
    reference_norm = float(reference64.norm())
    difference_norm = float((candidate64 - reference64).norm())
    return {
        "candidate_norm": candidate_norm,
        "reference_norm": reference_norm,
        "cosine": cosine(candidate64, reference64),
        "norm_ratio": candidate_norm / max(reference_norm, 1.0e-12),
        "relative_norm_difference": difference_norm / max(reference_norm, 1.0e-12),
    }


def declared_parameters(model: torch.nn.Module) -> tuple[tuple[str, ...], tuple[torch.nn.Parameter, ...]]:
    named = tuple((name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad)
    names = tuple(name for name, _ in named)
    parameters = tuple(parameter for _, parameter in named)
    if len(parameters) != EXPECTED_PARAMETER_TENSORS:
        raise RuntimeError(f"requires-grad tensor count drift: {len(parameters)}")
    scalar_count = sum(parameter.numel() for parameter in parameters)
    if scalar_count != EXPECTED_PARAMETER_SCALARS:
        raise RuntimeError(f"requires-grad scalar count drift: {scalar_count}")
    if len(set(names)) != len(names):
        raise RuntimeError("duplicate named parameter")
    return names, parameters


def flatten_gradient(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        tuple(parameters),
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    missing = [index for index, gradient in enumerate(gradients) if gradient is None]
    if missing:
        raise RuntimeError(f"unexpected None gradients at parameter indices {missing}")
    flattened = torch.cat(
        [gradient.detach().to(device="cpu", dtype=torch.float64).reshape(-1) for gradient in gradients]
    )
    if flattened.numel() != EXPECTED_PARAMETER_SCALARS:
        raise RuntimeError("flattened gradient scalar count drift")
    if not bool(torch.isfinite(flattened).all()):
        raise FloatingPointError("non-finite flattened gradient")
    return flattened


def component_objectives(
    prediction: torch.Tensor,
    target: torch.Tensor,
    parent: torch.Tensor,
    *,
    frame_weights: torch.Tensor,
    delta_weights: torch.Tensor,
    full_target_energy: float,
    full_target_delta_energy: float,
) -> OrderedDict[str, torch.Tensor]:
    common = dict(
        frame_weights=frame_weights,
        delta_weights=delta_weights,
        full_target_energy=full_target_energy,
        full_target_delta_energy=full_target_delta_energy,
        spectral_floor_fraction=SPECTRAL_FLOOR_FRACTION,
    )
    main, _ = record_energy_squared_loss(
        prediction,
        target,
        parent,
        derivative_weight=0.0,
        spectral_weight=0.0,
        nonworse_weight=0.0,
        correction_weight=0.0,
        **common,
    )
    raw = OrderedDict(main=main)
    switches = {
        "derivative": {"derivative_weight": 1.0},
        "spectral": {"spectral_weight": 1.0},
        "nonworse": {"nonworse_weight": 1.0},
        "correction": {"correction_weight": 1.0},
    }
    for name, enabled in switches.items():
        weights = {
            "derivative_weight": 0.0,
            "spectral_weight": 0.0,
            "nonworse_weight": 0.0,
            "correction_weight": 0.0,
        }
        weights.update(enabled)
        plus_main, _ = record_energy_squared_loss(
            prediction,
            target,
            parent,
            **weights,
            **common,
        )
        raw[name] = plus_main - main
    total, _ = record_energy_squared_loss(
        prediction,
        target,
        parent,
        derivative_weight=COMPONENT_WEIGHTS["derivative"],
        spectral_weight=COMPONENT_WEIGHTS["spectral"],
        nonworse_weight=COMPONENT_WEIGHTS["nonworse"],
        correction_weight=COMPONENT_WEIGHTS["correction"],
        **common,
    )
    reconstructed = sum(COMPONENT_WEIGHTS[name] * raw[name] for name in raw)
    if not torch.allclose(total.detach(), reconstructed.detach(), rtol=1.0e-6, atol=1.0e-8):
        raise RuntimeError("record-energy component reconstruction drift")
    raw["total"] = total
    return raw


def gradients_for_objectives(
    objectives: Mapping[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
) -> OrderedDict[str, torch.Tensor]:
    names = tuple(objectives)
    gradients: OrderedDict[str, torch.Tensor] = OrderedDict()
    # Total is derived from raw component gradients after all five exact raw
    # derivatives are evaluated.  This avoids a redundant sixth reverse pass.
    raw_names = tuple(name for name in names if name != "total")
    for index, name in enumerate(raw_names):
        gradients[name] = flatten_gradient(
            objectives[name], parameters, retain_graph=index + 1 < len(raw_names)
        )
    gradients["total"] = sum(COMPONENT_WEIGHTS[name] * gradients[name] for name in raw_names)
    return gradients


def objective_summary(
    objectives: Mapping[str, torch.Tensor], gradients: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    return {
        name: {
            "raw_value": float(objectives[name].detach()),
            "weight_in_total": 1.0 if name == "total" else COMPONENT_WEIGHTS[name],
            "gradient_norm": float(gradients[name].norm()),
        }
        for name in objectives
    }


def expected_window_length(start: int) -> int:
    if start not in STARTS:
        raise ValueError("unregistered block start")
    return min(ROLLOUT_BLOCKS * 32, TIME_COUNT - int(start))


def inclusion_mean_errors() -> dict[str, float]:
    frame_sum = torch.zeros(TIME_COUNT - FUTURE_START, dtype=torch.float64)
    delta_sum = torch.zeros(TIME_COUNT - FUTURE_START - 1, dtype=torch.float64)
    for start in STARTS:
        length = expected_window_length(start)
        frame, delta = unbiased_window_weights(
            loss_start=start,
            sample_length=length,
            time_count=TIME_COUNT,
            block_size=32,
            rollout_blocks=ROLLOUT_BLOCKS,
            dtype=torch.float64,
        )
        offset = start - FUTURE_START
        frame_sum[offset : offset + length] += frame
        delta_sum[offset : offset + length - 1] += delta
    frame_mean = frame_sum / len(STARTS)
    delta_mean = delta_sum / len(STARTS)
    return {
        "frame_max_absolute_error": float((frame_mean - 1.0).abs().max()),
        "delta_max_absolute_error": float((delta_mean - 1.0).abs().max()),
    }


def gradient_variance(gradients: Sequence[torch.Tensor]) -> dict[str, float]:
    stack = torch.stack(tuple(gradient.double() for gradient in gradients))
    mean = stack.mean(dim=0)
    mean_squared_deviation = float((stack - mean).square().sum(dim=1).mean())
    mean_norm = float(mean.norm())
    return {
        "mean_gradient_norm": mean_norm,
        "mean_squared_deviation": mean_squared_deviation,
        "sampling_cv": math.sqrt(max(mean_squared_deviation, 0.0)) / max(mean_norm, 1.0e-12),
    }


def quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("quantiles require non-empty finite input")
    return {
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
    }


def clip_statistics(norms: Sequence[float], threshold: float = CLIP_THRESHOLD) -> dict[str, Any]:
    values = np.asarray(norms, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or bool((values < 0).any()):
        raise ValueError("clip statistics require finite nonnegative norms")
    clipped = values > float(threshold)
    scales = np.minimum(1.0, float(threshold) / np.maximum(values, 1.0e-12))
    return {
        "threshold": float(threshold),
        "count": int(clipped.sum()),
        "total": int(values.size),
        "fraction": float(clipped.mean()),
        "norm_p50": float(np.quantile(values, 0.50)),
        "norm_p95": float(np.quantile(values, 0.95)),
        "norm_max": float(values.max()),
        "scale_p50": float(np.quantile(scales, 0.50)),
        "scale_p95": float(np.quantile(scales, 0.95)),
        "scale_min": float(scales.min()),
    }


@dataclass
class CompleteState:
    parameters: OrderedDict[str, torch.Tensor]
    buffers: OrderedDict[str, torch.Tensor]
    gradients: OrderedDict[str, torch.Tensor | None]
    cpu_rng: torch.Tensor
    cuda_rng: tuple[torch.Tensor, ...]
    training: bool


def capture_complete_state(module: torch.nn.Module) -> CompleteState:
    return CompleteState(
        parameters=OrderedDict(
            (name, value.detach().clone()) for name, value in module.named_parameters()
        ),
        buffers=OrderedDict((name, value.detach().clone()) for name, value in module.named_buffers()),
        gradients=OrderedDict(
            (name, None if value.grad is None else value.grad.detach().clone())
            for name, value in module.named_parameters()
        ),
        cpu_rng=torch.get_rng_state().clone(),
        cuda_rng=tuple(state.clone() for state in torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else (),
        training=bool(module.training),
    )


def restore_complete_state(module: torch.nn.Module, state: CompleteState) -> None:
    with torch.no_grad():
        current_parameters = OrderedDict(module.named_parameters())
        current_buffers = OrderedDict(module.named_buffers())
        if tuple(current_parameters) != tuple(state.parameters) or tuple(current_buffers) != tuple(state.buffers):
            raise RuntimeError("module state declaration changed")
        for name, parameter in current_parameters.items():
            parameter.copy_(state.parameters[name])
            saved_gradient = state.gradients[name]
            parameter.grad = None if saved_gradient is None else saved_gradient.clone().to(parameter.device)
        for name, buffer in current_buffers.items():
            buffer.copy_(state.buffers[name])
    module.train(state.training)
    torch.set_rng_state(state.cpu_rng)
    if state.cuda_rng:
        torch.cuda.set_rng_state_all(list(state.cuda_rng))


def assert_complete_state_equal(module: torch.nn.Module, state: CompleteState) -> None:
    current_parameters = OrderedDict(module.named_parameters())
    current_buffers = OrderedDict(module.named_buffers())
    if tuple(current_parameters) != tuple(state.parameters) or tuple(current_buffers) != tuple(state.buffers):
        raise RuntimeError("module state declaration differs from rollback snapshot")
    for name, parameter in current_parameters.items():
        if not torch.equal(parameter.detach().cpu(), state.parameters[name].cpu()):
            raise RuntimeError(f"parameter rollback failed: {name}")
        expected_gradient = state.gradients[name]
        if (parameter.grad is None) != (expected_gradient is None):
            raise RuntimeError(f"gradient presence rollback failed: {name}")
        if parameter.grad is not None and not torch.equal(parameter.grad.detach().cpu(), expected_gradient.cpu()):
            raise RuntimeError(f"gradient rollback failed: {name}")
    for name, buffer in current_buffers.items():
        if not torch.equal(buffer.detach().cpu(), state.buffers[name].cpu()):
            raise RuntimeError(f"buffer rollback failed: {name}")
    if not torch.equal(torch.get_rng_state(), state.cpu_rng):
        raise RuntimeError("CPU RNG rollback failed")
    if state.cuda_rng:
        current_cuda = torch.cuda.get_rng_state_all()
        if len(current_cuda) != len(state.cuda_rng) or any(
            not torch.equal(current, expected) for current, expected in zip(current_cuda, state.cuda_rng)
        ):
            raise RuntimeError("CUDA RNG rollback failed")


def apply_normalized_direction(
    module: torch.nn.Module,
    state: CompleteState,
    gradient: torch.Tensor,
    rho: float,
) -> dict[str, float]:
    flat = gradient.detach().cpu().double().flatten()
    parameter_norm = math.sqrt(
        sum(float(value.double().square().sum()) for value in state.parameters.values())
    )
    gradient_norm = float(flat.norm())
    scale = float(rho) * parameter_norm / max(gradient_norm, 1.0e-12)
    offset = 0
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            count = parameter.numel()
            delta = flat[offset : offset + count].reshape(parameter.shape).to(
                device=parameter.device, dtype=parameter.dtype
            )
            parameter.copy_(state.parameters[name].to(parameter.device) - scale * delta)
            offset += count
    if offset != flat.numel():
        raise RuntimeError("line-search direction length differs from parameters")
    return {
        "rho": float(rho),
        "parameter_norm": parameter_norm,
        "gradient_norm": gradient_norm,
        "applied_scale": scale,
    }


def classify(metrics: Mapping[str, Any]) -> dict[str, Any]:
    failures = list(metrics.get("invalid_failures", []))
    flags = OrderedDict(
        invalid=bool(failures),
        target_or_auxiliary_mismatch=(
            float(metrics["cosine_full_total_full_main"]) < 0.80
            or (
                float(metrics["training_window_total_best_main_improvement"]) < 0.001
                and float(metrics["full_main_best_main_improvement"]) >= 0.01
            )
        ),
        sampling_limited=(
            sum(float(value) > 1.0 for value in metrics["per_record_sampling_cv"]) >= 3
            or float(metrics["all_window_to_own_ht_cosine_median"]) < 0.50
        ),
        tbptt_or_rollout_graph_limited=(
            float(metrics["cosine_training_window_ht_main_full_main"]) < 0.80
            or float(np.median(np.asarray(metrics["per_record_tbptt_main_cosines"], dtype=np.float64))) < 0.70
        ),
        clip_limited=float(metrics["window_total_clip_fraction"]) >= 0.25,
        optimization_limited_supported=False,
        local_plateau_architecture_or_target_candidate=(
            float(metrics["full_main_best_main_improvement"]) < 0.001
        ),
    )
    flags["optimization_limited_supported"] = (
        not any(
            flags[name]
            for name in (
                "invalid",
                "target_or_auxiliary_mismatch",
                "sampling_limited",
                "tbptt_or_rollout_graph_limited",
                "clip_limited",
            )
        )
        and float(metrics["training_window_total_best_main_improvement"]) >= 0.01
        and float(metrics["training_window_total_best_composite_relative_change"]) <= 1.0e-8
        and float(metrics["training_window_total_max_per_record_main_regression"]) <= 0.01
    )
    priority = (
        "invalid",
        "target_or_auxiliary_mismatch",
        "sampling_limited",
        "tbptt_or_rollout_graph_limited",
        "clip_limited",
        "optimization_limited_supported",
        "local_plateau_architecture_or_target_candidate",
    )
    primary = next((name for name in priority if flags[name]), "inconclusive")
    return {"classification": primary, "flags": dict(flags), "invalid_failures": failures}


def verify_selection(
    manifest: Mapping[str, Any], cache: ParentBlockCache
) -> list[dict[str, Any]]:
    records = list(manifest["records"])
    if tuple(manifest.get("family_order", ())) != FAMILIES:
        raise RuntimeError("selection family order drift")
    if canonical_sha256(records) != AUTHORITY_RECORDS_SHA256:
        raise RuntimeError("authoritative seven-field selection digest drift")
    projected = [{key: value for key, value in row.items() if key != "cache_position"} for row in records]
    if canonical_sha256(projected) != COMPATIBILITY_RECORDS_SHA256:
        raise RuntimeError("compatibility six-field selection digest drift")
    if manifest["digests"]["authority"]["sha256"] != AUTHORITY_RECORDS_SHA256:
        raise RuntimeError("manifest authority digest declaration drift")
    if manifest["digests"]["compatibility_projection"]["sha256"] != COMPATIBILITY_RECORDS_SHA256:
        raise RuntimeError("manifest compatibility digest declaration drift")
    if len(records) != 4 or tuple(row["family"] for row in records) != FAMILIES:
        raise RuntimeError("selection must contain exactly one record per frozen family")
    observed = []
    for row in records:
        position = int(row["cache_position"])
        source_index = int(row["source_index"])
        if not 0 <= position < len(cache.sample_ids):
            raise RuntimeError("selection cache position out of range")
        source = cache.source
        decode = lambda value: value.decode() if isinstance(value, bytes) else str(value)
        actual = {
            "family": str(cache.families[position]),
            "sample_id": str(cache.sample_ids[position]),
            "source_index": int(cache.source_indices[position]),
            "group_id": str(cache.group_ids[position]),
            "role": str(cache.roles[position]),
            "cache_position": position,
            "sample_sha256": decode(source["sample_sha256"][source_index]),
        }
        if actual != row:
            raise RuntimeError(f"selection/cache/source binding drift for {row.get('sample_id')}")
        split = decode(source["split"][source_index])
        if split != "train" or row["role"] != "fit":
            raise RuntimeError("selection is not train/fit only")
        observed.append(actual)
    # Recompute the non-performance selection rule against every fit record.
    for family, selected in zip(FAMILIES, observed):
        eligible = [
            (int(cache.source_indices[index]), str(cache.sample_ids[index]), index)
            for index in cache.positions("fit")
            if str(cache.families[index]) == family
        ]
        expected = min(eligible)
        if expected != (selected["source_index"], selected["sample_id"], selected["cache_position"]):
            raise RuntimeError(f"minimum (source_index,sample_id) selection rule drift for {family}")
    return observed


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "transfer_dg_parent_anchored_block32_checkpoint_v1":
        raise RuntimeError("r3 checkpoint schema drift")
    if checkpoint.get("validation_opened") or checkpoint.get("test_id_opened"):
        raise RuntimeError("r3 checkpoint has an opened sealed split")
    if int(checkpoint.get("epoch", -1)) != 12 or int(checkpoint.get("update", -1)) != 816:
        raise RuntimeError("r3 checkpoint epoch/update drift")
    config = checkpoint["model_config"]
    model = ParentAnchoredBlockCorrector(
        condition_channels=int(config["condition_channels"]),
        block_size=int(config["block_size"]),
        width=int(config["width"]),
        spectral_rank=int(config["spectral_rank"]),
        modes=int(config["modes"]),
        depth=int(config["depth"]),
        maximum_correction_ratio=float(config["maximum_correction_ratio"]),
        boundary_blend_frames=int(config["boundary_blend_frames"]),
        activation_checkpointing=True,
        hard_free_surface=bool(config["hard_free_surface"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.train()  # Enables non-reentrant activation checkpointing; there is no dropout or batch norm.
    model.requires_grad_(True)
    if parameter_count(model) != EXPECTED_PARAMETER_SCALARS:
        raise RuntimeError("model parameter count drift")
    declared_parameters(model)
    return model


def full_parent_and_truth(
    cache: ParentBlockCache, position: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coefficients, condition = cache.parent_and_condition(position, device)
    parent = render_retained_rfft_frames(
        coefficients,
        time_count=cache.time_count,
        frame_indices=torch.arange(cache.time_count, device=device),
    )
    parent = parent.clone()
    parent[..., 0, :] = 0.0
    target = cache.truth(position, 0, cache.time_count, device)
    if parent.shape != target.shape or parent.shape != (1, TIME_COUNT, 201, 201):
        raise RuntimeError("full parent/target shape drift")
    if not bool(torch.isfinite(parent).all() and torch.isfinite(target).all()):
        raise FloatingPointError("non-finite full parent or train target")
    return coefficients, condition, parent, target


def same_graph_ht_main(
    prediction: torch.Tensor,
    target: torch.Tensor,
    parent: torch.Tensor,
    *,
    full_energy: float,
    full_delta_energy: float,
) -> torch.Tensor:
    values = []
    for start in STARTS:
        length = expected_window_length(start)
        local = slice(start, start + length)
        frame, delta = unbiased_window_weights(
            loss_start=start,
            sample_length=length,
            time_count=TIME_COUNT,
            block_size=32,
            rollout_blocks=ROLLOUT_BLOCKS,
            device=prediction.device,
            dtype=prediction.dtype,
        )
        objectives = component_objectives(
            prediction[:, local],
            target[:, local],
            parent[:, local],
            frame_weights=frame,
            delta_weights=delta,
            full_target_energy=full_energy,
            full_target_delta_energy=full_delta_energy,
        )
        values.append(objectives["main"])
    return torch.stack(values).mean()


def audit_full_graph(
    model: torch.nn.Module,
    cache: ParentBlockCache,
    position: int,
    parameters: Sequence[torch.nn.Parameter],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    coefficients, condition, parent, target = full_parent_and_truth(cache, position, device)
    prediction = model.rollout(parent, condition)
    future = slice(FUTURE_START, TIME_COUNT)
    full_energy, full_delta_energy = cache.full_future_energies(position)
    ones_frame = torch.ones(TIME_COUNT - FUTURE_START, device=device, dtype=prediction.dtype)
    ones_delta = torch.ones(TIME_COUNT - FUTURE_START - 1, device=device, dtype=prediction.dtype)
    objectives = component_objectives(
        prediction[:, future],
        target[:, future],
        parent[:, future],
        frame_weights=ones_frame,
        delta_weights=ones_delta,
        full_target_energy=full_energy,
        full_target_delta_energy=full_delta_energy,
    )
    ht_main = same_graph_ht_main(
        prediction,
        target,
        parent,
        full_energy=full_energy,
        full_delta_energy=full_delta_energy,
    )
    # Raw component gradients consume the graph, so evaluate B first while retaining it.
    ht_gradient = flatten_gradient(ht_main, parameters, retain_graph=True)
    gradients = gradients_for_objectives(objectives, parameters)
    relative = float(relative_rows(prediction[:, future], target[:, future])[0])
    sqrt_main = math.sqrt(max(float(objectives["main"].detach()), 0.0))
    algebra = vector_comparison(ht_gradient, gradients["main"])
    algebra["loss_relative_difference"] = abs(
        float(ht_main.detach()) - float(objectives["main"].detach())
    ) / max(abs(float(objectives["main"].detach())), 1.0e-12)
    gates = {
        "sqrt_full_main_vs_relative_rows": {
            "sqrt_full_main": sqrt_main,
            "relative_rows": relative,
            "passed": math.isclose(sqrt_main, relative, rel_tol=1.0e-5, abs_tol=1.0e-6),
        },
        "same_graph_ht_algebra": {
            **algebra,
            "passed": (
                algebra["cosine"] >= 0.999999
                and algebra["relative_norm_difference"] <= 1.0e-5
                and algebra["loss_relative_difference"] <= 1.0e-6
            ),
        },
    }
    result = {
        "relative_l2": relative,
        "full_objective": objective_summary(objectives, gradients),
        "same_graph_ht_main_value": float(ht_main.detach()),
        "gates": gates,
    }
    del prediction, target, parent, coefficients, condition, objectives, ht_main
    return result, {**gradients, "same_graph_ht_main": ht_gradient}, (torch.tensor(full_energy), torch.tensor(full_delta_energy))


def audit_training_windows(
    model: torch.nn.Module,
    cache: ParentBlockCache,
    position: int,
    parameters: Sequence[torch.nn.Parameter],
    device: torch.device,
    *,
    repeat_first: bool,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    full_energy, full_delta_energy = cache.full_future_energies(position)
    window_rows = []
    per_component: dict[str, list[torch.Tensor]] = {
        name: [] for name in (*COMPONENT_WEIGHTS.keys(), "total")
    }
    first_total = None
    for start in STARTS:
        coefficients, condition = cache.parent_and_condition(position, device)
        prediction, target, parent = training_window(
            model,
            coefficients,
            condition,
            cache,
            position,
            loss_start=start,
            rollout_blocks=ROLLOUT_BLOCKS,
        )
        length = int(prediction.shape[1])
        if length != expected_window_length(start):
            raise RuntimeError(f"training-window length drift at start {start}: {length}")
        frame, delta = unbiased_window_weights(
            loss_start=start,
            sample_length=length,
            time_count=TIME_COUNT,
            block_size=32,
            rollout_blocks=ROLLOUT_BLOCKS,
            device=device,
            dtype=prediction.dtype,
        )
        objectives = component_objectives(
            prediction,
            target,
            parent,
            frame_weights=frame,
            delta_weights=delta,
            full_target_energy=full_energy,
            full_target_delta_energy=full_delta_energy,
        )
        gradients = gradients_for_objectives(objectives, parameters)
        for name in per_component:
            per_component[name].append(gradients[name])
        total_norm = float(gradients["total"].norm())
        window_rows.append(
            {
                "start": start,
                "length": length,
                "stop_exclusive": start + length,
                "objectives": objective_summary(objectives, gradients),
                "total_gradient_norm": total_norm,
                "clip_scale": min(1.0, CLIP_THRESHOLD / max(total_norm, 1.0e-12)),
            }
        )
        if start == STARTS[0]:
            first_total = gradients["total"].clone()
        del coefficients, condition, prediction, target, parent, objectives, gradients
    mean_gradients = {
        name: torch.stack(values).mean(dim=0) for name, values in per_component.items()
    }
    for row, total_gradient in zip(window_rows, per_component["total"]):
        row["cosine_to_own_ht_total"] = cosine(total_gradient, mean_gradients["total"])
    repeatability = None
    if repeat_first:
        start = STARTS[0]
        coefficients, condition = cache.parent_and_condition(position, device)
        prediction, target, parent = training_window(
            model,
            coefficients,
            condition,
            cache,
            position,
            loss_start=start,
            rollout_blocks=ROLLOUT_BLOCKS,
        )
        frame, delta = unbiased_window_weights(
            loss_start=start,
            sample_length=int(prediction.shape[1]),
            time_count=TIME_COUNT,
            block_size=32,
            rollout_blocks=ROLLOUT_BLOCKS,
            device=device,
            dtype=prediction.dtype,
        )
        repeated_objectives = component_objectives(
            prediction,
            target,
            parent,
            frame_weights=frame,
            delta_weights=delta,
            full_target_energy=full_energy,
            full_target_delta_energy=full_delta_energy,
        )
        repeated_gradients = gradients_for_objectives(repeated_objectives, parameters)
        repeatability = vector_comparison(repeated_gradients["total"], first_total)
        repeatability["passed"] = (
            repeatability["cosine"] >= 0.999999
            and repeatability["relative_norm_difference"] <= 1.0e-5
        )
        del coefficients, condition, prediction, target, parent, repeated_objectives, repeated_gradients
    result = {
        "mean_ht_objective": {
            name: float(np.mean([row["objectives"][name]["raw_value"] for row in window_rows]))
            for name in (*COMPONENT_WEIGHTS.keys(), "total")
        },
        "gradient_variance": {
            name: gradient_variance(values) for name, values in per_component.items()
        },
        "window_to_own_ht_total_cosine_quantiles": quantiles(
            [row["cosine_to_own_ht_total"] for row in window_rows]
        ),
        "clip": clip_statistics([row["total_gradient_norm"] for row in window_rows]),
        "repeatability": repeatability,
        "windows": window_rows,
        "spectral_estimator_note": (
            "The local spectral estimator is retained exactly from record_energy_squared_loss; "
            "it is not claimed to be an unbiased estimator of the full-trajectory spectral objective."
        ),
    }
    return result, mean_gradients


@torch.no_grad()
def evaluate_full_objectives(
    model: torch.nn.Module,
    cache: ParentBlockCache,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    rows = []
    for record in records:
        position = int(record["cache_position"])
        _, condition, parent, target = full_parent_and_truth(cache, position, device)
        prediction = model.rollout(parent, condition)
        future = slice(FUTURE_START, TIME_COUNT)
        full_energy, full_delta_energy = cache.full_future_energies(position)
        frame = torch.ones(TIME_COUNT - FUTURE_START, device=device, dtype=prediction.dtype)
        delta = torch.ones(TIME_COUNT - FUTURE_START - 1, device=device, dtype=prediction.dtype)
        objectives = component_objectives(
            prediction[:, future],
            target[:, future],
            parent[:, future],
            frame_weights=frame,
            delta_weights=delta,
            full_target_energy=full_energy,
            full_target_delta_energy=full_delta_energy,
        )
        rows.append(
            {
                "sample_id": record["sample_id"],
                "family": record["family"],
                "main": float(objectives["main"]),
                "composite": float(objectives["total"]),
            }
        )
        del condition, parent, target, prediction, objectives
    model.train(was_training)
    return {
        "main": float(np.mean([row["main"] for row in rows])),
        "composite": float(np.mean([row["composite"] for row in rows])),
        "rows": rows,
    }


def line_search(
    model: torch.nn.Module,
    cache: ParentBlockCache,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    directions: Mapping[str, torch.Tensor],
    initial_state: CompleteState,
) -> dict[str, Any]:
    baseline = evaluate_full_objectives(model, cache, records, device)
    result: dict[str, Any] = {"rho_grid": list(RHO_GRID), "baseline": baseline, "directions": {}}
    baseline_by_id = {row["sample_id"]: row for row in baseline["rows"]}
    for name, gradient in directions.items():
        points = []
        for rho in RHO_GRID:
            restore_complete_state(model, initial_state)
            metadata = apply_normalized_direction(model, initial_state, gradient, rho)
            evaluation = evaluate_full_objectives(model, cache, records, device)
            main_improvement = 1.0 - evaluation["main"] / max(baseline["main"], 1.0e-16)
            composite_relative_change = (
                evaluation["composite"] - baseline["composite"]
            ) / max(abs(baseline["composite"]), 1.0e-16)
            per_record_regressions = [
                evaluation_row["main"] / max(baseline_by_id[evaluation_row["sample_id"]]["main"], 1.0e-16) - 1.0
                for evaluation_row in evaluation["rows"]
            ]
            points.append(
                {
                    **metadata,
                    **evaluation,
                    "main_improvement": main_improvement,
                    "composite_relative_change": composite_relative_change,
                    "maximum_per_record_main_regression": max(per_record_regressions),
                }
            )
            restore_complete_state(model, initial_state)
            assert_complete_state_equal(model, initial_state)
        best = min(points, key=lambda point: (point["main"], point["rho"]))
        result["directions"][name] = {
            "points": points,
            "oracle_diagnostic_best_rho": best["rho"],
            "oracle_diagnostic_best_main": best["main"],
            "oracle_diagnostic_best_main_improvement": best["main_improvement"],
            "oracle_diagnostic_best_composite_relative_change": best["composite_relative_change"],
            "oracle_diagnostic_best_maximum_per_record_main_regression": best[
                "maximum_per_record_main_regression"
            ],
        }
    restore_complete_state(model, initial_state)
    assert_complete_state_equal(model, initial_state)
    return result


def resolved_expected_paths(prereg: Mapping[str, Any]) -> dict[str, Path]:
    return {name: (ROOT / value).resolve() if not Path(value).is_absolute() else Path(value).resolve() for name, value in prereg["paths"].items()}


def observed_environment() -> dict[str, Any]:
    """Return the complete and deliberately narrow environment-report whitelist."""

    return {
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
    }


def environment_preflight(
    *,
    preregistration: Path,
    output: Path,
    candidate: str,
    mode: str,
    after_success: Callable[[], Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate process-start environment without making any CUDA API call."""

    preregistration_sha256_before = sha256(preregistration)
    observed = observed_environment()
    failures = []
    if observed["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        failures.append("CUBLAS_WORKSPACE_CONFIG")
    if observed["CUDA_VISIBLE_DEVICES"] != "0":
        failures.append("CUDA_VISIBLE_DEVICES")
    if failures:
        preregistration_sha256_after = sha256(preregistration)
        atomic_json(
            {
                "schema": SCHEMA,
                "candidate": candidate,
                "mode": mode,
                "status": "failed",
                "classification": {
                    "classification": "invalid",
                    "flags": {"invalid": True},
                    "invalid_failures": ["environment_preflight_failed"],
                },
                "failure_stage": "environment_preflight_failed",
                "environment_contract_failures": failures,
                "observed_environment": observed,
                "preregistration_sha256_observed_before": preregistration_sha256_before,
                "preregistration_sha256_observed_after": preregistration_sha256_after,
                "preregistration_hash_unchanged": (
                    preregistration_sha256_after == preregistration_sha256_before
                ),
                "diagnostic_entered": False,
                "promotion_authorized": False,
                "validation_opened": False,
                "test_id_opened": False,
            },
            output,
        )
        raise RuntimeError(
            "environment preflight failed: " + ",".join(failures)
        )
    if after_success is not None:
        after_success()
    return observed, preregistration_sha256_before


def cuda_preflight(
    *,
    device: torch.device,
    seed: int,
    output: Path,
    candidate: str,
    mode: str,
    input_hashes_before: Mapping[str, str],
    preregistration: Path,
    preregistration_sha256_before: str,
    environment: Mapping[str, Any],
    after_success: Callable[[], Any] | None = None,
) -> Any:
    """Initialize CUDA deterministically before any cache or model loader runs."""

    try:
        # This order is part of the frozen preflight contract.  In particular,
        # reset_peak_memory_stats is invalid before a CUDA context exists.
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        np.random.seed(int(seed))
    except Exception as error:
        preregistration_sha256_after = sha256(preregistration)
        atomic_json(
            {
                "schema": SCHEMA,
                "candidate": candidate,
                "mode": mode,
                "status": "failed",
                "classification": {
                    "classification": "invalid",
                    "flags": {"invalid": True},
                    "invalid_failures": ["cuda_initialization_failed"],
                },
                "failure_stage": "cuda_initialization_failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "input_hashes_before": dict(input_hashes_before),
                "observed_environment": dict(environment),
                "preregistration_sha256_observed_before": preregistration_sha256_before,
                "preregistration_sha256_observed_after": preregistration_sha256_after,
                "preregistration_hash_unchanged": (
                    preregistration_sha256_after == preregistration_sha256_before
                ),
                "diagnostic_entered": False,
                "promotion_authorized": False,
                "validation_opened": False,
                "test_id_opened": False,
            },
            output,
        )
        raise
    return None if after_success is None else after_success()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    mode_name = "run" if args.run else "smoke"
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing audit report")
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if prereg.get("schema") != "transfer_dg_r3_exact_gradient_sampling_preregistration_v1":
        raise RuntimeError("unexpected preregistration schema")
    expected_paths = resolved_expected_paths(prereg)
    supplied = {
        "preregistration": args.preregistration.resolve(),
        "selection_manifest": args.selection_manifest.resolve(),
        "cache": args.cache.resolve(),
        "source_h5": args.source_h5.resolve(),
        "checkpoint": args.checkpoint.resolve(),
    }
    for name, path in supplied.items():
        if name in expected_paths and path != expected_paths[name]:
            raise RuntimeError(f"unregistered path override: {name}")
    expected_output = expected_paths[f"{mode_name}_output"]
    if args.output.resolve() != expected_output:
        raise RuntimeError("unregistered output path override")
    environment, preregistration_sha256_before = environment_preflight(
        preregistration=args.preregistration.resolve(),
        output=args.output,
        candidate=prereg["candidate"],
        mode=mode_name,
    )
    bindings = prereg["bindings"]
    bound_paths = {
        "script_sha256": Path(__file__).resolve(),
        "test_sha256": (ROOT / "tests/test_transfer_dg_r3_exact_gradient_sampling.py").resolve(),
        "selection_manifest_sha256": args.selection_manifest.resolve(),
        "cache_sha256": args.cache.resolve(),
        "source_h5_sha256": args.source_h5.resolve(),
        "checkpoint_sha256": args.checkpoint.resolve(),
        "rollback_parent_sha256": expected_paths["rollback_parent"],
        "upstream_selection_manifest_sha256": expected_paths["upstream_selection_manifest"],
        "corrector_sha256": (ROOT / "saved_time_phase_operator_v4/parent_anchored_block.py").resolve(),
        "relative_loss_sha256": (ROOT / "saved_time_phase_operator_v4/parent_anchored_relative_loss.py").resolve(),
        "training_window_sha256": (ROOT / "scripts/train_transfer_dg_parent_anchored_block32.py").resolve(),
    }
    input_hashes_before = {name: sha256(path) for name, path in bound_paths.items()}
    drift = {
        name: {"expected": bindings[name], "observed": observed}
        for name, observed in input_hashes_before.items()
        if bindings.get(name) != observed
    }
    if drift:
        preregistration_sha256_after = sha256(args.preregistration)
        atomic_json(
            {
                "schema": SCHEMA,
                "candidate": prereg["candidate"],
                "mode": mode_name,
                "status": "failed",
                "classification": {
                    "classification": "invalid",
                    "flags": {"invalid": True},
                    "invalid_failures": [f"binding_drift:{name}" for name in sorted(drift)],
                },
                "binding_drift": drift,
                "input_hashes_before": input_hashes_before,
                "observed_environment": environment,
                "preregistration_sha256_observed_before": preregistration_sha256_before,
                "preregistration_sha256_observed_after": preregistration_sha256_after,
                "preregistration_hash_unchanged": (
                    preregistration_sha256_after == preregistration_sha256_before
                ),
                "promotion_authorized": False,
                "validation_opened": False,
                "test_id_opened": False,
            },
            args.output,
        )
        raise RuntimeError(f"binding drift: {drift}")
    def preflight_invalid(reason: str, details: Mapping[str, Any] | None = None) -> None:
        preregistration_sha256_after = sha256(args.preregistration)
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "candidate": prereg["candidate"],
            "mode": mode_name,
            "status": "failed",
            "classification": {
                "classification": "invalid",
                "flags": {"invalid": True},
                "invalid_failures": [reason],
            },
            "input_hashes_before": input_hashes_before,
            "observed_environment": environment,
            "preregistration_sha256_observed_before": preregistration_sha256_before,
            "preregistration_sha256_observed_after": preregistration_sha256_after,
            "preregistration_hash_unchanged": (
                preregistration_sha256_after == preregistration_sha256_before
            ),
            "promotion_authorized": False,
            "validation_opened": False,
            "test_id_opened": False,
        }
        if details is not None:
            payload["details"] = dict(details)
        atomic_json(payload, args.output)

    if prereg.get("validation_opened") or prereg.get("test_id_opened"):
        preflight_invalid("preregistration_sealed_split_flag")
        raise RuntimeError("preregistration indicates an opened sealed split")
    if tuple(prereg["diagnostic"]["starts"]) != STARTS:
        preflight_invalid("registered_start_grid_drift")
        raise RuntimeError("registered start grid drift")
    inclusion = inclusion_mean_errors()
    if max(inclusion.values()) > 1.0e-6:
        preflight_invalid("ht_inclusion_gate", inclusion)
        raise RuntimeError(f"HT inclusion gate failed: {inclusion}")
    device = torch.device("cuda:0")
    cuda_preflight(
        device=device,
        seed=int(prereg["environment"]["seed"]),
        output=args.output,
        candidate=prereg["candidate"],
        mode=mode_name,
        input_hashes_before=input_hashes_before,
        preregistration=args.preregistration.resolve(),
        preregistration_sha256_before=preregistration_sha256_before,
        environment=environment,
    )
    cache = None
    model = None
    initial_state = None
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "candidate": prereg["candidate"],
        "mode": mode_name,
        "status": "running",
        "promotion_authorized": False,
        "validation_opened": False,
        "test_id_opened": False,
        "input_hashes_before": input_hashes_before,
        "observed_environment": environment,
        "preregistration_sha256_observed_before": preregistration_sha256_before,
        "inclusion_gate": inclusion,
    }
    try:
        cache = ParentBlockCache(args.cache, args.source_h5)
        if cache.time_count != TIME_COUNT:
            raise RuntimeError("cache time count drift")
        selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        records = verify_selection(selection, cache)
        active_records = records if args.run else records[:1]
        model = load_model(args.checkpoint, device)
        parameter_names, parameters = declared_parameters(model)
        initial_state = capture_complete_state(model)
        record_reports = []
        full_gradients = []
        window_gradients = []
        all_window_norms = []
        all_window_cosines = []
        for record in active_records:
            position = int(record["cache_position"])
            full_report, full_gradient, _ = audit_full_graph(
                model, cache, position, parameters, device
            )
            window_report, window_gradient = audit_training_windows(
                model,
                cache,
                position,
                parameters,
                device,
                repeat_first=args.smoke,
            )
            full_report["gates"]["ht_inclusion"] = {
                **inclusion,
                "passed": max(inclusion.values()) <= 1.0e-6,
            }
            full_gradients.append(full_gradient)
            window_gradients.append(window_gradient)
            all_window_norms.extend(row["total_gradient_norm"] for row in window_report["windows"])
            all_window_cosines.extend(
                row["cosine_to_own_ht_total"] for row in window_report["windows"]
            )
            comparison = {
                "training_window_ht_main_vs_full_main": vector_comparison(
                    window_gradient["main"], full_gradient["main"]
                ),
                "training_window_ht_total_vs_full_total": vector_comparison(
                    window_gradient["total"], full_gradient["total"]
                ),
                "full_total_vs_full_main": vector_comparison(
                    full_gradient["total"], full_gradient["main"]
                ),
            }
            record_reports.append(
                {"selection": record, "full_graph": full_report, "training_window": window_report, "comparison": comparison}
            )
            assert_complete_state_equal(model, initial_state)
            peak = int(torch.cuda.max_memory_allocated(device))
            if peak >= int(prereg["budgets"]["peak_allocated_bytes_max"]):
                raise RuntimeError(f"peak allocated memory budget exceeded: {peak}")
        aggregate_full = {
            name: torch.stack([row[name] for row in full_gradients]).mean(dim=0)
            for name in (*COMPONENT_WEIGHTS.keys(), "total")
        }
        aggregate_window = {
            name: torch.stack([row[name] for row in window_gradients]).mean(dim=0)
            for name in (*COMPONENT_WEIGHTS.keys(), "total")
        }
        report["records"] = record_reports
        report["record_count"] = len(active_records)
        report["parameter_contract"] = {
            "requires_grad_tensor_count": len(parameter_names),
            "requires_grad_scalar_count": sum(parameter.numel() for parameter in parameters),
            "named_parameter_order": list(parameter_names),
            "flatten_dtype": "torch.float64",
            "flatten_device": "cpu",
            "allow_unused": True,
            "none_gradient_policy": "fail",
        }
        report["aggregate_gradient_comparisons"] = {
            "training_window_ht_main_vs_full_main": vector_comparison(
                aggregate_window["main"], aggregate_full["main"]
            ),
            "training_window_ht_total_vs_full_total": vector_comparison(
                aggregate_window["total"], aggregate_full["total"]
            ),
            "full_total_vs_full_main": vector_comparison(
                aggregate_full["total"], aggregate_full["main"]
            ),
        }
        report["aggregate_gradient_norms"] = {
            "full": {name: float(value.norm()) for name, value in aggregate_full.items()},
            "training_window_ht": {
                name: float(value.norm()) for name, value in aggregate_window.items()
            },
        }
        report["clip"] = clip_statistics(all_window_norms)
        report["all_window_to_own_ht_total_cosine_quantiles"] = quantiles(all_window_cosines)
        report["line_search"] = None
        report["classification"] = {
            "classification": "not_evaluated_smoke",
            "reason": "Frozen classification requires all four records and the run-only line search.",
        }
        if args.run:
            directions = OrderedDict(
                (
                    ("negative_aggregate_training_window_ht_total", aggregate_window["total"]),
                    ("negative_aggregate_full_total", aggregate_full["total"]),
                    ("negative_aggregate_full_main", aggregate_full["main"]),
                )
            )
            search = line_search(model, cache, records, device, directions, initial_state)
            report["line_search"] = search
            training_best = search["directions"]["negative_aggregate_training_window_ht_total"]
            full_main_best = search["directions"]["negative_aggregate_full_main"]
            classification_metrics = {
                "invalid_failures": [],
                "cosine_full_total_full_main": report["aggregate_gradient_comparisons"][
                    "full_total_vs_full_main"
                ]["cosine"],
                "training_window_total_best_main_improvement": training_best[
                    "oracle_diagnostic_best_main_improvement"
                ],
                "full_main_best_main_improvement": full_main_best[
                    "oracle_diagnostic_best_main_improvement"
                ],
                "per_record_sampling_cv": [
                    row["training_window"]["gradient_variance"]["total"]["sampling_cv"]
                    for row in record_reports
                ],
                "all_window_to_own_ht_cosine_median": report[
                    "all_window_to_own_ht_total_cosine_quantiles"
                ]["p50"],
                "cosine_training_window_ht_main_full_main": report[
                    "aggregate_gradient_comparisons"
                ]["training_window_ht_main_vs_full_main"]["cosine"],
                "per_record_tbptt_main_cosines": [
                    row["comparison"]["training_window_ht_main_vs_full_main"]["cosine"]
                    for row in record_reports
                ],
                "window_total_clip_fraction": report["clip"]["fraction"],
                "training_window_total_best_composite_relative_change": training_best[
                    "oracle_diagnostic_best_composite_relative_change"
                ],
                "training_window_total_max_per_record_main_regression": training_best[
                    "oracle_diagnostic_best_maximum_per_record_main_regression"
                ],
            }
            gate_failures = []
            for row in record_reports:
                for gate_name, gate in row["full_graph"]["gates"].items():
                    if not gate["passed"]:
                        gate_failures.append(f"{row['selection']['sample_id']}:{gate_name}")
            classification_metrics["invalid_failures"] = gate_failures
            report["classification_inputs"] = classification_metrics
            report["classification"] = classify(classification_metrics)
        rollback_failures = []
        for row in record_reports:
            repeatability = row["training_window"]["repeatability"]
            if repeatability is not None and not repeatability["passed"]:
                rollback_failures.append(f"{row['selection']['sample_id']}:repeatability")
        restore_complete_state(model, initial_state)
        assert_complete_state_equal(model, initial_state)
        report["rollback_gate"] = {"passed": not rollback_failures, "failures": rollback_failures}
        if rollback_failures:
            raise RuntimeError(f"repeatability/rollback gate failed: {rollback_failures}")
        input_hashes_after = {name: sha256(path) for name, path in bound_paths.items()}
        if input_hashes_after != input_hashes_before:
            raise RuntimeError("an input hash changed during the diagnostic")
        report["input_hashes_after"] = input_hashes_after
        preregistration_sha256_after = sha256(args.preregistration)
        report["preregistration_sha256_observed_after"] = preregistration_sha256_after
        report["preregistration_hash_unchanged"] = (
            preregistration_sha256_after == preregistration_sha256_before
        )
        if not report["preregistration_hash_unchanged"]:
            raise RuntimeError("preregistration changed during the diagnostic")
        elapsed = time.time() - started
        wall_limit = float(prereg["budgets"][f"{mode_name}_wall_seconds_max"])
        peak = int(torch.cuda.max_memory_allocated(device))
        report["resources"] = {
            "elapsed_seconds": elapsed,
            "wall_seconds_max": wall_limit,
            "peak_allocated_bytes": peak,
            "peak_allocated_bytes_max": int(prereg["budgets"]["peak_allocated_bytes_max"]),
        }
        if elapsed > wall_limit:
            raise RuntimeError(f"wall-time budget exceeded: {elapsed:.3f}s > {wall_limit:.3f}s")
        report["status"] = "complete"
        report["promotion_authorized"] = False
        report["validation_opened"] = False
        report["test_id_opened"] = False
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
        if len(encoded) > int(prereg["budgets"]["new_disk_bytes_max"]):
            raise RuntimeError("JSON report exceeds the registered new-disk budget")
        atomic_json(report, args.output)
        print(json.dumps({"status": "complete", "mode": mode_name, "output": str(args.output.resolve())}, sort_keys=True))
        return 0
    except Exception as error:
        if model is not None and initial_state is not None:
            try:
                restore_complete_state(model, initial_state)
                assert_complete_state_equal(model, initial_state)
            except Exception as rollback_error:
                report["rollback_error"] = repr(rollback_error)
        try:
            preregistration_sha256_after = sha256(args.preregistration)
            report["preregistration_sha256_observed_after"] = preregistration_sha256_after
            report["preregistration_hash_unchanged"] = (
                preregistration_sha256_after == preregistration_sha256_before
            )
        except Exception as preregistration_hash_error:
            report["preregistration_hash_after_error"] = repr(preregistration_hash_error)
        report.update(
            {
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "classification": {
                    "classification": "invalid",
                    "flags": {"invalid": True},
                    "invalid_failures": [repr(error)],
                },
                "promotion_authorized": False,
                "validation_opened": False,
                "test_id_opened": False,
            }
        )
        # Failure records are also atomic and contain no model state.
        atomic_json(report, args.output)
        raise
    finally:
        if cache is not None:
            cache.close()


if __name__ == "__main__":
    raise SystemExit(main())
