#!/usr/bin/env python3
"""Offline meta-training of the onset hypernetwork (stage B of the two-stage scheme).

Learns one shared "early snapshots -> full-field compensation" mapping across a
family-balanced set of train-split instances.  The full future wavefield is
opened ONLY for train episodes; the resulting checkpoint is loaded at deployment
time (``scripts/run_meta_instance_adaptation.py``), where the guarded adapter
still receives exactly the two early onset snapshots and only its per-instance
``latent_delta`` (the deployment LoRA) is fine-tuned.

The compensation is the coarse-bypassing full-field residual head
(``OnsetAdaptedV5.residual`` added on top of the corrected parent field), so
meta-training directly targets the confirmed structural bottleneck (Gibbs
ringing on sharp uniform-medium wavefronts and late-time frames).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
import torch.distributed as dist
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet_v3.data.index import ALLOWED_MEDIUM_TYPES, build_manifest
from saved_time_phase_operator_v4.instance_adaptation.adapters import (
    ADAPTER_SCHEMA_VERSION,
    OnsetAdaptedV5,
)
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.instance_adaptation.feature_modulation import (
    energy_balanced_relative_loss,
    metric_aligned_relative_loss,
)
from saved_time_phase_operator_v4.instance_adaptation.chonknoris import (
    supervise_cholesky_linearizations,
)
from saved_time_phase_operator_v4.instance_adaptation.trainer import (
    high_residual_spatial_indices,
    linearize_supervised_chonknoris_batch,
)
from scripts.run_v5_instance_adaptation import (
    _load_background_provider,
    _load_parent,
    resolve_saved_time_parent_config,
)
from scripts.train_v5_feature_meta import build_balanced_meta_episodes
from scripts.train_v5_residual_meta import (
    _sha256,
    _time_indices,
)


def _distributed_context(device: str) -> tuple[int, int, torch.device, bool]:
    """Initialize torchrun workers and bind each worker to one local GPU."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if world_size > 1:
        if not dist.is_available():
            raise RuntimeError("distributed training is unavailable in this PyTorch build")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("torchrun requested CUDA but CUDA is unavailable")
        backend = "nccl" if device == "cuda" else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
            initialized_here = True
        if device == "cuda":
            torch.cuda.set_device(local_rank)
            target_device = torch.device("cuda", local_rank)
        else:
            target_device = torch.device(device)
    else:
        target_device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
    return rank, world_size, target_device, initialized_here


def _broadcast_trainable_parameters(
    parameters: list[torch.nn.Parameter], *, world_size: int
) -> None:
    """Make the initial adapter state identical on every torchrun worker."""

    if world_size <= 1:
        return
    for parameter in parameters:
        dist.broadcast(parameter.data, src=0)


def _average_gradients(
    parameters: list[torch.nn.Parameter], *, world_size: int
) -> None:
    """Synchronize gradients for method-based forwards that bypass DDP.forward."""

    if world_size <= 1:
        return
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(float(world_size))


def _logical_batches(
    cached: list[dict[str, object]],
    *,
    batch_size: int,
    vectorized: bool,
) -> list[list[dict[str, object]]]:
    """Pack equal-length episodes together without changing sample coverage."""

    size = int(batch_size)
    if size <= 0:
        raise ValueError("logical batch size must be positive")
    if not vectorized:
        return [cached[start : start + size] for start in range(0, len(cached), size)]
    buckets: dict[int, list[dict[str, object]]] = {}
    for sample in cached:
        buckets.setdefault(int(len(sample["indices"])), []).append(sample)
    batches: list[list[dict[str, object]]] = []
    leftovers: list[dict[str, object]] = []
    for key in sorted(buckets):
        values = buckets[key]
        complete = (len(values) // size) * size
        batches.extend(
            values[start : start + size] for start in range(0, complete, size)
        )
        leftovers.extend(values[complete:])
    batches.extend(
        leftovers[start : start + size]
        for start in range(0, len(leftovers), size)
    )
    if sum(len(batch) for batch in batches) != len(cached):
        raise AssertionError("vectorized batch planner changed episode coverage")
    return batches


def _compatible_subgroups(
    samples: list[dict[str, object]], *, maximum_size: int
) -> list[list[dict[str, object]]]:
    """Split one optimizer batch into shape-compatible vmap chunks."""

    size = int(maximum_size)
    if size <= 0:
        raise ValueError("maximum vmap size must be positive")
    buckets: dict[int, list[dict[str, object]]] = {}
    for sample in samples:
        buckets.setdefault(int(len(sample["indices"])), []).append(sample)
    return [
        values[start : start + size]
        for key in sorted(buckets)
        for values in (buckets[key],)
        for start in range(0, len(values), size)
    ]


def _parent_normalized_at_times(
    parent,
    normalizer,
    record,
    device,
    time_s,
    *,
    frame_indices=None,
    background_provider=None,
    time_block: int = 1,
):
    """Parent field in NORMALIZED space (O(1) scale).

    Decoded pressure on this dataset is ~1e-9, which makes an L2 fit numerically
    ill-conditioned and lets a single episode blow the shared residual up.  All
    meta-training, deployment fine-tuning, and loss evaluation therefore happen
    in the normalizer's O(1) space; only the final deployed field is decoded.
    """
    velocity = record.velocity_mps.to(device).unsqueeze(0)
    source = record.source_parameters.to(device).unsqueeze(0)
    source_map = record.source_map.to(device).unsqueeze(0)
    prepared = parent.prepare_sources(
        parent.encode_medium(velocity, normalizer),
        source,
        source_map,
        normalizer,
        record_to_medium=torch.zeros(1, dtype=torch.long, device=device),
    )
    dense_grid = parent.prepare_dense_grid(
        prepared,
        x_m=record.x_m.to(device),
        z_m=record.z_m.to(device),
        travel_time_s=(
            None
            if record.dense_travel_time_s is None
            else record.dense_travel_time_s.to(device).unsqueeze(0)
        ),
    )
    block = int(time_block)
    if block <= 0:
        raise ValueError("parent inference time_block must be positive")
    with torch.no_grad():
        field = parent.dense_normalized(
            prepared,
            time_s.to(device),
            dense_grid=dense_grid,
            time_block=block,
        )
        if background_provider is not None:
            if frame_indices is None:
                raise ValueError("background parent prediction requires frame indices")
            indices = torch.as_tensor(frame_indices, dtype=torch.long).reshape(1, -1)
            pbg_phys = background_provider.physical(
                [record.sample_id], indices, device=device, dtype=field.dtype
            )
            field = field + normalizer.encode_pressure(
                pbg_phys, record.source_parameters.to(device)[None, 4]
            )
        return field


def residual_adaptive_time_indices(
    candidate_indices: torch.Tensor,
    frame_residual: torch.Tensor,
    observed_indices: tuple[int, int],
    count: int,
    *,
    high_fraction: float = 0.75,
) -> torch.Tensor:
    """Compress a train-only candidate pool while retaining residual hotspots.

    The two causal onset observations are mandatory.  ``high_fraction`` of the
    remaining budget takes the largest normalized frame residuals; the rest is
    assigned to deterministic full-axis anchors so difficult frames do not
    destroy temporal coverage.  Future truth is used only to build offline
    training episodes and never by validation or deployment.
    """

    candidates = torch.as_tensor(candidate_indices, dtype=torch.long).detach().cpu()
    scores = torch.as_tensor(frame_residual, dtype=torch.float64).detach().cpu()
    if candidates.ndim != 1 or scores.ndim != 1 or len(candidates) != len(scores):
        raise ValueError("candidate indices and residual scores must be equal vectors")
    if len(candidates) < 2 or candidates.tolist() != sorted(set(candidates.tolist())):
        raise ValueError("candidate time indices must be sorted and unique")
    if not torch.isfinite(scores).all() or bool((scores < 0).any()):
        raise ValueError("frame residual scores must be finite and nonnegative")
    requested = int(count)
    if requested < 2 or requested > len(candidates):
        raise ValueError("adaptive time count must lie in [2, candidate_count]")
    fraction = float(high_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("adaptive high-residual fraction must lie in [0,1]")

    onset = tuple(int(value) for value in observed_indices)
    if len(set(onset)) != 2 or any(value not in candidates.tolist() for value in onset):
        raise ValueError("both distinct observed time indices must be candidates")
    index_to_score = {
        int(index): float(score) for index, score in zip(candidates, scores)
    }
    selected = set(onset)
    remaining_budget = requested - len(selected)
    high_count = min(
        remaining_budget,
        int(math.ceil(remaining_budget * fraction)),
    )
    residual_rank = sorted(
        (int(index) for index in candidates if int(index) not in selected),
        key=lambda index: (-index_to_score[index], index),
    )
    selected.update(residual_rank[:high_count])

    coverage_count = remaining_budget - high_count
    if coverage_count:
        anchors = torch.linspace(
            float(candidates[0]), float(candidates[-1]), coverage_count
        ).tolist()
        for anchor in anchors:
            available = [
                int(index) for index in candidates if int(index) not in selected
            ]
            if not available:
                break
            selected.add(min(available, key=lambda index: (abs(index - anchor), index)))

    # Anchor collisions can leave unused capacity only in very small pools.
    # Fill it by residual rank so the output contract remains exactly ``count``.
    for index in residual_rank:
        if len(selected) >= requested:
            break
        selected.add(index)
    if len(selected) != requested:
        raise AssertionError("adaptive time selector did not preserve its budget")
    return torch.tensor(sorted(selected), dtype=torch.long)


def train_meta(
    config_path: str | Path,
    *,
    output: str | Path,
    device: str = "cuda",
    per_family: int = 4,
    epochs: int = 1,
    time_points: int = 48,
    learning_rate: float = 2.0e-4,
    physics_weight: float = 0.0,
    chonknoris_weight: float = 0.0,
    chonknoris_relaxations: tuple[float, ...] = (1.0e-3, 1.0e-2, 1.0e-1),
    chonknoris_pool_size: int = 4,
    chonknoris_residual_sampling: str = "average_pool",
    adaptive_time_candidates: int = 0,
    adaptive_high_fraction: float = 0.75,
    observed_weight: float = 0.25,
    closed_form_output_ridge: float = 0.0,
    per_rank_batch_size: int = 1,
    jacobian_vmap_size: int = 1,
    smoke: bool = False,
) -> Path:
    config = yaml.safe_load(Path(config_path).read_text())
    source_h5 = Path(config["source_h5"]).expanduser()
    manifest = build_manifest(source_h5)
    rank, world_size, target_device, initialized_here = _distributed_context(device)
    seed = int(config.get("seed", 17))
    torch.manual_seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    parent, normalizer = _load_parent(config, manifest, target_device)
    background_provider = _load_background_provider(config)
    adapter = OnsetAdaptedV5(
        parent,
        latent_dim=int(config.get("latent_dim", 32)),
        lora_rank=int(config.get("lora_rank", 4)),
    ).to(target_device)
    # Stage B trains the whole meta network (conditioner + residual); the
    # per-instance latent_delta stays zero and is optimized only at deployment.
    adapter._deployment_mode = False
    adapter.train()

    family_count = 1 if smoke else int(per_family)
    global_episodes = build_balanced_meta_episodes(
        manifest,
        split="train",
        per_family=family_count,
        seed=seed,
        allowed_sample_ids=(
            None if background_provider is None else background_provider.sample_ids
        ),
    )
    if smoke:
        # A single episode per run keeps the smoke path fast while still
        # exercising the full forward/backward/save loop.
        global_episodes = global_episodes[:1]
    global_episode_count = len(global_episodes)
    if world_size > 1 and global_episode_count % world_size:
        raise ValueError(
            "the global meta episode count must be divisible by WORLD_SIZE: "
            f"episodes={global_episode_count}, world_size={world_size}"
        )
    episodes = global_episodes[rank::world_size]

    factor_weight = max(0.0, float(chonknoris_weight))
    residual_sampling = str(chonknoris_residual_sampling).strip().lower()
    if residual_sampling not in {"average_pool", "topk"}:
        raise ValueError("CHONKNORIS residual sampling must be average_pool or topk")
    adaptive_candidates = int(adaptive_time_candidates)
    if adaptive_candidates < 0:
        raise ValueError("adaptive time candidate count cannot be negative")
    if adaptive_candidates and adaptive_candidates < int(time_points):
        raise ValueError("adaptive time candidates must be at least time_points")
    adaptive_fraction = float(adaptive_high_fraction)
    if not math.isfinite(adaptive_fraction) or not 0.0 <= adaptive_fraction <= 1.0:
        raise ValueError("adaptive high-residual fraction must lie in [0,1]")
    onset_weight = float(observed_weight)
    if not math.isfinite(onset_weight) or onset_weight < 0.0:
        raise ValueError("observed loss weight must be finite and nonnegative")
    output_ridge = float(closed_form_output_ridge)
    if not math.isfinite(output_ridge) or output_ridge < 0.0:
        raise ValueError("closed-form output ridge must be finite and nonnegative")
    rank_batch_size = int(per_rank_batch_size)
    if rank_batch_size <= 0:
        raise ValueError("per-rank batch size must be positive")
    vmap_size = int(jacobian_vmap_size)
    if vmap_size <= 0 or vmap_size > rank_batch_size:
        raise ValueError("Jacobian vmap size must lie in [1, per-rank batch size]")
    trainable = list(adapter.adapter_parameters())
    if factor_weight > 0.0:
        trainable.extend(adapter.chonknoris_parameters())
    _broadcast_trainable_parameters(trainable, world_size=world_size)
    optimizer = torch.optim.AdamW(
        trainable, lr=float(learning_rate), weight_decay=1.0e-5
    )
    saved_time_parent = resolve_saved_time_parent_config(config)
    travel_time_h5 = config.get("travel_time_h5")
    if travel_time_h5 is None and saved_time_parent is not None:
        travel_time_h5 = saved_time_parent.get("travel_time_h5")
    dataset = GuardedOnsetDataset(
        source_h5,
        manifest,
        split="train",
        travel_time_h5=travel_time_h5,
    )
    records_by_source = {
        record.source_index: index
        for index, record in enumerate(dataset.records.records)
    }
    history: list[dict[str, float | int | str]] = []
    cached: list[dict[str, object]] = []
    epoch_mean_losses: list[float] = []
    epoch_post_update_mean_losses: list[float] = []
    best_epoch = 0
    best_epoch_mean_loss = math.inf
    best_conditioner_state = None
    best_residual_state = None
    best_chonknoris_state = None

    def cache_tensor(value: torch.Tensor) -> torch.Tensor:
        cached_value = value.detach().contiguous().cpu()
        return (
            cached_value.pin_memory()
            if target_device.type == "cuda"
            else cached_value
        )

    def cat_cached(samples, key: str) -> torch.Tensor:
        return torch.cat(
            [
                sample[key].to(target_device, non_blocking=True)
                for sample in samples
            ]
        )

    def stack_cached(samples, key: str) -> torch.Tensor:
        return torch.stack(
            [
                sample[key].to(target_device, non_blocking=True)
                for sample in samples
            ]
        )

    try:
        with h5py.File(source_h5, "r", swmr=True) as h5:
            for local_episode_index, metadata in enumerate(episodes):
                record = dataset[records_by_source[metadata.source_index]]
                velocity = record.velocity_mps.to(target_device).unsqueeze(0)
                source = record.source_parameters.to(target_device).unsqueeze(0)
                observed = record.observed_wavefield.to(target_device).unsqueeze(0)
                time_s = record.time_s.to(target_device)
                candidate_count = (
                    adaptive_candidates if adaptive_candidates else int(time_points)
                )
                candidate_indices = _time_indices(
                    len(time_s), record.observed_indices, candidate_count
                )
                candidate_truth_phys = torch.as_tensor(
                    np.asarray(
                        h5["wavefield"][
                            metadata.source_index, candidate_indices.tolist()
                        ],
                        dtype=np.float32,
                    ),
                    device=target_device,
                ).unsqueeze(0)
                candidate_time = time_s[candidate_indices.to(target_device)]
                # Everything below lives in the normalizer's O(1) space.
                candidate_truth = normalizer.encode_pressure(
                    candidate_truth_phys, source[:, 4]
                )
                observed = normalizer.encode_pressure(observed, source[:, 4])
                candidate_parent = _parent_normalized_at_times(
                    parent,
                    normalizer,
                    record,
                    target_device,
                    candidate_time,
                    frame_indices=candidate_indices,
                    background_provider=background_provider,
                )
                candidate_frame_error = (
                    (candidate_parent.float() - candidate_truth.float())
                    .flatten(2)
                    .norm(dim=-1)[0]
                )
                candidate_frame_norm = candidate_truth.float().flatten(2).norm(dim=-1)[0]
                candidate_denominator = torch.maximum(
                    candidate_frame_norm,
                    candidate_frame_norm.amax() * 0.01,
                ).clamp_min(torch.finfo(candidate_frame_error.dtype).tiny)
                candidate_frame_residual = candidate_frame_error / candidate_denominator
                if adaptive_candidates:
                    selected_count = min(int(time_points), len(candidate_indices))
                    indices = residual_adaptive_time_indices(
                        candidate_indices,
                        candidate_frame_residual,
                        record.observed_indices,
                        selected_count,
                        high_fraction=adaptive_fraction,
                    )
                    candidate_position = {
                        int(index): position
                        for position, index in enumerate(candidate_indices.tolist())
                    }
                    selected_positions = torch.tensor(
                        [candidate_position[int(index)] for index in indices],
                        dtype=torch.long,
                        device=target_device,
                    )
                else:
                    indices = candidate_indices
                    selected_positions = torch.arange(
                        len(indices), dtype=torch.long, device=target_device
                    )
                truth = candidate_truth[:, selected_positions]
                selected_parent = candidate_parent[:, selected_positions]
                selected_time = candidate_time[selected_positions]
                selected_frame_residual = candidate_frame_residual[selected_positions]
                cached.append(
                    {
                        # A formal rank caches 48 parent/truth fields.  Keeping
                        # them on CUDA consumes ~0.8 GiB before the vmap(jacfwd)
                        # workspace exists and can push a safe vmap=2 run OOM.
                        # Locked host cache + nonblocking subgroup transfer keeps
                        # the Jacobian phase near full utilization without that
                        # persistent device-memory tax.
                        "velocity": cache_tensor(velocity),
                        "source": cache_tensor(source),
                        "observed": cache_tensor(observed),
                        "truth": cache_tensor(truth),
                        "selected_parent": cache_tensor(selected_parent),
                        "selected_time": cache_tensor(selected_time),
                        "observed_indices": record.observed_indices,
                        "indices": indices,
                        "medium_type": record.medium_type,
                        "sample_id": record.sample_id,
                        "group_id": record.group_id,
                        "source_index": int(record.source_index),
                        "episode_index": local_episode_index * world_size + rank,
                        "candidate_time_points": int(len(candidate_indices)),
                        "candidate_residual_mean": float(
                            candidate_frame_residual.mean().detach().cpu()
                        ),
                        "selected_residual_mean": float(
                            selected_frame_residual.mean().detach().cpu()
                        ),
                    }
                )
            # The frozen parent is never called after its fields are cached.
            # Offload it before allocating the Jacobian workspace and return its
            # CUDA reservation to the allocator.
            if target_device.type == "cuda":
                adapter.parent.to("cpu")
                torch.cuda.empty_cache()
            batch_plan = _logical_batches(
                cached,
                batch_size=rank_batch_size,
                vectorized=vmap_size > 1,
            )

            def evaluate_cached_field_objective() -> float:
                """Post-update train-only score used to select reproducible weights."""

                local_sum = 0.0
                local_count = 0
                with torch.no_grad():
                    for evaluation_batch in batch_plan:
                        for subgroup in _compatible_subgroups(
                            evaluation_batch, maximum_size=rank_batch_size
                        ):
                            subgroup_size = len(subgroup)
                            velocity = cat_cached(subgroup, "velocity")
                            source = cat_cached(subgroup, "source")
                            observed = cat_cached(subgroup, "observed")
                            truth = cat_cached(subgroup, "truth")
                            selected_parent = cat_cached(subgroup, "selected_parent")
                            selected_time = stack_cached(subgroup, "selected_time")
                            prediction = adapter.raw_wavefield(
                                selected_parent,
                                velocity,
                                source,
                                observed,
                                selected_time,
                            )
                            field_per_sample = (
                                (prediction.float() - truth.float())
                                .flatten(1)
                                .norm(dim=-1)
                                / truth.float().flatten(1).norm(dim=-1).clamp_min(1.0e-8)
                            )
                            onset_positions = torch.tensor(
                                [
                                    [
                                        int(
                                            (sample["indices"] == value)
                                            .nonzero(as_tuple=False)[0]
                                        )
                                        for value in sample["observed_indices"]
                                    ]
                                    for sample in subgroup
                                ],
                                dtype=torch.long,
                                device=target_device,
                            )
                            rows = torch.arange(
                                subgroup_size, device=target_device
                            )[:, None]
                            observed_error = (
                                prediction[rows, onset_positions] - observed
                            ).flatten(1).norm(dim=1)
                            record_norm = truth.flatten(1).norm(dim=1).clamp_min(1.0e-8)
                            score = (
                                field_per_sample
                                + onset_weight * observed_error / record_norm
                            )
                            if float(physics_weight) > 0.0:
                                target_norm = truth.flatten(2).norm(dim=-1)
                                denominator = torch.maximum(
                                    target_norm,
                                    target_norm.amax(dim=1, keepdim=True) * 0.01,
                                ).clamp_min(torch.finfo(prediction.dtype).tiny)
                                score = score + float(physics_weight) * (
                                    (prediction - truth).flatten(2).norm(dim=-1)
                                    / denominator
                                ).mean(dim=1)
                            local_sum += float(score.sum())
                            local_count += subgroup_size
                totals = torch.tensor(
                    [local_sum, float(local_count)],
                    dtype=torch.float64,
                    device=target_device,
                )
                if world_size > 1:
                    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                return float(totals[0] / totals[1].clamp_min(1.0))

            def fit_closed_form_residual_output(
                ridge_value: float,
            ) -> tuple[int, float]:
                """Fit the final 1x1 residual map on every selected hotspot."""

                feature_count = int(adapter.residual.output.in_channels) + 1
                normal = torch.zeros(
                    feature_count,
                    feature_count,
                    dtype=torch.float64,
                    device=target_device,
                )
                right = torch.zeros(
                    feature_count, dtype=torch.float64, device=target_device
                )
                row_count = 0
                point_count = int(chonknoris_pool_size) ** 2
                with torch.no_grad():
                    for fit_batch in batch_plan:
                        for subgroup in _compatible_subgroups(
                            fit_batch, maximum_size=rank_batch_size
                        ):
                            velocity = cat_cached(subgroup, "velocity")
                            source = cat_cached(subgroup, "source")
                            observed = cat_cached(subgroup, "observed")
                            truth = cat_cached(subgroup, "truth")
                            selected_parent = cat_cached(subgroup, "selected_parent")
                            selected_time = stack_cached(subgroup, "selected_time")
                            indices = high_residual_spatial_indices(
                                selected_parent - truth,
                                points_per_frame=point_count,
                            )
                            latent = adapter.conditioner(
                                velocity, source, observed
                            )
                            features = adapter.residual.sampled_linear_features(
                                velocity,
                                source,
                                observed,
                                latent,
                                selected_time,
                                indices,
                            )
                            scale = (
                                selected_parent.detach()
                                .float()
                                .flatten(1)
                                .std(dim=1)
                                .clamp_min(1.0e-8)
                            )
                            desired = (truth - selected_parent) / scale[
                                :, None, None, None
                            ]
                            desired = torch.gather(
                                desired.flatten(-2), -1, indices
                            ).clamp(-0.95, 0.95)
                            target_linear = torch.atanh(desired).reshape(-1).double()
                            design = features.reshape(-1, features.shape[-1]).double()
                            design = torch.cat(
                                (
                                    design,
                                    torch.ones(
                                        design.shape[0],
                                        1,
                                        dtype=design.dtype,
                                        device=design.device,
                                    ),
                                ),
                                dim=1,
                            )
                            normal.add_(design.mT @ design)
                            right.add_(design.mT @ target_linear)
                            row_count += int(design.shape[0])
                if world_size > 1:
                    dist.all_reduce(normal, op=dist.ReduceOp.SUM)
                    dist.all_reduce(right, op=dist.ReduceOp.SUM)
                    count_tensor = torch.tensor(
                        row_count, dtype=torch.int64, device=target_device
                    )
                    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
                    row_count = int(count_tensor)
                if row_count <= 0:
                    raise RuntimeError("closed-form residual fit received no points")
                normal.div_(float(row_count))
                right.div_(float(row_count))
                identity = torch.eye(
                    feature_count, dtype=normal.dtype, device=normal.device
                )
                regularized = normal + float(ridge_value) * identity
                factor, info = torch.linalg.cholesky_ex(regularized)
                if int(info) != 0:
                    regularized = regularized + max(
                        float(ridge_value), 1.0e-8
                    ) * identity
                    factor = torch.linalg.cholesky(regularized)
                solution = torch.cholesky_solve(
                    right[:, None], factor
                )[:, 0]
                if not torch.isfinite(solution).all():
                    raise FloatingPointError("closed-form residual output is nonfinite")
                with torch.no_grad():
                    adapter.residual.output.weight.copy_(
                        solution[:-1].to(adapter.residual.output.weight).reshape_as(
                            adapter.residual.output.weight
                        )
                    )
                    adapter.residual.output.bias.copy_(
                        solution[-1:].to(adapter.residual.output.bias)
                    )
                condition = float(torch.linalg.cond(regularized))
                return row_count, condition

            initial_mean_loss = evaluate_cached_field_objective()
            best_epoch_mean_loss = initial_mean_loss
            best_stage = "initial_parent"
            best_conditioner_state = {
                name: value.detach().cpu().clone()
                for name, value in adapter.conditioner.state_dict().items()
            }
            best_residual_state = {
                name: value.detach().cpu().clone()
                for name, value in adapter.residual.state_dict().items()
            }
            best_chonknoris_state = (
                {
                    name: value.detach().cpu().clone()
                    for name, value in adapter.chonknoris.state_dict().items()
                }
                if factor_weight > 0.0
                else None
            )
            ridge_prefit_mean_loss = None
            ridge_prefit_rows = 0
            ridge_prefit_condition = None
            ridge_prefit_selected_ridge = None
            ridge_prefit_sweep: list[dict[str, float | int]] = []
            if output_ridge > 0.0:
                initial_residual_state = best_residual_state
                for exponent in range(6):
                    adapter.residual.load_state_dict(
                        initial_residual_state, strict=True
                    )
                    ridge_candidate = output_ridge * (10.0 ** exponent)
                    rows, condition = fit_closed_form_residual_output(
                        ridge_candidate
                    )
                    score = evaluate_cached_field_objective()
                    ridge_prefit_sweep.append(
                        {
                            "ridge": ridge_candidate,
                            "mean_loss": score,
                            "rows": rows,
                            "condition_number": condition,
                        }
                    )
                    if ridge_prefit_mean_loss is None or score < ridge_prefit_mean_loss:
                        ridge_prefit_mean_loss = score
                        ridge_prefit_rows = rows
                        ridge_prefit_condition = condition
                        ridge_prefit_selected_ridge = ridge_candidate
                    if score < best_epoch_mean_loss:
                        best_epoch_mean_loss = score
                        best_stage = "closed_form_output_ridge"
                        best_residual_state = {
                            name: value.detach().cpu().clone()
                            for name, value in adapter.residual.state_dict().items()
                        }
                adapter.residual.load_state_dict(best_residual_state, strict=True)
                if rank == 0:
                    print(
                        json.dumps(
                            {
                                "event": "closed_form_output_fit",
                                "rows": ridge_prefit_rows,
                                "condition_number": ridge_prefit_condition,
                                "selected_ridge": ridge_prefit_selected_ridge,
                                "sweep": ridge_prefit_sweep,
                                "initial_mean_loss": initial_mean_loss,
                                "ridge_mean_loss": ridge_prefit_mean_loss,
                                "accepted_as_best": (
                                    best_stage == "closed_form_output_ridge"
                                ),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            optimizer_steps_per_epoch = len(batch_plan)
            for epoch in range(int(epochs)):
                local_samples_seen = 0
                local_epoch_loss_sum = 0.0
                local_epoch_sample_count = 0
                for optimizer_step, logical_batch in enumerate(batch_plan, start=1):
                    optimizer.zero_grad(set_to_none=True)
                    logical_batch_size = len(logical_batch)
                    logical_loss = torch.zeros((), device=target_device)
                    for subgroup in _compatible_subgroups(
                        logical_batch, maximum_size=vmap_size
                    ):
                        subgroup_size = len(subgroup)
                        velocity = cat_cached(subgroup, "velocity")
                        source = cat_cached(subgroup, "source")
                        observed = cat_cached(subgroup, "observed")
                        truth = cat_cached(subgroup, "truth")
                        selected_parent = cat_cached(subgroup, "selected_parent")
                        selected_time = stack_cached(subgroup, "selected_time")
                        prediction = adapter.raw_wavefield(
                            selected_parent,
                            velocity,
                            source,
                            observed,
                            selected_time,
                        )
                        field_loss = metric_aligned_relative_loss(prediction, truth)
                        field_per_sample = (
                            (prediction.float() - truth.float()).flatten(1).norm(dim=-1)
                            / truth.float().flatten(1).norm(dim=-1).clamp_min(1.0e-8)
                        )
                        onset_positions = torch.tensor(
                            [
                                [
                                    int(
                                        (sample["indices"] == value)
                                        .nonzero(as_tuple=False)[0]
                                    )
                                    for value in sample["observed_indices"]
                                ]
                                for sample in subgroup
                            ],
                            dtype=torch.long,
                            device=target_device,
                        )
                        rows = torch.arange(subgroup_size, device=target_device)[:, None]
                        observed_error = (
                            prediction[rows, onset_positions] - observed
                        ).flatten(1).norm(dim=1)
                        record_norm = truth.flatten(1).norm(dim=1).clamp_min(1.0e-8)
                        observed_per_sample = observed_error / record_norm
                        observed_loss = observed_per_sample.mean()
                        energy_loss = energy_balanced_relative_loss(prediction, truth)
                        target_norm = truth.flatten(2).norm(dim=-1)
                        denominator = torch.maximum(
                            target_norm,
                            target_norm.amax(dim=1, keepdim=True) * 0.01,
                        ).clamp_min(torch.finfo(prediction.dtype).tiny)
                        energy_per_sample = (
                            (prediction - truth).flatten(2).norm(dim=-1) / denominator
                        ).mean(dim=1)
                        chonknoris_loss = prediction.new_zeros(())
                        factor_relative_error = prediction.new_zeros(())
                        operator_relative_error = prediction.new_zeros(())
                        factor_condition = 0.0
                        if factor_weight > 0.0:
                            residuals, jacobians, contexts = (
                                linearize_supervised_chonknoris_batch(
                                    adapter,
                                    selected_parent,
                                    velocity,
                                    source,
                                    observed,
                                    selected_time,
                                    truth,
                                    pool_size=int(chonknoris_pool_size),
                                    residual_sampling=residual_sampling,
                                )
                            )
                            supervision = supervise_cholesky_linearizations(
                                adapter.chonknoris,
                                residuals,
                                jacobians,
                                contexts,
                                chonknoris_relaxations,
                            )
                            chonknoris_loss = supervision.loss
                            factor_relative_error = supervision.factor_relative_error
                            operator_relative_error = supervision.operator_relative_error
                            factor_condition = supervision.condition_number
                            del residuals, jacobians, contexts
                        loss = (
                            field_loss
                            + onset_weight * observed_loss
                            + float(physics_weight) * energy_loss
                            + factor_weight * chonknoris_loss
                        )
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite meta-training loss")
                        subgroup_weight = subgroup_size / float(logical_batch_size)
                        (loss * subgroup_weight).backward()
                        logical_loss = logical_loss + loss.detach() * subgroup_weight
                        factor_value = float(factor_relative_error.detach())
                        operator_value = float(operator_relative_error.detach())
                        chonknoris_value = float(chonknoris_loss.detach())
                        per_sample_loss = (
                            field_per_sample.detach()
                            + onset_weight * observed_per_sample.detach()
                            + float(physics_weight) * energy_per_sample.detach()
                            + factor_weight * chonknoris_value
                        )
                        local_epoch_loss_sum += float(per_sample_loss.sum())
                        local_epoch_sample_count += subgroup_size
                        for sample_index, sample in enumerate(subgroup):
                            episode_index = int(sample["episode_index"])
                            history.append(
                                {
                                    "epoch": epoch,
                                    "step": episode_index,
                                    "local_step": episode_index // world_size,
                                    "rank": rank,
                                    "medium_type": str(sample["medium_type"]),
                                    "sample_id": str(sample["sample_id"]),
                                    "group_id": str(sample["group_id"]),
                                    "source_index": int(sample["source_index"]),
                                    "loss": float(per_sample_loss[sample_index]),
                                    "field_loss": float(field_per_sample[sample_index]),
                                    "observed_loss": float(
                                        observed_per_sample[sample_index]
                                    ),
                                    "energy_loss": float(energy_per_sample[sample_index]),
                                    "chonknoris_loss": chonknoris_value,
                                    "chonknoris_factor_relative_error": factor_value,
                                    "chonknoris_operator_relative_error": operator_value,
                                    "chonknoris_condition_number": float(
                                        factor_condition
                                    ),
                                    "time_points": int(len(sample["indices"])),
                                    "jacobian_vmap_size": subgroup_size,
                                    "candidate_time_points": int(
                                        sample["candidate_time_points"]
                                    ),
                                    "candidate_residual_mean": float(
                                        sample["candidate_residual_mean"]
                                    ),
                                    "selected_residual_mean": float(
                                        sample["selected_residual_mean"]
                                    ),
                                    "selected_time_indices": [
                                        int(value) for value in sample["indices"].tolist()
                                    ],
                                }
                            )
                        local_samples_seen += subgroup_size
                    _average_gradients(trainable, world_size=world_size)
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    if rank == 0 and (
                        optimizer_step == 1
                        or optimizer_step % 4 == 0
                        or optimizer_step == optimizer_steps_per_epoch
                    ):
                        print(
                            json.dumps(
                                {
                                    "event": "train_progress",
                                    "epoch": epoch + 1,
                                    "epochs": int(epochs),
                                    "optimizer_step": optimizer_step,
                                    "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                                    "per_rank_batch_size": rank_batch_size,
                                    "jacobian_vmap_size": vmap_size,
                                    "global_batch_size": rank_batch_size * world_size,
                                    "effective_samples_seen": (
                                        epoch * global_episode_count
                                        + local_samples_seen * world_size
                                    ),
                                    "loss_rank0": float(logical_loss),
                                    "world_size": world_size,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                epoch_totals = torch.tensor(
                    [local_epoch_loss_sum, float(local_epoch_sample_count)],
                    dtype=torch.float64,
                    device=target_device,
                )
                if world_size > 1:
                    dist.all_reduce(epoch_totals, op=dist.ReduceOp.SUM)
                epoch_mean_loss = float(
                    epoch_totals[0] / epoch_totals[1].clamp_min(1.0)
                )
                epoch_mean_losses.append(epoch_mean_loss)
                post_update_mean_loss = evaluate_cached_field_objective()
                epoch_post_update_mean_losses.append(post_update_mean_loss)
                if post_update_mean_loss < best_epoch_mean_loss:
                    best_epoch = epoch + 1
                    best_epoch_mean_loss = post_update_mean_loss
                    best_stage = f"epoch_{epoch + 1}"
                    best_conditioner_state = {
                        name: value.detach().cpu().clone()
                        for name, value in adapter.conditioner.state_dict().items()
                    }
                    best_residual_state = {
                        name: value.detach().cpu().clone()
                        for name, value in adapter.residual.state_dict().items()
                    }
                    best_chonknoris_state = (
                        {
                            name: value.detach().cpu().clone()
                            for name, value in adapter.chonknoris.state_dict().items()
                        }
                        if factor_weight > 0.0
                        else None
                    )
                if rank == 0:
                    print(
                        json.dumps(
                            {
                                "event": "epoch_complete",
                                "epoch": epoch + 1,
                                "global_mean_loss": epoch_mean_loss,
                                "post_update_global_mean_loss": post_update_mean_loss,
                                "best_epoch": best_epoch,
                                "best_stage": best_stage,
                                "best_global_mean_loss": best_epoch_mean_loss,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            if best_conditioner_state is not None and best_residual_state is not None:
                adapter.conditioner.load_state_dict(best_conditioner_state, strict=True)
                adapter.residual.load_state_dict(best_residual_state, strict=True)
                if factor_weight > 0.0 and best_chonknoris_state is not None:
                    adapter.chonknoris.load_state_dict(
                        best_chonknoris_state, strict=True
                    )
    finally:
        dataset.close()
        if background_provider is not None:
            background_provider.close()

    if world_size > 1:
        # vmap(jacfwd) intentionally drives CUDA close to capacity.  NCCL's
        # object collectives stage serialized payloads on the current GPU and
        # can therefore OOM after every optimizer update has already finished.
        # Release inactive training reservations and gather Python history over
        # a CPU/Gloo group; gradient synchronization remains on NCCL.
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
            torch.cuda.empty_cache()
        gathered_history: list[list[dict[str, float | int | str]] | None] | None = (
            [None] * world_size if rank == 0 else None
        )
        object_group = dist.new_group(backend="gloo")
        try:
            dist.gather_object(
                history, gathered_history, dst=0, group=object_group
            )
        finally:
            dist.destroy_process_group(object_group)
        if rank == 0:
            history = [
                item
                for rank_history in gathered_history or []
                for item in rank_history or []
            ]
            history.sort(key=lambda item: (int(item["epoch"]), int(item["step"])))
        else:
            history = []

    peak_cuda_bytes = (
        int(torch.cuda.max_memory_allocated(target_device))
        if target_device.type == "cuda"
        else 0
    )
    if world_size > 1:
        peak_tensor = torch.tensor(
            peak_cuda_bytes, dtype=torch.int64, device=target_device
        )
        dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
        peak_cuda_bytes = int(peak_tensor.item())

    output_path = Path(output)
    if rank == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        parent_checkpoint = Path(str(config["parent_checkpoint"])).resolve()
        torch.save(
            {
                "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
                "conditioner_state": adapter.conditioner.state_dict(),
                "residual_state": adapter.residual.state_dict(),
                "chonknoris_state": (
                    adapter.chonknoris.state_dict() if factor_weight > 0.0 else None
                ),
                "latent_dim": int(config.get("latent_dim", 32)),
                "lora_rank": int(config.get("lora_rank", 4)),
                "source_h5_sha256": _sha256(source_h5),
                "manifest_digest": manifest.digest,
                "parent_kind": str(config.get("parent_kind", "saved_time_v4")),
                "parent_checkpoint": str(parent_checkpoint),
                "parent_checkpoint_sha256": _sha256(parent_checkpoint),
                "parent_run_identity": config.get(
                    "parent_run_identity", config.get("parent_checkpoint_identity")
                ),
                "background_cache": config.get("background_cache"),
                "future_truth_used_only_for_train_episode": True,
                "episodes": global_episode_count,
                "local_episodes": len(episodes),
                "per_family": family_count,
                "epochs": int(epochs),
                "world_size": world_size,
                "per_rank_batch_size": rank_batch_size,
                "jacobian_vmap_size": vmap_size,
                "peak_cuda_bytes": peak_cuda_bytes,
                "global_batch_size": rank_batch_size * world_size,
                "synchronous_optimizer_steps": (
                    math.ceil(len(episodes) / rank_batch_size) * int(epochs)
                ),
                "effective_sample_updates": global_episode_count * int(epochs),
                "time_points": int(time_points),
                "physics_weight": float(physics_weight),
                "chonknoris_weight": factor_weight,
                "chonknoris_relaxations": tuple(
                    float(value) for value in chonknoris_relaxations
                ),
                "chonknoris_pool_size": int(chonknoris_pool_size),
                "chonknoris_residual_sampling": residual_sampling,
                "adaptive_time_candidates": adaptive_candidates,
                "adaptive_high_fraction": adaptive_fraction,
                "observed_weight": onset_weight,
                "closed_form_output_ridge": output_ridge,
                "initial_mean_loss": initial_mean_loss,
                "ridge_prefit_mean_loss": ridge_prefit_mean_loss,
                "ridge_prefit_rows": ridge_prefit_rows,
                "ridge_prefit_condition_number": ridge_prefit_condition,
                "ridge_prefit_selected_ridge": ridge_prefit_selected_ridge,
                "ridge_prefit_sweep": ridge_prefit_sweep,
                "adaptive_sampling_uses_train_truth_only": bool(adaptive_candidates),
                "best_epoch": best_epoch,
                "best_stage": best_stage,
                "best_epoch_mean_loss": best_epoch_mean_loss,
                "epoch_mean_losses": epoch_mean_losses,
                "epoch_post_update_mean_losses": epoch_post_update_mean_losses,
                "adaptive_selection_manifest": [
                    {
                        "episode_index": int(sample["step"]),
                        "sample_id": str(sample["sample_id"]),
                        "group_id": str(sample["group_id"]),
                        "source_index": int(sample["source_index"]),
                        "medium_type": str(sample["medium_type"]),
                        "selected_time_indices": [
                            int(value) for value in sample["selected_time_indices"]
                        ],
                        "candidate_time_points": int(sample["candidate_time_points"]),
                        "candidate_residual_mean": float(
                            sample["candidate_residual_mean"]
                        ),
                        "selected_residual_mean": float(
                            sample["selected_residual_mean"]
                        ),
                    }
                    for sample in sorted(
                        (
                            value
                            for value in history
                            if int(value["epoch"]) == 0
                        ),
                        key=lambda value: int(value["step"]),
                    )
                ],
                "history": history,
            },
            output_path,
        )
    if world_size > 1:
        dist.barrier()
    if initialized_here:
        dist.destroy_process_group()
    return output_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--time-points", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--physics-weight", type=float, default=0.0)
    parser.add_argument("--chonknoris-weight", type=float, default=0.0)
    parser.add_argument(
        "--chonknoris-relaxation",
        action="append",
        type=float,
        help="Repeat to supervise multiple Tikhonov relaxations.",
    )
    parser.add_argument("--chonknoris-pool-size", type=int, default=4)
    parser.add_argument(
        "--chonknoris-residual-sampling",
        choices=("average_pool", "topk"),
        default="average_pool",
    )
    parser.add_argument("--adaptive-time-candidates", type=int, default=0)
    parser.add_argument("--adaptive-high-fraction", type=float, default=0.75)
    parser.add_argument("--observed-weight", type=float, default=0.25)
    parser.add_argument("--closed-form-output-ridge", type=float, default=0.0)
    parser.add_argument("--per-rank-batch-size", type=int, default=1)
    parser.add_argument("--jacobian-vmap-size", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    path = train_meta(
        args.config,
        output=args.output,
        device=args.device,
        per_family=args.per_family,
        epochs=args.epochs,
        time_points=args.time_points,
        learning_rate=args.learning_rate,
        physics_weight=args.physics_weight,
        chonknoris_weight=args.chonknoris_weight,
        chonknoris_relaxations=(
            tuple(args.chonknoris_relaxation)
            if args.chonknoris_relaxation
            else (1.0e-3, 1.0e-2, 1.0e-1)
        ),
        chonknoris_pool_size=args.chonknoris_pool_size,
        chonknoris_residual_sampling=args.chonknoris_residual_sampling,
        adaptive_time_candidates=args.adaptive_time_candidates,
        adaptive_high_fraction=args.adaptive_high_fraction,
        observed_weight=args.observed_weight,
        closed_form_output_ridge=args.closed_form_output_ridge,
        per_rank_batch_size=args.per_rank_batch_size,
        jacobian_vmap_size=args.jacobian_vmap_size,
        smoke=args.smoke,
    )
    if int(os.environ.get("RANK", "0")) != 0:
        return 0
    final = {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    history = payload.get("history", [])
    if history:
        final = {"final_loss": history[-1]["loss"], "steps": len(history)}
    print(json.dumps({"checkpoint": str(path), "future_truth_opened": "train_only", **final}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
