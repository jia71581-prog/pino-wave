#!/usr/bin/env python3
"""Train the shared multi-scale residual iterator on frozen parent predictions.

Future truth is opened only for train-split crop supervision.  The pretrained
parent is evaluated under ``torch.no_grad()``, moved off CUDA after caching, and
never included in the optimizer.  Each crop is a contiguous source-free time
window so the LWC-84 residual retains its discrete meaning.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
import torch.distributed as dist
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet_v3.data.index import ALLOWED_MEDIUM_TYPES, build_manifest
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.instance_adaptation.losses import lwc84_residual
from saved_time_phase_operator_v4.instance_adaptation.residual_iterator import (
    RESIDUAL_ITERATOR_SCHEMA_VERSION,
    MultiScaleResidualCorrector,
    ResidualIteratorLossWeights,
    align_lwc84_residual,
    refine_with_residual_iterator,
    residual_iterator_training_loss,
    unroll_residual_iterator,
)
from scripts.run_v5_instance_adaptation import _load_background_provider, _load_parent
from scripts.train_meta_hypernet import (
    _average_gradients,
    _broadcast_trainable_parameters,
    _distributed_context,
    _parent_normalized_at_times,
)
from scripts.train_v5_feature_meta import build_balanced_meta_episodes
from scripts.train_v5_residual_meta import _sha256


def _crop_coordinates(
    *,
    sample_index: int,
    epoch: int,
    seed: int,
    time_count: int,
    height: int,
    width: int,
    time_window: int,
    patch_size: int,
    source_free_start: int,
) -> tuple[int, int, int]:
    generator = torch.Generator().manual_seed(
        int(seed) + 1_000_003 * int(epoch) + 9_973 * int(sample_index)
    )
    maximum_time_start = time_count - time_window
    minimum_time_start = min(max(0, int(source_free_start)), maximum_time_start)
    time_start = int(
        torch.randint(
            minimum_time_start,
            maximum_time_start + 1,
            (1,),
            generator=generator,
        )
    )
    z_start = int(torch.randint(0, height - patch_size + 1, (1,), generator=generator))
    x_start = int(torch.randint(0, width - patch_size + 1, (1,), generator=generator))
    return time_start, z_start, x_start


def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (
        (prediction.float() - target.float()).flatten(1).norm(dim=1)
        / target.float().flatten(1).norm(dim=1).clamp_min(1.0e-8)
    )


def _family_weights_from_config(
    config: Mapping[str, object], key: str
) -> dict[str, float]:
    raw = config.get(key, {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{key} must be a mapping from family to positive weight")
    unknown = set(str(value) for value in raw) - set(ALLOWED_MEDIUM_TYPES)
    if unknown:
        raise ValueError(f"{key} contains unknown families: {sorted(unknown)}")
    result = {
        family: float(raw.get(family, 1.0)) for family in ALLOWED_MEDIUM_TYPES
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in result.values()):
        raise ValueError(f"{key} weights must be finite and positive")
    return result


def _background_supported_sample_ids(
    config: Mapping[str, object],
) -> tuple[str, ...] | None:
    """Return the cache support used to constrain offline training episodes."""

    if not config.get("background_cache"):
        return None
    provider = _load_background_provider(config)
    try:
        return tuple(provider.sample_ids)
    finally:
        provider.close()


def _select_smoke_episodes(episodes, *, count: int, family_weights: Mapping[str, float]):
    """Exercise the most emphasized families first in the memory smoke test."""

    requested = int(count)
    if requested <= 0:
        raise ValueError("smoke episode count must be positive")
    ranked = sorted(
        episodes,
        key=lambda item: (
            -float(family_weights[str(item.medium_type)]),
            tuple(ALLOWED_MEDIUM_TYPES).index(str(item.medium_type)),
        ),
    )
    if len(ranked) < requested:
        raise ValueError("smoke episode support is insufficient")
    return tuple(ranked[:requested])


def _weighted_epoch_indices(
    cached: list[dict[str, object]],
    *,
    family_weights: Mapping[str, float],
    seed: int,
    epoch: int,
) -> list[int]:
    """Build a deterministic family-weighted schedule with coverage guarantees."""

    if not cached:
        raise ValueError("cannot sample an empty residual-iterator cache")
    by_family = {
        family: [
            index
            for index, sample in enumerate(cached)
            if str(sample["medium_type"]) == family
        ]
        for family in ALLOWED_MEDIUM_TYPES
    }
    active = tuple(family for family, indices in by_family.items() if indices)
    generator = torch.Generator().manual_seed(
        int(seed) + 1_000_003 * int(epoch) + int(cached[0]["sample_index"])
    )
    active_values = [float(family_weights[family]) for family in active]
    if max(active_values) - min(active_values) <= 1.0e-12:
        return torch.randperm(len(cached), generator=generator).tolist()

    # Include every locally available family once, then allocate the remaining
    # update slots with replacement according to family importance.  Dividing
    # by family population makes the requested weights apply to families, not
    # accidentally to the number of cached records.
    schedule = []
    for family in active:
        indices = by_family[family]
        offset = int(torch.randint(len(indices), (1,), generator=generator))
        schedule.append(indices[offset])
    remaining = len(cached) - len(schedule)
    if remaining > 0:
        record_weights = torch.tensor(
            [
                float(family_weights[str(sample["medium_type"])])
                / len(by_family[str(sample["medium_type"])])
                for sample in cached
            ],
            dtype=torch.float64,
        )
        draws = torch.multinomial(
            record_weights,
            num_samples=remaining,
            replacement=True,
            generator=generator,
        ).tolist()
        schedule.extend(int(value) for value in draws)
    shuffle = torch.randperm(len(schedule), generator=generator).tolist()
    return [schedule[index] for index in shuffle]


def train_residual_iterator(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    parent_checkpoint: str | Path | None = None,
    parent_run_identity: str | Path | None = None,
    device: str = "cuda",
    per_family: int = 8,
    epochs: int = 5,
    time_window: int = 16,
    patch_size: int = 96,
    iterations: int = 4,
    width: int = 24,
    learning_rate: float = 2.0e-4,
    per_rank_batch_size: int = 1,
    smoke: bool = False,
) -> Path:
    config = yaml.safe_load(Path(config_path).read_text())
    if parent_checkpoint is not None:
        config["parent_checkpoint"] = str(Path(parent_checkpoint).resolve())
    if parent_run_identity is not None:
        config["parent_run_identity"] = str(Path(parent_run_identity).resolve())
    source_h5 = Path(config["source_h5"]).expanduser()
    manifest = build_manifest(source_h5)
    rank, world_size, target_device, initialized_here = _distributed_context(device)
    seed = int(config.get("seed", 17))
    family_loss_weights = _family_weights_from_config(config, "family_loss_weights")
    family_sampling_weights = _family_weights_from_config(
        config, "family_sampling_weights"
    )
    torch.manual_seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    rank_batch_size = int(per_rank_batch_size)
    if rank_batch_size <= 0:
        raise ValueError("per-rank residual-iterator batch size must be positive")
    family_count = 1 if smoke else int(per_family)
    background_sample_ids = _background_supported_sample_ids(config)
    global_episodes = build_balanced_meta_episodes(
        manifest,
        split="train",
        per_family=family_count,
        seed=seed,
        allowed_sample_ids=background_sample_ids,
    )
    if smoke:
        global_episodes = _select_smoke_episodes(
            global_episodes,
            count=rank_batch_size,
            family_weights=family_sampling_weights,
        )
    if world_size > 1 and len(global_episodes) % world_size:
        raise ValueError(
            "global iterator episodes must be divisible by WORLD_SIZE: "
            f"episodes={len(global_episodes)}, world_size={world_size}"
        )
    episodes = global_episodes[rank::world_size]
    if not episodes:
        raise ValueError("this rank received no iterator training episodes")
    if len(episodes) % rank_batch_size:
        raise ValueError(
            "per-rank iterator episodes must be divisible by the per-rank batch: "
            f"episodes={len(episodes)}, batch={rank_batch_size}"
        )

    parent, normalizer = _load_parent(config, manifest, target_device)
    background_provider = _load_background_provider(
        config, (metadata.sample_id for metadata in episodes)
    )
    corrector = MultiScaleResidualCorrector(width=int(width)).to(target_device)
    trainable = list(corrector.parameters())
    _broadcast_trainable_parameters(trainable, world_size=world_size)
    optimizer = torch.optim.AdamW(
        trainable, lr=float(learning_rate), weight_decay=1.0e-5
    )
    dataset = GuardedOnsetDataset(source_h5, manifest, split="train")
    records_by_source = {
        record.source_index: index
        for index, record in enumerate(dataset.records.records)
    }
    cached: list[dict[str, object]] = []

    def cache(value: torch.Tensor) -> torch.Tensor:
        result = value.detach().contiguous().cpu()
        return result.pin_memory() if target_device.type == "cuda" else result

    try:
        with h5py.File(source_h5, "r", swmr=True) as h5:
            for local_index, metadata in enumerate(episodes):
                record = dataset[records_by_source[metadata.source_index]]
                velocity = record.velocity_mps.to(target_device).unsqueeze(0)
                source = record.source_parameters.to(target_device).unsqueeze(0)
                time_s = record.time_s.to(target_device)
                all_indices = torch.arange(len(time_s), dtype=torch.long)
                truth_physical = torch.as_tensor(
                    np.asarray(h5["wavefield"][metadata.source_index], dtype=np.float32),
                    device=target_device,
                ).unsqueeze(0)
                truth = normalizer.encode_pressure(truth_physical, source[:, 4])
                starter = _parent_normalized_at_times(
                    parent,
                    normalizer,
                    record,
                    target_device,
                    time_s,
                    frame_indices=all_indices,
                    background_provider=background_provider,
                )
                dt = float(time_s[1] - time_s[0])
                # Ricker forcing is negligible after onset + 3/f.  Restrict
                # homogeneous residual crops to that source-free tail.
                source_free_time = float(source[0, 3]) + 3.0 / max(float(source[0, 2]), 1.0e-6)
                source_free_start = int(math.ceil(source_free_time / max(dt, 1.0e-12)))
                cached.append(
                    {
                        "sample_index": local_index * world_size + rank,
                        "sample_id": record.sample_id,
                        "medium_type": record.medium_type,
                        "velocity": cache(velocity),
                        "time_s": cache(time_s),
                        "truth": cache(truth),
                        "starter": cache(starter),
                        "source_free_start": source_free_start,
                    }
                )
        if target_device.type == "cuda":
            parent.to("cpu")
            torch.cuda.empty_cache()

        first_shape = cached[0]["starter"].shape
        _, available_times, height, spatial_width = first_shape
        window = min(int(time_window), int(available_times))
        patch = min(int(patch_size), int(height), int(spatial_width))
        if window < 4 or patch <= 8:
            raise ValueError("iterator crops require at least four times and a 9x9 grid")

        output = Path(output_dir)
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
        if world_size > 1:
            dist.barrier()
        metrics_path = output / "metrics.jsonl"
        started = time.perf_counter()
        best_score = math.inf
        best_epoch = 0
        best_state = None
        loss_weights = ResidualIteratorLossWeights()

        for epoch in range(int(epochs)):
            local_loss_sum = 0.0
            local_before_sum = 0.0
            local_after_sum = 0.0
            local_count = 0
            local_family_before = {family: 0.0 for family in ALLOWED_MEDIUM_TYPES}
            local_family_after = {family: 0.0 for family in ALLOWED_MEDIUM_TYPES}
            local_family_count = {family: 0 for family in ALLOWED_MEDIUM_TYPES}
            corrector.train()
            epoch_indices = _weighted_epoch_indices(
                cached,
                family_weights=family_sampling_weights,
                seed=seed,
                epoch=epoch,
            )
            for batch_start in range(0, len(epoch_indices), rank_batch_size):
                batch_indices = epoch_indices[batch_start : batch_start + rank_batch_size]
                samples = [cached[index] for index in batch_indices]
                starter_crops: list[torch.Tensor] = []
                truth_crops: list[torch.Tensor] = []
                velocity_crops: list[torch.Tensor] = []
                time_crops: list[torch.Tensor] = []
                for sample in samples:
                    starter_full = sample["starter"].to(target_device, non_blocking=True)
                    truth_full = sample["truth"].to(target_device, non_blocking=True)
                    velocity_full = sample["velocity"].to(target_device, non_blocking=True)
                    time_full = sample["time_s"].to(target_device, non_blocking=True)
                    t0, z0, x0 = _crop_coordinates(
                        sample_index=int(sample["sample_index"]),
                        epoch=epoch,
                        seed=seed,
                        time_count=starter_full.shape[1],
                        height=starter_full.shape[2],
                        width=starter_full.shape[3],
                        time_window=window,
                        patch_size=patch,
                        source_free_start=int(sample["source_free_start"]),
                    )
                    time_slice = slice(t0, t0 + window)
                    z_slice = slice(z0, z0 + patch)
                    x_slice = slice(x0, x0 + patch)
                    starter_crops.append(
                        starter_full[:, time_slice, z_slice, x_slice]
                    )
                    truth_crops.append(truth_full[:, time_slice, z_slice, x_slice])
                    velocity_crops.append(velocity_full[:, :, z_slice, x_slice])
                    time_crops.append(time_full[time_slice])
                starter = torch.cat(starter_crops, dim=0)
                truth = torch.cat(truth_crops, dim=0)
                velocity = torch.cat(velocity_crops, dim=0)
                times = torch.stack(time_crops, dim=0)

                optimizer.zero_grad(set_to_none=True)
                states, residuals = unroll_residual_iterator(
                    corrector,
                    starter,
                    velocity,
                    times,
                    (-1, 0),
                    iterations=int(iterations),
                    frozen_time_indices=(),
                )
                truth_residual = lwc84_residual(
                    truth,
                    velocity,
                    dt=float((times[:, 1:] - times[:, :-1]).mean()),
                    dx=10.0,
                    dz=10.0,
                    observed_indices=(-1, 0),
                )
                fixed_point_correction = corrector(
                    starter,
                    truth,
                    velocity,
                    align_lwc84_residual(truth_residual, truth, (-1, 0)),
                    times,
                )
                terms = residual_iterator_training_loss(
                    states,
                    residuals,
                    truth,
                    fixed_point_correction=fixed_point_correction,
                    sample_weights=torch.tensor(
                        [
                            family_loss_weights[str(sample["medium_type"])]
                            for sample in samples
                        ],
                        dtype=torch.float32,
                        device=target_device,
                    ),
                    weights=loss_weights,
                )
                loss = terms["total"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite residual-iterator training loss")
                loss.backward()
                _average_gradients(trainable, world_size=world_size)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()

                before_values = _relative_l2(starter, truth)
                after_values = _relative_l2(states[-1].detach(), truth)
                before = before_values.mean()
                after = after_values.mean()
                batch_records = len(samples)
                local_loss_sum += float(loss.detach()) * batch_records
                local_before_sum += float(before) * batch_records
                local_after_sum += float(after) * batch_records
                local_count += batch_records
                for sample_index, sample in enumerate(samples):
                    family = str(sample["medium_type"])
                    local_family_before[family] += float(before_values[sample_index])
                    local_family_after[family] += float(after_values[sample_index])
                    local_family_count[family] += 1

            totals = torch.tensor(
                [
                    local_loss_sum,
                    local_before_sum,
                    local_after_sum,
                    float(local_count),
                    *[local_family_before[family] for family in ALLOWED_MEDIUM_TYPES],
                    *[local_family_after[family] for family in ALLOWED_MEDIUM_TYPES],
                    *[float(local_family_count[family]) for family in ALLOWED_MEDIUM_TYPES],
                ],
                dtype=torch.float64,
                device=target_device,
            )
            if world_size > 1:
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            count_value = max(float(totals[3]), 1.0)
            epoch_loss = float(totals[0] / count_value)
            before_error = float(totals[1] / count_value)
            after_error = float(totals[2] / count_value)
            offset = 4
            family_before = {}
            family_after = {}
            family_draw_count = {}
            registered_family_count = len(ALLOWED_MEDIUM_TYPES)
            for family_index, family in enumerate(ALLOWED_MEDIUM_TYPES):
                family_records = float(
                    totals[offset + 2 * registered_family_count + family_index]
                )
                family_draw_count[family] = int(family_records)
                if family_records > 0.0:
                    family_before[family] = float(
                        totals[offset + family_index] / family_records
                    )
                    family_after[family] = float(
                        totals[offset + registered_family_count + family_index]
                        / family_records
                    )
                else:
                    family_before[family] = None
                    family_after[family] = None
            if after_error < best_score:
                best_score = after_error
                best_epoch = epoch + 1
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in corrector.state_dict().items()
                }
            event = {
                "event": "epoch_complete",
                "epoch": epoch + 1,
                "epochs": int(epochs),
                "mean_loss": epoch_loss,
                "mean_relative_l2_before": before_error,
                "mean_relative_l2_after_unroll": after_error,
                "best_epoch": best_epoch,
                "best_train_crop_relative_l2": best_score,
                "elapsed_seconds": time.perf_counter() - started,
                "world_size": world_size,
                "per_rank_batch_size": rank_batch_size,
                "global_batch_size": rank_batch_size * world_size,
                "family_loss_weights": family_loss_weights,
                "family_sampling_weights": family_sampling_weights,
                "sampled_family_count": family_draw_count,
                "family_relative_l2_before": family_before,
                "family_relative_l2_after_unroll": family_after,
            }
            if rank == 0:
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, sort_keys=True) + "\n")
                print(json.dumps(event, sort_keys=True), flush=True)
                torch.save(
                    {
                        "schema_version": RESIDUAL_ITERATOR_SCHEMA_VERSION,
                        "corrector_state": corrector.state_dict(),
                        "epoch": epoch + 1,
                        "event": event,
                    },
                    output / "latest.pt",
                )

        if best_state is None:
            raise RuntimeError("residual iterator produced no checkpoint state")
        corrector.load_state_dict(best_state, strict=True)
        parent_path = Path(str(config["parent_checkpoint"])).resolve()
        result_path = output / "best_residual_iterator.pt"
        if rank == 0:
            torch.save(
                {
                    "schema_version": RESIDUAL_ITERATOR_SCHEMA_VERSION,
                    "corrector_state": corrector.state_dict(),
                    "corrector_width": int(width),
                    "iterations": int(iterations),
                    "time_window": window,
                    "patch_size": patch,
                    "parent_checkpoint": str(parent_path),
                    "parent_checkpoint_sha256": _sha256(parent_path),
                    "source_h5": str(source_h5.resolve()),
                    "source_h5_sha256": _sha256(source_h5),
                    "manifest_digest": manifest.digest,
                    "future_truth_used_only_for_train_crops": True,
                    "source_free_training_crops": True,
                    "episodes": len(global_episodes),
                    "per_family": family_count,
                    "training_sample_ids": tuple(
                        metadata.sample_id for metadata in global_episodes
                    ),
                    "background_cache": (
                        None
                        if not config.get("background_cache")
                        else str(Path(str(config["background_cache"])).resolve())
                    ),
                    "epochs": int(epochs),
                    "world_size": world_size,
                    "per_rank_batch_size": rank_batch_size,
                    "global_batch_size": rank_batch_size * world_size,
                    "family_loss_weights": family_loss_weights,
                    "family_sampling_weights": family_sampling_weights,
                    "best_epoch": best_epoch,
                    "best_train_crop_relative_l2": best_score,
                    "peak_cuda_bytes": (
                        int(torch.cuda.max_memory_allocated(target_device))
                        if target_device.type == "cuda" else 0
                    ),
                },
                result_path,
            )
        if world_size > 1:
            dist.barrier()
        return result_path
    finally:
        dataset.close()
        if background_provider is not None:
            background_provider.close()
        if initialized_here:
            dist.destroy_process_group()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--parent-run-identity")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--per-family", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--time-window", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--per-rank-batch-size", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        payload = vars(args).copy()
        payload["config_exists"] = Path(args.config).is_file()
        payload["parent_checkpoint_exists"] = (
            None if args.parent_checkpoint is None else Path(args.parent_checkpoint).is_file()
        )
        payload["parent_run_identity_exists"] = (
            None if args.parent_run_identity is None else Path(args.parent_run_identity).is_file()
        )
        print(json.dumps(payload, sort_keys=True))
        return 0
    train_residual_iterator(
        args.config,
        output_dir=args.output_dir,
        parent_checkpoint=args.parent_checkpoint,
        parent_run_identity=args.parent_run_identity,
        device=args.device,
        per_family=args.per_family,
        epochs=args.epochs,
        time_window=args.time_window,
        patch_size=args.patch_size,
        iterations=args.iterations,
        width=args.width,
        learning_rate=args.learning_rate,
        per_rank_batch_size=args.per_rank_batch_size,
        smoke=args.smoke,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
